/* libdetour v0.2: int3-brokered patching, mprotect, trampoline allocation,
 * and ELF symbol lookup.
 *
 * The int3-brokered sequence is the de facto standard for cross-thread-
 * safe code patching on x86, used by the Linux kernel's text_poke_bp()
 * and by ftrace. Steps:
 *
 *   1. Atomically write 0xCC over byte 0 of the target. 1-byte aligned
 *      stores are atomic on x86.
 *   2. Synchronize (membarrier IPI to all cores running threads of this
 *      process; sched_yield fallback if membarrier returns EINVAL).
 *   3. Write the remaining 13 patch bytes. Non-atomic memcpy is safe
 *      because any thread fetching byte 0 sees int3 and traps before
 *      reaching bytes 1..13.
 *   4. Atomically write the new byte 0 (0xFF for the FF 25 patch form).
 *   5. Synchronize again.
 *
 * A core executing in the patched function during install traps to int3.
 * v0.2 does not install a SIGTRAP handler to emulate the not-yet-installed
 * prologue; that thread receives SIGTRAP. Threads not executing in the
 * target are unaffected. This is the documented v0.2 limitation.
 */

#include "detour_internal.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/membarrier.h>
#include <link.h>
#include <elf.h>
#include <errno.h>

/* Trampoline allocation */

void *detour_alloc_trampoline(void)
{
    size_t sz = (size_t)sysconf(_SC_PAGESIZE);
    if (sz < DETOUR_TRAMP_SIZE) sz = DETOUR_TRAMP_SIZE;

    /* MAP_32BIT first: keeps the trampoline within +/-2GB of low-memory
     * targets so RIP-relative displacements fit in 32 bits. */
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (p == MAP_FAILED) {
        /* Fall back to no hint (kernel picks the address). */
        p = mmap(NULL, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return NULL;
    }
    return p;
}

/* Allocate a trampoline near `hint`. Used by detour_create's retry path
 * when MAP_32BIT places the trampoline too far from a high-address
 * (PIE) target's RIP-relative data: passing target_fn +/-1GB as the
 * hint biases the kernel's placement into the +/-2GB window required
 * for RIP-relative displacement encoding. The kernel treats `hint` as
 * advisory (no MAP_FIXED); on conflict it picks another address, which
 * the caller must revalidate via build_trampoline. Returns NULL on
 * mmap failure. */
void *detour_alloc_trampoline_near(void *hint)
{
    size_t sz = (size_t)sysconf(_SC_PAGESIZE);
    if (sz < DETOUR_TRAMP_SIZE) sz = DETOUR_TRAMP_SIZE;

    void *p = mmap(hint, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
}

void detour_free_trampoline(void *trampoline)
{
    if (!trampoline) return;
    size_t sz = (size_t)sysconf(_SC_PAGESIZE);
    if (sz < DETOUR_TRAMP_SIZE) sz = DETOUR_TRAMP_SIZE;
    munmap(trampoline, sz);
}

/* mprotect helpers */

static int protect_range(void *addr, size_t len, int prot)
{
    long pagesize = sysconf(_SC_PAGESIZE);
    if (pagesize <= 0) return DETOUR_ERR_PROTECT_FAILED;

    uintptr_t start = (uintptr_t)addr;
    uintptr_t page_base = start & ~((uintptr_t)pagesize - 1);
    uintptr_t end = start + len;
    uintptr_t page_end = (end + pagesize - 1) & ~((uintptr_t)pagesize - 1);
    size_t span = page_end - page_base;

    if (mprotect((void *)page_base, span, prot) != 0)
        return DETOUR_ERR_PROTECT_FAILED;
    return DETOUR_OK;
}

int detour_make_writable(void *addr, size_t len)
{
    return protect_range(addr, len, PROT_READ | PROT_WRITE | PROT_EXEC);
}

int detour_make_executable(void *addr, size_t len)
{
    return protect_range(addr, len, PROT_READ | PROT_EXEC);
}

/* Cross-core synchronization */

void detour_sync_cores(void)
{
    /* membarrier(MEMBARRIER_CMD_GLOBAL) issues an IPI to all cores
     * running threads of this process. On x86, interrupt delivery is a
     * serializing event for the instruction stream, which is what SDM
     * Vol 3A 8.1.3 requires for cross-modifying code. */
    long r = syscall(SYS_membarrier, MEMBARRIER_CMD_GLOBAL, 0);
    if (r != 0) {
        /* Fallback for kernels without membarrier. sched_yield does not
         * synchronize other cores, but it gives the scheduler a chance
         * to migrate. This is a degraded mode. */
        (void)sched_yield();
    }
}

/* int3-brokered patch */

int detour_patch(void *target, const uint8_t *new_bytes, size_t len)
{
    volatile uint8_t *t = (volatile uint8_t *)target;

    /* Step 1: atomic 1-byte write of 0xCC. */
    t[0] = 0xCC;

    /* Step 2: synchronize. */
    detour_sync_cores();

    /* Step 3: write bytes 1..len-1. Non-atomic; safe because any thread
     * fetching byte 0 traps on int3 before reaching these bytes. */
    if (len > 1) {
        memcpy((void *)(t + 1), new_bytes + 1, len - 1);
    }

    /* Step 4: atomic 1-byte write of the new first byte. */
    t[0] = new_bytes[0];

    /* Step 5: synchronize. */
    detour_sync_cores();

    /* No-op on x86_64; marks intent for the ARM64 port. */
    __builtin___clear_cache((char *)target, (char *)target + len);

    return DETOUR_OK;
}

/* ELF symbol size lookup via dl_iterate_phdr */

struct lookup_ctx {
    uintptr_t target;
    size_t    found_size;
};

/* Walk the in-memory .dynsym of one ELF object to find a function
 * symbol whose address matches ctx->target. Returns 1 to stop
 * dl_iterate_phdr if found. */
static int scan_dynsym(uintptr_t base, const ElfW(Dyn) *dyn,
                       uintptr_t target, size_t *out_size)
{
    ElfW(Sym)  *symtab = NULL;
    const char *strtab = NULL;
    ElfW(Word) *hash     = NULL;
    ElfW(Word) *gnu_hash = NULL;

    for (const ElfW(Dyn) *d = dyn; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
            case DT_SYMTAB:   symtab    = (ElfW(Sym) *)d->d_un.d_ptr; break;
            case DT_STRTAB:   strtab    = (const char *)d->d_un.d_ptr; break;
            case DT_HASH:     hash      = (ElfW(Word) *)d->d_un.d_ptr; break;
            case DT_GNU_HASH: gnu_hash  = (ElfW(Word) *)d->d_un.d_ptr; break;
            default: break;
        }
    }
    if (!symtab) return 0;

    /* Determine the number of symbols in .dynsym. DT_HASH gives it
     * directly (nchain). DT_GNU_HASH requires walking the buckets and
     * chains to find the highest symbol index. If neither is present,
     * we cannot safely iterate. */
    ElfW(Word) nsyms = 0;
    if (hash) {
        nsyms = hash[1];
    } else if (gnu_hash) {
        uint32_t nbuckets   = ((uint32_t *)gnu_hash)[0];
        uint32_t symoffset  = ((uint32_t *)gnu_hash)[1];
        uint32_t bloom_size = ((uint32_t *)gnu_hash)[2];
        /* bloom_shift at [3]; not needed for counting. */
        const uint64_t *bloom   = (const uint64_t *)&gnu_hash[4];
        const uint32_t *buckets = (const uint32_t *)&bloom[bloom_size];
        const uint32_t *chain   = &buckets[nbuckets];

        uint32_t last = 0;
        for (uint32_t i = 0; i < nbuckets; i++) {
            if (buckets[i] > last) last = buckets[i];
        }
        if (last < symoffset) {
            nsyms = symoffset;
        } else {
            /* Walk the chain from `last` until a chain entry with the
             * low bit set (end of chain marker). */
            while ((chain[last - symoffset] & 1u) == 0) last++;
            nsyms = (ElfW(Word))last + 1;
        }
    } else {
        return 0;  /* no hash table; cannot bound iteration */
    }

    /* Skip [0] (the reserved null symbol). */
    for (ElfW(Word) i = 1; i < nsyms; i++) {
        ElfW(Sym) *s = &symtab[i];
        if (ELF64_ST_TYPE(s->st_info) != STT_FUNC) continue;
        if (s->st_value == 0) continue;
        uintptr_t addr = base + (uintptr_t)s->st_value;
        if (addr == target) {
            *out_size = (size_t)s->st_size;
            return 1;
        }
    }
    return 0;
    (void)strtab;
}

static int lookup_callback(struct dl_phdr_info *info, size_t sz, void *data)
{
    (void)sz;
    struct lookup_ctx *ctx = (struct lookup_ctx *)data;
    uintptr_t target = ctx->target;

    /* Check if target falls within any PT_LOAD segment of this object. */
    int in_module = 0;
    const ElfW(Dyn) *dyn = NULL;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_LOAD) continue;
        uintptr_t start = (uintptr_t)info->dlpi_addr + ph->p_vaddr;
        uintptr_t end   = start + ph->p_memsz;
        if (target >= start && target < end) {
            in_module = 1;
        }
    }
    if (!in_module) return 0;

    /* Find PT_DYNAMIC for this object. */
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type == PT_DYNAMIC) {
            dyn = (const ElfW(Dyn) *)((uintptr_t)info->dlpi_addr + ph->p_vaddr);
            break;
        }
    }
    if (!dyn) return 0;

    size_t size = 0;
    if (scan_dynsym((uintptr_t)info->dlpi_addr, dyn, target, &size)) {
        ctx->found_size = size;
        return 1;  /* stop iterating */
    }
    return 0;
}

size_t detour_lookup_fn_size(void *addr)
{
    struct lookup_ctx ctx;
    ctx.target = (uintptr_t)addr;
    ctx.found_size = 0;
    dl_iterate_phdr(lookup_callback, &ctx);
    return ctx.found_size;
}
