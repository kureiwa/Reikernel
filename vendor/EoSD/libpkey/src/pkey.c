/* libpkey: Intel Memory Protection Keys (MPK) wrapper.
 *
 * Wraps the pkey_alloc(2) / pkey_free(2) / pkey_mprotect(2) syscalls and
 * exposes the WRPKRU/RDPKRU userspace instructions (in src/pkey_x86_64.asm)
 * for ~20-cycle, no-syscall permission toggles on pages tagged with a key.
 *
 * Feature detection: pkey_available() probes CPUID 7:0:ECX[4] (OSPKE),
 * which is set when the OS has enabled CR4.PKE. CPU-only support
 * (CPUID 7:0:ECX[3] PKU) is not sufficient -- the WRPKRU/RDPKRU
 * instructions are no-ops when CR4.PKE=0, and pkey_alloc(2) returns
 * ENOSYS/EINVAL. The task brief's parenthetical "(OSPKE)" matches ECX[4];
 * the bit number [3] in the brief is the PKU (CPU support) bit, not OSPKE.
 * This module checks OSPKE ([4]) because that is what gates userspace
 * usability. See DESIGN.md "OSPKE vs PKU" for the full rationale.
 *
 * Syscall wrappers use raw syscall(SYS_pkey_*) rather than the glibc
 * wrappers (glibc 2.27+, <sys/mman.h> under _GNU_SOURCE). The glibc
 * wrappers have the same names as this module's public API but different
 * signatures (e.g. glibc pkey_alloc takes (flags, access_rights); ours
 * takes (void)). Including <sys/mman.h> under _GNU_SOURCE would conflict,
 * so pkey.c does NOT include <sys/mman.h> -- it forwards the caller's
 * `prot` argument opaquely to the syscall. The SYS_pkey_* numbers (329/
 * 330/331 on x86_64) are in <sys/syscall.h> on kernels 4.6+; fallback
 * #defines cover older glibc headers.
 *
 * _GNU_SOURCE is defined so <unistd.h> declares syscall(); pkey.c does
 * not include <sys/mman.h>, so glibc's pkey_alloc/pkey_free/pkey_mprotect
 * declarations are never pulled in and there is no signature clash.
 *
 * x86_64 only. On any other architecture the asm helpers are absent and
 * the build fails at link time (intentional -- MPK is x86_64-only). */

#define _GNU_SOURCE
#include "pkey.h"

#include <stdint.h>
#include <unistd.h>
#include <sys/syscall.h>

/* x86_64 asm helpers. Declared here, defined in src/pkey_x86_64.asm.
 * On non-x86_64 the build does not link these; libpkey is x86_64-only. */
extern uint32_t pkey_rdpkru(void);
extern void     pkey_wrpkru(uint32_t pkru);
extern void     pkey_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);

/* Fallback syscall numbers for older glibc headers (< 2.27) that may not
 * define SYS_pkey_*. The numbers are stable on x86_64 since Linux 4.6
 * (arch/x86/entry/syscalls/syscall_64.tbl: 329/330/331). */
#ifndef SYS_pkey_mprotect
#define SYS_pkey_mprotect 329
#endif
#ifndef SYS_pkey_alloc
#define SYS_pkey_alloc 330
#endif
#ifndef SYS_pkey_free
#define SYS_pkey_free 331
#endif

/* Number of hardware protection keys. PKRU is 32 bits, 2 bits per key. */
#define PKEY_NUM_KEYS 4

/* PKRU bit layout: 2 bits per key, AD at bit (pkey*2), WD at bit (pkey*2+1).
 * AD (access disable): no read or write. WD (write disable): read only.
 * AD dominates WD. Bits [31:8] are reserved (must be 0; WRPKRU ignores
 * them but RDPKRU returns them as 0). */
#define PKEY_AD_BIT(pkey) (1u << ((pkey) * 2u))
#define PKEY_WD_BIT(pkey) (2u << ((pkey) * 2u))
#define PKEY_PAIR_MASK(pkey) (3u << ((pkey) * 2u))

int pkey_available(void)
{
    uint32_t regs[4] = {0, 0, 0, 0};
    /* CPUID leaf 7, subleaf 0. ECX[3] = PKU (CPU support), ECX[4] = OSPKE
     * (OS has set CR4.PKE). We test OSPKE because it is the gating bit
     * for userspace WRPKRU/RDPKRU usability. PKU=1 && OSPKE=0 means the
     * CPU supports MPK but the OS has not enabled it; the instructions
     * execute as no-ops and pkey_alloc(2) returns ENOSYS. */
    pkey_cpuid(0x7u, 0x0u, regs);
    return (int)((regs[2] >> 4) & 1u);
}

int pkey_alloc(void)
{
    long rc;

    if (!pkey_available()) {
        return PKEY_ERR_UNSUPPORTED;
    }

    /* pkey_alloc(flags=0, access_rights=0). flags=0 means no restrictions
     * on the key (PKEY_DISABLE_ACCESS / PKEY_DISABLE_WRITE could be passed
     * to pre-disable, but we expose that via pkey_set_access instead).
     * Returns a key number >= 0, or -1 with errno set. */
    rc = syscall(SYS_pkey_alloc, 0u, 0u);
    if (rc < 0) {
        /* EINVAL (bad flags, or OSPKE not actually set despite CPUID) or
         * ENOSPC (no free keys; max 4 keys, key 0 reserved for default).
         * Both collapse to PKEY_ERR_ALLOC. */
        return PKEY_ERR_ALLOC;
    }
    return (int)rc;
}

int pkey_free(int pkey)
{
    long rc;

    if (!pkey_available()) {
        return PKEY_ERR_UNSUPPORTED;
    }
    if (pkey < 0 || pkey >= PKEY_NUM_KEYS) {
        return PKEY_ERR_INVALID;
    }

    rc = syscall(SYS_pkey_free, pkey);
    if (rc < 0) {
        /* EINVAL (bad pkey number, or pkey was not allocated). */
        return PKEY_ERR_INVALID;
    }
    return PKEY_OK;
}

int pkey_mprotect(void *addr, size_t len, int prot, int pkey)
{
    long rc;

    if (!pkey_available()) {
        return PKEY_ERR_UNSUPPORTED;
    }
    if (addr == NULL) {
        return PKEY_ERR_INVALID;
    }
    if (pkey < 0 || pkey >= PKEY_NUM_KEYS) {
        return PKEY_ERR_INVALID;
    }

    /* pkey_mprotect(addr, len, prot, pkey). Identical to mprotect except
     * it also tags the affected PTEs with `pkey`. addr must be page-aligned;
     * len is rounded up to a page multiple by the kernel. */
    rc = syscall(SYS_pkey_mprotect, addr, len, prot, pkey);
    if (rc < 0) {
        /* EINVAL (bad addr/prot/pkey), ENOMEM (VMA splitting failure),
         * EACCES (prot conflict), EAGAIN (temporarily unavailable). */
        return PKEY_ERR_MPROTECT;
    }
    return PKEY_OK;
}

int pkey_set_access(int pkey, int access_disable, int write_disable)
{
    uint32_t pkru;
    uint32_t bits;

    if (!pkey_available()) {
        return PKEY_ERR_UNSUPPORTED;
    }
    if (pkey < 0 || pkey >= PKEY_NUM_KEYS) {
        return PKEY_ERR_INVALID;
    }

    /* Read-modify-write PKRU: clear the 2 bits for this key, then set the
     * requested AD/WD. This preserves the bits for other keys (so a caller
     * managing multiple keys does not clobber them). */
    pkru = pkey_rdpkru();
    pkru &= ~PKEY_PAIR_MASK(pkey);

    bits = 0;
    if (access_disable) {
        bits |= PKEY_AD_BIT(pkey);
    }
    if (write_disable) {
        bits |= PKEY_WD_BIT(pkey);
    }
    pkru |= bits;

    pkey_wrpkru(pkru);
    return PKEY_OK;
}

int pkey_get_access(int pkey)
{
    uint32_t pkru;
    uint32_t ad, wd;

    if (!pkey_available()) {
        return PKEY_ERR_UNSUPPORTED;
    }
    if (pkey < 0 || pkey >= PKEY_NUM_KEYS) {
        return PKEY_ERR_INVALID;
    }

    pkru = pkey_rdpkru();
    ad = (pkru >> (pkey * 2u)) & 1u;
    wd = (pkru >> (pkey * 2u + 1u)) & 1u;

    /* AD dominates: if AD is set, the key has no access regardless of WD. */
    if (ad) {
        return 2;
    }
    if (wd) {
        return 1;
    }
    return 0;
}

int pkey_allow(int pkey)
{
    /* Clear both AD and WD -> full access. */
    return pkey_set_access(pkey, 0, 0);
}

int pkey_deny(int pkey)
{
    /* Set AD, clear WD -> no access (AD dominates, so WD is cleared for
     * canonical state; pkey_get_access returns 2 either way). */
    return pkey_set_access(pkey, 1, 0);
}

int pkey_readonly(int pkey)
{
    /* Set WD, clear AD -> read but not write. */
    return pkey_set_access(pkey, 0, 1);
}
