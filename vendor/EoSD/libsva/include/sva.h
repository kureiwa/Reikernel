#ifndef SVA_H
#define SVA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sva_region sva_region_t;

typedef enum {
    SVA_PROT_READ       = 1 << 0,
    SVA_PROT_WRITE      = 1 << 1,
    SVA_PROT_EXEC       = 1 << 2,
    /* v0.2: install a PROT_NONE guard page *before* the usable region in
     * addition to the overflow guard after it. Layout when set:
     *   [guard page][usable region][guard page]
     * Without the flag, only the overflow guard (after the region) is
     * installed, matching v0.1 behavior. */
    SVA_PROT_GUARD_BOTH = 1 << 3,
    /* v0.2: back the usable region with MAP_HUGETLB. The usable size is
     * rounded up to 2 MB and the guard page(s) are 2 MB-sized and
     * 2 MB-aligned (a 4 KB guard before a 2 MB huge page is useless
     * because the huge page's alignment requirement would not let the
     * two pack contiguously without reserving a full 2 MB slot for the
     * guard). */
    SVA_PROT_HUGETLB    = 1 << 4,
} sva_prot_flags_t;

typedef struct {
    void *(*alloc)(size_t size, void *user_data);
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} sva_allocator_t;

typedef enum {
    SVA_OK                 = 0,
    SVA_ERR_INVALID        = -1,
    SVA_ERR_MMAP_FAILED    = -2,
    SVA_ERR_PROTECT_FAILED = -3,
    SVA_ERR_EXEC_DENIED    = -4,
} sva_err_t;

/* Maps `size` bytes (rounded up to page size) with the given protection
 * flags, followed immediately by one PROT_NONE guard page.
 *
 * `allocator` is for libsva's internal bookkeeping structures (the
 * sva_region_t handle and any per-region metadata). NULL = libc malloc.
 * The mmap'd region itself is NEVER routed through this allocator.
 *
 * Return / error contract (callers MUST check both):
 *   non-NULL return + *out_err == SVA_OK                -> full success
 *   non-NULL return + *out_err == SVA_ERR_EXEC_DENIED   -> success-with-warning (exec was silently downgraded)
 *   NULL return    + *out_err < 0                       -> hard failure
 *
 * Thread-safety: safe to call concurrently from multiple threads; each
 * call operates on its own mmap'd region and its own bookkeeping
 * allocation. No internal shared state.
 */
sva_region_t *sva_map_guarded(size_t size, sva_prot_flags_t flags,
                              const sva_allocator_t *allocator,
                              sva_err_t *out_err);

/* Unmaps the region (usable bytes plus the guard page) and frees the
 * sva_region_t bookkeeping struct via the allocator recorded at map time.
 * Passing NULL is a no-op.
 *
 * Thread-safety: safe only when no other thread is concurrently accessing
 * the same region. The caller owns synchronization.
 */
void sva_unmap(sva_region_t *region);

/* Returns the usable base pointer. The guard page is not included and is
 * not caller-addressable; touching it is the overflow condition this
 * module exists to detect. Returns NULL if region is NULL.
 *
 * Thread-safety: safe to call concurrently with any other read of the same
 * region. Not safe to call concurrently with sva_unmap on the same region.
 */
void *sva_base(const sva_region_t *region);

/* Returns the usable size (excludes the guard page). Returns 0 if region
 * is NULL.
 *
 * Thread-safety: same as sva_base.
 */
size_t sva_size(const sva_region_t *region);

/* Returns the address of the guard page itself. Mainly useful for a
 * caller's own SIGSEGV handler to check whether a fault came from one of
 * its guarded regions by address-range comparison. Returns NULL if region
 * is NULL.
 *
 * Thread-safety: same as sva_base.
 */
void *sva_guard_page_addr(const sva_region_t *region);

/* v0.2: returns the address of the underflow guard page (the PROT_NONE
 * page installed immediately *before* the usable base when the region was
 * created with SVA_PROT_GUARD_BOTH). Returns NULL if the region was not
 * created with SVA_PROT_GUARD_BOTH, or if region is NULL.
 *
 * Thread-safety: same as sva_base.
 */
void *sva_underflow_guard_addr(const sva_region_t *region);

/* Flushes the TLB for the given region's usable range. On Linux this is
 * implemented as mprotect(base, size, PROT_NONE) followed by
 * mprotect(base, size, original_prot); both calls trigger
 * flush_tlb_range in the kernel. invlpg is privileged (ring 0) and is
 * never used. Cost is two syscalls plus the kernel-side shootdown.
 *
 * Most callers do NOT need this: mprotect, munmap, and mremap already
 * flush the TLB as a side effect.
 *
 * Returns 0 on success, negative on error.
 *
 * Thread-safety: safe to call concurrently with reads of the same region.
 * Concurrent writes from other threads during the PROT_NONE window will
 * fault; callers must coordinate if in-flight writes are expected.
 */
int sva_flush_tlb(sva_region_t *region);

#ifdef __cplusplus
}
#endif

#endif /* SVA_H */
