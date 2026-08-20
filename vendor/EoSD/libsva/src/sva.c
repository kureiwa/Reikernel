#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "sva.h"

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

/* x86_64 default huge page size. Read from /proc/meminfo at runtime would
 * be more flexible but the API.md spec fixes this at 2 MB for v0.2. */
#define SVA_HUGEPAGE_SIZE ((size_t)(2u * 1024u * 1024u))

struct sva_region {
    void *mmap_addr;          /* raw mmap base, for munmap */
    void *usable_base;        /* returned by sva_base */
    size_t usable_size;       /* returned by sva_size */
    size_t total_size;        /* full mmap length, for munmap */
    void *underflow_guard;    /* NULL when GUARD_BOTH not set */
    void *overflow_guard;     /* always non-NULL on a valid region */
    int prot;
    sva_allocator_t allocator;
};

static void *sva_default_alloc(size_t size, void *user_data)
{
    (void)user_data;
    return malloc(size);
}

static void sva_default_free(void *ptr, size_t size, void *user_data)
{
    (void)size;
    (void)user_data;
    free(ptr);
}

static int sva_to_prot(sva_prot_flags_t flags)
{
    int prot = 0;
    /* SVA_PROT_GUARD_BOTH and SVA_PROT_HUGETLB are behavior flags, not
     * PROT_* bits; they are handled separately in sva_map_guarded. */
    if (flags & SVA_PROT_READ)  prot |= PROT_READ;
    if (flags & SVA_PROT_WRITE) prot |= PROT_WRITE;
    if (flags & SVA_PROT_EXEC)  prot |= PROT_EXEC;
    return prot;
}

static size_t sva_round_up(size_t v, size_t m)
{
    return (v + m - 1u) & ~(m - 1u);
}

sva_region_t *sva_map_guarded(size_t size, sva_prot_flags_t flags,
                              const sva_allocator_t *allocator,
                              sva_err_t *out_err)
{
    sva_region_t *region;
    long page_size;
    size_t total_size;
    size_t usable_size;
    size_t guard_size;
    size_t guard_count;
    size_t align;
    void *raw;
    void *usable_base;
    void *underflow_guard;
    void *overflow_guard;
    int prot;
    int mmap_flags;
    int want_both;
    int want_huge;
    sva_allocator_t resolved;

    if (out_err != NULL) {
        *out_err = SVA_OK;
    }

    if (size == 0) {
        if (out_err != NULL) {
            *out_err = SVA_ERR_INVALID;
        }
        return NULL;
    }

    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        if (out_err != NULL) {
            *out_err = SVA_ERR_MMAP_FAILED;
        }
        return NULL;
    }

    want_both = (flags & SVA_PROT_GUARD_BOTH) ? 1 : 0;
    want_huge = (flags & SVA_PROT_HUGETLB) ? 1 : 0;

    /* For HUGETLB the usable size is rounded to the huge page size and
     * each guard slot is a full huge page (see SVA_PROT_HUGETLB comment
     * in sva.h). Otherwise the v0.1 page-size rounding applies. */
    if (want_huge) {
        guard_size = SVA_HUGEPAGE_SIZE;
        align = SVA_HUGEPAGE_SIZE;
    } else {
        guard_size = (size_t)page_size;
        align = (size_t)page_size;
    }

    /* Overflow guard on the rounding itself: sva_round_up(v,m) computes
     * (v+m-1)&~(m-1), which silently wraps to 0 when v is within m-1 of
     * SIZE_MAX. A caller passing a near-SIZE_MAX size would otherwise slip
     * past the size==0 rejection above and get a degenerate 0-byte region.
     * Reject before rounding so the wrap is impossible. */
    if (size > SIZE_MAX - (align - 1)) {
        if (out_err != NULL) {
            *out_err = SVA_ERR_INVALID;
        }
        return NULL;
    }
    usable_size = sva_round_up(size, align);

    guard_count = want_both ? 2u : 1u;

    /* Overflow check: usable_size + guard_count * guard_size must not wrap. */
    if (usable_size > SIZE_MAX - guard_count * guard_size) {
        if (out_err != NULL) {
            *out_err = SVA_ERR_INVALID;
        }
        return NULL;
    }
    total_size = usable_size + guard_count * guard_size;

    prot = sva_to_prot(flags);

    /* Resolve allocator: NULL pointer or NULL fields -> libc defaults.
     * The mmap'd region itself is never routed through this allocator. */
    if (allocator != NULL) {
        resolved = *allocator;
    } else {
        resolved.alloc = NULL;
        resolved.free = NULL;
        resolved.user_data = NULL;
    }
    if (resolved.alloc == NULL) {
        resolved.alloc = sva_default_alloc;
    }
    if (resolved.free == NULL) {
        resolved.free = sva_default_free;
    }

    region = (sva_region_t *)resolved.alloc(sizeof(sva_region_t),
                                            resolved.user_data);
    if (region == NULL) {
        if (out_err != NULL) {
            *out_err = SVA_ERR_MMAP_FAILED;
        }
        return NULL;
    }

    mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (want_huge) {
        mmap_flags |= MAP_HUGETLB;
    }

    raw = mmap(NULL, total_size, prot, mmap_flags, -1, 0);
    if (raw == MAP_FAILED) {
        int saved_errno = errno;
        /* W^X fallback: if EXEC was requested and the kernel rejected the
         * mapping with a permission error, retry without PROT_EXEC and
         * report success-with-warning via SVA_ERR_EXEC_DENIED. */
        if ((flags & SVA_PROT_EXEC) &&
            (saved_errno == EACCES || saved_errno == EPERM)) {
            int prot_no_exec = prot & ~PROT_EXEC;
            raw = mmap(NULL, total_size, prot_no_exec, mmap_flags, -1, 0);
            if (raw == MAP_FAILED) {
                resolved.free(region, sizeof(sva_region_t),
                              resolved.user_data);
                if (out_err != NULL) {
                    *out_err = SVA_ERR_MMAP_FAILED;
                }
                return NULL;
            }
            prot = prot_no_exec;
            if (out_err != NULL) {
                *out_err = SVA_ERR_EXEC_DENIED;
            }
        } else {
            resolved.free(region, sizeof(sva_region_t), resolved.user_data);
            if (out_err != NULL) {
                *out_err = SVA_ERR_MMAP_FAILED;
            }
            return NULL;
        }
    }

    /* Install the PROT_NONE guard page(s).
     *
     * Layout:
     *   GUARD_BOTH: [underflow guard][usable region][overflow guard]
     *   default:    [usable region][overflow guard]
     *
     * For HUGETLB each guard slot is a full huge page (2 MB). mprotect on
     * a sub-range of a MAP_HUGETLB VMA splits the VMA at huge-page
     * boundaries; the kernel does not require the PROT_NONE slot to be
     * backed by a huge page reservation, but it does occupy address space
     * within the same mapping.
     */
    if (want_both) {
        underflow_guard = raw;
        usable_base = (char *)raw + guard_size;
        overflow_guard = (char *)raw + guard_size + usable_size;

        if (mprotect(underflow_guard, guard_size, PROT_NONE) != 0) {
            munmap(raw, total_size);
            resolved.free(region, sizeof(sva_region_t), resolved.user_data);
            if (out_err != NULL) {
                *out_err = SVA_ERR_PROTECT_FAILED;
            }
            return NULL;
        }
    } else {
        underflow_guard = NULL;
        usable_base = raw;
        overflow_guard = (char *)raw + usable_size;
    }

    if (mprotect(overflow_guard, guard_size, PROT_NONE) != 0) {
        munmap(raw, total_size);
        resolved.free(region, sizeof(sva_region_t), resolved.user_data);
        if (out_err != NULL) {
            *out_err = SVA_ERR_PROTECT_FAILED;
        }
        return NULL;
    }

    region->mmap_addr = raw;
    region->usable_base = usable_base;
    region->usable_size = usable_size;
    region->total_size = total_size;
    region->underflow_guard = underflow_guard;
    region->overflow_guard = overflow_guard;
    region->prot = prot;
    region->allocator = resolved;

    return region;
}

void sva_unmap(sva_region_t *region)
{
    sva_allocator_t allocator;

    if (region == NULL) {
        return;
    }

    /* Cache the allocator before reading from `region` after free. */
    allocator = region->allocator;

    /* total_size covers the usable region plus all guard pages in a
     * single munmap, regardless of whether GUARD_BOTH / HUGETLB were set. */
    munmap(region->mmap_addr, region->total_size);

    allocator.free(region, sizeof(sva_region_t), allocator.user_data);
}

void *sva_base(const sva_region_t *region)
{
    if (region == NULL) {
        return NULL;
    }
    return region->usable_base;
}

size_t sva_size(const sva_region_t *region)
{
    if (region == NULL) {
        return 0;
    }
    return region->usable_size;
}

void *sva_guard_page_addr(const sva_region_t *region)
{
    if (region == NULL) {
        return NULL;
    }
    return region->overflow_guard;
}

void *sva_underflow_guard_addr(const sva_region_t *region)
{
    if (region == NULL) {
        return NULL;
    }
    return region->underflow_guard;
}

int sva_flush_tlb(sva_region_t *region)
{
    if (region == NULL) {
        return SVA_ERR_INVALID;
    }

    /* mprotect(base, size, PROT_NONE) then mprotect(base, size, original).
     * Both syscalls trigger flush_tlb_range in the kernel as a side effect
     * of changing the VMA protections. invlpg is privileged (ring 0) and
     * is never used. */
    if (mprotect(region->usable_base, region->usable_size, PROT_NONE) != 0) {
        return SVA_ERR_PROTECT_FAILED;
    }
    if (mprotect(region->usable_base, region->usable_size, region->prot) != 0) {
        return SVA_ERR_PROTECT_FAILED;
    }
    return 0;
}
