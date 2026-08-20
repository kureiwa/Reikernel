/* bench_flush_tlb: measure sva_flush_tlb latency.
 *
 * sva_flush_tlb does mprotect(base, size, PROT_NONE) followed by
 * mprotect(base, size, original_prot). Both syscalls trigger
 * flush_tlb_range in the kernel as a side effect of changing the VMA
 * protections. The cost is two syscalls plus the kernel-side shootdown.
 *
 * This bench maps a 4 KB region (default guard, no GUARD_BOTH) and
 * then calls sva_flush_tlb in a tight loop. 10,000 iterations gives
 * a stable median.
 *
 * For comparison, the bench also measures a bare mprotect pair
 * (PROT_NONE then back to PROT_READ|PROT_WRITE) on the same kind of
 * region, so the overhead of sva_flush_tlb's wrapper logic (NULL
 * check, field reads, error returns) can be isolated from the
 * syscall cost itself.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 199309L

#include "sva.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define ITERS 10000u

static uint64_t now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void)
{
    sva_err_t err;
    sva_region_t *r = sva_map_guarded(4096,
                                      SVA_PROT_READ | SVA_PROT_WRITE,
                                      NULL, &err);
    if (r == NULL || err != SVA_OK) {
        fprintf(stderr, "bench_flush_tlb: sva_map_guarded failed err=%d\n",
                err);
        return 1;
    }

    /* Warm up. */
    for (int i = 0; i < 100; i++) {
        if (sva_flush_tlb(r) != 0) {
            fprintf(stderr, "bench_flush_tlb: warmup flush failed\n");
            sva_unmap(r);
            return 1;
        }
    }

    /* Measure sva_flush_tlb. */
    uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        if (sva_flush_tlb(r) != 0) {
            fprintf(stderr, "bench_flush_tlb: iter %u failed\n", i);
            sva_unmap(r);
            return 1;
        }
    }
    uint64_t t1 = now_ns();
    double sva_ns = (double)(t1 - t0) / (double)ITERS;

    /* Measure bare mprotect pair on a separate 4 KB region, for
     * comparison. The bare path does no wrapper logic, no field reads,
     * no error returns -- just the two syscalls. */
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    void *raw = mmap(NULL, (size_t)ps, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        fprintf(stderr, "bench_flush_tlb: bare mmap failed\n");
        sva_unmap(r);
        return 1;
    }

    for (int i = 0; i < 100; i++) {
        mprotect(raw, (size_t)ps, PROT_NONE);
        mprotect(raw, (size_t)ps, PROT_READ | PROT_WRITE);
    }

    t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        mprotect(raw, (size_t)ps, PROT_NONE);
        mprotect(raw, (size_t)ps, PROT_READ | PROT_WRITE);
    }
    t1 = now_ns();
    double bare_ns = (double)(t1 - t0) / (double)ITERS;

    munmap(raw, (size_t)ps);
    sva_unmap(r);

    printf("bench_flush_tlb: %u iterations\n", ITERS);
    printf("bench_flush_tlb: sva_flush_tlb       : %.1f ns/op\n", sva_ns);
    printf("bench_flush_tlb: bare mprotect pair   : %.1f ns/op\n", bare_ns);
    printf("bench_flush_tlb:   wrapper overhead    : %.1f ns/op "
           "(sva - bare)\n", sva_ns - bare_ns);
    printf("bench_flush_tlb:   each call = 2x mprotect(PROT_NONE) + "
           "mprotect(orig) syscalls\n");
    return 0;
}
