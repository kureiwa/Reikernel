#include "sva.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* v0.2 huge-page bench.
 *
 * Compares map/unmap latency for:
 *   - 4 KB region, default (4 KB) guard page
 *   - 2 MB region, SVA_PROT_HUGETLB, 2 MB guard page
 *
 * The huge-page portion is skipped (exit 0 after the regular portion)
 * when /proc/sys/vm/nr_hugepages == 0 or when the first huge-page map
 * fails at runtime.
 */

#define ITERATIONS 200
#define SVA_HUGEPAGE_SIZE (2u * 1024u * 1024u)

static uint64_t now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int huge_pages_in_pool(void)
{
    FILE *f = fopen("/proc/sys/vm/nr_hugepages", "r");
    long n = 0;
    if (f == NULL) {
        return 0;
    }
    if (fscanf(f, "%ld", &n) != 1) {
        n = 0;
    }
    fclose(f);
    return n > 0;
}

int main(void)
{
    sva_err_t err;
    uint64_t start, end;
    uint64_t total_ns_regular, total_ns_huge;

    /* Regular 4 KB map/unmap. */
    start = now_ns();
    for (int i = 0; i < ITERATIONS; i++) {
        err = SVA_OK;
        sva_region_t *r = sva_map_guarded(4096,
                                          SVA_PROT_READ | SVA_PROT_WRITE,
                                          NULL, &err);
        if (r == NULL || err != SVA_OK) {
            fprintf(stderr, "bench_hugetlb: regular iter %d failed err=%d\n",
                    i, err);
            if (r != NULL) sva_unmap(r);
            return 1;
        }
        sva_unmap(r);
    }
    end = now_ns();
    total_ns_regular = end - start;

    printf("bench_hugetlb: regular 4KB map/unmap: %d iters, "
           "%" PRIu64 " ns total, %.1f ns/op\n",
           ITERATIONS, total_ns_regular,
           (double)total_ns_regular / (double)ITERATIONS);

    if (!huge_pages_in_pool()) {
        printf("bench_hugetlb: SKIP huge-page bench (nr_hugepages == 0)\n");
        return 0;
    }

    /* Probe with a single map first so we can SKIP cleanly if the kernel
     * refuses to hand out a huge page despite a non-zero nr_hugepages
     * (e.g. the pool was drained between the check and the map). */
    err = SVA_OK;
    sva_region_t *probe = sva_map_guarded(SVA_HUGEPAGE_SIZE,
                                          SVA_PROT_READ | SVA_PROT_WRITE |
                                          SVA_PROT_HUGETLB,
                                          NULL, &err);
    if (probe == NULL || err != SVA_OK) {
        printf("bench_hugetlb: SKIP huge-page bench (probe map failed "
               "err=%d)\n", err);
        if (probe != NULL) sva_unmap(probe);
        return 0;
    }
    sva_unmap(probe);

    /* 2 MB huge-page map/unmap. */
    start = now_ns();
    for (int i = 0; i < ITERATIONS; i++) {
        err = SVA_OK;
        sva_region_t *r = sva_map_guarded(SVA_HUGEPAGE_SIZE,
                                          SVA_PROT_READ | SVA_PROT_WRITE |
                                          SVA_PROT_HUGETLB,
                                          NULL, &err);
        if (r == NULL || err != SVA_OK) {
            /* Bail out mid-bench rather than fail; huge-page exhaustion
             * during the run is an environment issue, not a code bug. */
            printf("bench_hugetlb: SKIP huge-page bench after %d iters "
                   "(err=%d)\n", i, err);
            if (r != NULL) sva_unmap(r);
            return 0;
        }
        sva_unmap(r);
    }
    end = now_ns();
    total_ns_huge = end - start;

    printf("bench_hugetlb: huge 2MB map/unmap:   %d iters, "
           "%" PRIu64 " ns total, %.1f ns/op\n",
           ITERATIONS, total_ns_huge,
           (double)total_ns_huge / (double)ITERATIONS);

    double ratio = (double)total_ns_huge / (double)total_ns_regular;
    printf("bench_hugetlb: huge/regular ratio = %.2fx\n", ratio);
    return 0;
}
