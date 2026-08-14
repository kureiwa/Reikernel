#include "sva.h"

#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define ITERATIONS 1000

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
    uint64_t start, end;
    uint64_t total_ns;

    start = now_ns();
    for (int i = 0; i < ITERATIONS; i++) {
        err = SVA_OK;
        sva_region_t *r = sva_map_guarded(4096,
                                          SVA_PROT_READ | SVA_PROT_WRITE,
                                          NULL, &err);
        if (r == NULL || err != SVA_OK) {
            fprintf(stderr, "bench_map: iter %d failed err=%d\n", i, err);
            if (r != NULL) sva_unmap(r);
            return 1;
        }
        sva_unmap(r);
    }
    end = now_ns();

    total_ns = end - start;
    double ns_per_op = (double)total_ns / (double)ITERATIONS;
    printf("bench_map: %d iterations, %" PRIu64 " ns total, "
           "%.1f ns/op\n",
           ITERATIONS, total_ns, ns_per_op);
    return 0;
}
