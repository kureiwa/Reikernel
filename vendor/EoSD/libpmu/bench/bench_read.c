/* Measures pmu_read latency on a cycles counter fd over 1M reads and prints
 * ns/op.
 *
 * v0.3 note: pmu_open no longer returns NULL on PMU_ERR_PERM; it returns
 * a non-NULL dummy ctx. This bench detects the dummy via pmu_is_available
 * and reports the dummy-path cost (single-digit ns) instead of the
 * syscall-round-trip cost. On a real ctx, the expected range is
 * ~100-300 ns/op for the read(2) fallback; bench_rdpmc measures the
 * rdpmc fast path when the kernel exposes it.
 */

#define _POSIX_C_SOURCE 199309L   /* clock_gettime + CLOCK_MONOTONIC under -std=c11 */

#include "pmu.h"

#include <stdio.h>
#include <time.h>

#define ITERS 1000000ULL

int main(void)
{
    pmu_err_t err = PMU_OK;
    pmu_ctx_t *ctx = pmu_open(PMU_CYCLES, &err);
    if (!ctx) {
        printf("FAIL bench_read: pmu_open returned err=%d\n", (int)err);
        return 1;
    }

    if (pmu_start(ctx) != PMU_OK) {
        printf("FAIL bench_read: pmu_start failed\n");
        pmu_close(ctx);
        return 1;
    }

    int dummy = !pmu_is_available(ctx);

    /* Warm up: first read may take a cold-cache hit. */
    uint64_t v = 0;
    for (int i = 0; i < 1024; i++) {
        pmu_read(ctx, &v);
    }

    struct timespec t0, t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0) {
        printf("FAIL bench_read: clock_gettime failed\n");
        pmu_close(ctx);
        return 1;
    }

    for (uint64_t i = 0; i < ITERS; i++) {
        pmu_read(ctx, &v);
    }

    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
        printf("FAIL bench_read: clock_gettime failed\n");
        pmu_close(ctx);
        return 1;
    }

    uint64_t ns0 = (uint64_t)t0.tv_sec * 1000000000ULL + (uint64_t)t0.tv_nsec;
    uint64_t ns1 = (uint64_t)t1.tv_sec * 1000000000ULL + (uint64_t)t1.tv_nsec;
    double elapsed_ns = (double)(ns1 - ns0);
    double ns_per_op  = elapsed_ns / (double)ITERS;

    printf("bench_read: %llu pmu_read calls in %.0f ns\n",
           (unsigned long long)ITERS, elapsed_ns);
    printf("bench_read: %.1f ns/op (final counter value=%llu)\n",
           ns_per_op, (unsigned long long)v);
    if (dummy) {
        printf("bench_read: ctx is DUMMY (perf_event_open denied, "
               "PMU_ERR_PERM=%d)\n", err == PMU_ERR_PERM);
        printf("bench_read: expected ~0 ns/op on dummy (no-op path); "
               "see bench_dummy for details\n");
        printf("bench_read:   try: sudo sysctl -w kernel.perf_event_paranoid=1\n");
    } else {
        printf("bench_read: expected ~100-300 ns/op (read(2) syscall "
               "round-trip); ~6-13 ns/op if rdpmc fast path is active\n");
    }

    pmu_close(ctx);
    return 0;
}
