/*
 * bench_now: tick_now latency, isolated.
 *
 * Calls tick_now in a tight loop, reports ns/op. The drift auto-check
 * fires every 1024 calls (or every 5 s); the amortized syscall cost is
 * < 1 ns per call, so the reported number is dominated by rdtsc + the
 * Q20 conversion (shl + magic mul + shr) + the counter increment/compare.
 *
 * Expected: 5-20 ns/op on modern x86_64. The pre-magic-divide baseline
 * was ~25-30 ns/op (the DIVQ dominated).
 */
#define _POSIX_C_SOURCE 200112L

#include <tick.h>

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define NITERS 10000000

int main(void) {
    tick_ctx_t *ctx = tick_ctx_create(0);
    if (!ctx) {
        fprintf(stderr, "bench_now: ctx_create failed: %s\n", tick_last_error());
        return 1;
    }

    /* Warm up: prime calibration, caches, branch predictors. The 1024-
     * call drift counter wraps a few times during warmup so the bench
     * loop measures steady-state cost. */
    for (int i = 0; i < 8192; i++) {
        (void)tick_now(ctx);
    }

    /* XOR keeps each call's return value live without a memory load
     * between iterations. A volatile sink would force one. */
    uint64_t sink = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NITERS; i++) {
        sink ^= tick_now(ctx);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long ns = (long)((t1.tv_sec - t0.tv_sec) * 1000000000L
                   + (t1.tv_nsec - t0.tv_nsec));
    double ns_per_op = (double)ns / (double)NITERS;
    printf("bench_now: %d tick_now calls, %ld ns total, %.2f ns/op\n",
           NITERS, ns, ns_per_op);

    /* Reference: clock_gettime(CLOCK_MONOTONIC) ns/op for comparison. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NITERS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        sink ^= (uint64_t)ts.tv_nsec;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long cg_ns = (long)((t1.tv_sec - t0.tv_sec) * 1000000000L
                      + (t1.tv_nsec - t0.tv_nsec));
    printf("bench_now: reference clock_gettime %ld ns total, %.2f ns/op\n",
           cg_ns, (double)cg_ns / (double)NITERS);

    /* Anti-DCE. */
    if (sink == 0) {
        printf("(sink==0, impossible for real ticks)\n");
    }

    tick_ctx_destroy(ctx);
    return 0;
}
