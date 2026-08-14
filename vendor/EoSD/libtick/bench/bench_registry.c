/*
 * bench_registry (v0.2): register/cancel/wait_next throughput at 100, 1000,
 * and 10000 timers. The registry is a binary min-heap.
 *
 * Measures:
 *   - register cost per op (fill the registry from empty, random deadlines
 *     so each insert exercises the O(log n) sift-up path rather than the
 *     sorted-append fast path).
 *   - cancel cost per op (drain the full registry; cancel ids are tracked
 *     explicitly because the heap's free-list reuse order is not the same
 *     as v0.1's flat-array scan order after the first fill/drain cycle).
 *   - wait_next scan cost per call at 100, 1000, 10000 pending timers
 *     (timeout=0, all deadlines far in the future). With the heap, the
 *     scan is O(1) peek + O(1) timeout check; the v0.1 linear scan was
 *     O(n). At 1000 timers v0.1 measured ~795 ns/op; v0.2 should land in
 *     the ~100-200 ns range.
 */
#define _GNU_SOURCE
#include <tick.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define MAX_CAP 10000
#define N_SCAN_ITERS 100000

static long elapsed_ns(struct timespec *t0, struct timespec *t1) {
    return (long)((t1->tv_sec - t0->tv_sec) * 1000000000L
                + (t1->tv_nsec - t0->tv_nsec));
}

int main(void) {
    tick_ctx_t *ctx = tick_ctx_create(MAX_CAP);
    if (!ctx) {
        fprintf(stderr, "bench_registry: ctx_create failed: %s\n", tick_last_error());
        return 1;
    }

    uint64_t far = tick_now(ctx) + 100000000000ULL;  /* 100s in the future */
    struct timespec t0, t1;
    tick_timer_id_t ids[MAX_CAP];

    srand(42);

    /* Register throughput: fill MAX_CAP slots with random deadlines so each
     * insert exercises the sift-up path. */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < MAX_CAP; i++) {
        uint64_t dl = far + (uint64_t)rand();
        int rc = tick_register(ctx, dl, TICK_MODE_POLL, NULL, NULL, &ids[i]);
        if (rc < 0) {
            fprintf(stderr, "bench_registry: register failed at %d (rc=%d)\n", i, rc);
            tick_ctx_destroy(ctx);
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long reg_ns = elapsed_ns(&t0, &t1);
    printf("bench_registry: register %d timers (random deadlines, heap sift-up): "
           "%ld ns total, %.2f ns/op\n",
           MAX_CAP, reg_ns, (double)reg_ns / MAX_CAP);

    /* Cancel throughput: drain all MAX_CAP by tracked id. */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < MAX_CAP; i++) {
        int rc = tick_cancel(ctx, ids[i]);
        if (rc != 0) {
            fprintf(stderr, "bench_registry: cancel failed at %d (id=%d)\n",
                    i, (int)ids[i]);
            tick_ctx_destroy(ctx);
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long can_ns = elapsed_ns(&t0, &t1);
    printf("bench_registry: cancel %d timers (heap remove-at): "
           "%ld ns total, %.2f ns/op\n",
           MAX_CAP, can_ns, (double)can_ns / MAX_CAP);

    /* wait_next scan cost at 100, 1000, 10000 timers. With timeout=0 and all
     * deadlines far in the future, wait_next peeks the heap root, finds it is
     * not yet due, sees deadline_cap <= now, and returns 0 without sleeping
     * or firing. The dominant cost is the root peek + tick_now. */
    int sizes[] = { 100, 1000, 10000 };
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        for (int i = 0; i < n; i++) {
            uint64_t dl = far + (uint64_t)rand();
            int rc = tick_register(ctx, dl, TICK_MODE_POLL, NULL, NULL, &ids[i]);
            if (rc < 0) {
                fprintf(stderr, "bench_registry: scan-setup register failed at %d\n", i);
                tick_ctx_destroy(ctx);
                return 1;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N_SCAN_ITERS; i++) {
            tick_timer_id_t fired[16];
            int rc = tick_wait_next(ctx, fired, 16, 0);
            if (rc < 0) {
                fprintf(stderr, "bench_registry: wait_next error: %d\n", rc);
                tick_ctx_destroy(ctx);
                return 1;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long scan_ns = elapsed_ns(&t0, &t1);
        printf("bench_registry: wait_next scan (%d timers, %d iters, heap peek): "
               "%ld ns total, %.2f ns/op\n",
               n, N_SCAN_ITERS, scan_ns, (double)scan_ns / N_SCAN_ITERS);

        for (int i = 0; i < n; i++) {
            tick_cancel(ctx, ids[i]);
        }
    }

    tick_ctx_destroy(ctx);
    return 0;
}
