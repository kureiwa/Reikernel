/* libtick extreme tests: push the timer registry to its limits.
 *
 * Tests:
 * - Max capacity (register capacity-1 timers)
 * - All-same-deadline (1000 timers expire simultaneously)
 * - Rapid register/cancel (100K cycles, leak check)
 * - Tiny deadline (1 ns overshoot detection)
 * - Fuzz: random deadlines, verify all fire
 */

#include <tick.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "latency.h"

#define CAPACITY 1000

static int test_max_capacity(void)
{
    tick_ctx_t *ctx = tick_ctx_create(CAPACITY);
    if (!ctx) { fprintf(stderr, "FAIL max_capacity: create failed\n"); return 1; }

    tick_timer_id_t ids[CAPACITY];
    uint64_t now = tick_now(ctx);
    int ok = 1;

    /* Register CAPACITY-1 timers (leave room for the "full" test). */
    for (int i = 0; i < CAPACITY - 1; i++) {
        int rc = tick_register(ctx, now + 1000000000ULL, TICK_MODE_POLL, NULL, NULL, &ids[i]);
        if (rc < 0) {
            fprintf(stderr, "FAIL max_capacity: register %d returned %d\n", i, rc);
            ok = 0;
            break;
        }
    }

    /* One more should succeed (we're at capacity-1). */
    tick_timer_id_t extra;
    int rc = tick_register(ctx, now + 1000000000ULL, TICK_MODE_POLL, NULL, NULL, &extra);
    if (rc < 0) {
        fprintf(stderr, "FAIL max_capacity: register at capacity-1 returned %d\n", rc);
        ok = 0;
    }

    /* The next one should fail with TICK_ERR_FULL.
     * Note: tick_register returns -TICK_ERR_FULL (positive 2) on failure. */
    rc = tick_register(ctx, now + 1000000000ULL, TICK_MODE_POLL, NULL, NULL, &extra);
    if (rc != -TICK_ERR_FULL) {
        fprintf(stderr, "FAIL max_capacity: register at capacity returned %d (expected %d)\n",
                rc, -TICK_ERR_FULL);
        ok = 0;
    }

    tick_ctx_destroy(ctx);
    if (ok) printf("PASS max_capacity: %d timers registered, FULL at capacity\n", CAPACITY);
    return ok ? 0 : 1;
}

static int test_all_same_deadline(void)
{
    tick_ctx_t *ctx = tick_ctx_create(CAPACITY);
    if (!ctx) { fprintf(stderr, "FAIL same_deadline: create failed\n"); return 1; }

    uint64_t now = tick_now(ctx);
    uint64_t deadline = now + 50000000ULL; /* 50 ms */

    tick_timer_id_t ids[CAPACITY];
    for (int i = 0; i < CAPACITY; i++) {
        int rc = tick_register(ctx, deadline, TICK_MODE_POLL, NULL, NULL, &ids[i]);
        if (rc < 0) { fprintf(stderr, "FAIL same_deadline: register %d\n", i); tick_ctx_destroy(ctx); return 1; }
    }

    /* Wait for them to fire. All should fire in a single wait_next call. */
    tick_timer_id_t fired[CAPACITY];
    int n = tick_wait_next(ctx, fired, CAPACITY, deadline + 100000000ULL);
    if (n != CAPACITY) {
        fprintf(stderr, "FAIL same_deadline: expected %d fired, got %d\n", CAPACITY, n);
        tick_ctx_destroy(ctx);
        return 1;
    }

    tick_ctx_destroy(ctx);
    printf("PASS same_deadline: %d timers fired in one wait_next call\n", CAPACITY);
    return 0;
}

static int test_rapid_register_cancel(void)
{
    tick_ctx_t *ctx = tick_ctx_create(100);
    if (!ctx) { fprintf(stderr, "FAIL rapid: create failed\n"); return 1; }

    uint64_t now = tick_now(ctx);
    int ok = 1;

    for (int round = 0; round < 1000; round++) {
        tick_timer_id_t ids[100];
        for (int i = 0; i < 100; i++) {
            int rc = tick_register(ctx, now + 100000000000ULL, TICK_MODE_POLL, NULL, NULL, &ids[i]);
            if (rc < 0) { ok = 0; break; }
        }
        for (int i = 0; i < 100; i++) {
            tick_cancel(ctx, ids[i]);
        }
    }

    tick_ctx_destroy(ctx);
    if (ok) printf("PASS rapid_register_cancel: 100K register/cancel cycles, no leak\n");
    return ok ? 0 : 1;
}

static int test_tiny_deadline(void)
{
    tick_ctx_t *ctx = tick_ctx_create(0);
    if (!ctx) { fprintf(stderr, "FAIL tiny: create failed\n"); return 1; }

    /* Sleep until 1 ns in the past -- should report overshoot immediately. */
    uint64_t now = tick_now(ctx);
    uint64_t overshoot;
    int rc = tick_sleep_until(ctx, now - 1, &overshoot);
    if (rc != 1) {
        fprintf(stderr, "FAIL tiny_deadline: expected overshoot (rc=1), got %d\n", rc);
        tick_ctx_destroy(ctx);
        return 1;
    }
    if (overshoot == 0) {
        fprintf(stderr, "FAIL tiny_deadline: overshoot=0, expected >0\n");
        tick_ctx_destroy(ctx);
        return 1;
    }

    tick_ctx_destroy(ctx);
    printf("PASS tiny_deadline: past deadline detected, overshoot=%llu ns\n",
           (unsigned long long)overshoot);
    return 0;
}

static int test_fuzz_deadlines(void)
{
    tick_ctx_t *ctx = tick_ctx_create(CAPACITY);
    if (!ctx) { fprintf(stderr, "FAIL fuzz: create failed\n"); return 1; }

    srand(42);
    uint64_t now = tick_now(ctx);
    tick_timer_id_t ids[CAPACITY];

    /* Register timers with random deadlines in [now+1ms, now+100ms]. */
    for (int i = 0; i < CAPACITY; i++) {
        uint64_t dl = now + 1000000ULL + (rand() % 100000000ULL);
        int rc = tick_register(ctx, dl, TICK_MODE_POLL, NULL, NULL, &ids[i]);
        if (rc < 0) { fprintf(stderr, "FAIL fuzz: register %d\n", i); tick_ctx_destroy(ctx); return 1; }
    }

    /* Collect all fired timers until timeout. */
    int total_fired = 0;
    tick_timer_id_t fired[CAPACITY];
    uint64_t timeout = now + 200000000ULL; /* 200 ms */
    while (total_fired < CAPACITY) {
        int n = tick_wait_next(ctx, fired + total_fired, CAPACITY - total_fired, timeout);
        if (n <= 0) break;
        total_fired += n;
    }

    tick_ctx_destroy(ctx);
    if (total_fired != CAPACITY) {
        fprintf(stderr, "FAIL fuzz: expected %d fired, got %d\n", CAPACITY, total_fired);
        return 1;
    }
    printf("PASS fuzz_deadlines: %d random-deadline timers all fired\n", CAPACITY);
    return 0;
}

static int test_sleep_latency(void)
{
    tick_ctx_t *ctx = tick_ctx_create(0);
    if (!ctx) { fprintf(stderr, "FAIL sleep_latency: create\n"); return 1; }

    struct { const char *name; uint64_t deadline_ns; } cases[] = {
        {"1us",   1000ULL},
        {"10us",  10000ULL},
        {"100us", 100000ULL},
        {"1ms",   1000000ULL},
    };
    const size_t N = 100;
    uint64_t samples[100];

    printf("=== latency ===\n");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        for (size_t i = 0; i < N; i++) {
            uint64_t now = tick_now(ctx);
            uint64_t dl = now + cases[c].deadline_ns;
            uint64_t overshoot = 0;
            tick_sleep_until(ctx, dl, &overshoot);
            samples[i] = overshoot;
        }
        uint64_t p50, p99, max;
        latency_stats(samples, N, &p50, &p99, &max);
        char label[64];
        snprintf(label, sizeof(label), "tick_sleep_until %s overshoot",
                 cases[c].name);
        latency_print_ns(label, p50, p99, max, N);
    }

    tick_ctx_destroy(ctx);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_max_capacity();
    failures += test_all_same_deadline();
    failures += test_rapid_register_cancel();
    failures += test_tiny_deadline();
    failures += test_fuzz_deadlines();
    failures += test_sleep_latency();
    if (failures == 0) {
        printf("\nlibtick extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibtick extreme: %d FAILURE(S)\n", failures);
    return 1;
}
