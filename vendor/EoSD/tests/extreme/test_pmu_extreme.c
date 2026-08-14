/* libpmu extreme tests: exercise error paths and API edge cases.
 *
 * Note: perf_event_open is denied in many sandboxed environments
 * (perf_event_paranoid=2). These tests focus on error handling and
 * API correctness rather than counter values.
 *
 * Tests:
 * - All counter types (may SKIP if perf denied)
 * - Invalid counter type
 * - NULL context handling
 * - Open/close cycles
 * - Start/stop/read on invalid contexts
 */

#include <pmu.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "latency.h"

static int test_all_counter_types(void)
{
    pmu_counter_type_t types[] = {PMU_CYCLES, PMU_INSTRUCTIONS, PMU_CACHE_MISSES};
    const char *names[] = {"CYCLES", "INSTRUCTIONS", "CACHE_MISSES"};
    int skipped = 0;

    for (int i = 0; i < 3; i++) {
        pmu_err_t err;
        pmu_ctx_t *ctx = pmu_open(types[i], &err);
        if (!ctx) {
            if (err == PMU_ERR_PERM) {
                printf("SKIP %s: perf_event_open denied (paranoid=2)\n", names[i]);
                skipped++;
                continue;
            }
            fprintf(stderr, "FAIL %s: open returned err=%d\n", names[i], err);
            return 1;
        }

        if (pmu_start(ctx) != PMU_OK) {
            fprintf(stderr, "FAIL %s: start\n", names[i]);
            pmu_close(ctx);
            return 1;
        }

        /* Do some work. */
        volatile int sink = 0;
        for (int j = 0; j < 1000000; j++) sink += j;
        (void)sink;

        uint64_t val;
        if (pmu_stop_and_read(ctx, &val) != PMU_OK) {
            fprintf(stderr, "FAIL %s: stop_and_read\n", names[i]);
            pmu_close(ctx);
            return 1;
        }

        if (val == 0) {
            fprintf(stderr, "WARN %s: counter read 0 (expected >0)\n", names[i]);
        }

        pmu_close(ctx);
        if (err != PMU_ERR_PERM)
            printf("PASS %s: opened, started, counted %llu, closed\n",
                   names[i], (unsigned long long)val);
    }

    if (skipped == 3) {
        printf("SKIP all_counter_types: perf_event_open denied for all types\n");
        printf("  Run with: sudo sysctl -w kernel.perf_event_paranoid=1\n");
    }
    return 0;
}

static int test_invalid_type(void)
{
    pmu_err_t err;
    pmu_ctx_t *ctx = pmu_open((pmu_counter_type_t)999, &err);
    if (ctx != NULL) {
        fprintf(stderr, "FAIL invalid_type: expected NULL for type 999\n");
        pmu_close(ctx);
        return 1;
    }
    if (err != PMU_ERR_INVALID && err != PMU_ERR_UNAVAILABLE) {
        fprintf(stderr, "FAIL invalid_type: expected INVALID or UNAVAILABLE, got %d\n", err);
        return 1;
    }
    printf("PASS invalid_type: type 999 rejected with err=%d\n", err);
    return 0;
}

static int test_null_context(void)
{
    uint64_t val;
    if (pmu_start(NULL) != PMU_ERR_INVALID) {
        fprintf(stderr, "FAIL null: pmu_start(NULL) should return INVALID\n");
        return 1;
    }
    if (pmu_read(NULL, &val) != PMU_ERR_INVALID) {
        fprintf(stderr, "FAIL null: pmu_read(NULL) should return INVALID\n");
        return 1;
    }
    if (pmu_stop_and_read(NULL, &val) != PMU_ERR_INVALID) {
        fprintf(stderr, "FAIL null: pmu_stop_and_read(NULL) should return INVALID\n");
        return 1;
    }
    /* pmu_close(NULL) should be a no-op. */
    pmu_close(NULL);
    printf("PASS null_context: all NULL-context calls return INVALID (close is no-op)\n");
    return 0;
}

static int test_open_close_cycles(void)
{
    pmu_err_t err;
    pmu_ctx_t *ctx = pmu_open(PMU_CYCLES, &err);
    if (!ctx) {
        if (err == PMU_ERR_PERM) {
            printf("SKIP open_close_cycles: perf denied\n");
            return 0;
        }
        fprintf(stderr, "FAIL open_close: open err=%d\n", err);
        return 1;
    }

    for (int i = 0; i < 100; i++) {
        if (pmu_start(ctx) != PMU_OK) {
            fprintf(stderr, "FAIL open_close: start %d\n", i);
            return 1;
        }
        uint64_t val;
        if (pmu_stop_and_read(ctx, &val) != PMU_OK) {
            fprintf(stderr, "FAIL open_close: stop %d\n", i);
            return 1;
        }
    }

    pmu_close(ctx);
    printf("PASS open_close_cycles: 100 start/stop/read cycles on one context\n");
    return 0;
}

static int test_read_latency(void)
{
    pmu_err_t err;
    pmu_ctx_t *ctx = pmu_open(PMU_CYCLES, &err);
    if (!ctx) {
        fprintf(stderr, "FAIL read_latency: open err=%d\n", err);
        return 1;
    }
    if (!pmu_is_available(ctx)) {
        printf("SKIP read_latency: perf denied (dummy ctx)\n");
        pmu_close(ctx);
        return 0;
    }
    if (pmu_start(ctx) != PMU_OK) {
        fprintf(stderr, "FAIL read_latency: start\n");
        pmu_close(ctx);
        return 1;
    }

    const size_t N = 100000;
    uint64_t *samples = malloc(N * sizeof(uint64_t));
    if (!samples) { fprintf(stderr, "FAIL read_latency: malloc\n"); pmu_close(ctx); return 1; }

    for (size_t i = 0; i < N; i++) {
        uint64_t val;
        uint64_t t0 = latency_now_ns();
        pmu_read(ctx, &val);
        uint64_t t1 = latency_now_ns();
        samples[i] = t1 - t0;
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("pmu_read", p50, p99, max, N);

    free(samples);
    pmu_close(ctx);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_all_counter_types();
    failures += test_invalid_type();
    failures += test_null_context();
    failures += test_open_close_cycles();
    failures += test_read_latency();
    if (failures == 0) {
        printf("\nlibpmu extreme: ALL PASS (or SKIP)\n");
        return 0;
    }
    printf("\nlibpmu extreme: %d FAILURE(S)\n", failures);
    return 1;
}
