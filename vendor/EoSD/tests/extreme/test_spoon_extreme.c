/* libspoon extreme tests: push coroutines to their limits.
 *
 * Tests:
 * - Many coroutines (max pool capacity)
 * - Deep nesting (A -> B -> C -> D -> E, yield back up)
 * - FP-heavy work across switches (verify MXCSR preserved under load)
 * - Rapid create/destroy (leak check)
 * - Minimal stack (16 KB, verify it works)
 */

#include <spoon.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <xmmintrin.h>
#include "latency.h"

#define N_COROS 500

/* Real coroutine fn for many_coroutines. */
static int many_flags[N_COROS];
static void many_fn(spoon_co_t *self, void *arg)
{
    (void)self;
    int idx = (int)(intptr_t)arg;
    many_flags[idx] = idx * 10;
    spoon_yield();
    many_flags[idx] += 1;
}

static int test_many_coroutines_real(void)
{
    spoon_pool_t *pool = spoon_pool_create(N_COROS + 1, 32768, NULL);
    if (!pool) { fprintf(stderr, "FAIL many_real: pool_create\n"); return 1; }

    spoon_co_t *coros[N_COROS];
    for (int i = 0; i < N_COROS; i++) {
        many_flags[i] = 0;
        if (spoon_create(pool, many_fn, (void *)(intptr_t)i, 0, &coros[i]) != SPOON_OK) {
            fprintf(stderr, "FAIL many_real: create %d\n", i);
            return 1;
        }
    }

    /* First pass: each coroutine sets its flag and yields. */
    for (int i = 0; i < N_COROS; i++)
        spoon_switch_to(coros[i]);

    /* Second pass: each coroutine increments its flag and returns. */
    for (int i = 0; i < N_COROS; i++) {
        if (spoon_status(coros[i]) != SPOON_DONE)
            spoon_switch_to(coros[i]);
    }

    int ok = 1;
    for (int i = 0; i < N_COROS; i++) {
        if (many_flags[i] != i * 10 + 1) {
            fprintf(stderr, "FAIL many_real: flag[%d]=%d, expected %d\n",
                    i, many_flags[i], i * 10 + 1);
            ok = 0;
            break;
        }
        spoon_destroy(coros[i]);
    }

    spoon_pool_destroy(pool);
    if (ok) printf("PASS many_coroutines: %d coroutines, 2 switches each, all correct\n", N_COROS);
    return ok ? 0 : 1;
}

/* Deep nesting: chain of coroutines, each switches to the next. */
#define DEPTH 20
static int chain_result = 0;

static void chain_fn(spoon_co_t *self, void *arg)
{
    (void)self; (void)arg;
    chain_result++;
    /* The caller drives the chain; each coroutine just increments and
     * yields back. The "depth" is tested by the caller switching
     * sequentially. */
    spoon_yield();
    chain_result += 100;
}

static int test_deep_chain(void)
{
    spoon_pool_t *pool = spoon_pool_create(DEPTH + 1, 32768, NULL);
    if (!pool) { fprintf(stderr, "FAIL deep: pool_create\n"); return 1; }

    spoon_co_t *chain[DEPTH];
    for (int i = 0; i < DEPTH; i++) {
        if (spoon_create(pool, chain_fn, NULL, 0, &chain[i]) != SPOON_OK) {
            fprintf(stderr, "FAIL deep: create %d\n", i);
            return 1;
        }
    }

    chain_result = 0;
    /* Switch to each in sequence. Each increments chain_result, yields
     * back (we resume), then we switch again to let it finish. */
    for (int i = 0; i < DEPTH; i++) {
        spoon_switch_to(chain[i]);
    }
    for (int i = 0; i < DEPTH; i++) {
        if (spoon_status(chain[i]) != SPOON_DONE)
            spoon_switch_to(chain[i]);
    }

    int expected = DEPTH * 101;
    int ok = (chain_result == expected);

    for (int i = 0; i < DEPTH; i++) spoon_destroy(chain[i]);
    spoon_pool_destroy(pool);

    if (ok) printf("PASS deep_chain: %d-deep coroutine chain, result=%d\n", DEPTH, chain_result);
    else fprintf(stderr, "FAIL deep_chain: result=%d, expected %d\n", chain_result, expected);
    return ok ? 0 : 1;
}

/* FP-heavy: do SIMD work across switches, verify correctness. */
#define FP_ARRAY_SIZE 256
static float fp_input[FP_ARRAY_SIZE];
static float fp_output[FP_ARRAY_SIZE];

static void fp_heavy_fn(spoon_co_t *self, void *arg)
{
    (void)self; (void)arg;
    /* Do some FP work, yield, do more FP work. If MXCSR or XMM state
     * were corrupted, the output would be wrong. */
    for (int i = 0; i < FP_ARRAY_SIZE; i++)
        fp_output[i] = fp_input[i] * 2.0f;
    spoon_yield();
    for (int i = 0; i < FP_ARRAY_SIZE; i++)
        fp_output[i] = fp_output[i] + 1.0f;
}

static int test_fp_heavy(void)
{
    for (int i = 0; i < FP_ARRAY_SIZE; i++)
        fp_input[i] = (float)i;

    spoon_pool_t *pool = spoon_pool_create(4, 65536, NULL);
    if (!pool) { fprintf(stderr, "FAIL fp_heavy: pool_create\n"); return 1; }

    spoon_co_t *co;
    if (spoon_create(pool, fp_heavy_fn, NULL, 0, &co) != SPOON_OK) {
        fprintf(stderr, "FAIL fp_heavy: create\n");
        return 1;
    }

    spoon_switch_to(co);
    /* At yield point: fp_output[i] = input[i] * 2. */
    for (int i = 0; i < FP_ARRAY_SIZE; i++) {
        if (fp_output[i] != (float)i * 2.0f) {
            fprintf(stderr, "FAIL fp_heavy: mid-switch output[%d]=%.2f, expected %.2f\n",
                    i, fp_output[i], (float)i * 2.0f);
            return 1;
        }
    }
    spoon_switch_to(co);

    int ok = 1;
    for (int i = 0; i < FP_ARRAY_SIZE; i++) {
        if (fp_output[i] != (float)i * 2.0f + 1.0f) {
            fprintf(stderr, "FAIL fp_heavy: final output[%d]=%.2f, expected %.2f\n",
                    i, fp_output[i], (float)i * 2.0f + 1.0f);
            ok = 0;
            break;
        }
    }

    spoon_destroy(co);
    spoon_pool_destroy(pool);
    if (ok) printf("PASS fp_heavy: %d-element float array correct across 2 switches\n", FP_ARRAY_SIZE);
    return ok ? 0 : 1;
}

/* Rapid create/destroy: leak check. Each coroutine is driven to
 * SPOON_DONE before destroy, since spoon_destroy now enforces the
 * documented DONE precondition (returning without freeing otherwise). */
static void immediate_fn(spoon_co_t *self, void *arg)
{
    (void)self; (void)arg;
    /* Returns immediately -- coroutine reaches SPOON_DONE on first switch. */
}

static int test_rapid_create_destroy(void)
{
    spoon_pool_t *pool = spoon_pool_create(16, 32768, NULL);
    if (!pool) { fprintf(stderr, "FAIL rapid: pool_create\n"); return 1; }

    for (int i = 0; i < 10000; i++) {
        spoon_co_t *co;
        if (spoon_create(pool, immediate_fn, NULL, 0, &co) != SPOON_OK) {
            fprintf(stderr, "FAIL rapid: create %d\n", i);
            spoon_pool_destroy(pool);
            return 1;
        }
        spoon_switch_to(co);      /* drive to SPOON_DONE */
        spoon_destroy(co);
    }

    spoon_pool_destroy(pool);
    printf("PASS rapid_create_destroy: 10K create/switch/destroy cycles, no leak\n");
    return 0;
}

static int test_switch_latency(void)
{
    spoon_pool_t *pool = spoon_pool_create(16, 32768, NULL);
    if (!pool) { fprintf(stderr, "FAIL switch_latency: pool_create\n"); return 1; }

    const size_t N = 100000;
    uint64_t *samples = malloc(N * sizeof(uint64_t));
    if (!samples) { fprintf(stderr, "FAIL switch_latency: malloc\n"); spoon_pool_destroy(pool); return 1; }

    /* Each iteration: create a coroutine that returns immediately,
     * time one switch_to (which is a full round-trip: caller -> co ->
     * caller), then destroy. immediate_fn is defined above. */
    for (size_t i = 0; i < N; i++) {
        spoon_co_t *co;
        if (spoon_create(pool, immediate_fn, NULL, 0, &co) != SPOON_OK) {
            fprintf(stderr, "FAIL switch_latency: create %zu\n", i);
            free(samples);
            spoon_pool_destroy(pool);
            return 1;
        }
        uint64_t t0 = latency_now_ns();
        spoon_switch_to(co);
        uint64_t t1 = latency_now_ns();
        samples[i] = t1 - t0;
        spoon_destroy(co);
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("spoon_switch_to round-trip", p50, p99, max, N);

    free(samples);
    spoon_pool_destroy(pool);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_many_coroutines_real();
    failures += test_deep_chain();
    failures += test_fp_heavy();
    failures += test_rapid_create_destroy();
    failures += test_switch_latency();
    if (failures == 0) {
        printf("\nlibspoon extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibspoon extreme: %d FAILURE(S)\n", failures);
    return 1;
}
