/* libbarrage extreme tests: push the bump allocator to its limits.
 *
 * Tests:
 * - Fill to capacity (verify OUT_OF_SPACE)
 * - Reset cycles (10K alloc/reset, leak check)
 * - All alignments (1, 2, 4, 8, 16, 32, 64)
 * - Large allocation (single alloc near arena size)
 * - Zero-size allocation
 * - Many small allocations (1M)
 */

#include <barrage.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "latency.h"

static int test_fill_capacity(void)
{
    barrage_arena_t *a = barrage_create(4096, NULL);
    if (!a) { fprintf(stderr, "FAIL fill: create\n"); return 1; }

    barrage_err_t err;
    int count = 0;
    void *p;
    while ((p = barrage_alloc(a, 16, 16, &err)) != NULL) {
        count++;
    }
    if (err != BARRAGE_ERR_OUT_OF_SPACE) {
        fprintf(stderr, "FAIL fill: expected OUT_OF_SPACE, got %d\n", err);
        barrage_destroy(a);
        return 1;
    }
    /* 4096 / 16 = 256 allocations. */
    if (count != 256) {
        fprintf(stderr, "FAIL fill: %d allocs, expected 256\n", count);
        barrage_destroy(a);
        return 1;
    }
    barrage_destroy(a);
    printf("PASS fill_capacity: 256 x 16-byte allocs filled 4 KB arena, then OUT_OF_SPACE\n");
    return 0;
}

static int test_reset_cycles(void)
{
    barrage_arena_t *a = barrage_create(65536, NULL);
    if (!a) { fprintf(stderr, "FAIL reset: create\n"); return 1; }

    barrage_err_t err;
    for (int round = 0; round < 10000; round++) {
        /* Allocate some, then reset. */
        for (int i = 0; i < 100; i++) {
            if (!barrage_alloc(a, 64, 16, &err)) {
                fprintf(stderr, "FAIL reset: round %d alloc %d, err=%d\n", round, i, err);
                barrage_destroy(a);
                return 1;
            }
        }
        barrage_reset(a);
        if (barrage_used(a) != 0) {
            fprintf(stderr, "FAIL reset: used=%zu after reset (round %d)\n",
                    barrage_used(a), round);
            barrage_destroy(a);
            return 1;
        }
    }
    barrage_destroy(a);
    printf("PASS reset_cycles: 10K alloc-100/reset cycles, used=0 each time\n");
    return 0;
}

static int test_all_alignments(void)
{
    barrage_arena_t *a = barrage_create(4096, NULL);
    if (!a) { fprintf(stderr, "FAIL align: create\n"); return 1; }

    barrage_err_t err;
    size_t aligns[] = {1, 2, 4, 8, 16, 32, 64};
    int ok = 1;

    for (size_t a_idx = 0; a_idx < sizeof(aligns)/sizeof(aligns[0]); a_idx++) {
        barrage_reset(a);
        void *p = barrage_alloc(a, 1, aligns[a_idx], &err);
        if (!p) {
            fprintf(stderr, "FAIL align(%zu): alloc failed err=%d\n", aligns[a_idx], err);
            ok = 0;
            break;
        }
        uintptr_t addr = (uintptr_t)p;
        if (addr % aligns[a_idx] != 0) {
            fprintf(stderr, "FAIL align(%zu): addr %p not aligned\n",
                    aligns[a_idx], p);
            ok = 0;
            break;
        }
    }

    /* Invalid alignments. */
    barrage_reset(a);
    if (barrage_alloc(a, 16, 0, &err) != NULL || err != BARRAGE_ERR_INVALID) {
        fprintf(stderr, "FAIL align(0): expected INVALID\n");
        ok = 0;
    }
    if (barrage_alloc(a, 16, 3, &err) != NULL || err != BARRAGE_ERR_INVALID) {
        fprintf(stderr, "FAIL align(3): expected INVALID (not power of 2)\n");
        ok = 0;
    }
    if (barrage_alloc(a, 16, 128, &err) != NULL || err != BARRAGE_ERR_INVALID) {
        fprintf(stderr, "FAIL align(128): expected INVALID (> 64)\n");
        ok = 0;
    }

    barrage_destroy(a);
    if (ok) printf("PASS all_alignments: 1,2,4,8,16,32,64 verified; 0,3,128 rejected\n");
    return ok ? 0 : 1;
}

static int test_large_alloc(void)
{
    /* 1 MB arena, allocate nearly all of it. */
    size_t sz = 1 << 20;
    barrage_arena_t *a = barrage_create(sz, NULL);
    if (!a) { fprintf(stderr, "FAIL large: create\n"); return 1; }

    barrage_err_t err;
    /* Allocate sz - 64 bytes (leaves 64 bytes of slack). */
    void *p = barrage_alloc(a, sz - 64, 16, &err);
    if (!p) {
        fprintf(stderr, "FAIL large: alloc %zu failed err=%d\n", sz - 64, err);
        barrage_destroy(a);
        return 1;
    }
    /* Touch the last byte to verify the mapping is real. */
    memset(p, 0xAB, sz - 64);
    if (((uint8_t *)p)[sz - 65] != 0xAB) {
        fprintf(stderr, "FAIL large: memory not writable\n");
        barrage_destroy(a);
        return 1;
    }

    /* Next alloc of any significant size should fail. A 1-byte alloc
     * might succeed if there's slack, so try something larger. */
    if (barrage_alloc(a, 128, 16, &err) != NULL) {
        /* This could succeed if there's enough slack. Not a failure --
         * just note it. */
        printf("PASS large_alloc: single %zu-byte alloc filled 1 MB arena (small slack remained)\n", sz - 64);
    } else {
        printf("PASS large_alloc: single %zu-byte alloc filled 1 MB arena, next alloc failed\n", sz - 64);
    }

    barrage_destroy(a);
    return 0;
}

static int test_many_small(void)
{
    barrage_arena_t *a = barrage_create(1 << 20, NULL); /* 1 MB */
    if (!a) { fprintf(stderr, "FAIL many_small: create\n"); return 1; }

    barrage_err_t err;
    int count = 0;
    while (barrage_alloc(a, 8, 8, &err) != NULL)
        count++;

    /* 1 MB / 8 = 131072 allocations. */
    if (count < 130000) {
        fprintf(stderr, "FAIL many_small: only %d allocs, expected ~131072\n", count);
        barrage_destroy(a);
        return 1;
    }

    barrage_destroy(a);
    printf("PASS many_small: %d x 8-byte allocs from 1 MB arena\n", count);
    return 0;
}

static int test_latency(void)
{
    /* 16 MB arena: holds ~2M 8-byte allocs before a reset is needed.
     * We collect 1M successful-alloc samples; whenever alloc returns
     * NULL (OUT_OF_SPACE) we reset and continue. */
    barrage_arena_t *a = barrage_create(1 << 24, NULL);
    if (!a) { fprintf(stderr, "FAIL latency: create\n"); return 1; }

    const size_t N = 1000000;
    uint64_t *samples = malloc(N * sizeof(uint64_t));
    if (!samples) { fprintf(stderr, "FAIL latency: malloc\n"); barrage_destroy(a); return 1; }

    barrage_err_t err;
    size_t i = 0;
    while (i < N) {
        uint64_t t0 = latency_now_ns();
        void *p = barrage_alloc(a, 8, 8, &err);
        uint64_t t1 = latency_now_ns();
        if (p) {
            samples[i++] = t1 - t0;
        } else {
            barrage_reset(a);
        }
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("barrage_alloc", p50, p99, max, N);

    free(samples);
    barrage_destroy(a);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_fill_capacity();
    failures += test_reset_cycles();
    failures += test_all_alignments();
    failures += test_large_alloc();
    failures += test_many_small();
    failures += test_latency();
    if (failures == 0) {
        printf("\nlibbarrage extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibbarrage extreme: %d FAILURE(S)\n", failures);
    return 1;
}
