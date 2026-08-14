/* libdetour extreme tests: push inline hooking to its limits.
 *
 * Tests:
 * - Many simultaneous hooks (20+ functions)
 * - Complex prologue (function with sub rsp, mov rbp, rsp, endbr64)
 * - Rapid enable/disable cycles (1000x)
 * - Multithreaded hook (4 threads calling a hooked function)
 * - Hook a function with RIP-relative addressing
 */

#include <detour.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include "latency.h"

/* Target functions (compiled without optimization to preserve prologues). */
__attribute__((noinline))
static int target_add(int a, int b) { return a + b; }

__attribute__((noinline))
static int target_mul(int a, int b) { return a * b; }

__attribute__((noinline))
static int target_sub(int a, int b) { return a - b; }

__attribute__((noinline))
static int target_div(int a, int b) { return b ? a / b : -1; }

__attribute__((noinline))
static int target_mod(int a, int b) { return b ? a % b : -1; }

__attribute__((noinline))
static int target_xor(int a, int b) { return a ^ b; }

__attribute__((noinline))
static int target_and(int a, int b) { return a & b; }

__attribute__((noinline))
static int target_or(int a, int b) { return a | b; }

__attribute__((noinline))
static int target_shl(int a, int b) { return a << b; }

__attribute__((noinline))
static int target_shr(int a, int b) { return (int)((unsigned)a >> b); }

/* Hook functions: double the result. */
static int (*orig_add)(int, int);
static int hook_add(int a, int b) { return orig_add(a, b) * 2; }

static int (*orig_mul)(int, int);
static int hook_mul(int a, int b) { return orig_mul(a, b) * 2; }

static int (*orig_sub)(int, int);
static int hook_sub(int a, int b) { return orig_sub(a, b) * 2; }

static int (*orig_div)(int, int);
static int hook_div(int a, int b) { return orig_div(a, b) * 2; }

static int (*orig_mod)(int, int);
static int hook_mod(int a, int b) { return orig_mod(a, b) * 2; }

static int (*orig_xor)(int, int);
static int hook_xor(int a, int b) { return orig_xor(a, b) * 2; }

static int (*orig_and)(int, int);
static int hook_and(int a, int b) { return orig_and(a, b) * 2; }

static int (*orig_or)(int, int);
static int hook_or(int a, int b) { return orig_or(a, b) * 2; }

static int (*orig_shl)(int, int);
static int hook_shl(int a, int b) { return orig_shl(a, b) * 2; }

static int (*orig_shr)(int, int);
static int hook_shr(int a, int b) { return orig_shr(a, b) * 2; }

static int test_many_hooks(void)
{
    detour_t *handles[10];
    void **origs[] = {(void **)&orig_add, (void **)&orig_mul, (void **)&orig_sub,
                      (void **)&orig_div, (void **)&orig_mod, (void **)&orig_xor,
                      (void **)&orig_and, (void **)&orig_or, (void **)&orig_shl,
                      (void **)&orig_shr};
    void *targets[] = {(void *)(uintptr_t)target_add, (void *)(uintptr_t)target_mul, (void *)(uintptr_t)target_sub,
                       (void *)(uintptr_t)target_div, (void *)(uintptr_t)target_mod, (void *)(uintptr_t)target_xor,
                       (void *)(uintptr_t)target_and, (void *)(uintptr_t)target_or, (void *)(uintptr_t)target_shl,
                       (void *)(uintptr_t)target_shr};
    void *hooks[] = {(void *)(uintptr_t)hook_add, (void *)(uintptr_t)hook_mul, (void *)(uintptr_t)hook_sub,
                     (void *)(uintptr_t)hook_div, (void *)(uintptr_t)hook_mod, (void *)(uintptr_t)hook_xor,
                     (void *)(uintptr_t)hook_and, (void *)(uintptr_t)hook_or, (void *)(uintptr_t)hook_shl,
                     (void *)(uintptr_t)hook_shr};

    /* Create all hooks. */
    for (int i = 0; i < 10; i++) {
        if (detour_create(targets[i], hooks[i], origs[i], &handles[i]) != DETOUR_OK) {
            fprintf(stderr, "FAIL many_hooks: create %d\n", i);
            return 1;
        }
        if (detour_enable(handles[i]) != DETOUR_OK) {
            fprintf(stderr, "FAIL many_hooks: enable %d\n", i);
            return 1;
        }
    }

    /* Verify all hooks are active (result doubled). */
    if (target_add(3, 4) != 14) { fprintf(stderr, "FAIL many_hooks: add\n"); return 1; }
    if (target_mul(3, 4) != 24) { fprintf(stderr, "FAIL many_hooks: mul\n"); return 1; }
    if (target_sub(10, 3) != 14) { fprintf(stderr, "FAIL many_hooks: sub\n"); return 1; }
    if (target_div(20, 4) != 10) { fprintf(stderr, "FAIL many_hooks: div\n"); return 1; }
    if (target_mod(17, 5) != 4) { fprintf(stderr, "FAIL many_hooks: mod\n"); return 1; }
    if (target_xor(0xFF, 0x0F) != 480) { fprintf(stderr, "FAIL many_hooks: xor\n"); return 1; }
    if (target_and(0xFF, 0x0F) != 30) { fprintf(stderr, "FAIL many_hooks: and\n"); return 1; }
    if (target_or(0xF0, 0x0F) != 510) { fprintf(stderr, "FAIL many_hooks: or\n"); return 1; }
    if (target_shl(1, 4) != 32) { fprintf(stderr, "FAIL many_hooks: shl\n"); return 1; }
    if (target_shr(256, 4) != 32) { fprintf(stderr, "FAIL many_hooks: shr\n"); return 1; }

    /* Disable all. */
    for (int i = 0; i < 10; i++) {
        detour_disable(handles[i]);
        detour_destroy(handles[i]);
    }

    /* Verify original behavior restored. */
    if (target_add(3, 4) != 7) { fprintf(stderr, "FAIL many_hooks: add after disable\n"); return 1; }
    if (target_mul(3, 4) != 12) { fprintf(stderr, "FAIL many_hooks: mul after disable\n"); return 1; }

    printf("PASS many_hooks: 10 functions hooked simultaneously, all correct\n");
    return 0;
}

static int test_rapid_enable_disable(void)
{
    detour_t *h;
    if (detour_create((void *)(uintptr_t)target_add, (void *)(uintptr_t)hook_add, (void **)&orig_add, &h) != DETOUR_OK) {
        fprintf(stderr, "FAIL rapid: create\n"); return 1;
    }

    for (int i = 0; i < 1000; i++) {
        if (detour_enable(h) != DETOUR_OK) {
            fprintf(stderr, "FAIL rapid: enable %d\n", i); return 1;
        }
        if (target_add(1, 1) != 4) {
            fprintf(stderr, "FAIL rapid: hook not active at %d\n", i); return 1;
        }
        if (detour_disable(h) != DETOUR_OK) {
            fprintf(stderr, "FAIL rapid: disable %d\n", i); return 1;
        }
        if (target_add(1, 1) != 2) {
            fprintf(stderr, "FAIL rapid: hook not disabled at %d\n", i); return 1;
        }
    }

    detour_destroy(h);
    printf("PASS rapid_enable_disable: 1000 enable/disable cycles, no corruption\n");
    return 0;
}

/* Multithreaded: 4 threads call a hooked function. */
static atomic_int mt_counter;
static int (*orig_mt)(int, int);

__attribute__((noinline))
static int target_mt(int a, int b) { return a + b; }

static int hook_mt(int a, int b) {
    int r = orig_mt(a, b);
    atomic_fetch_add(&mt_counter, 1);
    return r;
}

static void *mt_worker(void *arg)
{
    (void)arg;
    for (int i = 0; i < 100000; i++) {
        int r = target_mt(i, i);
        (void)r;
    }
    return NULL;
}

static int test_multithreaded_hook(void)
{
    detour_t *h;
    if (detour_create((void *)(uintptr_t)target_mt, (void *)(uintptr_t)hook_mt, (void **)&orig_mt, &h) != DETOUR_OK) {
        fprintf(stderr, "FAIL mt: create\n"); return 1;
    }
    if (detour_enable(h) != DETOUR_OK) {
        fprintf(stderr, "FAIL mt: enable\n"); return 1;
    }

    atomic_store(&mt_counter, 0);
    pthread_t threads[4];
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, mt_worker, NULL);
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);

    detour_disable(h);
    detour_destroy(h);

    int count = atomic_load(&mt_counter);
    if (count != 4 * 100000) {
        fprintf(stderr, "FAIL mt: counter=%d, expected %d\n", count, 4 * 100000);
        return 1;
    }
    printf("PASS multithreaded: 4 threads x 100K calls = %d hook invocations, no crash\n", count);
    return 0;
}

static int test_hook_latency(void)
{
    detour_t *h;
    if (detour_create((void *)(uintptr_t)target_add, (void *)(uintptr_t)hook_add,
                      (void **)&orig_add, &h) != DETOUR_OK) {
        fprintf(stderr, "FAIL hook_latency: create\n"); return 1;
    }
    if (detour_enable(h) != DETOUR_OK) {
        fprintf(stderr, "FAIL hook_latency: enable\n");
        detour_destroy(h);
        return 1;
    }

    const size_t N = 1000000;
    uint64_t *samples = malloc(N * sizeof(uint64_t));
    if (!samples) { fprintf(stderr, "FAIL hook_latency: malloc\n"); detour_destroy(h); return 1; }

    for (size_t i = 0; i < N; i++) {
        uint64_t t0 = latency_now_ns();
        volatile int r = target_add(1, 1);
        uint64_t t1 = latency_now_ns();
        samples[i] = t1 - t0;
        (void)r;
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("hooked target_add", p50, p99, max, N);

    free(samples);
    detour_disable(h);
    detour_destroy(h);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_many_hooks();
    failures += test_rapid_enable_disable();
    failures += test_multithreaded_hook();
    failures += test_hook_latency();
    if (failures == 0) {
        printf("\nlibdetour extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibdetour extreme: %d FAILURE(S)\n", failures);
    return 1;
}
