/* bench_overhead: measure the per-call cost of calling a hooked function
 * (patched target -> hook -> trampoline -> original body -> hook ->
 * return) versus a direct call to the same function.
 *
 * Uses rdtsc for cycle-accurate timing. The target function is written
 * to be > 14 bytes (the patch size) and to use RIP-relative addressing
 * for a global store, so the bench also exercises the RIP-relative
 * relocation path. */

#include "detour.h"

#include <stdio.h>
#include <stdint.h>

volatile int g_bench_sink = 0;

__attribute__((noinline, noclone))
int bench_target(int x)
{
    /* Three volatile stores force a non-trivial body (>14 bytes) and
     * exercise RIP-relative mov [rip+disp], reg. */
    g_bench_sink = x;
    g_bench_sink = x + 1;
    g_bench_sink = x + 2;
    return x + 1;
}

static int (*bench_orig)(int) = NULL;

__attribute__((noinline, noclone))
int bench_hook(int x)
{
    /* The sink store prevents the compiler from turning this into a
     * tail-call jmp to bench_orig, which would mask the real cost of
     * the hook -> trampoline -> original round-trip. */
    int r = bench_orig(x);
    g_bench_sink = r;
    return r;
}

static inline uint64_t rdtsc_fenced(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtscp"
                         : "=a"(lo), "=d"(hi)
                         :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}

#define ITERS 5000000UL
#define WARMUP 10000

int main(void)
{
    volatile int sink = 0;

    /* Warm up the direct path. */
    for (int i = 0; i < WARMUP; i++) sink += bench_target(i);

    uint64_t t0 = rdtsc_fenced();
    for (uint64_t i = 0; i < ITERS; i++) {
        sink += bench_target((int)i);
    }
    uint64_t t1 = rdtsc_fenced();
    double direct_cyc = (double)(t1 - t0) / (double)ITERS;

    /* Install the hook. */
    detour_t *h = NULL;
    int rc = detour_create((void *)bench_target, (void *)bench_hook,
                           (void **)&bench_orig, &h);
    if (rc != DETOUR_OK) {
        printf("FAIL: detour_create returned %d\n", rc);
        return 1;
    }
    rc = detour_enable(h);
    if (rc != DETOUR_OK) {
        printf("FAIL: detour_enable returned %d\n", rc);
        detour_destroy(h);
        return 1;
    }

    /* Warm up the hooked path. */
    for (int i = 0; i < WARMUP; i++) sink += bench_target(i);

    t0 = rdtsc_fenced();
    for (uint64_t i = 0; i < ITERS; i++) {
        sink += bench_target((int)i);
    }
    t1 = rdtsc_fenced();
    double hooked_cyc = (double)(t1 - t0) / (double)ITERS;

    detour_destroy(h);

    __asm__ volatile("" :: "r"(sink) : "memory");

    printf("bench_overhead: %lu direct calls, %lu hooked calls\n",
           ITERS, ITERS);
    printf("bench_overhead: direct = %.1f cycles/call\n", direct_cyc);
    printf("bench_overhead: hooked = %.1f cycles/call\n", hooked_cyc);
    printf("bench_overhead: hook overhead = %.1f cycles/call\n",
           hooked_cyc - direct_cyc);
    return 0;
}
