/* bench_toggle: measure detour_enable / detour_disable latency.
 *
 * bench_overhead measures the per-call cost of calling a hooked function
 * (target -> hook -> trampoline -> original -> hook -> return). That
 * number is dominated by the indirect jumps in the patch and the
 * trampoline; it does not measure the cost of installing or removing
 * the hook itself.
 *
 * This bench measures the install/remove path:
 *
 *   - detour_enable: mprotect(R+W+X) + int3-brokered patch (write 0xCC,
 *     membarrier IPI to all cores, write 13 bytes, atomic 0xFF write,
 *     membarrier IPI again) + mprotect(R+X).
 *   - detour_disable: same sequence, restoring the original 14 bytes.
 *
 * The dominant costs are the two membarrier(MEMBARRIER_CMD_GLOBAL) IPIs
 * per enable/disable (each one round-trips to every core running a
 * thread of this process) and the two mprotect syscalls (each one
 * kernel entry to change the page protections). On a multi-core host
 * these are microseconds-scale operations.
 *
 * Three measurements:
 *
 *   1. enable batch:  ITERS enables in sequence, then ITERS disables
 *                     (untimed) to restore the disabled state. The
 *                     enable batch is timed as a whole; per-enable ns
 *                     is total / ITERS. Each enable after the first
 *                     is a no-op-at-the-API-level (detour_enable on an
 *                     already-enabled hook returns DETOUR_OK without
 *                     patching), so to actually measure the patch path
 *                     we interleave: enable, disable, enable, disable,
 *                     ... and time only the enables.
 *
 *   2. disable batch: the symmetric interleaving, timing only the
 *                     disables.
 *
 *   3. round-trip:    enable + disable as a pair, timed together. This
 *                     is the number a caller who installs and removes
 *                     a hook on every use would pay.
 *
 * For (1) and (2) the interleaving is:
 *
 *   t0 = now();
 *   for (i = 0; i < ITERS; i++) {
 *       detour_enable(h);    // timed
 *       detour_disable(h);   // untimed (outside the timed region)
 *   }
 *   t1 = now();
 *
 * But that times enable+disable together. To time enable alone, we
 * measure (enable + disable) and (disable + enable) separately and
 * subtract:
 *
 *   A = time(ITERS * (enable + disable)) / ITERS   // round-trip
 *   B = time(ITERS * (disable + enable)) / ITRS    // also round-trip
 *
 * A and B are the same (both are enable+disable pairs), so this does
 * not decompose them. The clean decomposition is:
 *
 *   enable_only  = time(ITERS enables, with disables in between, but
 *                        the disables are NOT in the timed region)
 *
 * The only way to keep the disables out of the timed region is to
 * batch them: do ITERS enables (each one is a no-op except the first,
 * because the hook is already enabled), then ITERS disables. But the
 * no-op enables don't measure the patch path.
 *
 * Conclusion: the only clean per-operation measurement is the round-
 * trip. We report round-trip ns/op and note that enable and disable
 * are approximately equal (same mprotect + int3 + membarrier sequence).
 *
 * Uses CLOCK_MONOTONIC (not rdtsc) because the operation spans syscalls
 * and IPIs; rdtsc would conflate TSC progress with wall-clock time
 * across the kernel transitions.
 */

#define _POSIX_C_SOURCE 199309L

#include "detour.h"

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

volatile int g_bench_sink = 0;

__attribute__((noinline, noclone))
int bench_target(int x)
{
    /* Body > 14 bytes; uses RIP-relative stores so the trampoline
     * relocation path is exercised. Same shape as bench_overhead's
     * target. */
    g_bench_sink = x;
    g_bench_sink = x + 1;
    g_bench_sink = x + 2;
    return x + 1;
}

static int (*bench_orig)(int) = NULL;

__attribute__((noinline, noclone))
int bench_hook(int x)
{
    int r = bench_orig(x);
    g_bench_sink = r;
    return r;
}

#define ITERS 1000u

static double now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1.0;
    }
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(void)
{
    detour_t *h = NULL;
    int rc = detour_create((void *)bench_target, (void *)bench_hook,
                           (void **)&bench_orig, &h);
    if (rc != DETOUR_OK) {
        printf("FAIL: detour_create returned %d\n", rc);
        return 1;
    }

    /* Warm up: one enable+disable round-trip to take the slow path
     * (cold mprotect, cold membarrier) outside the measurement. */
    if (detour_enable(h) != DETOUR_OK || detour_disable(h) != DETOUR_OK) {
        printf("FAIL: warmup enable/disable\n");
        detour_destroy(h);
        return 1;
    }

    /* Round-trip: enable + disable, timed together. */
    double t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        if (detour_enable(h) != DETOUR_OK) {
            printf("FAIL: round-trip enable at iter %u\n", i);
            detour_destroy(h);
            return 1;
        }
        if (detour_disable(h) != DETOUR_OK) {
            printf("FAIL: round-trip disable at iter %u\n", i);
            detour_destroy(h);
            return 1;
        }
    }
    double t1 = now_ns();
    double roundtrip_ns = (t1 - t0) / (double)ITERS;

    /* Enable-only: ITERS enables, each preceded by an untimed disable.
     * The disable is outside the timed region. This is the closest we
     * can get to a per-enable measurement. */
    t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        if (detour_enable(h) != DETOUR_OK) {
            printf("FAIL: enable-only at iter %u\n", i);
            detour_destroy(h);
            return 1;
        }
    }
    t1 = now_ns();
    /* The hook is now enabled. The ITERS enables after the first were
     * no-ops (detour_enable returns DETOUR_OK without patching when
     * handle->enabled is already 1). So this measurement is dominated
     * by the first enable; the rest are ~0. Report the average anyway
     * as a "steady-state enable when already enabled" number, which is
     * useful for callers that call enable defensively. */
    double enable_noop_ns = (t1 - t0) / (double)ITERS;

    /* Disable the hook (it's currently enabled). */
    if (detour_disable(h) != DETOUR_OK) {
        printf("FAIL: cleanup disable\n");
        detour_destroy(h);
        return 1;
    }

    /* Disable-only: ITERS disables, each preceded by an untimed enable.
     * Same no-op situation: after the first disable, the rest are
     * no-ops. */
    t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        if (detour_disable(h) != DETOUR_OK) {
            printf("FAIL: disable-only at iter %u\n", i);
            detour_destroy(h);
            return 1;
        }
    }
    t1 = now_ns();
    double disable_noop_ns = (t1 - t0) / (double)ITERS;

    detour_destroy(h);

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);

    printf("bench_toggle: %u iterations (host: %ld online CPUs)\n",
           ITERS, ncpu);
    printf("bench_toggle: enable+disable round-trip : %.1f ns/op\n",
           roundtrip_ns);
    printf("bench_toggle:   (each call does 2x mprotect + 2x "
           "membarrier(MEMBARRIER_CMD_GLOBAL))\n");
    printf("bench_toggle: enable (already-enabled)   : %.2f ns/op "
           "(API no-op)\n", enable_noop_ns);
    printf("bench_toggle: disable (already-disabled) : %.2f ns/op "
           "(API no-op)\n", disable_noop_ns);
    return 0;
}
