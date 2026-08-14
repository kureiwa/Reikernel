/* bench_wrpkru: measure WRPKRU instruction latency.
 *
 * Three measurements on a freshly allocated pkey:
 *
 *   1. pkey_wrpkru (bare asm helper, src/pkey_x86_64.asm): the raw
 *      instruction cost with no C-side read-modify-write overhead.
 *      Alternates between two PKRU values to prevent the CPU from
 *      detecting a no-op and eliding the write.
 *
 *   2. pkey_allow / pkey_deny (convenience wrappers): each does an
 *      RDPKRU + bit-clear/set + WRPKRU. This is what a real caller
 *      pays to toggle a key's protection.
 *
 *   3. pkey_set_access direct: same as (2) but via the explicit API,
 *      to confirm the convenience wrappers add no overhead.
 *
 * Expected: when OSPKE is set, (1) is ~20 cycles (~6-8 ns at 3GHz) and
 * (2)/(3) are ~40-60 cycles (~12-20 ns) for the RDPKRU+WRPKRU pair.
 *
 * clock_gettime(CLOCK_MONOTONIC) (vDSO on x86-64) over 1 000 000 iters;
 * per-op cost = total / iters. The vDSO overhead (~20-30 ns/call) is
 * amortized across the loop body so it does not dominate the per-op ns.
 *
 * Skips (exit 0) if pkey_available() returns 0.
 *
 * _DEFAULT_SOURCE (not _GNU_SOURCE) is used for consistency with the
 * rest of the module; bench_wrpkru does not include <sys/mman.h> so the
 * glibc pkey_* clash does not apply, but _DEFAULT_SOURCE is what gives
 * clock_gettime + syscall() visibility under -std=c11. */

#define _DEFAULT_SOURCE

#include "pkey.h"

#include <stdio.h>
#include <stdint.h>
#include <time.h>

/* The bare asm helper, declared in src/pkey.c and defined in
 * src/pkey_x86_64.asm. The bench links the static archive so the global
 * symbol is visible; declaring it here lets the bench time the raw
 * instruction sequence without the C-side read-modify-write overhead. */
extern void pkey_wrpkru(uint32_t pkru);

#define ITERS 1000000ULL

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
    if (!pkey_available()) {
        printf("SKIP bench_wrpkru: MPK not available (OSPKE not set)\n");
        printf("  Try a host with Intel Skylake-X+ and Linux 4.6+ with "
               "CR4.PKE enabled.\n");
        return 0;
    }

    int pkey = pkey_alloc();
    if (pkey < 0) {
        printf("FAIL bench_wrpkru: pkey_alloc=%d\n", pkey);
        return 1;
    }

    /* Two distinct PKRU values that toggle only this key's bits. The
     * bare-helper loop alternates between them so the CPU cannot coalesce
     * the writes (each WRPKRU observes a different value from the prior). */
    uint32_t v_allow = 0;                           /* AD=0, WD=0 for all keys */
    uint32_t v_deny  = 1u << (unsigned)(pkey * 2);  /* AD=1 for our key */

    /* --- measurement 1: bare pkey_wrpkru ----------------------------- */
    for (int i = 0; i < 1024; i++) {
        pkey_wrpkru(v_allow);
        pkey_wrpkru(v_deny);
    }
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        pkey_wrpkru(v_allow);
        pkey_wrpkru(v_deny);
    }
    double t1 = now_ns();
    double ns_bare = (t1 - t0) / (double)(ITERS * 2u);  /* 2 WRPKRU per iter */

    /* --- measurement 2: pkey_allow / pkey_deny ----------------------- */
    for (int i = 0; i < 1024; i++) {
        pkey_allow(pkey);
        pkey_deny(pkey);
    }
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        pkey_allow(pkey);
        pkey_deny(pkey);
    }
    t1 = now_ns();
    double ns_wrap = (t1 - t0) / (double)(ITERS * 2u);  /* 2 ops per iter */

    /* --- measurement 3: pkey_set_access direct ----------------------- */
    for (int i = 0; i < 1024; i++) {
        pkey_set_access(pkey, 0, 0);
        pkey_set_access(pkey, 1, 0);
    }
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        pkey_set_access(pkey, 0, 0);
        pkey_set_access(pkey, 1, 0);
    }
    t1 = now_ns();
    double ns_set = (t1 - t0) / (double)(ITERS * 2u);

    /* Restore the key to full access before freeing. */
    pkey_allow(pkey);
    pkey_free(pkey);

    printf("bench_wrpkru: bare pkey_wrpkru    : %.2f ns/op\n", ns_bare);
    printf("bench_wrpkru:   expected ~20 cycles (~6-8 ns at 3GHz)\n");
    printf("bench_wrpkru: pkey_allow/deny     : %.2f ns/op\n", ns_wrap);
    printf("bench_wrpkru:   expected ~40-60 cycles (~12-20 ns) "
           "for RDPKRU+WRPKRU\n");
    printf("bench_wrpkru: pkey_set_access     : %.2f ns/op\n", ns_set);
    printf("bench_wrpkru:   expected same as pkey_allow/deny "
           "(convenience wrappers add no overhead)\n");
    return 0;
}
