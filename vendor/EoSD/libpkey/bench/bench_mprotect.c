/* bench_mprotect: measure mprotect latency for comparison with WRPKRU.
 *
 * Measures the cost of toggling a page's protection via mprotect(2), the
 * pre-MPK way to change page permissions. Two flavors:
 *
 *   1. mprotect PROT_NONE <-> PROT_READ|PROT_WRITE round-trip (two
 *      syscalls per iteration). This is the apples-to-apples comparison
 *      against pkey_deny + pkey_allow in bench_wrpkru.
 *
 *   2. mprotect to the same prot (PROT_READ|PROT_WRITE -> PROT_READ|
 *      PROT_WRITE, a no-op-flavor mprotect). Measures the syscall floor
 *      without the kernel's VMA-splitting cost on a real prot change.
 *
 * Expected: ~1000-3000 ns/op for (1) (two syscalls + VMA manipulation),
 * ~500-1500 ns/op for (2) (two syscalls, no VMA split). The headline
 * number for the WRPKRU comparison is (1) vs bench_wrpkru's bare/helper
 * numbers: ~100x faster.
 *
 * This bench runs unconditionally (no MPK dependency); mprotect is always
 * available. Pair with bench_wrpkru on an MPK-capable host for the ratio.
 *
 * clock_gettime(CLOCK_MONOTONIC) (vDSO on x86-64) over 1 000 000 iters.
 *
 * _DEFAULT_SOURCE exposes MAP_ANONYMOUS under -std=c11 (glibc gates it
 * on __USE_MISC). bench_mprotect does not include pkey.h so there is no
 * glibc pkey_* signature clash here; _DEFAULT_SOURCE is used for
 * consistency with the rest of the module. */

#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

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
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) {
        printf("FAIL bench_mprotect: sysconf _SC_PAGESIZE=%ld\n", ps);
        return 1;
    }
    void *page = mmap(NULL, (size_t)ps, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        printf("FAIL bench_mprotect: mmap MAP_FAILED\n");
        return 1;
    }

    /* --- measurement 1: real prot change round-trip ------------------ */
    for (int i = 0; i < 1024; i++) {
        mprotect(page, (size_t)ps, PROT_NONE);
        mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
    }
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        mprotect(page, (size_t)ps, PROT_NONE);
        mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
    }
    double t1 = now_ns();
    double ns_real = (t1 - t0) / (double)(ITERS * 2u);

    /* --- measurement 2: same-prot mprotect (syscall floor) ----------- */
    for (int i = 0; i < 1024; i++) {
        mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
        mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
    }
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
        mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
    }
    t1 = now_ns();
    double ns_same = (t1 - t0) / (double)(ITERS * 2u);

    mprotect(page, (size_t)ps, PROT_READ | PROT_WRITE);
    munmap(page, (size_t)ps);

    printf("bench_mprotect: real prot change (NONE<->RW): %.2f ns/op\n",
           ns_real);
    printf("bench_mprotect:   expected ~200-1500 ns/op (two syscalls; "
           "higher if VMA split)\n");
    printf("bench_mprotect: same-prot mprotect (RW->RW)  : %.2f ns/op\n",
           ns_same);
    printf("bench_mprotect:   expected ~80-400 ns/op (syscall floor, "
           "no VMA split)\n");
    printf("bench_mprotect: pair with bench_wrpkru on an MPK host for the "
           "WRPKRU/mprotect ratio (~20-100x depending on VMA split)\n");
    return 0;
}
