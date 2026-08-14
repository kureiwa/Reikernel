/* bench_install: measure crash_install / crash_uninstall latency.
 *
 * crash_install does:
 *   - sigaltstack (one syscall, plus a malloc for the 64 KiB altstack)
 *   - sigaction per signal (5 default signals: SIGSEGV, SIGABRT,
 *     SIGFPE, SIGILL, SIGBUS), each one syscall
 *   - field publishes (no syscalls)
 *
 * crash_uninstall does:
 *   - sigaction per installed signal to restore the saved handler
 *   - sigaltstack to restore the old altstack
 *   - free of the altstack buffer
 *   - spinlock + memset of the user-blob array
 *
 * Total per round-trip: ~12 syscalls (5 sigaction + 1 sigaltstack +
 * 5 sigaction + 1 sigaltstack) + 1 malloc + 1 free + memset.
 *
 * 1000 iterations. Reports ns per install, ns per uninstall, and ns
 * per round-trip.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <crash.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define ITERS 1000u

static char buf[8192 + CRASH_MAX_USER_BLOBS *
                      (CRASH_MAX_BLOB_KEY + 8 + CRASH_MAX_BLOB_SIZE)];

static uint64_t now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void)
{
    /* Suppress core files: crash_install installs SIGSEGV etc.
     * handlers; if the bench is interrupted, we do not want a core. */
    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);

    /* Open a dummy fd for the dump writer. No crash will actually
     * happen during this bench, but crash_install requires a valid
     * fd >= 0. */
    int fd = open("/dev/null", O_WRONLY);
    if (fd < 0) {
        perror("open /dev/null");
        return 1;
    }

    /* Warm up: one install + uninstall round-trip to take the cold
     * malloc/sigaction path outside the measurement. */
    if (crash_install(buf, sizeof(buf), fd, NULL, 0,
                      CRASH_AFTER_RERAISE) != CRASH_OK) {
        printf("FAIL: warmup crash_install\n");
        close(fd);
        return 1;
    }
    crash_uninstall();

    /* Measure round-trip: install + uninstall as a pair. */
    uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        if (crash_install(buf, sizeof(buf), fd, NULL, 0,
                          CRASH_AFTER_RERAISE) != CRASH_OK) {
            printf("FAIL: crash_install at iter %u\n", i);
            close(fd);
            return 1;
        }
        crash_uninstall();
    }
    uint64_t t1 = now_ns();
    double roundtrip_ns = (double)(t1 - t0) / (double)ITERS;

    /* Measure the already-installed fast-exit path: install once, then
     * call crash_install ITERS more times. Each subsequent call returns
     * CRASH_ERR_ALREADY_INSTALLED via the early-exit check at the top
     * of crash_install_elf (g_buf != NULL -> return without touching
     * sigaction/sigaltstack/malloc). This is the cost a caller pays
     * if it defensively re-installs. */
    if (crash_install(buf, sizeof(buf), fd, NULL, 0,
                      CRASH_AFTER_RERAISE) != CRASH_OK) {
        printf("FAIL: setup for install-noop\n");
        close(fd);
        return 1;
    }
    t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        crash_install(buf, sizeof(buf), fd, NULL, 0,
                      CRASH_AFTER_RERAISE);
    }
    t1 = now_ns();
    double install_noop_ns = (double)(t1 - t0) / (double)ITERS;
    crash_uninstall();

    /* Measure uninstall-only: install once, then time ITERS uninstalls.
     * After the first uninstall, the rest are no-ops (g_buf == NULL &&
     * g_altstack_mem == NULL -> early return). */
    if (crash_install(buf, sizeof(buf), fd, NULL, 0,
                      CRASH_AFTER_RERAISE) != CRASH_OK) {
        printf("FAIL: setup for uninstall-only\n");
        close(fd);
        return 1;
    }
    t0 = now_ns();
    for (uint32_t i = 0; i < ITERS; i++) {
        crash_uninstall();
    }
    t1 = now_ns();
    double uninstall_noop_ns = (double)(t1 - t0) / (double)ITERS;

    close(fd);

    printf("bench_install: %u iterations\n", ITERS);
    printf("bench_install: install+uninstall round-trip : %.1f ns/op\n",
           roundtrip_ns);
    printf("bench_install:   (each round-trip = 10 sigaction + 2 "
           "sigaltstack + malloc + free)\n");
    printf("bench_install: install (already-installed)   : %.2f ns/op "
           "(early-exit path)\n", install_noop_ns);
    printf("bench_install: uninstall (not-installed)     : %.2f ns/op "
           "(early-exit path)\n", uninstall_noop_ns);
    return 0;
}
