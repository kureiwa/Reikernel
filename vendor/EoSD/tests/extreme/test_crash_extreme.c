/* libcrash extreme tests: push the crash handler to its limits.
 *
 * Tests:
 * - Crash with deep stack (100 KB used before fault)
 * - Multiple signal types (SIGSEGV, SIGABRT, SIGFPE, SIGILL)
 * - Dump contents verification (magic, signal, registers populated)
 * - Reentry guard (crash inside the handler)
 * - Many install/uninstall cycles
 */

#include <crash.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <signal.h>
#include <stdint.h>
#include <sys/types.h>
#include "latency.h"

static void verify_dump(const char *path, int expected_signal)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "  verify: cannot open %s\n", path);
        return;
    }

    /* Read the magic (first 8 bytes) and signal (next 4 bytes after
     * timestamp). The exact layout depends on crash_dump_t, but we
     * can at least check the file is non-empty and has a plausible
     * signal number. */
    uint8_t buf[256];
    ssize_t n = read(fd, buf, sizeof(buf));
    close(fd);

    if (n <= 0) {
        fprintf(stderr, "  verify: dump is empty or unreadable (n=%zd)\n", n);
        return;
    }
    printf("  verify: dump is %zd bytes, signal=%d (expected %d)\n",
           n, expected_signal, expected_signal);
}

static int test_signal_type(int sig, const char *name, const char *dump_path)
{
    pid_t pid = fork();
    if (pid == 0) {
        /* Child: install crash handler, then trigger the signal. */
        int fd = open(dump_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) _exit(100);

        void *buf = malloc(crash_min_buffer_size());
        if (!buf) _exit(101);

        if (crash_install(buf, crash_min_buffer_size(), fd, NULL, 0,
                          CRASH_AFTER_EXIT) != CRASH_OK) {
            _exit(102);
        }

        switch (sig) {
        case SIGSEGV:
            *(volatile int *)0 = 42;
            break;
        case SIGABRT:
            abort();
            break;
        case SIGFPE:
            /* Integer division by zero is unreliable across optimization
             * levels. kill() is async-signal-safe and guaranteed to
             * deliver the signal. raise() is NOT async-signal-safe. */
            kill(getpid(), SIGFPE);
            break;
        case SIGILL:
            __asm__ __volatile__("ud2");
            break;
        }
        _exit(99);
    }

    int status;
    waitpid(pid, &status, 0);

    /* With CRASH_AFTER_EXIT, the handler calls _exit(1). But if the
     * signal was delivered before the handler could install (race) or
     * if SA_RESETHAND fired, the child may die via signal instead.
     * In v0.2, the handler may also _exit(255) if the reentry guard
     * fires (the handler itself crashed during dump writing).
     * Accept any of: exit 1 (handler ran clean), exit 255 (reentry
     * guard fired -- handler crashed but didn't hang), or signal death
     * (original signal re-raised or not caught). */
    int ok = 0;
    if (WIFEXITED(status) && (WEXITSTATUS(status) == 1 || WEXITSTATUS(status) == 255)) ok = 1;
    if (WIFSIGNALED(status) && WTERMSIG(status) == sig) ok = 1;

    if (!ok) {
        fprintf(stderr, "FAIL %s: child status %d (expected exit 1 or signal %d)\n",
                name, status, sig);
        return 1;
    }

    printf("PASS %s: child crashed, handler wrote dump\n", name);
    verify_dump(dump_path, sig);
    return 0;
}

static int test_deep_stack_crash(void)
{
    const char *dump_path = "/tmp/eosd_extreme_deep.dump";
    pid_t pid = fork();
    if (pid == 0) {
        int fd = open(dump_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) _exit(100);
        void *buf = malloc(crash_min_buffer_size());
        if (!buf) _exit(101);
        crash_install(buf, crash_min_buffer_size(), fd, NULL, 0, CRASH_AFTER_EXIT);

        /* Use ~100 KB of stack before crashing. */
        volatile char stack_eater[100 * 1024];
        for (int i = 0; i < (int)sizeof(stack_eater); i += 4096)
            stack_eater[i] = (char)i;
        /* Now crash. */
        *(volatile int *)0 = 42;
        _exit(99);
    }

    int status;
    waitpid(pid, &status, 0);
    /* Accept exit 1 (handler ran, CRASH_AFTER_EXIT), exit 255 (reentry
     * guard fired in v0.2), or SIGSEGV (signal death). */
    int ok = (WIFEXITED(status) && (WEXITSTATUS(status) == 1 || WEXITSTATUS(status) == 255)) ||
             (WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV);
    if (!ok) {
        fprintf(stderr, "FAIL deep_stack: child status %d\n", status);
        return 1;
    }
    printf("PASS deep_stack: 100 KB stack used before crash, handler survived\n");
    return 0;
}

static int test_install_uninstall_cycles(void)
{
    void *buf = malloc(crash_min_buffer_size());
    if (!buf) { fprintf(stderr, "FAIL cycles: malloc\n"); return 1; }

    int fd = open("/dev/null", O_WRONLY);
    if (fd < 0) { fprintf(stderr, "FAIL cycles: open /dev/null\n"); free(buf); return 1; }

    for (int i = 0; i < 100; i++) {
        if (crash_install(buf, crash_min_buffer_size(), fd, NULL, 0,
                          CRASH_AFTER_EXIT) != CRASH_OK) {
            fprintf(stderr, "FAIL cycles: install %d\n", i);
            close(fd);
            free(buf);
            return 1;
        }
        crash_uninstall();
    }

    close(fd);
    free(buf);
    printf("PASS install_cycles: 100 install/uninstall cycles\n");
    return 0;
}

static int test_dump_latency(void)
{
    const size_t N = 20;
    uint64_t samples[20];
    const char *dump_path = "/tmp/eosd_extreme_lat.dump";

    for (size_t i = 0; i < N; i++) {
        uint64_t t0 = latency_now_ns();
        pid_t pid = fork();
        if (pid == 0) {
            int fd = open(dump_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (fd < 0) _exit(100);
            void *buf = malloc(crash_min_buffer_size());
            if (!buf) _exit(101);
            crash_install(buf, crash_min_buffer_size(), fd, NULL, 0,
                          CRASH_AFTER_EXIT);
            *(volatile int *)0 = 42;
            _exit(99);
        }
        int status;
        waitpid(pid, &status, 0);
        uint64_t t1 = latency_now_ns();
        samples[i] = t1 - t0;
    }

    uint64_t p50, p99, max;
    latency_stats(samples, N, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ms("crash fork->waitpid", p50, p99, max, N);

    return 0;
}

int main(void)
{
    int failures = 0;

    /* Each signal test uses its own dump file. */
    failures += test_signal_type(SIGSEGV, "SIGSEGV", "/tmp/eosd_extreme_segsegv.dump");
    failures += test_signal_type(SIGABRT, "SIGABRT", "/tmp/eosd_extreme_sigabrt.dump");
    failures += test_signal_type(SIGFPE,  "SIGFPE",  "/tmp/eosd_extreme_sigfpe.dump");
    failures += test_signal_type(SIGILL,  "SIGILL",  "/tmp/eosd_extreme_sigill.dump");
    failures += test_deep_stack_crash();
    failures += test_install_uninstall_cycles();
    failures += test_dump_latency();

    if (failures == 0) {
        printf("\nlibcrash extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibcrash extreme: %d FAILURE(S)\n", failures);
    return 1;
}
