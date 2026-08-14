/* bench_dump: measure the wall-clock cost of producing one crash dump
 * via the CRASH_AFTER_FORK out-of-process path.
 *
 * The measurement is the parent's waitpid() time from just before the
 * child crashes to just after waitpid() returns. This spans:
 *
 *   - the kernel delivering SIGSEGV to the child
 *   - the child's signal handler running on the altstack
 *   - the child's _Fork() of the grandchild (dump writer)
 *   - the grandchild's crash_write_dump (build dump in buffer + write(2))
 *   - the grandchild's _exit(0)
 *   - the child's waitpid() of the grandchild
 *   - the child's crash_reraise() (restore SIG_DFL + kill(getpid(), SIGSEGV))
 *   - the kernel terminating the child
 *   - the parent's waitpid() returning
 *
 * The grandparent (this process) forks a child for each iteration,
 * so per-iteration cost also includes one fork() in the grandparent.
 * The fork is outside the timed region (it happens before t0).
 *
 * 50 iterations is enough for a stable median without flooding /tmp
 * with dump files. Each iteration uses a fresh temp file (mkstemp +
 * unlink) so the dump goes to a real fd but the file is cleaned up
 * automatically when the fd is closed.
 *
 * The bench reports ns per dump, plus the dump size written (so the
 * caller can compute MB/s of dump throughput if desired).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <crash.h>

#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ITERS 50u

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
    /* Suppress core files: the child re-raises SIGSEGV after writing
     * the dump, and we do not want a real core file per iteration. */
    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);

    /* Scratch buffer for crash_install. Sized to crash_min_buffer_size(). */
    static char buf[8192 + CRASH_MAX_USER_BLOBS *
                          (CRASH_MAX_BLOB_KEY + 8 + CRASH_MAX_BLOB_SIZE)];

    uint64_t total_ns = 0;
    uint64_t dump_size = 0;
    uint32_t done = 0;

    for (uint32_t i = 0; i < ITERS; i++) {
        char tmpfile[] = "/tmp/libcrash_bench_XXXXXX";
        int fd = mkstemp(tmpfile);
        if (fd < 0) {
            perror("mkstemp");
            return 1;
        }
        /* Unlink immediately so the file is cleaned up when the last
         * fd closes. The fd is inherited by the child and grandchild
         * (they share the same file description). */
        unlink(tmpfile);

        pid_t child = fork();
        if (child < 0) {
            perror("fork");
            close(fd);
            return 1;
        }
        if (child == 0) {
            /* Child (the _Fork parent during the crash). */
            if (crash_install(buf, sizeof(buf), fd, NULL, 0,
                              CRASH_AFTER_FORK) != CRASH_OK) {
                _exit(100);
            }
            /* Trigger SIGSEGV. */
            volatile int *p = NULL;
            *p = 42;
            _exit(99);  /* unreachable */
        }

        /* Parent. Time the waitpid. */
        uint64_t t0 = now_ns();
        int status;
        if (waitpid(child, &status, 0) < 0) {
            perror("waitpid");
            close(fd);
            return 1;
        }
        uint64_t t1 = now_ns();
        total_ns += (t1 - t0);
        done++;

        /* Read the dump size (first 8 bytes = magic, but we just
         * want the file size). */
        struct stat st;
        if (fstat(fd, &st) == 0) {
            dump_size = (uint64_t)st.st_size;
        }
        close(fd);
    }

    double ns_per = (double)total_ns / (double)done;
    printf("bench_dump: %u iterations (CRASH_AFTER_FORK path)\n", done);
    printf("bench_dump: %.1f ns/dump (%.3f ms/dump)\n", ns_per,
           ns_per / 1e6);
    if (dump_size > 0) {
        printf("bench_dump: dump size: %" PRIu64 " bytes (last iter)\n",
               dump_size);
    }
    printf("bench_dump:   spans: kernel SIGSEGV delivery + child handler "
           "+ _Fork +\n");
    printf("bench_dump:          grandchild dump-write + _exit + child "
           "waitpid + re-raise\n");
    return 0;
}
