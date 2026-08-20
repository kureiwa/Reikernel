/* example: crash + spoon + sva -- catch a coroutine stack overflow.
 *
 * Installs a libcrash handler, creates a libspoon coroutine with a
 * small libsva-guarded stack, and deliberately overflows the stack.
 * The guard page SIGSEGV is caught by libcrash, which writes a dump
 * to a file. The process then exits with the crash info.
 *
 * This is the documented three-module integration pattern from
 * EoSD-SPEC.md §3.
 */

#include <crash.h>
#include <sva.h>
#include <spoon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

static void overflow_coro(spoon_co_t *self, void *arg)
{
    (void)self; (void)arg;
    /* Recurse with large stack frames until we hit the guard page. */
    char buf[4096];
    memset(buf, 0xAA, sizeof(buf));
    /* "Use" buf to prevent tail-call optimization. */
    printf("  touching stack at %p\n", (void *)buf);
    fflush(stdout);
    /* Recursive call -- will eventually hit the guard page. */
    void (*recurse)(spoon_co_t *, void *) = (void (*)(spoon_co_t *, void *))overflow_coro;
    recurse(self, arg);
}

/* sva-backed stack allocator for spoon. */
typedef struct sva_stack {
    sva_region_t *region;
} sva_stack_t;

static sva_stack_t g_stack;  /* track for the demo */

static void *sva_stack_alloc(size_t size, void *ud)
{
    (void)ud;
    sva_err_t err;
    g_stack.region = sva_map_guarded(size, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
    if (!g_stack.region) return NULL;
    return sva_base(g_stack.region);
}

static void sva_stack_free(void *ptr, size_t size, void *ud)
{
    (void)ptr; (void)size; (void)ud;
    /* Process is about to exit; skip unmap. */
}

int main(void)
{
    /* Open a file for the crash dump. */
    int fd = open("/tmp/eosd_crash_demo.dump", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    /* Pre-allocate the crash scratch buffer. */
    size_t buf_size = crash_min_buffer_size();
    void *buf = malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Install the crash handler. After dump, _exit(1). */
    if (crash_install(buf, buf_size, fd, NULL, 0, CRASH_AFTER_EXIT) != CRASH_OK) {
        fprintf(stderr, "crash_install failed\n");
        return 1;
    }

    printf("Crash handler installed. Dump file: /tmp/eosd_crash_demo.dump\n");
    printf("Creating coroutine with 16 KB guarded stack...\n");
    printf("Expect a SIGSEGV when the stack overflows into the guard page.\n");
    printf("The crash handler attempts to write a dump then _exit(1).\n");
    printf("If the overflow is too severe, the handler itself may not survive.\n\n");

    spoon_pool_t *pool = spoon_pool_create(4, 65536, NULL);
    if (!pool) {
        fprintf(stderr, "spoon_pool_create failed\n");
        return 1;
    }

    spoon_co_t *co = NULL;
    /* 16 KB stack -- small enough to overflow quickly. */
    int rc = spoon_create_with_stack(pool, overflow_coro, NULL, 16 * 1024,
                                     sva_stack_alloc, sva_stack_free, NULL, &co);
    if (rc != SPOON_OK) {
        fprintf(stderr, "spoon_create_with_stack failed: %d\n", rc);
        return 1;
    }

    printf("Starting coroutine...\n");
    fflush(stdout);
    spoon_switch_to(co);

    /* If we get here, the coroutine overflowed but the crash handler
     * caught it and _exit'd, or the overflow didn't happen. Either
     * way, exit non-zero since this is a crash demo. */
    fprintf(stderr, "ERROR: coroutine returned without crashing\n");
    return 1;
}
