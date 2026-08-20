/* example: spoon + sva -- coroutine with a guarded stack.
 *
 * Creates a coroutine whose stack is backed by a libsva guarded region.
 * If the coroutine overflows its stack, it hits the PROT_NONE guard page
 * and crashes cleanly (SIGSEGV) instead of corrupting adjacent memory.
 *
 * Build: see example/Makefile
 */

#include <sva.h>
#include <spoon.h>
#include <stdio.h>
#include <string.h>

/* The coroutine writes to its stack recursively until it either
 * finishes (shallow recursion) or hits the guard page (deep
 * recursion). With a small stack and deep enough recursion, this
 * demonstrates the guard page catching an overflow. */

static volatile int depth_reached = 0;

static void recurse(spoon_co_t *self, int depth, void *stack_base)
{
    char buf[1024];
    memset(buf, depth & 0xFF, sizeof(buf));  /* touch the stack */
    depth_reached = depth;

    if (depth < 200) {
        recurse(self, depth + 1, stack_base);
    }
    spoon_yield();
}

static void guarded_coro(spoon_co_t *self, void *arg)
{
    void *base = arg;
    recurse(self, 0, base);
}

/* sva alloc hook for spoon_create_with_stack. */
static void *sva_alloc(size_t size, void *ud)
{
    sva_err_t err;
    sva_region_t *r = sva_map_guarded(size, SVA_PROT_READ | SVA_PROT_WRITE, NULL, &err);
    if (!r) return NULL;
    /* Stash the region handle for free_hook. In a real app you'd use a
     * side table; here we just leak it for the demo's lifetime. */
    (void)ud;
    return sva_base(r);
}

static void sva_free(void *ptr, size_t size, void *ud)
{
    /* In a real app you'd look up the sva_region_t* and call sva_unmap.
     * Skipped here since the process exits immediately after. */
    (void)ptr; (void)size; (void)ud;
}

int main(void)
{
    spoon_pool_t *pool = spoon_pool_create(4, 65536, NULL);
    if (!pool) {
        fprintf(stderr, "spoon_pool_create failed\n");
        return 1;
    }

    spoon_co_t *co = NULL;
    /* 32 KB stack -- small enough to overflow if the recursion is deep,
     * large enough to complete if it's shallow. */
    int rc = spoon_create_with_stack(pool, guarded_coro, NULL, 32 * 1024,
                                     sva_alloc, sva_free, NULL, &co);
    if (rc != SPOON_OK) {
        fprintf(stderr, "spoon_create_with_stack failed: %d\n", rc);
        return 1;
    }

    printf("Starting coroutine with 32 KB guarded stack...\n");
    spoon_switch_to(co);
    printf("Coroutine yielded at depth %d without overflow.\n", depth_reached);

    spoon_destroy(co);
    spoon_pool_destroy(pool);
    return 0;
}
