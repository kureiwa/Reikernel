/* libspoon: asymmetric coroutine context switch.
 *
 * Hand-written NASM does the actual register save/restore. This file
 * handles pool management, coroutine lifecycle, and initial stack
 * setup.
 *
 * The registry is single-threaded: one pool per thread, enforced by
 * convention. The "current coroutine" is tracked in a thread-local
 * so spoon_switch_to knows which context to save.
 *
 * Asymmetric invariant: each coroutine has at most one caller (the
 * context that most recently switched to it). The caller chain forms
 * a tree rooted at the main thread. Nested asymmetric switches
 * (A->B->C->A, where C switches back to A while A is still suspended
 * waiting for B) are forbidden -- they would create a cycle in the
 * caller chain and corrupt saved contexts when co->caller is
 * overwritten. spoon_switch_to detects and rejects such switches
 * with SPOON_ERR_INVALID. Asymmetric coroutines cannot be nested
 * symmetrically; callers wanting symmetric handoff must build it
 * explicitly on top of the yield/switch primitives.
 */

#include "spoon_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Thread-local: the currently-running coroutine, or NULL for the main
 * thread. Set by spoon_switch_to before the NASM switch. */
static _Thread_local spoon_co_t *tl_current = NULL;

/* Thread-local: a scratch context for the main thread (when it calls
 * spoon_switch_to, we need somewhere to save its callee-saved regs). */
static _Thread_local spoon_co_t tl_main_ctx;

/* Minimum stack size. PTHREAD_STACK_MIN (16 KB on glibc x86_64) is the
 * safe floor for signal delivery. */
#define SPOON_MIN_STACK 16384

void spoon_entry(spoon_co_t *co)
{
    /* Called by the NASM trampoline on first switch. Run fn, mark
     * DONE, switch back to caller. fn must not return into the
     * trampoline -- if it does, we switch away and never come back. */
    spoon_fn fn = co->fn;
    void *arg = co->arg;
    fn(co, arg);

    co->status = SPOON_DONE;
    /* Switch back to whoever started us. */
    spoon_co_t *caller = co->caller;
    tl_current = caller;
    spoon_switch(co, caller);
    /* Not reached. */
    abort();
}

spoon_pool_t *spoon_pool_create(size_t capacity, size_t default_stack_size,
                                 const spoon_allocator_t *allocator)
{
    if (capacity == 0 || default_stack_size < SPOON_MIN_STACK)
        return NULL;

    spoon_pool_t *pool = calloc(1, sizeof(*pool));
    if (!pool) return NULL;

    pool->coroutines = calloc(capacity, sizeof(spoon_co_t *));
    if (!pool->coroutines) {
        free(pool);
        return NULL;
    }
    pool->capacity = capacity;
    pool->count = 0;
    pool->default_stack_size = default_stack_size;
    if (allocator) {
        pool->allocator = *allocator;
    } else {
        pool->allocator.alloc = NULL;
        pool->allocator.free = NULL;
        pool->allocator.user_data = NULL;
    }
    return pool;
}

/* Force-free a coroutine's resources without checking status. Used by
 * spoon_destroy (after the DONE precondition check) and by
 * spoon_pool_destroy (which forcibly tears down all remaining
 * coroutines regardless of status). */
static void co_force_free(spoon_co_t *co)
{
    if (!co) return;
    /* Remove from pool. */
    if (co->pool) {
        for (size_t i = 0; i < co->pool->capacity; i++) {
            if (co->pool->coroutines[i] == co) {
                co->pool->coroutines[i] = NULL;
                co->pool->count--;
                break;
            }
        }
    }
    /* Free the stack. */
    if (co->stack && co->free_hook) {
        co->free_hook(co->stack, co->stack_size, co->free_hook_user_data);
    }
    free(co);
}

void spoon_pool_destroy(spoon_pool_t *pool)
{
    if (!pool) return;
    /* Forcibly tear down every remaining coroutine regardless of
     * status. This bypasses the DONE precondition enforced by
     * spoon_destroy (the user-facing API for orderly teardown of a
     * single finished coroutine). Pool teardown is "release
     * everything"; callers are expected to have driven in-flight
     * coroutines to completion themselves, but any they forgot will
     * be freed here rather than leaked. */
    for (size_t i = 0; i < pool->capacity; i++) {
        if (pool->coroutines[i]) {
            co_force_free(pool->coroutines[i]);
        }
    }
    free(pool->coroutines);
    free(pool);
}

static void *default_alloc(size_t size, void *ud)
{
    (void)ud;
    return malloc(size);
}

static void default_free(void *ptr, size_t size, void *ud)
{
    (void)size; (void)ud;
    free(ptr);
}

int spoon_create(spoon_pool_t *pool, spoon_fn fn, void *arg,
                  size_t stack_size, spoon_co_t **out_co)
{
    return spoon_create_with_stack(pool, fn, arg, stack_size,
                                    NULL, NULL, NULL, out_co);
}

int spoon_create_with_stack(spoon_pool_t *pool, spoon_fn fn, void *arg,
                             size_t stack_size,
                             void *(*alloc_hook)(size_t, void *),
                             void (*free_hook)(void *, size_t, void *),
                             void *user_data,
                             spoon_co_t **out_co)
{
    if (!pool || !fn || !out_co) return SPOON_ERR_INVALID;
    if (pool->count >= pool->capacity) return SPOON_ERR_FULL;
    if (stack_size == 0) stack_size = pool->default_stack_size;
    /* Reject sub-minimum stack sizes outright. Previously this was a
     * silent clamp to SPOON_MIN_STACK, which is safe when libspoon
     * allocates the stack itself but catastrophic when a caller
     * supplies an alloc_hook backed by a pre-allocated buffer of
     * exactly stack_size bytes -- the clamp caused libspoon to write
     * past the end of the user's buffer. Reject so callers learn at
     * create time. */
    if (stack_size < SPOON_MIN_STACK) return SPOON_ERR_INVALID;

    /* Round stack_size up to 16 bytes. */
    stack_size = (stack_size + 15) & ~(size_t)15;

    /* Allocate the coroutine struct. */
    spoon_co_t *co = calloc(1, sizeof(*co));
    if (!co) return SPOON_ERR_STACK_ALLOC;

    /* Allocate the stack. */
    void *stack;
    if (alloc_hook) {
        stack = alloc_hook(stack_size, user_data);
        co->free_hook = free_hook;
        co->free_hook_user_data = user_data;
    } else if (pool->allocator.alloc) {
        stack = pool->allocator.alloc(stack_size, pool->allocator.user_data);
        co->free_hook = pool->allocator.free;
        co->free_hook_user_data = pool->allocator.user_data;
    } else {
        stack = default_alloc(stack_size, NULL);
        co->free_hook = default_free;
        co->free_hook_user_data = NULL;
    }
    if (!stack) {
        free(co);
        return SPOON_ERR_STACK_ALLOC;
    }

    co->stack = stack;
    co->stack_size = stack_size;
    co->fn = fn;
    co->arg = arg;
    co->pool = pool;
    co->status = SPOON_READY;
    co->caller = NULL;

    /* Set up initial saved context for the first switch.
     *
     * The NASM switch will:
     *   mov rsp, [co + OFF_SP]
     *   restore callee-saved regs from co
     *   jmp [co + OFF_RIP]
     *
     * So we set:
     *   co->sp  = stack_top (16-byte aligned)
     *   co->rip = spoon_trampoline
     *   co->rbx = (uint64_t)co  (trampoline reads co from rbx)
     *   co->mxcsr = default
     *   co->x87_cw = default
     */
    uintptr_t stack_top = (uintptr_t)stack + stack_size;
    stack_top &= ~(uintptr_t)15;  /* 16-byte align */
    co->sp = (void *)stack_top;
    co->rip = (uint64_t)(uintptr_t)spoon_trampoline;
    co->rbx = (uint64_t)(uintptr_t)co;
    co->rbp = 0;
    co->r12 = 0;
    co->r13 = 0;
    co->r14 = 0;
    co->r15 = 0;
    co->mxcsr = SPOON_DEFAULT_MXCSR;
    co->x87_cw = SPOON_DEFAULT_X87CW;

    /* Add to pool. */
    for (size_t i = 0; i < pool->capacity; i++) {
        if (!pool->coroutines[i]) {
            pool->coroutines[i] = co;
            pool->count++;
            break;
        }
    }

    *out_co = co;
    return SPOON_OK;
}

int spoon_switch_to(spoon_co_t *co)
{
    if (!co) return SPOON_ERR_INVALID;
    if (co->status == SPOON_DONE) return SPOON_ERR_INVALID;

    /* Determine the "from" context: the current coroutine, or the
     * main thread's scratch context if tl_current is NULL. */
    spoon_co_t *from = tl_current ? tl_current : &tl_main_ctx;

    /* Asymmetric invariant: the caller chain forms a tree rooted at
     * the main thread. If `co` appears anywhere in `from`'s caller
     * chain (including as `from` itself), switching to `co` would
     * create a cycle: `co` would have its `caller` field overwritten
     * to point at `from`, while `from` is already downstream of `co`.
     * When `co` later finishes and switches back through its caller
     * chain, it would eventually re-enter a context whose saved
     * state has already been consumed -- a SIGABRT (verified by the
     * audit's probe_nested). The same check also covers the self-
     * switch case (co == from). */
    for (spoon_co_t *p = from; p; p = p->caller) {
        if (p == co) return SPOON_ERR_INVALID;
    }

    /* Record who started `co` so it can yield back. */
    co->caller = from;

    /* Set tl_current before the switch. When the switch returns (we've
     * been switched back to), tl_current was set by whoever initiated
     * the switch back to us. */
    tl_current = co;
    co->status = SPOON_RUNNING;

    spoon_switch(from, co);

    /* We've been resumed. tl_current was set by the resuming code
     * before its switch. Nothing to do here. */
    return SPOON_OK;
}

spoon_status_t spoon_status(const spoon_co_t *co)
{
    if (!co) return SPOON_DONE;
    return co->status;
}

int spoon_yield(void)
{
    spoon_co_t *cur = tl_current;
    if (!cur) return SPOON_ERR_INVALID;  /* called from main thread */
    spoon_co_t *caller = cur->caller;
    if (!caller) return SPOON_ERR_INVALID;
    cur->status = SPOON_SUSPENDED;
    tl_current = caller;
    spoon_switch(cur, caller);
    /* Resumed. */
    return SPOON_OK;
}

void spoon_destroy(spoon_co_t *co)
{
    if (!co) return;
    /* Precondition: the coroutine must have completed (SPOON_DONE) so
     * that no live saved context references its stack. Destroying a
     * SUSPENDED coroutine would free its stack while its saved RSP
     * still points into that stack; destroying a RUNNING coroutine
     * would free the stack the caller is currently executing on.
     * Returns silently without freeing if the precondition is not
     * met; spoon_pool_destroy is the escape hatch for forcibly
     * tearing down non-DONE coroutines during pool teardown. */
    if (co->status != SPOON_DONE) return;
    co_force_free(co);
}
