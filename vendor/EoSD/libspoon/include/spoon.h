#ifndef SPOON_H
#define SPOON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct spoon_pool spoon_pool_t;
typedef struct spoon_co   spoon_co_t;

typedef enum {
    SPOON_OK               = 0,
    SPOON_ERR_INVALID      = -1,
    SPOON_ERR_FULL         = -2,
    SPOON_ERR_STACK_ALLOC  = -3,
} spoon_err_t;

typedef enum {
    SPOON_READY,
    SPOON_RUNNING,
    SPOON_SUSPENDED,
    SPOON_DONE,
} spoon_status_t;

/* Coroutine entry function. Receives the coroutine handle and the
 * caller-supplied arg. When fn returns, the coroutine is marked DONE
 * and control switches back to the caller. */
typedef void (*spoon_fn)(spoon_co_t *self, void *arg);

/* Pluggable allocator for coroutine stacks. NULL fields default to
 * libc malloc/free. */
typedef struct {
    void *(*alloc)(size_t size, void *user_data);
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} spoon_allocator_t;

/* Creates a pool that can hold up to `capacity` coroutines, each with
 * `default_stack_size` bytes of stack (overridable per-coroutine).
 * `allocator` may be NULL for libc malloc/free.
 * Thread-safety: not thread-safe. One pool per thread. */
spoon_pool_t *spoon_pool_create(size_t capacity, size_t default_stack_size,
                                 const spoon_allocator_t *allocator);
void spoon_pool_destroy(spoon_pool_t *pool);

/* Creates a coroutine. stack_size == 0 means use pool default.
 * Returns SPOON_OK and writes the handle to *out_co on success. */
int spoon_create(spoon_pool_t *pool, spoon_fn fn, void *arg,
                  size_t stack_size, spoon_co_t **out_co);

/* Variant for wiring in an externally-mapped stack (e.g. from libsva).
 * alloc_hook receives stack_size and must return a pointer to at least
 * that much memory. free_hook is called on destroy. */
int spoon_create_with_stack(spoon_pool_t *pool, spoon_fn fn, void *arg,
                             size_t stack_size,
                             void *(*alloc_hook)(size_t size, void *user_data),
                             void (*free_hook)(void *ptr, size_t size, void *user_data),
                             void *user_data,
                             spoon_co_t **out_co);

/* Switches execution from the calling context into `co`. Returns when
 * `co` yields back (via spoon_yield) or completes. After return,
 * co->status is SPOON_SUSPENDED or SPOON_DONE.
 *
 * Returns SPOON_ERR_INVALID if co is NULL, already SPOON_DONE, the
 * current coroutine itself (self-switch), or already in the current
 * context's caller chain (nested asymmetric switch A->B->C->A).
 * Thread-safety: not thread-safe. Must be called from the thread that
 * owns the pool. */
int spoon_switch_to(spoon_co_t *co);

/* Yields from the current coroutine back to its caller. Equivalent to
 * spoon_switch_to(caller) but does not require the caller's handle.
 * Only valid when called from inside a coroutine (not from the main
 * thread). Returns SPOON_ERR_INVALID if called from the main thread
 * or if the current coroutine has no caller. */
int spoon_yield(void);

spoon_status_t spoon_status(const spoon_co_t *co);

/* Only valid when status == SPOON_DONE. Frees the coroutine's stack
 * back to the pool and removes it from the pool's slot table. Returns
 * silently without freeing if status is not SPOON_DONE (READY,
 * RUNNING, or SUSPENDED). To forcibly tear down non-DONE coroutines,
 * release the entire pool with spoon_pool_destroy. */
void spoon_destroy(spoon_co_t *co);

#ifdef __cplusplus
}
#endif

#endif /* SPOON_H */
