# libspoon: API (v0.3, shipped)

Status: shipped. Implementation in `src/spoon.c` + `src/spoon_x86_64.asm`
(x86_64 only, System V AMD64 ABI). Tests in `tests/`, benches in `bench/`,
extreme tests in `tests/extreme/`. All signatures match `include/spoon.h`.

## Overview

Hand-rolled asymmetric coroutine context switch. No built-in scheduler: the
caller drives all switches explicitly. Each coroutine has at most one caller
at a time (the context that most recently switched to it); the caller chain
forms a tree rooted at the main thread, and nested asymmetric switches
(A->B->C->A) are rejected (see `spoon_switch_to`). Stack overflow protection
is out of scope; pair with `libsva` guard pages via `spoon_create_with_stack`
if needed (documented pattern, no linkage).

## Types

```c
typedef struct spoon_pool spoon_pool_t;   // opaque, owns stack memory
typedef struct spoon_co   spoon_co_t;     // opaque, one coroutine handle

typedef enum {
    SPOON_OK               = 0,
    SPOON_ERR_INVALID      = -1,
    SPOON_ERR_FULL         = -2,
    SPOON_ERR_STACK_ALLOC  = -3,
} spoon_err_t;

typedef enum {
    SPOON_READY,      // created, not yet started
    SPOON_RUNNING,    // currently executing
    SPOON_SUSPENDED,  // switched away from mid-execution
    SPOON_DONE,       // function returned
} spoon_status_t;

typedef void (*spoon_fn)(spoon_co_t *self, void *arg);
```

## Pool / allocator

```c
typedef struct {
    void *(*alloc)(size_t size, void *user_data);
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} spoon_allocator_t;   // NULL fields = default to libc malloc/free

// default_stack_size applies unless overridden per-coroutine at spoon_create().
// default_stack_size below SPOON_MIN_STACK (16 KB) is rejected (returns NULL).
spoon_pool_t *spoon_pool_create(size_t capacity, size_t default_stack_size,
                                 const spoon_allocator_t *allocator /* nullable */);

// Forcibly tears down the pool. Every remaining coroutine is freed
// regardless of status (including SUSPENDED and RUNNING ones that
// spoon_destroy would refuse); this is the escape hatch for pool
// teardown and is safe because no caller code can resume a coroutine
// after the pool is gone.
void spoon_pool_destroy(spoon_pool_t *pool);
```

## Coroutine lifecycle

```c
// stack_size == 0 means "use pool default".
// stack_size below SPOON_MIN_STACK (16 KB) is rejected with
// SPOON_ERR_INVALID. There is no silent clamping: a caller passing an
// alloc_hook backed by a fixed-size buffer must allocate at least
// SPOON_MIN_STACK, otherwise libspoon would overrun the buffer.
int spoon_create(spoon_pool_t *pool, spoon_fn fn, void *arg,
                  size_t stack_size, spoon_co_t **out_co);

// Variant for wiring in an externally-mapped stack (e.g. from libsva's
// guarded mmap wrapper): caller passes alloc_hook/free_hook instead of using
// the pool's allocator for this one coroutine's stack. alloc_hook receives
// stack_size and must return a pointer to at least that much memory;
// free_hook is called with the same pointer once the coroutine is destroyed.
// This is the documented integration point for pairing with libsva.
// libspoon itself has no libsva dependency, it just accepts any
// caller-supplied memory via these hooks.
//
// stack_size below SPOON_MIN_STACK is rejected with SPOON_ERR_INVALID
// before alloc_hook is invoked, so the hook never receives a size it
// would have to clamp.
int spoon_create_with_stack(spoon_pool_t *pool, spoon_fn fn, void *arg,
                             size_t stack_size,
                             void *(*alloc_hook)(size_t size, void *user_data),
                             void (*free_hook)(void *ptr, size_t size, void *user_data),
                             void *user_data,
                             spoon_co_t **out_co);

// Switches execution from the calling context into `co`. Returns when
// `co` yields back (via spoon_yield on the original caller) or completes.
// After return, co->status is SPOON_SUSPENDED or SPOON_DONE.
//
// Returns SPOON_ERR_INVALID if:
//   - co is NULL,
//   - co is already SPOON_DONE,
//   - co is the current coroutine (self-switch),
//   - co appears in the caller chain of the current context (nested
//     asymmetric switch A->B->C->A). The asymmetric invariant forbids
//     cycles in the caller chain; attempting one would corrupt saved
//     contexts. Asymmetric coroutines cannot be nested symmetrically --
//     callers wanting symmetric handoff must build it explicitly on
//     top of spoon_switch_to and spoon_yield.
int spoon_switch_to(spoon_co_t *co);

// Yields from the current coroutine back to its caller. Equivalent to
// spoon_switch_to(caller) but does not require the caller's handle.
// Only valid when called from inside a coroutine (not from the main
// thread). Returns SPOON_ERR_INVALID if called from the main thread
// (tl_current == NULL) or if the current coroutine has no caller.
int spoon_yield(void);

spoon_status_t spoon_status(const spoon_co_t *co);

// Only valid to call once status == SPOON_DONE. Frees the coroutine's
// stack back to the pool and removes it from the pool's slot table.
//
// If status is not SPOON_DONE (READY, RUNNING, or SUSPENDED), returns
// silently without freeing -- destroying a SUSPENDED coroutine would
// free its stack while its saved RSP still points into that stack;
// destroying a RUNNING coroutine would free the stack the caller is
// currently executing on. To forcibly tear down non-DONE coroutines,
// release the entire pool with spoon_pool_destroy.
void spoon_destroy(spoon_co_t *co);
```

## Non-goals

- No scheduler, no timeslicing, no priorities: purely a manual context
  switch primitive.
- No symmetric coroutine handoff. Asymmetric coroutines cannot be
  nested symmetrically (A->B->C->A is rejected); callers wanting
  symmetric semantics must build them on top of spoon_switch_to and
  spoon_yield.
- No stack overflow detection built in: pair with `libsva` guard pages if
  needed (documented pattern, not linked).
- No cross-thread coroutine migration: a coroutine created in one thread's
  pool must be switched to only from that thread.
