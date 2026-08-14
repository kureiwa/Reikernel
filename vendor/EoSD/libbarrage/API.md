# libbarrage: API (v0.3)

Status: shipped. Implementation in src/barrage.c, header in include/barrage.h,
tests in tests/.

## Overview

Per-thread bump allocator with batch reset. The arena is a heap-allocated
struct passed by caller, not a `_Thread_local` variable. The contract is
one arena per thread, enforced by documentation; a debug-mode TID check on
`barrage_alloc` may be added in a later version to catch cross-thread
misuse at runtime. The backing store size is fixed at creation; allocating
past capacity is an error, and there's no growth. Reset just moves the top
pointer back to base. There are no destructors or finalizers, no
per-allocation cleanup; it's entirely the caller's responsibility not to
hold dangling pointers after a reset. Strictly single-instance-per-thread
(enforced by convention, not internal locking).

## Types

```c
typedef struct barrage_arena barrage_arena_t;   // opaque

typedef struct {
    void *(*alloc)(size_t size, void *user_data);   // backing store allocation
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} barrage_allocator_t;   // NULL fields = default: mmap for arenas >= 1 MiB,
                        // libc malloc for smaller (1 MiB is the inclusive
                        // mmap threshold). alloc and free must both be
                        // non-NULL or both be NULL; if exactly one is set,
                        // the allocator is rejected and the default path is
                        // used. This prevents the asymmetric-dispatch bug
                        // where a default-alloc'd region is freed via a
                        // user-provided free (or vice versa).

typedef enum {
    BARRAGE_OK              = 0,
    BARRAGE_ERR_INVALID     = -1,
    BARRAGE_ERR_OUT_OF_SPACE = -2,
} barrage_err_t;
```

## API

```c
// Reserves `size` bytes as the arena's backing store (one allocation via
// `allocator`, or default). Fixed for the arena's lifetime; it never grows.
barrage_arena_t *barrage_create(size_t size, const barrage_allocator_t *allocator /* nullable */);

void barrage_destroy(barrage_arena_t *arena);

// Bump-allocates `size` bytes with `align`-byte alignment. `align` must be
// a power of two and <= 64 (alignof(T) for any standard type satisfies this;
// the 64-byte cap matches the struct alignment, and any alignment <= 64 can
// be satisfied from a 64-aligned base). Returns NULL and sets *out_err =
// BARRAGE_ERR_INVALID if `align` is 0, not a power of two, or > 64. Returns
// NULL with *out_err = BARRAGE_ERR_OUT_OF_SPACE if the aligned request does
// not fit. The fast path is a few extra instructions for alignment vs. the
// always-16-aligned variant (lea/neg/add/and on x86_64), so callers
// requesting alignas(64) (cache line) or alignas(32) (AVX) pay negligible
// cost. `size == 0` returns a valid aligned pointer and does not advance
// top, so two consecutive zero-size allocs return the same address. This
// differs from malloc(0), which may return distinct pointers.
void *barrage_alloc(barrage_arena_t *arena, size_t size, size_t align,
                    barrage_err_t *out_err);

// Moves the top pointer back to base. All previously returned pointers
// become invalid immediately. No cleanup callbacks are invoked, no
// madvise(2) is issued; callers wanting MADV_DONTNEED or MADV_FREE on the
// used range must call madvise themselves.
void barrage_reset(barrage_arena_t *arena);

// Introspection, useful for capacity planning / debugging.
size_t barrage_used(const barrage_arena_t *arena);
size_t barrage_capacity(const barrage_arena_t *arena);
```

## Cache-line alignment

`barrage_arena_t` is `alignas(64)`-padded: the first member carries
`alignas(64)`, the struct inherits the 64-byte alignment requirement, and
`sizeof` is a multiple of 64. `_Static_assert`s in src/barrage.c enforce
both. This prevents false sharing between per-thread arenas allocated from
the same malloc heap. The arena struct is allocated separately from the
backing store via `aligned_alloc(64, sizeof(*arena))`. Callers embedding
the arena pointer in their own structs are unaffected (the struct is
opaque and always accessed through a pointer), but anyone who copies the
struct by value inherits the 64-byte alignment requirement.

## Backing store and destroy dispatch

The backing store is one allocation, captured at `barrage_create` time and
released as one allocation in `barrage_destroy`. The create-side dispatch:

  - if the caller's `allocator.alloc` is non-NULL (after pair validation),
    use it; `backing_via_mmap` is set to 0;
  - else if `size >= 1 MiB`, mmap an anonymous private region
    (`MAP_PRIVATE | MAP_ANONYMOUS`); `backing_via_mmap` is set to 1;
  - else libc malloc; `backing_via_mmap` is set to 0.

The destroy-side dispatch uses `backing_via_mmap` to remember which
default path was taken, then falls through to the user's free, then to
libc free, in that priority order:

  1. `backing_via_mmap` -> `munmap(base, size)`;
  2. else `allocator.free` non-NULL -> call it with `(base, size, user_data)`;
  3. else `free(base)`.

The `backing_via_mmap` flag exists because the default mmap path does not
record a `free` callback, so destroy cannot dispatch on `allocator.free`
alone. Pair validation at create time guarantees that `allocator.free` is
non-NULL only when `allocator.alloc` was actually used (in which case
`backing_via_mmap` is 0), so the three-way dispatch never faces an
ambiguous case.

## Non-goals

- No growth past the fixed backing store size: `barrage_create` capacity
  is a hard ceiling in v0.3.
- No per-allocation free; arenas only free in bulk via `barrage_reset`.
- No finalizer/destructor callbacks on reset.
- No internal thread-safety/locking: one arena per thread, enforced by
  convention/documentation, not code.
- No `madvise(2)` calls from `barrage_reset`: callers wanting
  `MADV_DONTNEED` (zero-fill on next access, drops RSS) or `MADV_FREE`
  (lazy free, faster reset) on reset should call `madvise` themselves on
  the arena's `[base, top)` range. libbarrage does not manage RSS.
