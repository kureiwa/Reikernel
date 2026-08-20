#ifndef BARRAGE_H
#define BARRAGE_H

#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque arena handle. The struct definition lives in src/barrage.c. */
typedef struct barrage_arena barrage_arena_t;

/* Pluggable backing-store allocator. Pass NULL to barrage_create (or a
 * struct with NULL function pointers) to use the defaults: mmap for arenas
 * >= 1 MiB, libc malloc for smaller. alloc and free must both be non-NULL
 * or both be NULL; if exactly one is non-NULL, the allocator is treated as
 * default. This avoids the mismatch of freeing default-alloc'd memory via
 * a user-provided free (or vice versa). */
typedef struct {
    void *(*alloc)(size_t size, void *user_data);
    void  (*free)(void *ptr, size_t size, void *user_data);
    void  *user_data;
} barrage_allocator_t;

/* Error codes stay in negative integer space so callers can uniformly
 * check `if (rc < 0)` per EoSD-SPEC.md section 4. */
typedef enum {
    BARRAGE_OK               = 0,
    BARRAGE_ERR_INVALID      = -1,
    BARRAGE_ERR_OUT_OF_SPACE = -2,
} barrage_err_t;

/* Reserves `size` bytes as the arena's backing store (one allocation via
 * the chosen allocator). Fixed for the arena's lifetime; never grows.
 * Returns NULL on allocation failure (errno from malloc/mmap is not
 * preserved; callers needing detail should check errno themselves).
 * Thread-safety: safe to call from multiple threads only if each call
 * targets a different arena; the returned handle is single-thread. */
barrage_arena_t *barrage_create(size_t size, const barrage_allocator_t *allocator);

/* Releases the backing store and the arena struct. NULL is a no-op.
 * Thread-safety: caller must ensure no concurrent barrage_alloc/reset on
 * the same handle. */
void barrage_destroy(barrage_arena_t *arena);

/* Bump-allocates `size` bytes with `align`-byte alignment. `align` must be
 * a power of two and <= 64; otherwise returns NULL and sets *out_err to
 * BARRAGE_ERR_INVALID. Returns NULL with *out_err = BARRAGE_ERR_OUT_OF_SPACE
 * if the aligned request does not fit. `out_err` may be NULL.
 * Thread-safety: NOT safe to call concurrently from multiple threads on
 * the same arena. */
void *barrage_alloc(barrage_arena_t *arena, size_t size, size_t align,
                    barrage_err_t *out_err);

/* Moves the top pointer back to base. All previously returned pointers
 * become invalid immediately. No cleanup callbacks are invoked, no
 * madvise(2) is issued; callers wanting MADV_DONTNEED or MADV_FREE on the
 * used range must call madvise themselves.
 * Thread-safety: same contract as barrage_alloc. */
void barrage_reset(barrage_arena_t *arena);

/* Introspection. Both return 0 for a NULL arena. */
size_t barrage_used(const barrage_arena_t *arena);
size_t barrage_capacity(const barrage_arena_t *arena);

#ifdef __cplusplus
}
#endif

#endif /* BARRAGE_H */
