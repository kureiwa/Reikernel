/* _DEFAULT_SOURCE exposes MAP_ANONYMOUS (a BSD extension, not POSIX). */
#define _DEFAULT_SOURCE

#include <barrage.h>

#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* mmap is the default backing store for arenas >= 1 MiB. Below that,
 * malloc is cheaper (no page-table churn) and the arena is unlikely to
 * benefit from madvise. */
#define BARRAGE_MMAP_THRESHOLD (1ULL << 20)  /* 1 MiB */

/* Cache-line alignment (64 bytes) prevents false sharing between per-thread
 * arenas allocated from the same malloc heap. The first member carries
 * alignas(64); the struct's alignment is the strictest member alignment,
 * so the struct inherits the 64-byte requirement. sizeof is a multiple of
 * 64. */
struct barrage_arena {
    alignas(64) void *base;
    void *top;
    void *end;
    barrage_allocator_t allocator;  /* copy of caller's struct or zeroed defaults */
    _Bool backing_via_mmap;         /* true: munmap on destroy; false: caller free or libc free */
};

_Static_assert(_Alignof(struct barrage_arena) >= 64,
               "barrage_arena must be at least 64-byte aligned");
_Static_assert(sizeof(struct barrage_arena) % 64 == 0,
               "barrage_arena must be padded to a multiple of 64");

barrage_arena_t *barrage_create(size_t size, const barrage_allocator_t *allocator)
{
    /* The arena struct is allocated separately from the backing store and
     * kept 64-byte aligned via aligned_alloc. sizeof is already a multiple
     * of 64 thanks to alignas(64) on the first member. */
    struct barrage_arena *arena = aligned_alloc(64, sizeof(*arena));
    if (!arena) {
        return NULL;
    }
    memset(arena, 0, sizeof(*arena));

    /* Copy the caller's allocator (if any). NULL function pointers in a
     * user-provided struct mean "use the default" per API.md, the same as
     * passing NULL for the whole struct. alloc and free must be supplied as
     * a pair: if exactly one is non-NULL, the allocator is rejected and the
     * default (mmap/malloc) is used instead. This prevents the
     * asymmetric-dispatch bug where a default-alloc'd region would be freed
     * via the user's free (or vice versa). */
    if (allocator) {
        arena->allocator = *allocator;
        if (!!arena->allocator.alloc != !!arena->allocator.free) {
            arena->allocator.alloc = NULL;
            arena->allocator.free = NULL;
        }
    }

    void *base;
    if (arena->allocator.alloc) {
        base = arena->allocator.alloc(size, arena->allocator.user_data);
        arena->backing_via_mmap = 0;
    } else if (size >= BARRAGE_MMAP_THRESHOLD) {
        /* MAP_ANONYMOUS|MAP_PRIVATE: zero-initialized, copy-on-write, not
         * backed by any file. fd is ignored but must be -1 per POSIX.
         * The kernel chooses a page-aligned address; page alignment
         * subsumes the 16/32/64-byte alignment contracts. */
        base = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            free(arena);
            return NULL;
        }
        arena->backing_via_mmap = 1;
    } else {
        /* glibc malloc returns memory aligned to alignof(max_align_t), which
         * is 16 on x86_64. That satisfies the documented alignment contract. */
        base = malloc(size);
        if (!base) {
            free(arena);
            return NULL;
        }
        arena->backing_via_mmap = 0;
    }

    arena->base = base;
    arena->top = base;
    arena->end = (char *)base + size;

    return arena;
}

void barrage_destroy(barrage_arena_t *arena)
{
    if (!arena) {
        return;
    }
    size_t size = (size_t)((char *)arena->end - (char *)arena->base);

    /* Dispatch in priority order: mmap first (only the default large-arena
     * path sets backing_via_mmap), then a caller-provided free, then libc
     * free for the default small-arena path. */
    if (arena->backing_via_mmap) {
        munmap(arena->base, size);
    } else if (arena->allocator.free) {
        arena->allocator.free(arena->base, size, arena->allocator.user_data);
    } else {
        free(arena->base);
    }
    free(arena);
}

void *barrage_alloc(barrage_arena_t *arena, size_t size, size_t align,
                    barrage_err_t *out_err)
{
    if (out_err) {
        *out_err = BARRAGE_OK;
    }
    if (!arena) {
        if (out_err) {
            *out_err = BARRAGE_ERR_INVALID;
        }
        return NULL;
    }

    /* Validate alignment: power of two and <= 64. C11 requires valid
     * alignments to be powers of two, so callers passing alignof(T) are
     * safe. The 64-byte cap matches the struct alignment; any alignment
     * <= 64 can be satisfied from a 64-aligned base. */
    if (align == 0 || (align & (align - 1)) != 0 || align > 64) {
        if (out_err) {
            *out_err = BARRAGE_ERR_INVALID;
        }
        return NULL;
    }

    /* Fast path: align BEFORE advancing.
     *
     *   aligned = align_up(top, align)
     *   new_top = aligned + size
     *   if new_top > end: out of space
     *   top = new_top
     *   return aligned
     *
     * Aligning before advancing means the returned pointer is the aligned
     * base of the new allocation, and the bounds check tests the aligned
     * end. The earlier (buggy) sequence aligned AFTER advancing, returning
     * a pointer into the next allocation's territory. */
    uintptr_t top_u = (uintptr_t)arena->top;
    uintptr_t aligned = (top_u + (align - 1)) & ~((uintptr_t)align - 1);
    uintptr_t end_u = (uintptr_t)arena->end;

    if (aligned > end_u) {
        if (out_err) {
            *out_err = BARRAGE_ERR_OUT_OF_SPACE;
        }
        return NULL;
    }
    /* size > end - aligned catches both overflow (aligned + size wrapping)
     * and genuine exhaustion. Unsigned subtraction is safe here because
     * aligned <= end_u was checked above. */
    if (size > (end_u - aligned)) {
        if (out_err) {
            *out_err = BARRAGE_ERR_OUT_OF_SPACE;
        }
        return NULL;
    }

    uintptr_t new_top = aligned + size;
    arena->top = (void *)new_top;
    return (void *)aligned;
}

void barrage_reset(barrage_arena_t *arena)
{
    if (!arena) {
        return;
    }
    arena->top = arena->base;
}

size_t barrage_used(const barrage_arena_t *arena)
{
    if (!arena) {
        return 0;
    }
    return (size_t)((char *)arena->top - (char *)arena->base);
}

size_t barrage_capacity(const barrage_arena_t *arena)
{
    if (!arena) {
        return 0;
    }
    return (size_t)((char *)arena->end - (char *)arena->base);
}
