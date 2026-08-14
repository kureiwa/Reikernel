/* clock_gettime needs _POSIX_C_SOURCE >= 199309L. */
#define _POSIX_C_SOURCE 200112L

#include <barrage.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_ALLOCS 1000000u
#define ALLOC_SIZE 32
/* 256 MiB arena so 1M 32-byte allocations with up to 15 bytes of alignment
 * padding (~47 MiB worst case) fit without overflow. */
#define ARENA_SIZE (256ULL * 1024 * 1024)

/* Volatile sink prevents the compiler from eliding allocations whose
 * return values are otherwise unused. */
static volatile void *sink;

int main(void)
{
    struct timespec t0, t1;
    barrage_err_t err;

    barrage_arena_t *arena = barrage_create((size_t)ARENA_SIZE, NULL);
    if (!arena) {
        fprintf(stderr, "bench: barrage_create failed\n");
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < N_ALLOCS; i++) {
        void *p = barrage_alloc(arena, ALLOC_SIZE, 16, &err);
        if (!p) {
            fprintf(stderr, "bench: barrage_alloc failed at %zu\n", i);
            barrage_destroy(arena);
            return 1;
        }
        sink = p;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double barrage_ns =
        (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
        (double)(t1.tv_nsec - t0.tv_nsec);

    barrage_reset(arena);
    barrage_destroy(arena);

    /* malloc comparison: keep all pointers so the allocator does the
     * actual work (a free immediately after each malloc would let tcache
     * absorb the cost, masking the difference). */
    static void *ptrs[N_ALLOCS];

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < N_ALLOCS; i++) {
        ptrs[i] = malloc(ALLOC_SIZE);
        if (!ptrs[i]) {
            fprintf(stderr, "bench: malloc failed at %zu\n", i);
            return 1;
        }
        sink = ptrs[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double malloc_ns =
        (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
        (double)(t1.tv_nsec - t0.tv_nsec);

    for (size_t i = 0; i < N_ALLOCS; i++) {
        free(ptrs[i]);
    }

    printf("barrage_alloc (%u x %d bytes): %.2f ns/op\n",
           N_ALLOCS, ALLOC_SIZE, barrage_ns / (double)N_ALLOCS);
    printf("malloc        (%u x %d bytes): %.2f ns/op\n",
           N_ALLOCS, ALLOC_SIZE, malloc_ns / (double)N_ALLOCS);
    return 0;
}
