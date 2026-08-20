/* bench_switch.c -- measure spoon_switch latency.
 *
 * One coroutine increments a counter and yields. The main thread
 * resumes it. Each round-trip is 2 switches (main->co, co->main).
 * This is the correct asymmetric pattern: the coroutine always yields
 * back to the main thread, never to another coroutine. */

#include "spoon.h"
#include <stdio.h>
#include <time.h>

#define N_ROUNDS 2000000

static int counter = 0;

static void worker_fn(spoon_co_t *self, void *arg)
{
    (void)self; (void)arg;
    while (counter < N_ROUNDS) {
        counter++;
        spoon_yield();
    }
}

int main(void)
{
    spoon_pool_t *pool = spoon_pool_create(4, 65536, NULL);
    if (!pool) { fprintf(stderr, "pool_create failed\n"); return 1; }

    spoon_co_t *co = NULL;
    if (spoon_create(pool, worker_fn, NULL, 0, &co) != SPOON_OK) {
        fprintf(stderr, "create failed\n"); return 1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Drive the coroutine to completion. Each spoon_switch_to resumes
     * the coroutine; it yields back after incrementing. */
    while (counter < N_ROUNDS) {
        spoon_switch_to(co);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    /* Each round-trip is 2 switches: main->co (switch_to) and co->main (yield). */
    double switches = (double)N_ROUNDS * 2.0;
    double ns_per_switch = elapsed_ns / switches;

    printf("bench_switch: %d round-trips (%.0f switches)\n", N_ROUNDS, switches);
    printf("  total: %.2f ms\n", elapsed_ns / 1e6);
    printf("  per switch: %.1f ns\n", ns_per_switch);

    spoon_destroy(co);
    spoon_pool_destroy(pool);
    return 0;
}
