/* example: tick + spoon -- timer-driven coroutine scheduler.
 *
 * A simple cooperative scheduler that uses libtick for timing and
 * libspoon for coroutines. Each coroutine runs for a fixed time slice
 * (10 ms), then yields. The scheduler resumes the next coroutine on
 * the next tick.
 *
 * This demonstrates the documented integration pattern: libtick does
 * not call libspoon (no link-time dep), but the application wires
 * them together.
 */

#include <tick.h>
#include <spoon.h>
#include <stdio.h>
#include <stdlib.h>

#define N_WORKERS 3
#define SLICE_NS   (10 * 1000 * 1000ULL)  /* 10 ms */
#define ROUNDS     3

static spoon_co_t *workers[N_WORKERS];
static spoon_co_t *current;
static int worker_idx = 0;

static void worker_fn(spoon_co_t *self, void *arg)
{
    (void)self;
    const char *name = (const char *)arg;
    for (int i = 0; i < ROUNDS; i++) {
        printf("  [%s] round %d starting\n", name, i);
        spoon_yield();  /* yield back to scheduler */
        printf("  [%s] round %d resumed\n", name, i);
        spoon_yield();
    }
    printf("  [%s] done\n", name);
}

int main(void)
{
    tick_ctx_t *tick = tick_ctx_create(0);
    if (!tick) {
        fprintf(stderr, "tick_ctx_create failed: %s\n", tick_last_error());
        return 1;
    }

    spoon_pool_t *pool = spoon_pool_create(N_WORKERS + 1, 65536, NULL);
    if (!pool) {
        fprintf(stderr, "spoon_pool_create failed\n");
        return 1;
    }

    const char *names[] = {"alpha", "beta", "gamma"};
    for (int i = 0; i < N_WORKERS; i++) {
        if (spoon_create(pool, worker_fn, (void *)names[i], 0, &workers[i]) != SPOON_OK) {
            fprintf(stderr, "spoon_create %d failed\n", i);
            return 1;
        }
    }

    printf("Timer-driven coroutine scheduler: %d workers, %d rounds, %lld ns slices\n",
           N_WORKERS, ROUNDS, (unsigned long long)SLICE_NS);

    for (int i = 0; i < N_WORKERS; i++) {
        worker_idx = i;
        current = workers[i];
        printf("=== scheduling worker %d (%s) ===\n", i, names[i]);
        spoon_switch_to(current);
        /* Sleep for the slice before next worker. */
        uint64_t now = tick_now(tick);
        tick_sleep_until(tick, now + SLICE_NS, NULL);
    }

    /* Second pass to let workers finish. */
    for (int i = 0; i < N_WORKERS; i++) {
        if (spoon_status(workers[i]) != SPOON_DONE) {
            spoon_switch_to(workers[i]);
        }
    }

    for (int i = 0; i < N_WORKERS; i++) {
        spoon_destroy(workers[i]);
    }
    spoon_pool_destroy(pool);
    tick_ctx_destroy(tick);

    printf("All workers completed.\n");
    return 0;
}
