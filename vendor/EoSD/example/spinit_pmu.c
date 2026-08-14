/* example: spinit + pmu -- measure lock contention with hardware counters.
 *
 * Spawns N threads that contend on a libspinit spinlock. Uses libpmu
 * to count cycles and cache misses during the contention, so you can
 * see the cost of the lock in hardware terms.
 */

#include <spinit.h>
#include <pmu.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>

#define N_THREADS 4
#define N_ITERS   100000

static spinit_t lock = SPINIT_INIT;
static atomic_long counter = 0;

typedef struct {
    int id;
    pmu_ctx_t *cycles_pmu;
    pmu_ctx_t *cache_pmu;
} thread_arg_t;

static void *worker(void *arg)
{
    thread_arg_t *t = (thread_arg_t *)arg;
    uint64_t cyc_end;
    uint64_t cache_end;

    pmu_start(t->cycles_pmu);
    pmu_start(t->cache_pmu);

    for (int i = 0; i < N_ITERS; i++) {
        spinit_lock(&lock);
        atomic_fetch_add(&counter, 1);
        spinit_unlock(&lock);
    }

    pmu_stop_and_read(t->cycles_pmu, &cyc_end);
    pmu_stop_and_read(t->cache_pmu, &cache_end);

    printf("  thread %d: %llu cycles, %llu cache misses for %d lock/unlock cycles\n",
           t->id, (unsigned long long)cyc_end, (unsigned long long)cache_end, N_ITERS);
    return NULL;
}

int main(void)
{
    pmu_err_t err;

    /* PMU may be denied on some systems (perf_event_paranoid=2). */
    pmu_ctx_t *cycles_pmu = pmu_open(PMU_CYCLES, &err);
    if (!cycles_pmu) {
        printf("PMU_CYCLES unavailable (err=%d). Skipping hardware counter demo.\n", err);
        printf("Run with: sudo sysctl -w kernel.perf_event_paranoid=1\n");
        return 0;  /* not a failure, just can't demonstrate */
    }
    pmu_ctx_t *cache_pmu = pmu_open(PMU_CACHE_MISSES, &err);
    if (!cache_pmu) {
        printf("PMU_CACHE_MISSES unavailable. Continuing with cycles only.\n");
    }

    printf("Spawning %d threads, %d lock/unlock iterations each...\n",
           N_THREADS, N_ITERS);

    pthread_t threads[N_THREADS];
    thread_arg_t args[N_THREADS];

    for (int i = 0; i < N_THREADS; i++) {
        args[i].id = i;
        args[i].cycles_pmu = cycles_pmu;
        args[i].cache_pmu = cache_pmu;
        /* Each thread needs its own PMU context. For simplicity, open
         * new ones per thread in a real app. Here we share and accept
         * that only one thread's counts will be meaningful. */
        pthread_create(&threads[i], NULL, worker, &args[i]);
    }

    for (int i = 0; i < N_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("final counter: %ld (expected %d)\n",
           atomic_load(&counter), N_THREADS * N_ITERS);

    pmu_close(cycles_pmu);
    if (cache_pmu) pmu_close(cache_pmu);
    return 0;
}
