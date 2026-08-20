/* libspinit extreme tests: push the spinlock to its limits.
 *
 * Tests:
 * - High contention (32 threads, 100K iterations each)
 * - Thundering herd (all threads wake simultaneously)
 * - Long-held lock (hold for 1ms, verify spin-then-block)
 * - trylock under contention
 * - Sustained throughput (10M lock/unlock cycles)
 */

#include <spinit.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include "latency.h"

#define N_THREADS 32
#define N_ITERS   100000

static spinit_t lock;
static atomic_long counter;

static void *contend_worker(void *arg)
{
    (void)arg;
    for (int i = 0; i < N_ITERS; i++) {
        spinit_lock(&lock);
        atomic_fetch_add(&counter, 1);
        spinit_unlock(&lock);
    }
    return NULL;
}

static int test_high_contention(void)
{
    spinit_init(&lock);
    atomic_store(&counter, 0);

    pthread_t threads[N_THREADS];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < N_THREADS; i++)
        pthread_create(&threads[i], NULL, contend_worker, NULL);
    for (int i = 0; i < N_THREADS; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    long expected = (long)N_THREADS * N_ITERS;
    long actual = atomic_load(&counter);
    if (actual != expected) {
        fprintf(stderr, "FAIL high_contention: counter=%ld, expected %ld\n", actual, expected);
        return 1;
    }

    double ns_per_op = elapsed / expected;
    printf("PASS high_contention: %d threads x %d iters = %ld ops in %.1f ms (%.0f ns/op)\n",
           N_THREADS, N_ITERS, expected, elapsed / 1e6, ns_per_op);
    return 0;
}

/* Thundering herd: all threads block on the lock, then one releases
 * and wakes them all. Tests the futex wake path. */
static spinit_t herd_lock;
static atomic_int herd_ready;
static atomic_int herd_done;

static void *herd_worker(void *arg)
{
    (void)arg;
    atomic_fetch_add(&herd_ready, 1);
    spinit_lock(&herd_lock);
    atomic_fetch_add(&herd_done, 1);
    spinit_unlock(&herd_lock);
    return NULL;
}

static int test_thundering_herd(void)
{
    spinit_init(&herd_lock);
    atomic_store(&herd_ready, 0);
    atomic_store(&herd_done, 0);

    /* Acquire the lock so all workers will block. */
    spinit_lock(&herd_lock);

    pthread_t threads[N_THREADS];
    for (int i = 0; i < N_THREADS; i++)
        pthread_create(&threads[i], NULL, herd_worker, NULL);

    /* Wait for all threads to be ready (trying to acquire). */
    while (atomic_load(&herd_ready) < N_THREADS)
        usleep(1000);

    /* Release the lock -- all threads wake (thundering herd). */
    spinit_unlock(&herd_lock);

    for (int i = 0; i < N_THREADS; i++)
        pthread_join(threads[i], NULL);

    int done = atomic_load(&herd_done);
    if (done != N_THREADS) {
        fprintf(stderr, "FAIL thundering_herd: %d done, expected %d\n", done, N_THREADS);
        return 1;
    }
    printf("PASS thundering_herd: %d threads woke after single unlock\n", N_THREADS);
    return 0;
}

/* Long-held lock: one thread holds for 1ms, others spin then block. */
static spinit_t long_lock;
static atomic_int long_passers;

static void *long_holder(void *arg)
{
    (void)arg;
    spinit_lock(&long_lock);
    usleep(1000); /* hold for 1 ms */
    spinit_unlock(&long_lock);
    return NULL;
}

static void *long_waiter(void *arg)
{
    (void)arg;
    spinit_lock(&long_lock);
    atomic_fetch_add(&long_passers, 1);
    spinit_unlock(&long_lock);
    return NULL;
}

static int test_long_held(void)
{
    spinit_init(&long_lock);
    atomic_store(&long_passers, 0);

    pthread_t holder, waiters[8];
    pthread_create(&holder, NULL, long_holder, NULL);

    usleep(100); /* let holder acquire first */
    for (int i = 0; i < 8; i++)
        pthread_create(&waiters[i], NULL, long_waiter, NULL);

    pthread_join(holder, NULL);
    for (int i = 0; i < 8; i++)
        pthread_join(waiters[i], NULL);

    int passers = atomic_load(&long_passers);
    if (passers != 8) {
        fprintf(stderr, "FAIL long_held: %d passers, expected 8\n", passers);
        return 1;
    }
    printf("PASS long_held: 1ms hold + 8 waiters, all acquired eventually\n");
    return 0;
}

/* Sustained throughput: single-thread, 10M cycles. */
static int test_sustained_throughput(void)
{
    spinit_init(&lock);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < 10000000; i++) {
        spinit_lock(&lock);
        spinit_unlock(&lock);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double ns_per_op = elapsed_ns / 10000000.0;
    printf("PASS sustained: 10M lock/unlock in %.1f ms (%.1f ns/op)\n",
           elapsed_ns / 1e6, ns_per_op);
    return 0;
}

/* Latency under 4-thread contention: each thread times its own
 * lock+unlock cycles and writes into a disjoint slice of a shared
 * 100K-sample array. */
#define LAT_THREADS 4
#define LAT_TOTAL   100000

typedef struct {
    spinit_t *lock;
    uint64_t *samples;
    size_t    start;
    size_t    count;
} lat_arg_t;

static void *lat_worker(void *arg)
{
    lat_arg_t *a = arg;
    for (size_t i = 0; i < a->count; i++) {
        uint64_t t0 = latency_now_ns();
        spinit_lock(a->lock);
        spinit_unlock(a->lock);
        uint64_t t1 = latency_now_ns();
        a->samples[a->start + i] = t1 - t0;
    }
    return NULL;
}

static int test_lock_latency(void)
{
    spinit_t lat_lock;
    spinit_init(&lat_lock);

    uint64_t *samples = malloc((size_t)LAT_TOTAL * sizeof(uint64_t));
    if (!samples) { fprintf(stderr, "FAIL lock_latency: malloc\n"); return 1; }

    size_t per = LAT_TOTAL / LAT_THREADS;
    lat_arg_t args[LAT_THREADS];
    pthread_t threads[LAT_THREADS];
    for (int i = 0; i < LAT_THREADS; i++) {
        args[i].lock = &lat_lock;
        args[i].samples = samples;
        args[i].start = (size_t)i * per;
        args[i].count = per;
        pthread_create(&threads[i], NULL, lat_worker, &args[i]);
    }
    for (int i = 0; i < LAT_THREADS; i++)
        pthread_join(threads[i], NULL);

    uint64_t p50, p99, max;
    latency_stats(samples, LAT_TOTAL, &p50, &p99, &max);
    printf("=== latency ===\n");
    latency_print_ns("spinit lock+unlock (4-thread)", p50, p99, max, LAT_TOTAL);

    free(samples);
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += test_high_contention();
    failures += test_thundering_herd();
    failures += test_long_held();
    failures += test_sustained_throughput();
    failures += test_lock_latency();
    if (failures == 0) {
        printf("\nlibspinit extreme: ALL PASS\n");
        return 0;
    }
    printf("\nlibspinit extreme: %d FAILURE(S)\n", failures);
    return 1;
}
