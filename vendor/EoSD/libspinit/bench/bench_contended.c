/*
 * bench_contended: 2/4/8 threads, each doing 1M lock+unlock cycles on a
 * shared counter. Reports ns/op per thread configuration.
 *
 * On an oversubscribed machine (more threads than cores) the futex path
 * dominates and per-op cost rises into the microsecond range. On a
 * cores-matching-or-exceeding-threads machine, expect tens to low
 * hundreds of ns/op.
 */
#include <spinit.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NITERS 1000000
#define MAX_THREADS 8

static spinit_t lock;
static unsigned long counter;

struct worker_arg {
    int niters;
};

static void *worker(void *arg) {
    struct worker_arg *a = (struct worker_arg *)arg;
    for (int i = 0; i < a->niters; i++) {
        spinit_lock(&lock);
        counter++;
        spinit_unlock(&lock);
    }
    return NULL;
}

static double run(int nthreads) {
    counter = 0;
    spinit_init(&lock);

    pthread_t ts[MAX_THREADS];
    struct worker_arg a = { .niters = NITERS };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < nthreads; i++) {
        int rc = pthread_create(&ts[i], NULL, worker, &a);
        if (rc != 0) {
            fprintf(stderr, "bench_contended: pthread_create: %d\n", rc);
            exit(1);
        }
    }
    for (int i = 0; i < nthreads; i++) {
        pthread_join(ts[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    long total_ops = (long)nthreads * NITERS;
    return (double)ns / (double)total_ops;
}

int main(void) {
    int configs[] = {2, 4, 8};
    int n = (int)(sizeof(configs) / sizeof(configs[0]));

    printf("bench_contended: %d iterations per thread\n", NITERS);
    for (int i = 0; i < n; i++) {
        double ns_op = run(configs[i]);
        printf("  %d threads: %.2f ns/op\n", configs[i], ns_op);
    }
    return 0;
}
