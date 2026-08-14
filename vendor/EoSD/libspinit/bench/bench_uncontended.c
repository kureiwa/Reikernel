/*
 * bench_uncontended: single-thread 10M lock+unlock cycles. Reports
 * ns/op. Expected ~10-20ns (one lock cmpxchg + one store).
 */
#include <spinit.h>

#include <stdio.h>
#include <time.h>

#define NITERS 10000000

int main(void) {
    spinit_t lock = SPINIT_INIT;
    struct timespec t0, t1;

    /* Warm up calibration and caches. */
    for (int i = 0; i < 1000; i++) {
        spinit_lock(&lock);
        spinit_unlock(&lock);
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NITERS; i++) {
        spinit_lock(&lock);
        spinit_unlock(&lock);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    double ns_per_op = (double)ns / (double)NITERS;
    printf("bench_uncontended: %d lock+unlock cycles, %ld ns total, %.2f ns/op\n",
           NITERS, ns, ns_per_op);
    return 0;
}
