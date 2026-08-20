/*
 * bench_enqueue: single-producer enqueue latency with a background
 * drainer. Reports ns/op.
 *
 * The drainer runs on a separate thread so the producer's LOCK XADD
 * never blocks on a full ring (the 8192-slot capacity absorbs timing
 * hiccups). The reported number is therefore the pure producer-side
 * cost: LOCK XADD + spin-check (almost always satisfied first try) +
 * 56-byte copy + release store. Expected ~10-30 ns/op on modern
 * x86_64.
 */
#include <flume.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define NITERS 1000000
#define CAP    8192

static flume_t *g_flume;
static atomic_int g_stop = 0;

static void *drainer(void *arg)
{
    (void)arg;
    flume_msg_t batch[64];
    while (!atomic_load(&g_stop)) {
        size_t n = flume_drain(g_flume, batch, 64);
        if (n == 0) {
            __builtin_ia32_pause();
        }
    }
    /* Final drain: clear anything left in the ring so the producer's
     * last enqueues are not counted against the drainer's exit. */
    for (;;) {
        size_t n = flume_drain(g_flume, batch, 64);
        if (n == 0) {
            break;
        }
    }
    return NULL;
}

int main(void)
{
    g_flume = flume_create(CAP);
    if (!g_flume) {
        fprintf(stderr, "bench_enqueue: flume_create failed\n");
        return 1;
    }

    pthread_t dr;
    if (pthread_create(&dr, NULL, drainer, NULL) != 0) {
        fprintf(stderr, "bench_enqueue: pthread_create\n");
        flume_destroy(g_flume);
        return 1;
    }

    /* Warm up: prime caches and let the drainer reach steady state. */
    for (int i = 0; i < 1000; i++) {
        uint8_t msg[FLUME_MSG_SIZE] = {0};
        flume_enqueue(g_flume, msg, sizeof(msg), UINT64_MAX);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NITERS; i++) {
        uint8_t msg[FLUME_MSG_SIZE] = {0};
        /* UINT64_MAX: never abandon. The drainer keeps the ring from
         * filling in steady state, so the spin almost never fires. */
        flume_enqueue(g_flume, msg, sizeof(msg), UINT64_MAX);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    atomic_store(&g_stop, 1);
    pthread_join(dr, NULL);
    flume_destroy(g_flume);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    double ns_per_op = (double)ns / (double)NITERS;
    printf("bench_enqueue: %d enqueues, %ld ns total, %.2f ns/op\n",
           NITERS, ns, ns_per_op);
    return 0;
}
