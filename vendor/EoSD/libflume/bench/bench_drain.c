/*
 * bench_drain: single-consumer drain throughput with a background
 * producer. Reports messages/sec and ns/msg.
 *
 * The producer runs on a separate thread and pushes NITERS messages.
 * The consumer drains in batches of up to 64 and is the only thread
 * timed. The reported numbers therefore isolate the consumer-side
 * cost: per-slot acquire-load + 56-byte copy + release-store + relaxed
 * read_index advance, amortized across a batch. Expected single-digit
 * ns/msg (tens of M msgs/s) on modern x86_64.
 */
#include <flume.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define NITERS 1000000
#define CAP    8192

static flume_t *g_flume;
static atomic_int g_producer_done = 0;

static void *producer(void *arg)
{
    (void)arg;
    for (uint32_t i = 0; i < (uint32_t)NITERS; i++) {
        uint8_t buf[FLUME_MSG_SIZE];
        memset(buf, 0, sizeof(buf));
        memcpy(buf, &i, sizeof(i));
        /* UINT64_MAX: never abandon. The consumer keeps the ring from
         * filling in steady state. */
        flume_enqueue(g_flume, buf, sizeof(buf), UINT64_MAX);
    }
    atomic_store(&g_producer_done, 1);
    return NULL;
}

int main(void)
{
    g_flume = flume_create(CAP);
    if (!g_flume) {
        fprintf(stderr, "bench_drain: flume_create failed\n");
        return 1;
    }

    pthread_t prod;
    if (pthread_create(&prod, NULL, producer, NULL) != 0) {
        fprintf(stderr, "bench_drain: pthread_create\n");
        flume_destroy(g_flume);
        return 1;
    }

    /* Let the producer prime the ring so the first drain has work. */
    while (flume_lag(g_flume) < 64) {
        __builtin_ia32_pause();
    }

    uint64_t drained = 0;
    flume_msg_t batch[64];
    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (!(atomic_load(&g_producer_done) && flume_lag(g_flume) == 0)) {
        size_t n = flume_drain(g_flume, batch, 64);
        if (n == 0) {
            __builtin_ia32_pause();
            continue;
        }
        drained += n;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    pthread_join(prod, NULL);
    flume_destroy(g_flume);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    double ns_per_msg = drained > 0 ? (double)ns / (double)drained : 0.0;
    double msgs_per_sec = ns > 0 ? (double)drained / ((double)ns / 1e9) : 0.0;
    printf("bench_drain: %llu messages drained, %ld ns total, "
           "%.2f ns/msg, %.2f M msgs/s\n",
           (unsigned long long)drained, ns, ns_per_msg,
           msgs_per_sec / 1e6);
    return 0;
}
