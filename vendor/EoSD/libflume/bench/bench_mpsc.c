/*
 * bench_mpsc: multi-producer throughput at 1/2/4/8 producers with a
 * single background consumer. Reports ns/enqueue and total M msgs/s.
 *
 * Producers claim slots via LOCK XADD on write_index (wait-free); the
 * consumer drains in batches of 64. The ring is sized at 8192 so the
 * producers do not stall on the consumer in steady state.
 *
 * Expected: ~50-100 ns/enqueue at 1 producer (single LOCK XADD + spin-
 * check + 56-byte copy + release store); degrades gracefully under
 * contention as XADD serializes but the per-slot spin absorbs jitter.
 */
#define _GNU_SOURCE
#include <flume.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAP 8192
#define NITERS_PER_PRODUCER 500000

static flume_t *g_flume;
static atomic_int g_stop = 0;
static atomic_long g_total_drained = 0;

struct producer_arg {
    int id;
    int niters;
};

static void *producer(void *arg) {
    struct producer_arg *a = (struct producer_arg *)arg;
    for (int i = 0; i < a->niters; i++) {
        uint8_t buf[FLUME_MSG_SIZE];
        memset(buf, 0, sizeof(buf));
        /* Encode producer id + sequence so the consumer could verify
         * per-producer FIFO if desired (this bench does not check). */
        uint32_t tag = ((uint32_t)a->id << 16) | (uint16_t)i;
        memcpy(buf, &tag, sizeof(tag));
        flume_enqueue(g_flume, buf, sizeof(buf), UINT64_MAX);
    }
    return NULL;
}

static void *consumer(void *arg) {
    (void)arg;
    flume_msg_t batch[64];
    long drained = 0;
    while (!atomic_load(&g_stop)) {
        size_t n = flume_drain(g_flume, batch, 64);
        if (n == 0) {
            __builtin_ia32_pause();
            continue;
        }
        drained += (long)n;
    }
    /* Final drain. */
    for (;;) {
        size_t n = flume_drain(g_flume, batch, 64);
        if (n == 0) break;
        drained += (long)n;
    }
    atomic_store(&g_total_drained, drained);
    return NULL;
}

static double run(int nproducers) {
    g_flume = flume_create(CAP);
    if (!g_flume) {
        fprintf(stderr, "bench_mpsc: flume_create failed\n");
        exit(1);
    }
    atomic_store(&g_stop, 0);
    atomic_store(&g_total_drained, 0);

    pthread_t cons;
    pthread_create(&cons, NULL, consumer, NULL);

    /* Let the consumer start. */
    struct timespec sleep = { 0, 10 * 1000 * 1000L };
    nanosleep(&sleep, NULL);

    pthread_t ts[8];
    struct producer_arg args[8];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < nproducers; i++) {
        args[i].id = i;
        args[i].niters = NITERS_PER_PRODUCER;
        pthread_create(&ts[i], NULL, producer, &args[i]);
    }
    for (int i = 0; i < nproducers; i++) {
        pthread_join(ts[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    atomic_store(&g_stop, 1);
    pthread_join(cons, NULL);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    long total_enq = (long)nproducers * NITERS_PER_PRODUCER;
    double ns_per_enq = (double)ns / (double)total_enq;
    double msgs_per_sec = ns > 0 ? (double)total_enq / ((double)ns / 1e9) : 0.0;

    printf("  %d producers: %.2f ns/enqueue, %.2f M msgs/s, drained=%ld/%ld\n",
           nproducers, ns_per_enq, msgs_per_sec / 1e6,
           atomic_load(&g_total_drained), total_enq);

    flume_destroy(g_flume);
    return ns_per_enq;
}

int main(void) {
    int configs[] = { 1, 2, 4, 8 };
    printf("bench_mpsc: %d iters/producer, cap=%d, %zu-byte msgs\n",
           NITERS_PER_PRODUCER, CAP, (size_t)FLUME_MSG_SIZE);
    for (size_t i = 0; i < sizeof(configs)/sizeof(configs[0]); i++) {
        run(configs[i]);
    }
    return 0;
}
