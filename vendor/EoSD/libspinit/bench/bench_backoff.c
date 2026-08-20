/*
 * bench_backoff: compares spinit_lock (exponential backoff, cap=64)
 * against an inline naive test-and-test-and-set spinlock with no backoff
 * (one PAUSE per iteration) at 2/4/8 threads.
 *
 * Both locks use the same three-state futex fallback so the only varying
 * factor is the backoff in the spin window. The naive spin count is fixed
 * at SPINIT_FALLBACK_ITERATIONS (500); spinit's calibrated count may be
 * lower on constant-TSC hosts, which biases the comparison in spinit's
 * favor on hosts where calibration produces a smaller count.
 *
 * Expected: spinit matches or beats naive at 2+ threads; the gap widens
 * with thread count because the naive spin hammers the cache line every
 * iteration while spinit backs off exponentially.
 */
#define _GNU_SOURCE
#include <spinit.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <errno.h>

#define N_ITERS   500000
#define MAX_THREADS 8

/* Naive lock: same layout as spinit_t (cache-line-aligned state) and the
 * same three-state futex fallback, but the spin loop issues a single
 * PAUSE per iteration (no exponential backoff). Isolates the backoff
 * variable. */
typedef struct { alignas(64) _Atomic int state; } naive_lock_t;

#define NAIVE_SPIN_ITERS 500

static void naive_lock(naive_lock_t *l) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&l->state, &expected, 1)) {
        return;
    }
    for (int i = 0; i < NAIVE_SPIN_ITERS; i++) {
        __builtin_ia32_pause();
        if (atomic_load(&l->state) != 0) {
            continue;
        }
        expected = 0;
        if (atomic_compare_exchange_strong(&l->state, &expected, 1)) {
            return;
        }
    }
    int waited = 0;
    for (;;) {
        int new_state = waited ? 2 : 1;
        expected = 0;
        if (atomic_compare_exchange_strong(&l->state, &expected, new_state)) {
            return;
        }
        if (expected == 1) {
            int e = 1;
            atomic_compare_exchange_strong(&l->state, &e, 2);
        }
        if (atomic_load(&l->state) != 2) {
            continue;
        }
        long rc = syscall(SYS_futex, &l->state,
                          FUTEX_WAIT | FUTEX_PRIVATE_FLAG, 2,
                          NULL, NULL, 0);
        (void)rc;
        waited = 1;
    }
}

static void naive_unlock(naive_lock_t *l) {
    int prev = atomic_exchange(&l->state, 0);
    if (prev == 2) {
        long rc = syscall(SYS_futex, &l->state,
                          FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1,
                          NULL, NULL, 0);
        (void)rc;
    }
}

/* ---- bench driver ---- */

static spinit_t     g_sp_lock;
static naive_lock_t g_na_lock;
static atomic_long  g_counter;

static void *sp_worker(void *arg) {
    int niters = *(int *)arg;
    for (int i = 0; i < niters; i++) {
        spinit_lock(&g_sp_lock);
        atomic_fetch_add(&g_counter, 1);
        spinit_unlock(&g_sp_lock);
    }
    return NULL;
}

static void *na_worker(void *arg) {
    int niters = *(int *)arg;
    for (int i = 0; i < niters; i++) {
        naive_lock(&g_na_lock);
        atomic_fetch_add(&g_counter, 1);
        naive_unlock(&g_na_lock);
    }
    return NULL;
}

static double run(void *(*fn)(void *), int nthreads) {
    atomic_store(&g_counter, 0);
    pthread_t ts[MAX_THREADS];
    int niters = N_ITERS;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&ts[i], NULL, fn, &niters);
    }
    for (int i = 0; i < nthreads; i++) {
        pthread_join(ts[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    long total = (long)nthreads * N_ITERS;
    return (double)ns / (double)total;
}

int main(void) {
    spinit_init(&g_sp_lock);
    atomic_store(&g_na_lock.state, 0);

    int configs[] = { 2, 4, 8 };

    printf("bench_backoff: %d iters/thread, spinit cap=64 vs naive (no backoff)\n",
           N_ITERS);
    for (size_t i = 0; i < sizeof(configs)/sizeof(configs[0]); i++) {
        int t = configs[i];
        double sp = run(sp_worker, t);
        double na = run(na_worker, t);
        printf("  %d threads: spinit=%.2f ns/op  naive=%.2f ns/op  ratio=%.2fx\n",
               t, sp, na, na / sp);
    }
    return 0;
}
