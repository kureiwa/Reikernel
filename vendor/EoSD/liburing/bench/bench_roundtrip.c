/*
 * bench_roundtrip: full submit+enter+reap round-trip latency per nop.
 *
 * For each of NITERS iterations:
 *   - uring_prep_nop
 *   - uring_enter(to_submit=1, min_complete=1, GETEVENTS)
 *   - uring_reap_cqe
 *
 * This is the worst-case latency path: one syscall per SQE, no batching,
 * the kernel must process the SQE and post a CQE before io_uring_enter
 * returns. Expected ~1-3 us/op on modern x86_64 (one io_uring_enter
 * syscall, plus the CQE reaping). The uring_enter(GETEVENTS) call is
 * the dominant cost; the prep and reap are tens of nanoseconds each.
 *
 * If io_uring is unavailable, the bench prints SKIP and exits 0.
 */
#define _GNU_SOURCE
#include <uring.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/syscall.h>
#include <linux/io_uring.h>

#define NITERS 100000
#define CAP    256

static int io_uring_unavailable(void)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    long rc = syscall(SYS_io_uring_setup, 1, &p);
    if (rc >= 0) {
        close((int)rc);
        return 0;
    }
    return 1;
}

int main(void)
{
    uring_err_t err = UREING_OK;
    uring_t *r = uring_create(CAP, &err);
    if (!r) {
        if (err == UREING_ERR_SETUP && io_uring_unavailable()) {
            printf("SKIP: io_uring not available\n");
            return 0;
        }
        fprintf(stderr, "bench_roundtrip: uring_create err=%d\n", err);
        return 1;
    }

    /* Warm-up. */
    for (int i = 0; i < 256; i++) {
        uring_prep_nop(r, (uint64_t)i);
        uring_enter(r, 1, 1, UREING_ENTER_GETEVENTS);
        int res; uint64_t ud;
        uring_reap_cqe(r, &res, &ud);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < NITERS; i++) {
        int rc = uring_prep_nop(r, (uint64_t)i);
        if (rc != UREING_OK) {
            fprintf(stderr, "bench_roundtrip: prep rc=%d\n", rc);
            uring_destroy(r);
            return 1;
        }
        int n = uring_enter(r, 1, 1, UREING_ENTER_GETEVENTS);
        if (n < 0) {
            perror("uring_enter");
            uring_destroy(r);
            return 1;
        }
        int res = -1; uint64_t ud = 0;
        rc = uring_reap_cqe(r, &res, &ud);
        if (rc != UREING_OK) {
            fprintf(stderr, "bench_roundtrip: reap rc=%d at iter %d\n", rc, i);
            uring_destroy(r);
            return 1;
        }
        if (res != 0) {
            fprintf(stderr, "bench_roundtrip: nop res=%d at iter %d\n", res, i);
            uring_destroy(r);
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    double ns_per_op = (double)ns / (double)NITERS;
    printf("bench_roundtrip: %d ops, %ld ns total, %.2f ns/op (%.2f us/op)\n",
           NITERS, ns, ns_per_op, ns_per_op / 1000.0);
    uring_destroy(r);
    return 0;
}
