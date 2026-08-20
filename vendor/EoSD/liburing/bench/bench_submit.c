/*
 * bench_submit: pure submission-side latency. Prepares N nop SQEs in a
 * tight loop. Calls uring_enter only at the end of each batch to clear
 * the SQ (otherwise the SQ fills at `entries` and uring_prep_nop starts
 * returning UREING_ERR_SQ_FULL). The reported number is therefore the
 * cost of:
 *
 *   - next_sqe: 2 atomic loads (sq_head, sq_tail) + bounds check
 *   - memset the 64-byte SQE
 *   - 5-6 field stores (opcode, fd, addr, len, off, user_data)
 *   - publish_sqe: 1 store to sq_array + 1 release store to sq_tail
 *
 * Per-batch uring_enter is amortized over BATCH SQEs so it does not
 * dominate. The expected range is 30-80 ns/SQE depending on cache state
 * and the cost of the stdatomic release store (plain MOV on x86_64).
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

#define NITERS 2000000
#define CAP    1024
#define BATCH  256

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
        fprintf(stderr, "bench_submit: uring_create err=%d\n", err);
        return 1;
    }

    /* Warm-up: prime caches and let the kernel's io-wq reach steady state. */
    for (int i = 0; i < 1024; i++) {
        uring_prep_nop(r, (uint64_t)i);
        if ((i & 0xFF) == 0xFF) {
            uring_enter(r, 256, 0, 0);
            int res; uint64_t ud;
            while (uring_reap_cqe(r, &res, &ud) == UREING_OK) { /* drain */ }
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned pending = 0;
    for (int i = 0; i < NITERS; i++) {
        int rc = uring_prep_nop(r, (uint64_t)i);
        if (rc != UREING_OK) {
            /* SQ full: flush and retry this iteration. */
            uring_enter(r, pending, 0, 0);
            int res; uint64_t ud;
            while (uring_reap_cqe(r, &res, &ud) == UREING_OK) { /* drain */ }
            pending = 0;
            rc = uring_prep_nop(r, (uint64_t)i);
            if (rc != UREING_OK) {
                fprintf(stderr, "bench_submit: prep failed after flush rc=%d\n", rc);
                uring_destroy(r);
                return 1;
            }
        }
        pending++;
        if (pending == BATCH) {
            uring_enter(r, pending, 0, 0);
            int res; uint64_t ud;
            while (uring_reap_cqe(r, &res, &ud) == UREING_OK) { /* drain */ }
            pending = 0;
        }
    }
    /* Final flush. */
    if (pending > 0) {
        uring_enter(r, pending, 0, 0);
        int res; uint64_t ud;
        while (uring_reap_cqe(r, &res, &ud) == UREING_OK) { /* drain */ }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
            + (t1.tv_nsec - t0.tv_nsec);
    double ns_per_sqe = (double)ns / (double)NITERS;
    printf("bench_submit: %d SQEs, %ld ns total, %.2f ns/SQE\n",
           NITERS, ns, ns_per_sqe);
    uring_destroy(r);
    return 0;
}
