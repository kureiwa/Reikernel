#ifndef UREING_H
#define UREING_H

/*
 * liburing: a thin, low-level wrapper around the Linux io_uring interface.
 * See API.md and DESIGN.md for the design rationale and the v0.1 non-goals
 * (no event loop, no scheduler, no integration with other EoSD modules,
 * no SQPOLL, no buffer registration, no registered files).
 *
 * Scope: SQE setup, CQE polling, and the io_uring_enter(2) syscall. The
 * caller batches submissions and calls uring_enter explicitly; the caller
 * polls / reaps completions. Three mmaps (SQ ring, CQ ring, SQE array)
 * per the io_uring ABI; we use IORING_OFF_SQ_RING / IORING_OFF_CQ_RING /
 * IORING_OFF_SQES as documented in io_uring_setup(2).
 *
 * Requires kernel 5.1+ (io_uring was added in 5.1). If io_uring_setup
 * returns ENOSYS, uring_create fails with UREING_ERR_SETUP and the caller
 * is expected to skip tests rather than treat it as a hard failure. The
 * tests in tests/ do exactly that.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque ring handle. The struct definition lives in src/uring.c. */
typedef struct uring uring_t;

/* Error codes stay in negative integer space so callers can uniformly
 * check `if (rc < 0)` per EoSD-SPEC.md section 4. The "UREING_" prefix
 * is intentional (the public symbol namespace is `uring_*`; the error
 * enum constants were named to avoid colliding with a possible future
 * `URING_` macro from the kernel header on systems that ship it). */
typedef enum {
    UREING_OK            = 0,
    UREING_ERR_INVALID   = -1,  /* NULL handle, non-power-of-two entries, etc. */
    UREING_ERR_SETUP     = -2,  /* io_uring_setup(2) failed (ENOSYS, EINVAL, EMFILE, ...) */
    UREING_ERR_MMAP      = -3,  /* mmap of one of the three ring regions failed */
    UREING_ERR_SQ_FULL   = -4,  /* submission queue full (no free SQE) */
    UREING_ERR_CQ_EMPTY  = -5,  /* completion queue empty at the head index */
} uring_err_t;

/* io_uring_enter(2) flag passthroughs. We expose the two callers actually
 * need rather than re-importing the whole flag namespace. */
#define UREING_ENTER_GETEVENTS  0x1u  /* IORING_ENTER_GETEVENTS */
#define UREING_ENTER_SQ_WAKEUP  0x2u  /* IORING_ENTER_SQ_WAKEUP  */
#define UREING_ENTER_SQ_WAIT    0x4u  /* IORING_ENTER_SQ_WAIT    */

/*
 * Creates an io_uring with `entries` SQ slots. `entries` must be a power
 * of two and >= 1; the kernel rounds up to the next power of two anyway
 * but we reject non-power-of-two values locally so the caller gets a
 * deterministic error instead of a silent resize. The CQ ring is sized
 * by the kernel (typically 2 * entries) and reported in params.cq_entries
 * after the setup syscall returns.
 *
 * Internally: io_uring_setup(entries, &params), then three mmaps:
 *   - SQ ring at offset IORING_OFF_SQ_RING (size = SQ ring bytes)
 *   - CQ ring at offset IORING_OFF_CQ_RING (size = CQ ring bytes; with
 *     IORING_FEAT_SINGLE_MMAP this aliases the SQ ring mapping)
 *   - SQE array at offset IORING_OFF_SQES  (size = sq_entries * sizeof(io_uring_sqe))
 *
 * On success returns a heap-allocated uring_t * and, if `err` is non-NULL,
 * writes UREING_OK. On failure returns NULL and writes one of
 * UREING_ERR_INVALID, UREING_ERR_SETUP, UREING_ERR_MMAP to *err.
 *
 * Thread-safety: safe to call concurrently with other uring_create calls;
 * each returns an independent ring. The returned handle is then safe for
 * a single submitter and a single reaper; concurrent submit or concurrent
 * reap on the same handle requires external coordination (the SQ head/
 * tail and CQ head/tail are NOT multi-producer/multi-consumer).
 */
uring_t *uring_create(unsigned entries, uring_err_t *err);

/*
 * Queues an IORING_OP_NOP. The nop SQE completes immediately when
 * io_uring_enter is called; `user_data` is echoed back in the CQE. Useful
 * for latency probes and for warming up the ring. Returns 0 on success,
 * UREING_ERR_SQ_FULL if the SQ has no free slot, UREING_ERR_INVALID if
 * `r` is NULL.
 *
 * Does NOT call io_uring_enter. The caller batches submissions and
 * invokes uring_enter explicitly.
 *
 * Thread-safety: NOT safe to call concurrently from multiple threads on
 * the same ring; the SQ tail is advanced with a release store that
 * assumes a single submitter.
 */
int uring_prep_nop(uring_t *r, uint64_t user_data);

/*
 * Queues an IORING_OP_READ. Reads `len` bytes from `fd` at `offset`
 * (a byte offset for regular files; ignored for sockets / pipes -- pass 0)
 * into `buf`. On completion the CQE's `res` field holds the number of
 * bytes read (>= 0) or a negative errno. `user_data` is echoed back.
 *
 * Returns 0 on success, UREING_ERR_SQ_FULL if the SQ has no free slot,
 * UREING_ERR_INVALID if `r` is NULL or `buf` is NULL with `len` > 0.
 *
 * Does NOT call io_uring_enter.
 *
 * Thread-safety: same as uring_prep_nop (single submitter).
 */
int uring_prep_read(uring_t *r, int fd, void *buf, unsigned len,
                    uint64_t offset, uint64_t user_data);

/*
 * Queues an IORING_OP_WRITE. Writes `len` bytes from `buf` to `fd` at
 * `offset`. On completion the CQE's `res` field holds the number of bytes
 * written (>= 0) or a negative errno. `user_data` is echoed back.
 *
 * Returns 0 on success, UREING_ERR_SQ_FULL if the SQ has no free slot,
 * UREING_ERR_INVALID if `r` is NULL or `buf` is NULL with `len` > 0.
 *
 * Does NOT call io_uring_enter.
 *
 * Thread-safety: same as uring_prep_nop (single submitter).
 */
int uring_prep_write(uring_t *r, int fd, const void *buf, unsigned len,
                     uint64_t offset, uint64_t user_data);

/*
 * Submits queued SQEs to the kernel. Calls io_uring_enter(ring_fd,
 * to_submit, min_complete, flags). `to_submit` is the number of SQEs the
 * caller believes are pending (typically uring_sq_pending(r)); the kernel
 * will consume up to that many. `min_complete` is the minimum number of
 * CQEs to wait for, only meaningful when `flags` includes
 * UREING_ENTER_GETEVENTS. `flags` is a bitwise OR of UREING_ENTER_*.
 *
 * Returns the value returned by io_uring_enter(2): the number of SQEs
 * actually submitted (>= 0) on success, or a negative errno-style code
 * on failure. Callers can distinguish "submitted fewer than requested"
 * (a non-negative return < to_submit) from a hard error (negative return).
 *
 * Thread-safety: NOT safe to call concurrently from multiple threads on
 * the same ring without external coordination.
 */
int uring_enter(uring_t *r, unsigned to_submit, unsigned min_complete,
                unsigned flags);

/*
 * Reaps one CQE. Reads the CQ entry at the head index (acquire load on
 * the CQ head, pairing with the kernel's release store on the CQ tail),
 * copies out the result and user_data, and advances the head (release
 * store, pairing with the kernel's acquire load of the CQ head).
 *
 * Returns 0 on success, UREING_ERR_CQ_EMPTY if no CQE is ready at the
 * head, UREING_ERR_INVALID if `r` is NULL.
 *
 * Thread-safety: NOT safe to call concurrently from multiple threads on
 * the same ring (single reaper).
 */
int uring_reap_cqe(uring_t *r, int *res, uint64_t *user_data);

/*
 * Reaps up to `max_count` CQEs into caller-provided arrays. `res_out`
 * and `user_data_out` must each point to at least `max_count` ints /
 * uint64_t values. Returns the number of CQEs reaped (0..max_count).
 * Stops at the first empty slot. A `max_count` of 0 is a no-op and
 * returns 0.
 *
 * Thread-safety: same as uring_reap_cqe (single reaper).
 */
unsigned uring_drain_cqes(uring_t *r, int *res_out, uint64_t *user_data_out,
                          unsigned max_count);

/*
 * Number of SQEs pending (written by uring_prep_* but not yet consumed
 * by io_uring_enter). Computed as sq_tail - sq_head with both pointers
 * acquired atomically. Returns 0 if `r` is NULL.
 *
 * Thread-safety: safe to call from any thread; the result is a snapshot
 * and may change before the caller inspects it.
 */
unsigned uring_sq_pending(const uring_t *r);

/*
 * Number of CQEs ready to reap (not yet consumed by uring_reap_cqe /
 * uring_drain_cqes). Computed as cq_tail - cq_head with both pointers
 * acquired atomically. Returns 0 if `r` is NULL.
 *
 * Thread-safety: safe to call from any thread; the result is a snapshot.
 */
unsigned uring_cq_ready(const uring_t *r);

/*
 * Tears down the ring: munmaps the three regions, closes the io_uring
 * fd, frees the handle. Safe to call with NULL (no-op). Pending SQEs
 * and unreaped CQEs are lost; the kernel does not flush them on close.
 *
 * Thread-safety: caller must ensure no concurrent submit / reap on the
 * same handle.
 */
void uring_destroy(uring_t *r);

#ifdef __cplusplus
}
#endif

#endif /* UREING_H */
