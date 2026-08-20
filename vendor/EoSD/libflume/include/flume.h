#ifndef FLUME_H
#define FLUME_H

/*
 * libflume: wait-free MPSC ring buffer (LOCK XADD producers, single
 * consumer). See API.md and DESIGN.md for the design rationale and the
 * v0.1 non-goals (no MPMC, no blocking waits, no variable-length msgs,
 * no MPMC).
 *
 * Slot size is fixed at 64 bytes (one cache line): 8-byte sequence
 * number + 56-byte payload. Capacity must be a power of two.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLUME_MSG_SIZE 56

/* Public message type. Callers may also pass a raw 56-byte buffer to
 * flume_enqueue; flume_drain always produces flume_msg_t. The struct
 * has no padding so an array of N flume_msg_t is exactly N * 56 bytes
 * and can be passed directly to a bulk deserializer. */
typedef struct {
    uint8_t data[FLUME_MSG_SIZE];
} flume_msg_t;

/* Opaque ring handle. The struct definition lives in src/flume.c. */
typedef struct flume_ring flume_t;

/* Error codes stay in negative integer space so callers can uniformly
 * check `if (rc < 0)` per EoSD-SPEC.md section 4. */
typedef enum {
    FLUME_OK             = 0,
    FLUME_ERR_INVALID    = -1,  /* NULL handle, non-power-of-two capacity,
                                 * size > FLUME_MSG_SIZE, etc. */
    FLUME_ERR_FULL       = -2,  /* enqueue timed out after spin_timeout_ns */
    FLUME_ERR_EMPTY      = -3,  /* dequeue found no ready slot */
    FLUME_ERR_NOMEM      = -4,  /* mmap or alloc failure in flume_create */
} flume_err_t;

/*
 * Allocates and initializes a ring with `capacity` slots. Capacity must
 * be a power of two and >= 1; otherwise returns NULL. The ring is
 * mmap'd (anonymous, MAP_PRIVATE) and 64-byte aligned. Returns NULL on
 * allocation failure.
 *
 * Thread-safety: safe to call concurrently with other flume_create
 * calls; each returns an independent ring. The returned handle is then
 * safe for concurrent producers and a single consumer.
 */
flume_t *flume_create(size_t capacity);

/*
 * Attaches a ring to a caller-provided region. The region must be at
 * least flume_ring_bytes(capacity) bytes, 64-byte aligned, and
 * writable. libflume does not take ownership of the region; the caller
 * must keep it mapped for the lifetime of the ring. Use flume_detach
 * (not flume_destroy) to release the handle without unmapping.
 *
 * This is the integration point for libsva: the caller mmaps a guarded
 * region via sva_map_guarded and passes sva_base(region) here. A
 * producer bug that writes past the end of the ring then hits the
 * PROT_NONE guard page and raises SIGSEGV. See DESIGN.md for the
 * wiring example.
 *
 * Thread-safety: same as flume_create.
 */
flume_t *flume_attach(void *base, size_t capacity);

/*
 * Releases a ring created by flume_create. munmaps the underlying
 * region and frees the bookkeeping struct. NULL is a no-op. Must not
 * be called on a ring returned by flume_attach; use flume_detach for
 * those.
 *
 * Thread-safety: caller must ensure no concurrent enqueue/dequeue/
 * drain on the same handle.
 */
void flume_destroy(flume_t *f);

/*
 * Releases a ring created by flume_attach without unmapping the
 * backing region. NULL is a no-op. The caller owns the region's
 * lifetime.
 *
 * Thread-safety: same as flume_destroy.
 */
void flume_detach(flume_t *f);

/*
 * Enqueues one message. `size` must be <= FLUME_MSG_SIZE; bytes beyond
 * `size` in the slot are left unmodified (callers should not rely on
 * zero-padding). Producers acquire a slot via LOCK XADD on
 * write_index and spin (with PAUSE) until the slot's sequence number
 * signals the previous lap's data has been drained.
 *
 * `spin_timeout_ns`:
 *   0          -> try once, give up immediately if the slot is not
 *                 ready. No abandonment: the producer has not yet
 *                 written anything, so the slot stays in its previous
 *                 state.
 *   UINT64_MAX -> spin forever. The producer never abandons the slot.
 *                 Use this when the caller retries enqueue on FULL or
 *                 when message loss is unacceptable.
 *   other      -> spin for at most `spin_timeout_ns` nanoseconds.
 *
 * Returns 0 on success, FLUME_ERR_FULL if the slot did not become
 * ready within the spin window, FLUME_ERR_INVALID if `f` is NULL,
 * `msg` is NULL with size > 0, or `size` > FLUME_MSG_SIZE.
 *
 * Wait-free: a single LOCK XADD orders this producer relative to
 * every other producer. The post-XADD spin is per-slot, not
 * per-producer, and completes in bounded time once the consumer
 * drains the previous lap.
 *
 * Abandonment hazard: if flume_enqueue returns FLUME_ERR_FULL after
 * a finite spin, the producer has claimed a slot (via LOCK XADD) but
 * not published to it. The slot is left "ready for producer idx" with
 * no producer coming to fill it. The next producer to claim idx +
 * capacity will spin forever waiting for the consumer to drain the
 * abandoned lap, which it never does. The ring stalls. Callers that
 * cannot tolerate this must pass UINT64_MAX. See DESIGN.md.
 *
 * Thread-safety: safe to call concurrently from multiple producers.
 */
int flume_enqueue(flume_t *f, const void *msg, size_t size,
                  uint64_t spin_timeout_ns);

/*
 * Dequeues one message. Copies up to `out_size` bytes (or
 * FLUME_MSG_SIZE, whichever is smaller) into `out_msg`. Returns 0 on
 * success, FLUME_ERR_EMPTY if the ring is empty at the read index,
 * FLUME_ERR_INVALID if `f` is NULL.
 *
 * Single-consumer only: no internal synchronization. Concurrent calls
 * from two consumers race on read_index and corrupt the ring.
 *
 * Thread-safety: NOT safe to call concurrently from multiple threads.
 */
int flume_dequeue(flume_t *f, void *out_msg, size_t out_size);

/*
 * Drains up to `max_count` messages into `out_msgs`. Returns the
 * number drained (0..max_count). Stops at the first empty slot. The
 * caller may pass &out_msgs[0].data directly to a bulk deserializer
 * (e.g. libpack) since each flume_msg_t is 56 contiguous bytes with
 * no padding.
 *
 * Thread-safety: same as flume_dequeue (single consumer).
 */
size_t flume_drain(flume_t *f, flume_msg_t *out_msgs, size_t max_count);

/*
 * Current queue depth: write_index - read_index. Monotonically
 * accurate as of the read; the value may change before the caller
 * inspects it. Convert to ns of "lag time" via libtick's TSC by
 * sampling tick_now() before and after a producer burst and dividing
 * the lag by the TSC rate (see DESIGN.md).
 *
 * Thread-safety: safe to call from any thread.
 */
uint64_t flume_lag(const flume_t *f);

/*
 * Returns the ring's capacity in slots. Returns 0 if `f` is NULL.
 *
 * Thread-safety: safe to call from any thread.
 */
size_t flume_capacity(const flume_t *f);

/*
 * Returns the number of bytes a ring of `capacity` slots occupies,
 * including the cache-line-aligned index header. Use this to size a
 * region passed to flume_attach. Returns 0 if capacity is not a power
 * of two or is zero.
 */
size_t flume_ring_bytes(size_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* FLUME_H */
