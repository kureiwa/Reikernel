/*
 * libflume implementation. See include/flume.h for the public contract
 * and DESIGN.md for the rationale.
 *
 * Ring layout (offsets in bytes from base):
 *
 *   offset   0 : _Atomic uint64_t write_index   (cache line 0)
 *   offset  64 : _Atomic uint64_t read_index    (cache line 1)
 *   offset 128 : slot[0]  { uint64_t sequence; uint8_t data[56]; }
 *   offset 192 : slot[1]
 *   ...
 *   offset 128 + 64 * capacity : end
 *
 * Producers (wait-free):
 *   idx  = atomic_fetch_add(&write_index, 1)         # LOCK XADD
 *   slot = &slots[idx & (capacity - 1)]
 *   spin until slot->sequence == idx                   # previous lap drained
 *   copy 56 bytes into slot->data
 *   atomic_store_release(&slot->sequence, idx + 1)    # publish
 *
 * Consumer (single, lock-free):
 *   ridx = atomic_load(&read_index)
 *   slot = &slots[ridx & (capacity - 1)]
 *   seq  = atomic_load_acquire(&slot->sequence)
 *   if seq == ridx + 1:                                # data ready
 *       copy 56 bytes out of slot->data
 *       atomic_store_release(&slot->sequence, ridx + capacity)  # next lap
 *       atomic_store(&read_index, ridx + 1)
 *       return OK
 *   if seq == ridx: return EMPTY
 *
 * Sequence invariant: a slot at index i (i.e. slots[i & (cap-1)] on the
 * lap that claims index i) holds:
 *   i + 1        when a producer has published and the consumer has not
 *                yet drained it.
 *   i + capacity when the consumer has drained and is waiting for the
 *                producer that claims index i on the next lap.
 * A producer that has just claimed index i spins until sequence == i,
 * which the consumer sets to i + capacity on the previous lap's drain.
 * Initial state: slot[k].sequence = k for all k in [0, capacity), so the
 * first producer that claims index k sees sequence == k immediately.
 */

#define _GNU_SOURCE
#include <flume.h>

#include <stdatomic.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include <xmmintrin.h>  /* _mm_pause */

/* Assembly helpers in flume_x86_64.asm. flume_xadd_uint64 is the LOCK
 * XADD wrapper used by the producer fast path; flume_copy_56 is a 56-
 * byte unaligned copy via movdqu + movq (SSE2, baseline x86_64, no AVX
 * needed). C11 atomic_fetch_add would also lower to LOCK XADD; the asm
 * symbol makes the fast path explicit in the disassembly and gives the
 * bench a stable comparison target. */
extern uint64_t flume_xadd_uint64(_Atomic uint64_t *p, uint64_t inc);
extern void     flume_copy_56(void *dst, const void *src);

#define FLUME_CACHELINE 64

typedef struct {
    alignas(FLUME_CACHELINE) _Atomic uint64_t sequence;
    uint8_t data[FLUME_MSG_SIZE];
} flume_slot_t;

_Static_assert(sizeof(flume_slot_t) == FLUME_CACHELINE,
               "flume_slot_t must occupy exactly one cache line");
_Static_assert(_Alignof(flume_slot_t) == FLUME_CACHELINE,
               "flume_slot_t must be 64-byte aligned");

/* Two cache lines for the indices, then the slot array. alignas(64) on
 * write_index puts the struct on a 64-byte boundary; alignas(64) on
 * read_index forces 56 bytes of padding between them so producers
 * (touching write_index) and the consumer (touching read_index) never
 * share a cache line. */
struct flume_ring {
    alignas(FLUME_CACHELINE) _Atomic uint64_t write_index;
    alignas(FLUME_CACHELINE) _Atomic uint64_t read_index;
    alignas(FLUME_CACHELINE) flume_slot_t slots[];
};

_Static_assert(offsetof(struct flume_ring, read_index) == FLUME_CACHELINE,
               "write_index and read_index must be on separate cache lines");
_Static_assert(offsetof(struct flume_ring, slots) == 2 * FLUME_CACHELINE,
               "slot array must start on the third cache line");

/* Bookkeeping struct allocated separately from the mmap'd ring so the
 * caller can attach a ring in a foreign region (flume_attach) without
 * libflume owning that region. */
struct flume_handle {
    struct flume_ring *ring;
    size_t capacity;
    void *base;          /* mmap base for flume_destroy; NULL for flume_detach */
    size_t map_size;     /* munmap size for flume_destroy; 0 for flume_detach */
};

static int is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

size_t flume_ring_bytes(size_t capacity) {
    if (!is_power_of_two(capacity)) {
        return 0;
    }
    /* 2 * cache line for the index header + capacity * cache line for slots. */
    return 2 * FLUME_CACHELINE + capacity * FLUME_CACHELINE;
}

static void ring_init(struct flume_ring *ring, size_t capacity) {
    atomic_store_explicit(&ring->write_index, 0, memory_order_relaxed);
    atomic_store_explicit(&ring->read_index, 0, memory_order_relaxed);
    for (size_t i = 0; i < capacity; i++) {
        /* Slot k is initially "drained", waiting for producer k. The
         * first producer that claims index k sees sequence == k. */
        atomic_store_explicit(&ring->slots[i].sequence, i,
                              memory_order_relaxed);
    }
}

flume_t *flume_create(size_t capacity) {
    if (!is_power_of_two(capacity)) {
        return NULL;
    }
    size_t map_size = flume_ring_bytes(capacity);
    if (map_size == 0) {
        return NULL;
    }
    /* mmap returns page-aligned (>= 4 KiB), which subsumes 64-byte
     * alignment. MAP_ANONYMOUS|MAP_PRIVATE: zero-init, no fd. */
    void *base = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }
    struct flume_ring *ring = (struct flume_ring *)base;
    ring_init(ring, capacity);

    struct flume_handle *h = malloc(sizeof(*h));
    if (!h) {
        munmap(base, map_size);
        return NULL;
    }
    h->ring = ring;
    h->capacity = capacity;
    h->base = base;
    h->map_size = map_size;
    return (flume_t *)h;
}

flume_t *flume_attach(void *base, size_t capacity) {
    if (!base || !is_power_of_two(capacity)) {
        return NULL;
    }
    /* Reject a misaligned base. A misaligned ring would still be
     * internally consistent (slot offsets are relative), but every
     * load/store would cross a cache line and the cross-line atomic
     * would be much slower. Better to fail loudly here than ship a
     * silently-slow ring. */
    if (((uintptr_t)base % FLUME_CACHELINE) != 0) {
        return NULL;
    }
    struct flume_ring *ring = (struct flume_ring *)base;
    ring_init(ring, capacity);

    struct flume_handle *h = malloc(sizeof(*h));
    if (!h) {
        return NULL;
    }
    h->ring = ring;
    h->capacity = capacity;
    h->base = NULL;       /* caller owns the region */
    h->map_size = 0;
    return (flume_t *)h;
}

void flume_destroy(flume_t *f) {
    struct flume_handle *h = (struct flume_handle *)f;
    if (!h) {
        return;
    }
    if (h->base && h->map_size) {
        munmap(h->base, h->map_size);
    }
    free(h);
}

void flume_detach(flume_t *f) {
    struct flume_handle *h = (struct flume_handle *)f;
    if (!h) {
        return;
    }
    /* Caller owns the region; do not munmap. */
    free(h);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int flume_enqueue(flume_t *f, const void *msg, size_t size,
                  uint64_t spin_timeout_ns) {
    struct flume_handle *h = (struct flume_handle *)f;
    if (!h || !h->ring) {
        return FLUME_ERR_INVALID;
    }
    if (size > FLUME_MSG_SIZE) {
        return FLUME_ERR_INVALID;
    }
    if (!msg && size > 0) {
        return FLUME_ERR_INVALID;
    }

    /* Wait-free slot claim. LOCK XADD orders this producer against
     * every other concurrent producer; the returned idx is this
     * producer's exclusive slot index for the entire enqueue. */
    uint64_t idx = flume_xadd_uint64(&h->ring->write_index, 1);
    flume_slot_t *slot = &h->ring->slots[idx & (h->capacity - 1)];

    /* Spin until the consumer has drained the previous lap's data for
     * this slot. The slot's sequence is set to (idx + capacity) by the
     * consumer on drain, and to (idx + 1) by the producer on publish;
     * waiting for sequence == idx means waiting for the drain.
     *
     * spin_timeout_ns semantics:
     *   0          -> try once, give up immediately if not ready.
     *   UINT64_MAX -> spin forever (no abandonment).
     *   other      -> spin until now_ns() >= deadline.
     *
     * Abandonment hazard: if a producer times out and returns
     * FLUME_ERR_FULL without publishing, the slot at idx & (cap-1) is
     * left "ready for producer idx" forever. The next producer to
     * claim idx + capacity will spin waiting for sequence == idx +
     * capacity, which the consumer never sets (it never drained the
     * abandoned lap). The ring stalls. Callers that retry enqueue on
     * FULL must pass UINT64_MAX (or otherwise ensure no abandonment),
     * or accept that the ring is single-use after a FULL. See
     * DESIGN.md. */
    uint64_t deadline = 0;
    if (spin_timeout_ns > 0 && spin_timeout_ns != UINT64_MAX) {
        deadline = now_ns() + spin_timeout_ns;
    }
    for (;;) {
        uint64_t seq = atomic_load_explicit(&slot->sequence,
                                            memory_order_acquire);
        if (seq == idx) {
            break;
        }
        /* seq < idx: consumer has not yet drained the previous lap
         * (ring full from this slot's perspective). seq > idx is
         * impossible for a correct ring. */
        if (spin_timeout_ns == 0) {
            return FLUME_ERR_FULL;
        }
        if (spin_timeout_ns != UINT64_MAX && now_ns() >= deadline) {
            return FLUME_ERR_FULL;
        }
        _mm_pause();
    }

    /* Publish: copy the payload, then release-store sequence = idx + 1.
     * The release store pairs with the consumer's acquire-load of
     * slot->sequence, establishing that the payload writes are visible
     * to the consumer before it sees sequence == idx + 1. */
    if (size == FLUME_MSG_SIZE) {
        flume_copy_56(slot->data, msg);
    } else if (size > 0) {
        memcpy(slot->data, msg, size);
    }
    atomic_store_explicit(&slot->sequence, idx + 1, memory_order_release);
    return FLUME_OK;
}

int flume_dequeue(flume_t *f, void *out_msg, size_t out_size) {
    struct flume_handle *h = (struct flume_handle *)f;
    if (!h || !h->ring) {
        return FLUME_ERR_INVALID;
    }
    uint64_t ridx = atomic_load_explicit(&h->ring->read_index,
                                         memory_order_relaxed);
    flume_slot_t *slot = &h->ring->slots[ridx & (h->capacity - 1)];
    uint64_t seq = atomic_load_explicit(&slot->sequence,
                                        memory_order_acquire);
    if (seq == ridx) {
        /* Slot was last published at ridx and has not been refilled;
         * ring is empty at this index. */
        return FLUME_ERR_EMPTY;
    }
    /* seq == ridx + 1 means the slot holds a published message. seq is
     * never any other value for a correct ring (the consumer never
     * reads a slot whose sequence is ahead of read_index + 1). */
    size_t n = out_size < FLUME_MSG_SIZE ? out_size : FLUME_MSG_SIZE;
    if (out_msg && n > 0) {
        if (n == FLUME_MSG_SIZE) {
            flume_copy_56(out_msg, slot->data);
        } else {
            memcpy(out_msg, slot->data, n);
        }
    }
    /* Mark the slot ready for the next lap: the producer that claims
     * index ridx + capacity will spin until sequence == ridx + capacity.
     * The release store pairs with the producer's acquire-load. */
    atomic_store_explicit(&slot->sequence, ridx + h->capacity,
                          memory_order_release);
    atomic_store_explicit(&h->ring->read_index, ridx + 1,
                          memory_order_relaxed);
    return FLUME_OK;
}

size_t flume_drain(flume_t *f, flume_msg_t *out_msgs, size_t max_count) {
    struct flume_handle *h = (struct flume_handle *)f;
    if (!h || !h->ring || !out_msgs || max_count == 0) {
        return 0;
    }
    uint64_t ridx = atomic_load_explicit(&h->ring->read_index,
                                         memory_order_relaxed);
    size_t count = 0;
    while (count < max_count) {
        flume_slot_t *slot = &h->ring->slots[ridx & (h->capacity - 1)];
        uint64_t seq = atomic_load_explicit(&slot->sequence,
                                            memory_order_acquire);
        if (seq != ridx + 1) {
            break;  /* empty at this index */
        }
        flume_copy_56(out_msgs[count].data, slot->data);
        atomic_store_explicit(&slot->sequence, ridx + h->capacity,
                              memory_order_release);
        ridx++;
        count++;
    }
    if (count > 0) {
        atomic_store_explicit(&h->ring->read_index, ridx,
                              memory_order_relaxed);
    }
    return count;
}

uint64_t flume_lag(const flume_t *f) {
    const struct flume_handle *h = (const struct flume_handle *)f;
    if (!h || !h->ring) {
        return 0;
    }
    /* Two relaxed loads. The pair is not a single atomic snapshot, so
     * the result can under-count (if read_index advances between the
     * two loads) or over-count (if write_index advances). For the
     * documented use case (queue-depth telemetry + tick-based lag
     * estimation) the approximation is fine. */
    uint64_t w = atomic_load_explicit(&h->ring->write_index,
                                      memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&h->ring->read_index,
                                      memory_order_relaxed);
    return w - r;
}

size_t flume_capacity(const flume_t *f) {
    const struct flume_handle *h = (const struct flume_handle *)f;
    if (!h) {
        return 0;
    }
    return h->capacity;
}
