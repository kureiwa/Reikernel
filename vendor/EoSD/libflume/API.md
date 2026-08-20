# libflume: API (v0.3, shipped)

Status: shipped. Implementation in `src/flume.c` + `src/flume_x86_64.asm`
(x86_64 only, System V AMD64 ABI). Tests in `tests/`, benches in `bench/`.
All signatures match `include/flume.h`. The public API has been stable
since v0.1; the v0.3 bump reflects the toolkit-wide version alignment
and the addition of `bench/bench_mpsc.c` (multi-producer contention
sweep). No behavioral change to the API surface.

## Overview

Wait-free MPSC (multi-producer, single-consumer) ring buffer. Producers
acquire a slot with a single `LOCK XADD` on `write_index` and publish
via a release store on the slot's sequence number. The single consumer
drains one or more slots with no atomic RMW of its own -- only an
acquire load of the slot sequence and a relaxed advance of
`read_index`. Slot size is fixed at 64 bytes (one cache line): 8-byte
sequence + 56-byte payload. Capacity must be a power of two.

Designed for real-time latency-sensitive pipelines where the consumer
needs deterministic O(1) dequeue and the producer cannot tolerate CAS
retry loops under contention.

## Types

```c
#define FLUME_MSG_SIZE 56

typedef struct {
    uint8_t data[FLUME_MSG_SIZE];
} flume_msg_t;

typedef struct flume_ring flume_t;   /* opaque */

typedef enum {
    FLUME_OK             = 0,
    FLUME_ERR_INVALID    = -1,
    FLUME_ERR_FULL       = -2,
    FLUME_ERR_EMPTY      = -3,
    FLUME_ERR_NOMEM      = -4,
} flume_err_t;
```

`flume_msg_t` has no padding: an array of N `flume_msg_t` is exactly
`N * 56` bytes and `&out[0].data` can be passed directly to a bulk
deserializer.

## API

```c
flume_t *flume_create(size_t capacity);
flume_t *flume_attach(void *base, size_t capacity);
void     flume_destroy(flume_t *f);
void     flume_detach(flume_t *f);

int      flume_enqueue(flume_t *f, const void *msg, size_t size,
                       uint64_t spin_timeout_ns);
int      flume_dequeue(flume_t *f, void *out_msg, size_t out_size);
size_t   flume_drain(flume_t *f, flume_msg_t *out_msgs, size_t max_count);

uint64_t flume_lag(const flume_t *f);
size_t   flume_capacity(const flume_t *f);
size_t   flume_ring_bytes(size_t capacity);
```

### flume_create / flume_destroy

`flume_create(capacity)` mmaps an anonymous, 64-byte-aligned region of
`flume_ring_bytes(capacity)` bytes, initializes the ring (each slot's
sequence set to its index), and returns an opaque handle. `capacity`
must be a power of two and >= 1; otherwise returns NULL. Returns NULL
on mmap failure.

`flume_destroy(f)` munmaps the region and frees the bookkeeping
struct. NULL is a no-op. Must not be called on a handle returned by
`flume_attach`; use `flume_detach` for those.

### flume_attach / flume_detach

`flume_attach(base, capacity)` initializes a ring in a caller-provided
region. The region must be at least `flume_ring_bytes(capacity)` bytes,
64-byte aligned, and writable. libflume does not take ownership of the
region. Use `flume_detach` (not `flume_destroy`) to release the handle.

This is the integration point for libsva's guard-page regions. See
DESIGN.md for the wiring example.

### flume_enqueue

```c
int flume_enqueue(flume_t *f, const void *msg, size_t size,
                  uint64_t spin_timeout_ns);
```

Enqueues one message. `size` must be <= `FLUME_MSG_SIZE`. The producer
acquires a slot via `LOCK XADD` on `write_index`, then spins (with
`PAUSE`) until the slot's sequence number signals the previous lap's
data has been drained, then copies the payload and release-stores the
new sequence.

`spin_timeout_ns`:
- `0` -- try once, give up immediately if the slot is not ready.
- `UINT64_MAX` -- spin forever. Never abandons the slot.
- other -- spin for at most `spin_timeout_ns` nanoseconds.

Returns:
- `0` on success.
- `FLUME_ERR_FULL` if the slot did not become ready within the spin
  window.
- `FLUME_ERR_INVALID` if `f` is NULL, `msg` is NULL with `size > 0`,
  or `size > FLUME_MSG_SIZE`.

**Abandonment hazard.** If `flume_enqueue` returns `FLUME_ERR_FULL`
after a finite spin, the producer has claimed a slot (via `LOCK XADD`)
but not published to it. The slot is left "ready for producer idx"
with no producer coming to fill it. The next producer to claim
`idx + capacity` will spin forever waiting for the consumer to drain
the abandoned lap, which it never does. The ring stalls.

Callers that retry `flume_enqueue` on `FLUME_ERR_FULL`, or that
cannot tolerate message loss, must pass `UINT64_MAX`. Callers that
treat `FLUME_ERR_FULL` as a hard error and tear down the ring can use
a finite timeout. See DESIGN.md.

Thread-safety: safe to call concurrently from multiple producers.

### flume_dequeue

```c
int flume_dequeue(flume_t *f, void *out_msg, size_t out_size);
```

Dequeues one message. Copies up to `out_size` bytes (or
`FLUME_MSG_SIZE`, whichever is smaller) into `out_msg`.

Returns:
- `0` on success.
- `FLUME_ERR_EMPTY` if the ring is empty at the read index.
- `FLUME_ERR_INVALID` if `f` is NULL.

Single-consumer only. Concurrent calls from two consumers race on
`read_index` and corrupt the ring. No internal synchronization.

Thread-safety: NOT safe to call concurrently from multiple threads.

### flume_drain

```c
size_t flume_drain(flume_t *f, flume_msg_t *out_msgs, size_t max_count);
```

Drains up to `max_count` messages into `out_msgs`. Returns the number
drained (0..`max_count`). Stops at the first empty slot. `read_index`
is advanced once at the end of the batch (or not at all if zero
slots were drained).

The caller may pass `&out_msgs[0].data` directly to a bulk
deserializer since each `flume_msg_t` is 56 contiguous bytes with no
padding.

Thread-safety: same as `flume_dequeue` (single consumer).

### flume_lag

```c
uint64_t flume_lag(const flume_t *f);
```

Returns `write_index - read_index` (the current queue depth). Two
relaxed atomic loads; not a single atomic snapshot. The value may
under- or over-count by a small amount under concurrent producers.
For telemetry and lag-time estimation this is fine; for exact
accounting, take the read index lock externally (libflume does not
provide one).

Thread-safety: safe to call from any thread.

### flume_capacity / flume_ring_bytes

```c
size_t flume_capacity(const flume_t *f);
size_t flume_ring_bytes(size_t capacity);
```

`flume_capacity(f)` returns the ring's capacity in slots. Returns 0
if `f` is NULL.

`flume_ring_bytes(capacity)` returns the number of bytes a ring of
`capacity` slots occupies, including the cache-line-aligned index
header (`2 * 64 + capacity * 64`). Returns 0 if `capacity` is not a
power of two or is zero. Use this to size a region passed to
`flume_attach`.

## Minimal usage example

```c
#include <flume.h>
#include <stdio.h>

int main(void) {
    flume_t *f = flume_create(1024);
    if (!f) return 1;

    uint8_t msg[FLUME_MSG_SIZE] = {0};
    msg[0] = 0x42;
    flume_enqueue(f, msg, sizeof(msg), UINT64_MAX);

    uint8_t out[FLUME_MSG_SIZE];
    if (flume_dequeue(f, out, sizeof(out)) == 0) {
        printf("got: 0x%02x\n", out[0]);
    }

    flume_destroy(f);
    return 0;
}
```

## Memory ordering

The producer publishes with a release store on `slot->sequence`; the
consumer observes via an acquire load of the same field. This pairing
guarantees the 56-byte payload is visible to the consumer before it
sees the published sequence. The producer's `LOCK XADD` on
`write_index` is a full barrier on x86_64 (locked instruction). The
consumer's `read_index` advance is a relaxed store (single consumer,
no cross-core visibility requirement). `flume_lag` uses two relaxed
loads and is not an atomic snapshot. See DESIGN.md for the per-step
breakdown.

## Non-goals

- No MPMC. The consumer side is single-threaded by contract; a
  multi-consumer variant would need CAS on `read_index` and would
  lose the lock-free consumer property.
- No blocking waits. Producers spin; consumers poll. Callers wanting
  futex/event-based blocking should wrap `flume_lag` in their own
  wait primitive (see DESIGN.md for the libspinit + libtick pattern).
- No variable-length messages. Slot payload is fixed at 56 bytes.
  Larger payloads should be split across multiple slots or
  referenced by an out-of-line pointer.
- No persistence. The ring lives in anonymous memory; process exit
  loses all in-flight messages.
- No Windows/macOS support in v0.3 (Linux/x86_64 only; `nasm -f
  elf64`).
- No recovery from the abandonment hazard. A stalled ring must be
  destroyed and recreated.
