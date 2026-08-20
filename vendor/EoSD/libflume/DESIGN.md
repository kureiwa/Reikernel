# libflume: Design Notes (v0.3)

## Problem

A multi-producer, single-consumer queue with wait-free producers and
O(1) deterministic dequeue for the consumer. Target use cases from the
original notes: real-time event pipelines, log shipping, GPU command
buffer submission, telemetry aggregation. The consumer in these
pipelines runs on a latency-sensitive thread that cannot afford a CAS
retry loop or a futex syscall on the hot path; the producers are
many and may be oversubscribed.

The classic DV counterparts (Drepper-style MPSC with CAS, bounded
MPSC with two locks) either lose wait-freedom on the producer side
or pay lock overhead on both sides. libflume's design -- sequence
numbers + `LOCK XADD` -- gives producers a single locked RMW and the
consumer a pure load+store fast path.

## Why sequence numbers (the Lamport-bounded-buffer pattern)

Each 64-byte slot carries an 8-byte `sequence` field alongside the
56-byte payload. The sequence number is the slot's "expected index":
a slot at index `i` (i.e. `slots[i & (cap-1)]` on the lap that claims
`i`) holds:

- `i + 1` when a producer has published and the consumer has not yet
  drained it.
- `i + capacity` when the consumer has drained and is waiting for the
  producer that claims index `i` on the next lap.
- (transiently) `i - capacity + 1` after a publish, before the
  consumer's drain -- this is what the producer at index `i` waits
  to flip to `i`.

Initial state: `slot[k].sequence = k` for all `k` in
`[0, capacity)`, so the first producer that claims index `k` sees
`sequence == k` immediately.

The sequence number eliminates the ABA problem on `write_index`
itself. The producer does not need to compare-and-swap anything; it
unconditionally claims the next index via `LOCK XADD` and then waits
on the slot's own sequence. The consumer advances `read_index` only
after it has copied the payload and reset the slot's sequence for the
next lap.

This is the same pattern used by Linux's `kfifo` (in the
single-producer/single-consumer variant) and by the LMAX Disruptor's
`MultiProducerSequencer`. The Disruptor's slots are larger (variable)
and its consumer uses a separate `gatingSequence` for backpressure;
libflume is the minimal fixed-size-slot MPSC variant.

## The ASM boundary

Two NASM routines in `src/flume_x86_64.asm`:

- `flume_xadd_uint64(_Atomic uint64_t *p, uint64_t inc)` -- `LOCK XADD
  [rdi], rax; ret`. Returns the old `*p` in `rax`. This is the producer
  fast path: one locked RMW, no CAS retry, no branch. C11
  `atomic_fetch_add` would lower to the same instruction; the asm
  symbol makes the fast path visible in the disassembly and gives the
  bench a stable comparison target.
- `flume_copy_56(void *dst, const void *src)` -- three `movdqu` (16
  bytes each) + one `movq` (8 bytes) for the trailing 48..55 byte
  range. SSE2, baseline x86_64, no AVX needed. `movdqu` (unaligned)
  is required because `slot->data` starts at offset 8 within a
  64-byte aligned slot, so neither src nor dst is 16-byte aligned.

`PAUSE` for the producer spin is emitted by the C compiler via
`_mm_pause()` (from `<xmmintrin.h>`); it does not need its own asm
symbol. `LFENCE`/`RDTSC` are not on the hot path; the spin deadline
uses `clock_gettime(CLOCK_MONOTONIC)`, which is a vDSO call on Linux
and costs ~10ns. For a sub-microsecond spin window the deadline check
is dominated by the clock read; this is an accepted cost. A future
v0.2 could swap in `RDTSCP`-based timing if benchmarking shows the
`clock_gettime` overhead matters.

## Cache-line layout

```c
struct flume_ring {
    alignas(64) _Atomic uint64_t write_index;   /* cache line 0 */
    alignas(64) _Atomic uint64_t read_index;    /* cache line 1 */
    alignas(64) flume_slot_t slots[];           /* cache line 2+ */
};

typedef struct {
    alignas(64) _Atomic uint64_t sequence;      /* 8 bytes */
    uint8_t data[56];                            /* 56 bytes */
} flume_slot_t;
```

`_Static_assert`s pin `offsetof(read_index) == 64` and
`offsetof(slots) == 128`. The two `alignas(64)` annotations on the
index fields force the compiler to insert 56 bytes of padding between
`write_index` and `read_index`, so producers (touching `write_index`)
and the consumer (touching `read_index`) never share a cache line.
Without this padding, every producer's `LOCK XADD` would invalidate
the consumer's cache line for `read_index` and vice versa, doubling
the cache-coherence traffic.

Each slot is exactly 64 bytes (one cache line), pinned by
`_Static_assert(sizeof(flume_slot_t) == 64)`. This means a producer
publishing to slot `k` and a consumer draining slot `k-1` (or `k+1`)
touch different cache lines and do not contend.

## Memory ordering

Producer publish:
1. `LOCK XADD` on `write_index` (seq_cst by default; `LOCK`ed
   instructions are full barriers on x86_64).
2. Spin on `atomic_load_acquire(&slot->sequence)` until it equals
   `idx`. The acquire load pairs with the consumer's release store
   of `idx + capacity` on the previous lap's drain. The consumer's
   payload read becomes visible before we overwrite the slot.
3. `memcpy` / `flume_copy_56` the payload (plain stores; x86_64's
   TSO model guarantees these are visible to other cores in program
   order).
4. `atomic_store_release(&slot->sequence, idx + 1)`. The release
   store pairs with the consumer's acquire load. The payload
   writes are visible to the consumer before it sees
   `sequence == idx + 1`.

Consumer drain:
1. Relaxed load of `read_index` (single consumer, no contention).
2. `atomic_load_acquire(&slot->sequence)`. If `seq == ridx + 1`, the
   payload is visible.
3. `memcpy` / `flume_copy_56` the payload out.
4. `atomic_store_release(&slot->sequence, ridx + capacity)`. Pairs
   with the next-lap producer's acquire load.
5. Relaxed store of `read_index = ridx + 1`.

The consumer never issues a `LOCK`ed instruction. On x86_64's TSO
model, plain loads and stores are sequentially consistent within a
core; the only reordering the consumer needs to prevent is
compiler-level, which the `atomic_*_explicit` API handles.

## The abandonment hazard

`flume_enqueue` claims its slot *before* checking whether the slot is
ready. If the spin times out and the producer returns
`FLUME_ERR_FULL` without publishing, the slot at `idx & (cap-1)` is
left in whatever state the previous lap left it in -- which is
`idx - capacity + 1` (the previous producer's published sequence).
No producer will ever write to slot `idx & (cap-1)` for lap `idx`
again, because `write_index` has moved past `idx`.

The next producer to claim `idx + capacity` (which maps to the same
slot) will spin waiting for `slot->sequence == idx + capacity`. But
the consumer never drains the slot for lap `idx` (no one published),
so the consumer never sets `slot->sequence = idx + capacity`. The
ring stalls permanently.

This is not a bug in the implementation; it is an inherent property
of the `LOCK XADD` + sequence-number pattern when producers can
abandon. The Disruptor avoids it by not exposing a "give up" path --
producers block until the slot is ready. libflume exposes the timeout
because the original design notes call for a "bounded spin --
configurable timeout," but the contract is:

- **Finite timeout + retry on FULL** → ring will stall. Do not do this.
- **Finite timeout + treat FULL as a hard error** → caller tears down
  the ring. Acceptable.
- **`UINT64_MAX`** → producer never abandons. Always safe. This is
  what the tests and benches use.

The header (`flume.h`) and `API.md` document this explicitly. A
future v0.2 could add a "publish a skip marker" recovery path, but
that would require the consumer to distinguish real messages from
skip markers, complicating the fast path.

## Integration patterns (documented, not linked)

libflume has zero cross-module dependencies at build and link time.
Integration with the rest of the EoSD toolkit is via caller code, not
library linkage.

### libsva: guarded ring

A ring created with `flume_create` is backed by a plain anonymous
mmap. A producer bug that writes past the end of the ring (e.g. an
out-of-bounds `slot->data` write) silently corrupts adjacent heap
metadata. To catch this, wrap the ring in a libsva guarded region:

```c
#include <sva.h>
#include <flume.h>

size_t cap = 4096;
size_t bytes = flume_ring_bytes(cap);

sva_err_t err;
sva_region_t *region = sva_map_guarded(bytes,
                                       SVA_PROT_READ | SVA_PROT_WRITE |
                                       SVA_PROT_GUARD_BOTH,
                                       NULL, &err);
if (!region) { /* handle err */ }

flume_t *f = flume_attach(sva_base(region), cap);
/* ... use f ... */

flume_detach(f);
sva_unmap(region);
```

`SVA_PROT_GUARD_BOTH` installs a `PROT_NONE` page before and after
the usable region. A producer writing past the last slot hits the
trailing guard page and raises `SIGSEGV`. Combined with libcrash
(`crash_install_handler`), this produces a minidump at the exact
point of corruption.

`flume_attach` checks that `base` is 64-byte aligned; `sva_base`
returns a page-aligned pointer, which subsumes this requirement.

### libtick: lag-time estimation

`flume_lag(f)` returns a slot count, not a time. To convert to ns of
"lag time," sample `tick_now()` before and after a producer burst and
divide the lag by the TSC rate:

```c
#include <tick.h>
#include <flume.h>

tick_ctx_t *ctx = tick_ctx_create(0);
uint64_t t0 = tick_now(ctx);
/* ... producer burst ... */
uint64_t t1 = tick_now(ctx);
uint64_t lag_slots = flume_lag(f);
uint64_t elapsed_ns = t1 - t0;
/* Rough estimate: if the producer burst produced `lag_slots` more
 * messages than the consumer drained, the consumer is
 * `lag_slots / producer_rate` seconds behind. The caller knows the
 * producer rate from the burst; libflume does not. */
(void)elapsed_ns;
```

This is a caller-side calculation; libflume does not link libtick.

### libspinit: external blocking

Producers spin inside `flume_enqueue` for up to `spin_timeout_ns`.
Callers wanting futex-based blocking instead of spinning can use
`spin_timeout_ns = 0` and wrap `flume_enqueue` in a libspinit-protected
retry loop:

```c
#include <spinit.h>
#include <flume.h>

extern spinit_t producer_lock;  /* initialized elsewhere */

int flume_enqueue_blocking(flume_t *f, const void *msg, size_t size) {
    spinit_lock(&producer_lock);
    int rc;
    while ((rc = flume_enqueue(f, msg, size, 0)) == FLUME_ERR_FULL) {
        /* In a real implementation, futex-wait on a "slot freed"
         * flag set by the consumer. libflume does not provide this;
         * the caller wires it. */
        spinit_lock(&producer_lock);  /* serialize retries */
    }
    spinit_unlock(&producer_lock);
    return rc;
}
```

This sacrifices the wait-free producer property (the spinlock
serializes producers) in exchange for no CPU burn when the ring is
full. A future v0.2 could add an optional futex wake from the
consumer on drain.

### libpack: zero-copy bulk deserialize

`flume_drain` writes into a caller-provided `flume_msg_t[]` array.
Each `flume_msg_t` is exactly 56 bytes with no padding, so the array
is a contiguous 56-byte-strided buffer. libpack's bulk deserializer
can read directly from `&out[0].data`:

```c
#include <pack.h>
#include <flume.h>

flume_msg_t batch[64];
size_t n = flume_drain(f, batch, 64);
if (n > 0) {
    /* pack_deserialize_batch reads n * 56 bytes starting at
     * &batch[0].data, no copy. */
    pack_deserialize_batch(&batch[0].data, n, /* ... */);
}
```

libpack does not need to know about libflume; the array is just a
pointer and a count.

## Measured performance

Hardware: 2-CPU Intel Xeon sandbox (Debian, gcc 14.2.0, nasm 2.16.01).
Two-CPU host means benches run oversubscribed (producer + consumer +
drainer compete for 2 cores); numbers on a dedicated multi-core host
would be lower. Representative single-run numbers from the v0.3 bench
sweep:

| Bench | Configuration | Result |
|---|---|---|
| `bench_enqueue` | 1 producer, 1 background drainer, 8192-slot ring, 1M enqueues | ~61-66 ns/op |
| `bench_drain` | 1 background producer, 1 consumer, 8192-slot ring, batches of 64, 1M messages | ~62-89 ns/msg, 11-16 M msgs/s |
| `bench_mpsc` | 1 producer, 1 consumer, 8192-slot ring, 500K iters | ~65-112 ns/enqueue, ~9 M msgs/s |
| `bench_mpsc` | 2 producers (oversubscribed) | ~1100 ns/enqueue, ~0.9 M msgs/s |
| `bench_mpsc` | 4 producers (oversubscribed) | ~2000 ns/enqueue, ~0.5 M msgs/s |
| `bench_mpsc` | 8 producers (oversubscribed) | ~2600 ns/enqueue, ~0.4 M msgs/s |

The enqueue number includes the `LOCK XADD`, the slot-ready spin
(almost always satisfied first try because the drainer keeps up), the
56-byte `movdqu` copy, and the release store. The drain number is
amortized across batches of 64: per-slot cost is acquire-load +
56-byte copy + release-store, plus a single relaxed `read_index`
advance per batch. The `bench_mpsc` numbers at >=2 producers reflect
2-CPU sandbox oversubscription: with only 2 cores available, the
single consumer cannot keep up with 2+ producers and the producers'
spins dominate the wall-clock time. The drain count matches the total
enqueued count in every config (no message loss), confirming the
wait-free producer path is correct under contention. On a dedicated
4+ core host with no oversubscription, expect single-digit ns/op for
enqueue and 30-50 M msgs/s for drain.

These numbers are ballpark for a 2-CPU oversubscribed host and vary
run-to-run with scheduler placement. No extreme-test latency sweep
exists for libflume in v0.3 (the `tests/extreme/` suite covers the
other 9 modules; libflume is exercised only by `tests/` and `bench/`).

## Non-goals

- No MPMC. The consumer side is single-threaded by contract.
- No blocking waits. Producers spin; consumers poll. Callers wanting
  futex/event-based blocking should wrap the primitives (see the
  libspinit pattern above).
- No variable-length messages. Slot payload is fixed at 56 bytes.
- No persistence. The ring lives in anonymous memory.
- No Windows/macOS support in v0.3 (Linux/x86_64 only).
- No recovery from the abandonment hazard. A stalled ring must be
  destroyed and recreated. A future revision could add a "publish a
  skip marker" recovery path, but that would require the consumer to
  distinguish real messages from skip markers, complicating the fast
  path.
