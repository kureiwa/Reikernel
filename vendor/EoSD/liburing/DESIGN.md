# liburing: Design Notes (v0.1)

## Problem

The EoSD toolkit needs a low-level io_uring wrapper for
latency-sensitive I/O: file reads, pipe writes, network send/recv on
rings sized to fit in L1/L2. The existing libflume covers userspace
MPSC; liburing is the kernel-shared counterpart -- a ring whose other
end is the kernel's io-wq worker pool or the inline completion path.

The library is deliberately thin. It does not provide an event loop
(that is the caller's job, and the right shape depends on the
application -- single-thread reactor, thread-per-core, libspoon
coroutines). It does not provide opcode wrappers beyond nop / read /
write (those are the three needed for the v0.1 use cases; extending
the set is a local change). It does not register buffers or files
(that is a setup-time decision that depends on workload and is best
done by the caller). The library's value is the ring setup, the mmap
layout, the memory ordering, and the single-submitter / single-reaper
fast paths.

## Why io_uring

io_uring (Linux 5.1+) is the kernel's asynchronous I/O interface. Two
shared-memory ring buffers (SQ and CQ) plus a single syscall
(`io_uring_enter`) to push work in and pull completions out. Compared
to `epoll` + blocking I/O:

- No per-operation syscall when the SQ has room. The kernel polls the
  SQ tail (with `IORING_SETUP_SQPOLL`) or reads it on
  `io_uring_enter`. In the latter case one syscall submits N SQEs.
- Completions land in the CQ without a syscall when the kernel polls
  (SQPOLL) or are batched into a single `io_uring_enter` with
  `IORING_ENTER_GETEVENTS`.
- No file-descriptor table pressure: io_uring operates on raw fds
  the caller already holds; the ring itself is one fd.

The cost model is the right one for EoSD's latency targets: a
round-trip nop is ~150-200 ns on modern x86_64 (one
`io_uring_enter` + one CQE reap; the bench in `bench/` measures this).

## The mmap layout

io_uring_setup(2) returns a fd and fills `struct io_uring_params`
with the offsets needed to mmap three regions:

```
mmap(fd, size, RW, MAP_SHARED|MAP_POPULATE, IORING_OFF_SQ_RING) -> SQ ring
mmap(fd, size, RW, MAP_SHARED|MAP_POPULATE, IORING_OFF_CQ_RING) -> CQ ring
mmap(fd, size, RW, MAP_SHARED|MAP_POPULATE, IORING_OFF_SQES)    -> SQE array
```

The SQ ring region contains the head, tail, mask, ring_entries,
flags, dropped, and the `array` field (byte offset within the region
of the `__u32` SQ index array). The CQ ring region contains the
head, tail, mask, ring_entries, overflow, and the `cqes` field (byte
offset of the CQE array). The SQE array is a flat array of
`struct io_uring_sqe` (64 bytes each in the standard layout).

Since kernel 5.3 the `IORING_FEAT_SINGLE_MMAP` feature bit is always
advertised: the SQ ring and CQ ring share a single mmap at
`IORING_OFF_SQ_RING`. The library handles both cases (single and
separate) for portability with 5.1-5.2; on 5.3+ the separate path is
dead code.

The region size computation is defensive. The kernel's
`io_uring_mmap` helper sizes the region to cover the highest offset
reported in `sq_off` / `cq_off`. We do the same: compute
`max(sq_off.array + sq_entries * 4, cq_off.cqes + cq_entries *
sizeof(io_uring_cqe))`, round up to a page. Rounding up is harmless
-- the kernel accepts a larger size and only the requested pages are
mapped. We never size below the highest offset; doing so would cause
the kernel to reject the mmap with `EINVAL`.

## SQE publication: the SQ index array

The SQ ring's `sq_array` is an array of `__u32` indices into the SQE
array. The SQE at index `sq_array[tail & mask]` is the next SQE the
kernel will consume. This indirection exists so the kernel can
reorder SQEs for I/O merging without copying the SQE structs; it also
lets userspace write SQEs in any order (the index array defines the
submission order).

liburing always uses the identity mapping `sq_array[idx] = idx` where
`idx = tail & mask`. This means the submission order is the order in
which `uring_prep_*` is called, which is what every caller expects.
The `IORING_SETUP_NO_SQARRAY` flag (5.19+) eliminates the index array
entirely, but we do not set it: the indirection is one cache line on
the submission path and removing it would break 5.1-5.18 compatibility.

## Memory ordering

The SQ and CQ rings are shared between userspace and the kernel. The
kernel reads `sq_tail` (acquire on its side) and writes `sq_head`
(release on its side); we write `sq_tail` (release on our side) and
read `sq_head` (acquire on our side). The CQ ring is the mirror:
kernel writes `cq_tail` (release), reads `cq_head` (acquire); we read
`cq_tail` (acquire), write `cq_head` (release).

```c
/* SQ publication: write SQE fields, then index into sq_array, then
 * release-store sq_tail. The release pairs with the kernel's acquire
 * load of sq_tail. */
sqes[idx] = ...;
sq_array[idx] = idx;
atomic_store_explicit(&sq_tail, tail + 1, memory_order_release);

/* CQ reap: acquire-load cq_tail (pairs with kernel's release store),
 * read CQE fields, release-store cq_head + 1 (pairs with kernel's
 * acquire load). */
tail = atomic_load_explicit(&cq_tail, memory_order_acquire);
if (head == tail) return EMPTY;
res = cqes[head & mask].res;
atomic_store_explicit(&cq_head, head + 1, memory_order_release);
```

On x86_64 these compile to plain `MOV`s (TSO: every store is a
release store, every load is an acquire load). The C11 atomics are
required for correctness on weakly-ordered architectures (future
ARM64 port) and for documentation -- the intent is explicit in the
source rather than implicit in the platform's memory model.

The `uring_barrier()` symbol in `src/uring_x86_64.asm` (sfence +
lfence) is provided for callers that want an explicit full barrier
without going through stdatomic. The library itself does not call
`uring_barrier()` on its hot paths; the stdatomic release / acquire
operations are sufficient. The asm symbol exists as a stable
comparison target for benchmarks and as a fallback for callers that
manipulate ring memory via volatile casts.

## The ASM boundary

One NASM routine in `src/uring_x86_64.asm`:

- `void uring_barrier(void)` -- `sfence; lfence; ret`. A full memory
  barrier suitable for the io_uring kernel-shared-memory ring.
  Equivalent to `__atomic_thread_fence(__ATOMIC_SEQ_CST)` on x86_64
  (gcc lowers that to `mfence`); we emit `sfence + lfence` instead
  because the Intel SDM documents `sfence` as ordering non-temporal
  stores, which `mfence` does not strictly do. For the io_uring ring
  (ordinary WB memory) the two are equivalent.

No `RDTSC` / `RDPMC` / `CPUID` asm symbols: the library does not do
inline timing or feature detection. The kernel reports all needed
state (entries, offsets, features) via `io_uring_params`. If a future
version adds TSC-based timeout helpers, they will follow the libtick
/ libpmu pattern (LFENCE;RDTSC in asm, calibration in C).

## Caching the offsets

`io_uring_params` reports the SQ and CQ ring offsets once at setup
time. The library caches them in the `uring_t` handle as `sq_head_off`,
`sq_tail_off`, `sq_mask_off`, `sq_array_off`, `cq_head_off`,
`cq_tail_off`, `cq_mask_off`, `cq_cqes_off`. Every prep / reap /
pending / ready access computes `(char *)ring_base + off` directly,
with no params indirection on the hot path.

`sq_mask` and `cq_mask` are also cached as `unsigned` (entries - 1)
so the `& mask` fast path is a single AND, not a load from the ring.

## Region teardown

`uring_destroy` munmaps the three regions. When
`IORING_FEAT_SINGLE_MMAP` is in effect (the only case on 5.3+), the
SQ ring and CQ ring alias the same mapping; munmapping twice would
return `EINVAL` on the second call. The `single_mmap` flag in the
handle controls this: when set, only one `munmap` is issued for the
combined region. The SQE array is always a separate mapping and is
always unmapped separately.

After `munmap`, `close(ring_fd)` tears down the kernel-side ring
state. Any SQEs not yet consumed by the kernel are lost; any CQEs not
yet reaped are lost. The kernel does not flush pending work on close.
Callers that need to drain pending completions before teardown should
call `uring_enter(0, UINT_MAX, GETEVENTS)` in a loop until
`uring_cq_ready` returns 0, then `uring_destroy`.

## Single-submitter / single-reaper model

The SQ tail is advanced with a release store that assumes a single
submitter: there is no compare-and-swap on `sq_tail`, only a
`load-relaxed` + `store-release`. Two threads calling `uring_prep_nop`
concurrently would both read the same `sq_tail`, both write to the
same SQE slot, and both store `tail + 1` (one of them losing the
update). The library does not detect this; the resulting corruption
is the caller's responsibility to prevent.

The same applies on the reap side: `cq_head` is advanced with a
release store that assumes a single reaper. Multi-threaded reaping
requires external serialization.

This is consistent with how io_uring is meant to be used: one ring
per producer thread (or per producer-reaper pair), not one ring
shared across many producers. Applications that need many producers
typically use one ring per thread and reap from each independently;
the alternative (a single ring with external locking around
batches) loses io_uring's latency advantage.

The shipped tests and benches are all single-threaded. Multi-threaded
submission patterns are documented in DESIGN.md (below) as
integration examples, not library code.

## Benchmarks

Test host: 2-CPU Intel Xeon sandbox, kernel 5.10.134, gcc 14.2.0,
nasm 2.16.01. Build: `make -C liburing clean && make -C liburing &&
make -C liburing bench`. Numbers are from a single representative
run; run-to-run variance is ~5-15% on this oversubscribed host.

| Bench | Metric | Result |
|---|---|---|
| bench_submit    | ns per SQE (nop, batched 256, no GETEVENTS) | 54.84 ns/SQE |
| bench_roundtrip | ns per op (prep + enter + GETEVENTS + reap)  | 170.00 ns/op (0.17 us/op) |

`bench_submit` measures the pure submission-side cost: `next_sqe` (2
atomic loads), `memset` of the 64-byte SQE, the SQE field stores, and
`publish_sqe` (1 store to `sq_array` + 1 release store to `sq_tail`).
The `io_uring_enter` call is amortized over the 256-SQE batch so it
contributes ~10-20 ns/SQE; the rest is userspace ring bookkeeping.

`bench_roundtrip` is the worst-case latency path: one
`io_uring_enter(GETEVENTS, min_complete=1)` per SQE, no batching. The
syscall is the dominant cost (~120-150 ns); the prep and reap add
~20-30 ns each. The result is consistent with io_uring's documented
nop round-trip latency on modern x86_64.

A dedicated multi-core host with a pinned SQ poll thread
(IORING_SETUP_SQPOLL, not exposed in v0.1) would push
`bench_roundtrip` below 100 ns/op by eliminating the syscall on the
hot path. That is out of scope for v0.1.

## Known limitations

- **No SQPOLL.** The `IORING_SETUP_SQPOLL` flag is not exposed. Every
  `uring_enter` is a syscall. Callers needing sub-microsecond
  submission latency under sustained load should extend the setup
  flags locally (or wait for a v0.2 that adds SQPOLL support).
- **No IOPOLL.** `IORING_SETUP_IOPOLL` (kernel-side polling for
  O_DIRECT reads / writes) is not exposed. It is rarely the right
  choice outside of database workloads.
- **No buffer / file registration.** Fixed buffers (`IORING_REGISTER_BUFFERS`)
  and fixed files (`IORING_REGISTER_FILES`) eliminate per-SQE fd
  lookup and enable zero-copy I/O. Not wrapped in v0.1.
- **No timeout / cancel / linked SQE helpers.** The SQE setup helper
  set covers nop, read, write. Callers needing `IORING_OP_TIMEOUT`,
  `IORING_OP_ASYNC_CANCEL`, `IOSQE_IO_LINK`, etc. can build the SQE
  by hand (the `struct io_uring_sqe` layout is in
  `/usr/include/linux/io_uring.h`) and call the library's internal
  `publish_sqe` equivalent, or extend the prep helper set locally.
- **Single submitter / single reaper only.** See "Memory ordering"
  and "Single-submitter / single-reaper model" above.
- **No Windows / macOS.** io_uring is Linux-only. The NASM file is
  `elf64`-only. A future port would require `kqueue` (macOS) or IOCP
  (Windows) backends with substantially different semantics.

## Integration patterns (not implemented)

Per EoSD-SPEC.md section 3 (zero cross-deps), the following
integrations are documented patterns, not library code:

### liburing + libtopo (when libtopo ships)

Pin the io_uring SQ poll thread to a specific core via
`IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF` and
`params.sq_thread_cpu = libtopo_core(...)`. This avoids
scheduler-induced latency spikes on the submission path. Without
libtopo, the SQ poll thread is placed by the kernel's scheduler and
may migrate.

### liburing + libtick

Use `IORING_OP_TIMEOUT` (not wrapped in v0.1) to bound the
`io_uring_enter(GETEVENTS)` wait. Submit a timeout SQE alongside the
real work with `ts = libtick_now() + deadline_ns`. The kernel posts a
CQE for the timeout when it fires; the reaper breaks out of a
`GETEVENTS` wait on that CQE. Without libtick, use `CLOCK_MONOTONIC` directly.

### liburing + libflume

libflume as a userspace MPSC feeder: producers enqueue work to a
libflume ring; a single consumer thread drains the libflume ring,
converts each message to an io_uring SQE via `uring_prep_*`, and
batches submissions. This decouples producer-side contention
(libflume's LOCK XADD) from kernel-side submission (io_uring's
single-submitter model). The libflume `user_data` becomes the
io_uring `user_data` for completion routing.

### liburing + libpmu

Measure io_uring latency with hardware counters: open a libpmu
context for `PMU_CYCLES` or `PMU_CACHE_MISSES`, sample around the
`io_uring_enter` call, attribute cycles to the syscall vs the
userspace ring bookkeeping. Useful for tuning batch size and SQ poll
thread placement.

### liburing + libsva

Guard the io_uring mmap regions with libsva guard pages. A bug in the
library or the caller that writes past the end of the SQE array hits
a `PROT_NONE` guard page and raises `SIGSEGV` instead of corrupting
adjacent memory. The pattern: `sva_map_guarded(size, ...)` then
pass the resulting base to a future `uring_attach(base, entries)` API
(not in v0.1; the library always mmaps its own regions).
