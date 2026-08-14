# libtick: Design Notes (v0.3, shipped)

## Problem

Precise sleep-until for event loops with roughly 100-1000 timers, not
100,000/ms. Game loops (60Hz frame pacing with overshoot detection), media
players (vsync scheduling with catch-up skipping), and simple heartbeat-style
scheduling (Raft-like), all without hand-configuring `timerfd` per platform.

## Calibration

`tick_ctx_create()` first probes `CPUID.15H` (TSC frequency leaf) for a
zero-cost frequency in Hz. If the leaf is non-zero and both EAX and ECX are
non-zero, the frequency is computed as `ECX * EBX / EAX` (core crystal clock
times the EBX/EAX ratio) and the runtime calibration loop is skipped. If the
leaf is absent or returns zero (older CPUs, some VMs), it falls back to two
5ms `rdtsc`-vs-`clock_gettime(CLOCK_MONOTONIC)` busy-wait samples (10ms total)
and takes the minimum of the two, using `lfence; rdtsc` (or `rdtscp` when
CPUID.80000001H:EDX[27] is set; `mfence` is not used -- it is not the SDM-
recommended fence for `rdtsc`).

The calibration target is TSC ticks per nanosecond (HZ-independent), not
ticks per OS timer quantum. The runtime HZ value (for the spin-vs-syscall
decision) is detected by parsing `/proc/interrupts` and summing the per-CPU
counts on the `LOC` line over a 100ms window; the per-second rate is
`delta * 10`. If `/proc/interrupts` is absent, unparseable, or the measured
rate falls outside [100, 10000], the fallback is 1000 Hz. There is no
`timerfd_create`-based HZ probe; the implementation relies solely on
`/proc/interrupts`. This keeps the calibration result valid regardless of
`CONFIG_HZ_1000` vs. `CONFIG_HZ_250/300` vs. `NO_HZ_FULL`.

The TSC frequency is cached in the ctx as `tsc_hz` (ticks per second) and
`tsc_per_ns` (Q20 fixed-point: `(tsc_hz << 20) / 1e9`), consumed by
`tick_now` to convert TSC deltas to ns. Separately, `runtime_hz` (the
detected kernel HZ) drives the spin-vs-syscall threshold in
`tick_sleep_until`: `clock_nanosleep` gets within ~1 jiffy, then a brief
TSC spin tightens the last sub-jiffy gap. The two values are independent:
`tsc_per_ns` is a property of the TSC clock, `runtime_hz` is a property of
the kernel's timer interrupt.

The implementation accepts the calibration result if it falls in the
[100 MHz, 10 GHz] range. There is no separate "wildly inconsistent readings"
guard: the min-of-two-samples approach already suppresses single-sample
preemption noise, and the bounds check rejects implausible values. On systems
where TSC is unavailable or the result is out of bounds, `tick_ctx_create()`
returns NULL; the caller then calls `tick_last_error()` to find out why.

### constant_tsc

`tick_now()` is monotonic across CPU migrations as long as the platform
provides a constant-rate TSC. The implementation does NOT verify
CPUID.01H:EDX[8] (`constant_tsc`) at calibration time; modern x86_64 hosts
(all Intel since Nehalem, all AMD since Bulldozer) and the common VM
hypervisors set it. On a system without `constant_tsc` (rare), `tick_now()`
may report non-monotonic values across a CPU migration. This is documented
as a platform requirement rather than detected, on the assumption that the
target audience (game/media loops on a modern host) always has it.

### Granlund-Montgomery magic-number divide

`tick_now`'s fast path divides a 64-bit TSC delta (shifted left by 20 to
form a Q20 value) by `tsc_per_ns`, a 32-bit constant (max ~10.5M for a
10 GHz TSC). A `DIVQ` on this path costs ~20-30 cycles and dominates the
fast path. The library replaces it with a precomputed Granlund-Montgomery
magic number: for divisor `d` with `L = floor(log2(d))` and `d` not a
power of two, `m = ceil(2^(64+L) / d)` fits in 64 bits and
`floor(n / d) = floor(n * m / 2^(64+L))` using 128-bit multiplication.
For power-of-two `d`, `m = 1` and `shift = L` gives `n >> L`.

`compute_div_magic` runs once at `tick_ctx_create`, after `tsc_per_ns` is
known, and stores `{mul, shift}` in `ctx->tsc_per_ns_magic`. The fast path
becomes a `MULQ` (3 cycles) + `SHR` (1 cycle), replacing the ~20-cycle
`DIVQ`. Disassembly of the fast path: `shl; mulq; mov cl,[rbx+0x18];
shrd; shr; and; cmov; jmp`. The same magic is used in
`tick_ctx_check_drift`'s TSC-to-ns conversion so the two paths stay
byte-for-byte consistent.

The fast path is overflow-safe for deltas < 2^40 ticks (~97 min at 3 GHz,
~58 min at 5 GHz); larger deltas take the slow path, a two-step divide
that is overflow-safe for any realistic ctx lifetime (delta * 1e9 < 2^64
for delta < ~580 years at 3 GHz).

Bench impact (`bench_now`, 10M calls): 28-29 ns/op, vs ~22-23 ns/op for
`clock_gettime` via vDSO on the same host. The pre-magic baseline was
~25-30 ns/op with `DIVQ` dominating; the saving is modest because
`rdtscp` latency (~30 cycles) and the indirect call to `ctx->read_tsc`
dominate the fast path, not the divide.

## Drift defense (v0.3)

`constant_tsc` guarantees the TSC frequency does not change with P-states,
but it does not guarantee that the TSC offset between cores is identical.
On multi-socket NUMA boxes, or with certain deep C-state power-saving
modes, the TSC can skew slightly between cores. A thread that migrates
between cores (whether by scheduler decision or by CPU affinity changes)
may see its TSC-predicted monotonic time shift relative to
`CLOCK_MONOTONIC`. The original v0.2 design calibrated once at
`tick_ctx_create` and trusted that calibration forever; on affected
systems this caused `tick_now` to drift by sub-ms to multi-ms over minutes
to hours.

### Mechanism

`tick_ctx_check_drift(ctx)` reads `clock_gettime(CLOCK_MONOTONIC)` and
`rdtsc` close together using the same interleave-and-average trick as
calibration (two `rdtsc` reads bracketing `clock_gettime`, averaged to
estimate the TSC at the instant `clock_gettime`'s internal read occurred).
It computes the TSC-predicted monotonic time using the current
calibration:

```
predicted = monotonic_base_ns + tsc_to_ns(rdtsc - tsc_base)
drift     = |clock_gettime_now - predicted|
```

If `drift > drift_threshold_ns` (default 1 ms), the calibration is
replaced: `monotonic_base_ns` and `tsc_base` are set to the freshly
observed pair. `tsc_hz` and `tsc_per_ns` are never modified, because the
TSC frequency is constant on `constant_tsc` platforms; only the offset
drifts. The function returns the observed drift (pre-recalibration) so
callers can log or monitor it.

The check calls `clock_gettime` (a syscall on most kernels) and is
therefore NOT async-signal-safe. It is intended to run from the
application's event loop, not from a signal handler.

### Automatic check in tick_now

`tick_now` triggers `tick_ctx_check_drift` every 1024 calls or every 5 s
of wall time, whichever comes first. The fast-path cost is one increment
of `tick_now_call_count` plus one compare against 1024; the 5 s timer is
a fallback (subtract `last_drift_check_ns` from the just-computed `result`
and compare against 5e9) that only evaluates when the counter has not yet
tripped, so it is hidden behind the branch predictor's "not taken" path.
The actual check runs at most once per 1024 calls and costs ~1 us (one
`clock_gettime` + two `rdtsc`), amortizing to < 1 ns per `tick_now`.

The 5 s timer uses the `result` already computed for the caller, so it
adds no extra `rdtsc` read on the fast path. After every check (automatic
or manual), `last_drift_check_ns` is set to the freshly observed
`clock_gettime` value, restarting the 5 s window.

### What recalibration does NOT do

- It does not re-measure `tsc_hz` or `tsc_per_ns`. The TSC frequency is
  constant; re-measuring it would add ~10 ms of `measure_tsc_hz_via_clock`
  busy-wait per recalibration, which is unacceptable in a hot path.
- It does not re-detect `runtime_hz`. The kernel HZ does not change at
  runtime on a fixed system.
- It does not move the thread back to its original core. The drift is
  corrected by adjusting the offset, not by undoing the migration.
- It does not run on a background thread. The check is synchronous and
  single-threaded, matching the rest of the library's "one ctx per
  thread" contract.

### Test hook

`tick_test_inject_drift_ns(ctx, drift_ns)` (declared in `tick.c`, not in
`tick.h`) shifts `monotonic_base_ns` by `drift_ns` to simulate TSC skew
without touching `tsc_hz` / `tsc_per_ns`. A subsequent
`tick_ctx_check_drift` observes `|drift_ns|` of drift and, if it exceeds
the threshold, recalibrates. This lets `tests/test_drift.c` exercise the
recalibration path deterministically without waiting for real NUMA skew.

## Epoch

`tick_now()` and all deadline values share an epoch with `CLOCK_MONOTONIC`.
At calibration time the implementation captures a base pair:
`(clock_gettime(CLOCK_MONOTONIC) in ns, rdtsc ticks)` and offsets subsequent
`rdtsc` reads by that pair. The two `rdtsc` reads bracketing
`clock_gettime` are averaged to estimate the TSC at the instant
`clock_gettime`'s internal read occurred. This means callers can compute
deadlines from `clock_gettime(CLOCK_MONOTONIC)` and pass them directly to
`tick_sleep_until`.

```c
const char *tick_last_error(void);   /* thread-local, valid until the next libtick call on this thread */
```

`tick_last_error()` is cleared at the entry of every public libtick function
except `tick_last_error()` itself and `tick_ctx_destroy()`. A successful call
therefore leaves the buffer empty (`""`), matching the documented "valid
until next libtick call" contract.

## Why single-threaded

Locking a scheduling primitive callable from a hot loop defeats its purpose.
The intended fix for multi-threaded use is "one `tick_ctx_t` per thread," not
internal synchronization. This keeps `tick_sleep_until` and the registry
operations lock-free by construction.

## Registry: two binary min-heaps keyed by deadline

The registry is a fixed-size slot array of capacity `capacity`, indexed by
the public `tick_timer_id_t` (a stable slot index). Two parallel binary
min-heaps, `poll_heap` and `cb_heap`, hold slot indices for `TICK_MODE_POLL`
and `TICK_MODE_CALLBACK` timers respectively, ordered by the `deadline_ns`
of the slot each index points at. Each slot carries its current `heap_pos`
(stored in the overloaded `link` field) so `tick_cancel` is O(log n) without
a linear search.

Free slots are tracked via a singly-linked free list stored in the slot's
`link` field (reused as `next_free` when not in use). `free_list_init`
chains slots 0 -> 1 -> ... -> capacity-1 -> SIZE_MAX, so the first
`tick_register` calls return ids 0, 1, 2, ... in registration order.

Operation costs:

| Operation        | Cost                                   |
|------------------|----------------------------------------|
| `tick_register`  | pop free slot, fill, `heap_push` (sift up). O(log n). |
| `tick_cancel`    | `heap_remove_at(slot.heap_pos)`, push to free list. O(log n). |
| `tick_wait_next` | peek `poll_heap` root (min deadline), sleep until it, pop all expired. O(k log n) where k = expired count. |
| `tick_run_pending` | pop all expired from `cb_heap` into a dispatch array, then iterate that array. O(k log n + k) where k = fired count. |

The heap-based design replaced the v0.1 "flat array, O(capacity) linear
scan" approach when scaling to 1000+ timers became a concern. Bench
results (bench_registry.c, this host): register ~37 ns/op, cancel ~30 ns/op,
`wait_next` heap peek ~31 ns/op flat across 100/1000/10000 timers,
confirming O(1) peek and O(log n) mutation. No reversion to linear scan is
planned.

## tick_run_pending dispatch order and snapshot

`tick_run_pending` runs in two passes:

1. **Pop pass.** Pop every `cb_heap` root whose deadline has elapsed, recording
   each popped slot index into a `malloc`'d dispatch array in pop order. Each
   popped slot transitions from `in_use=1` to `firing=1` (not yet free). The
   dispatch array is sized to the pre-pop `cb_heap_size`, which is the upper
   bound on the fired count.

2. **Dispatch pass.** Iterate the dispatch array in order, invoking each
   callback. `cb` and `user_data` are read into locals before the slot is
   returned to the free list, so a callback that (incorrectly) re-registers
   into the same slot cannot corrupt the in-flight dispatch.

Callbacks therefore fire in **non-decreasing deadline order**, not slot-index
order. This matches what a caller would expect from a min-heap timer library
and is now verified by `tests/test_edge.c` (staggered-deadline dispatch
test). The dispatch pass is O(k), not O(capacity) -- only the k popped slots
are visited, not the full slot array.

## Registry reentrancy

The registry is **not reentrant**. `tick_register` / `tick_cancel` must not
be called from inside a callback fired by `tick_run_pending`. Pass 1 of
`tick_run_pending` captures the dispatch set (into the dispatch array) before
any callback runs, so a callback that (incorrectly) mutates the registry
cannot affect *which* slots fire, *nor the order* in which they fire. But
the slot itself may be returned to the free list and then re-consumed by a
re-entrant `tick_register`, leaving the dispatch's `cb`/`user_data` locals
correct but the slot's stored state inconsistent. Re-add the timer after
`tick_run_pending` returns, or use a separate context for periodic work.

## Batching: dropped

The original v0.1 notes mentioned firing expired timers "in groups of 8-16."
On review, this added complexity without a clear benefit at this timer-count
scale. v0.3 fires all expired timers on a single call instead. May revisit
if benchmarking under stress reveals a real reason to cap it.

## Non-goals

- Not a general-purpose task scheduler. No periodic timers, no priorities
  beyond deadline order, no async I/O integration.
- Not trying to replace `timerfd`/`kqueue`. This operates in userspace on top
  of whatever blocking sleep primitive the OS offers, calibrated for accuracy.
- Not reentrant: registry mutations from inside a fired callback are
  forbidden (see above).
- No dynamic registry growth; capacity is fixed at creation.
- No cross-thread timer registration/cancellation; one `tick_ctx_t` per thread.
