# libspinit: Design Notes (v0.3)

## Problem

A spinlock that doesn't waste CPU spinning indefinitely under contention,
but also doesn't pay futex-syscall latency on the common uncontended/
briefly-contended case. Target use cases from the original notes: thread
pool task queues, GPU command buffer submission (microsecond critical
sections), serialized log writes.

## Why single global calibration, not per-lock

The 500ns spin target is a property of the CPU's current TSC frequency, not
of any individual lock instance. Calibrating once per process (lazily, on
first use) and sharing the result across every `spinit_t` in the process
avoids redundant ~microsecond-scale calibration work per lock, and there's
no use case in the original notes that needs different locks to spin for
different durations. The 500ns figure is a fixed target for the
iteration count; per-iteration PAUSE count adapts via exponential backoff
(see below), but the iteration cap itself does not adapt to
contention or to whether the lock owner is making progress.

This is a third strategy, distinct from both glibc's adaptive mutex (which
uses a fixed `MAX_ADAPTIVE_COUNT` of 100 plus a per-mutex spin estimator
stored in `mutex->__data.__spins`, updated after each acquisition -- it
does NOT call `__sched_getcpu` or probe cache hit rate) and the kernel's
`mutex_optimistic_spin` (which checks owner-on-cpu via `mutex_can_spin_on_owner`).
libspinit's fixed iteration count is simpler and lower-overhead than
either. Per-iteration PAUSE count adapts via exponential backoff;
the iteration cap itself does not adapt. This is an accepted, deliberate
tradeoff.

The original notes mention adapting spin duration "even when the CPU
throttles down" (dynamic re-calibration). **Resolved:** libspinit does not
re-calibrate after the first measurement. This is an accepted, deliberate
scope cut. Revisit only if benchmarking shows real-world frequency scaling
meaningfully hurts spin accuracy in practice.

## The ASM boundary

`LOCK CMPXCHG` for the atomic compare-exchange. `spinit_rdtsc`
(src/spinit_x86_64.asm) wraps `LFENCE;RDTSC` and is used during
calibration only -- never on the hot path, which spins on a relaxed
`atomic_load` of `state` with a `PAUSE` hint between attempts. `LFENCE`
before `RDTSC` is the Intel SDM Vol 2B recipe for ordering prior loads
and stores before the TSC read. `spinit_cpuid` saves RDX into R8 before
issuing CPUID (CPUID clobbers RAX/RBX/RCX/RDX, and RDX carries the `out`
pointer on entry); RBX is pushed/popped as callee-saved. This lines up
with how the original spec describes it: a "textbook implementation"
needing only backoff constant tuning, not novel design.

## Calibration

`spin_iterations` is set once on the first `spinit_lock` call in the
process, via `pthread_once`. The value is shared across every `spinit_t`
in the process and is never updated. Algorithm (`calibrate()` in
src/spinit.c):

1. Read CPUID leaf 80000007H, bit EDX[8] (`constant_tsc`). If clear,
   skip to step 5. Without `constant_tsc` the TSC frequency can change
   under power management and any measured count would be wrong in
   absolute time; fall through to the fixed default.
2. Read CPUID leaf 15H. If ECX (the core crystal clock in Hz) and EAX
   are both non-zero, TSC frequency is `ECX * EBX / EAX` (the Intel SDM
   recipe). Populated on most modern Intel client and server parts.
3. If leaf 15H is unpopulated (ECX == 0), fall back to measuring TSC
   frequency by pairing `spinit_rdtsc` with `clock_gettime(CLOCK_MONOTONIC)`
   over a ~1ms busy wait. Two samples, take the minimum, to reduce
   preemption noise.
4. Compute the iteration cap: `target_ticks = hz / 2_000_000` (ticks
   in 500ns); `ticks_per_iter` from a 256-iteration probe that matches
   the minimum-cost hot-path spin body (one `PAUSE` + one cmpxchg-fail);
   `spin_iterations = target_ticks / ticks_per_iter`, clamped to >= 1.
5. If `constant_tsc` is absent, or any of the above produced 0,
   `spin_iterations = SPINIT_FALLBACK_ITERATIONS` (500). The spin window
   will be wrong in absolute time on non-constant-TSC parts but the
   lock remains correct.

The probe measures the floor cost of one spin iteration (one `PAUSE` +
one cmpxchg-fail). Under exponential backoff the hot path may issue up
to 64 `PAUSE`s per iteration, so the wall-clock spin window can exceed
500ns under contention. `spin_iterations` is an upper bound on attempts,
not a strict wall-clock guarantee. See "Exponential backoff within the
spin window".

## Exponential backoff within the spin window

Without backoff the spin loop issues one PAUSE per iteration for the
entire calibrated ~500ns window. Under N-way contention (8+ cores
hammering the same lock), every spinner issues a load of `state` on
every iteration. Even with the test-and-test-and-set pattern (which keeps
the cache line shared while spinning, avoiding the per-iter RMW
ping-pong of pure test-and-set), N spinners issuing N loads per ~10ns
interval is N× the cache-coherency traffic of the same loop with a
longer per-iteration pause. The fix is exponential backoff inside the
spin loop: start with 1 PAUSE per iteration, double on each failed
iteration, cap at 64 PAUSEs.

The doubling sequence is 1, 2, 4, 8, 16, 32, 64, 64, 64, ..., implemented
as `backoff = backoff < 32 ? backoff * 2 : 64` so the cap is hit cleanly
on the iteration that would have produced 64 (32×2). The cap of 64 was
chosen as a balance: large enough that a heavily-contended spinner polls
the cache line roughly every ~200ns (64 PAUSEs ≈ 200ns on modern x86,
where one PAUSE is ~3-4ns on Skylake-and-later), small enough that a
spinner notices an unlock within a single PAUSE window once the lock
releases. Above 64 the polling latency starts to dominate acquire latency
for briefly-held locks; below 64 the cache-line traffic reduction under
heavy contention is insufficient.

Backoff is local to each `spinit_lock` call -- there is no per-lock or
per-thread state carried between calls. A successful acquire returns
immediately; the next call starts again at `backoff = 1`. A briefly-
contended lock does not penalize later briefly-uncontended acquires.

The iteration cap (`spin_iterations`, calibrated to ~500ns without
backoff) is unchanged. The backoff stays inside the existing spin window:
once `spin_iterations` iterations are exhausted, the loop falls through to
the futex path as before. Under heavy contention the wall-clock spin
window can exceed 500ns because each iteration may issue more PAUSEs than
calibration assumed (calibration probes one PAUSE + one cmpxchg per
iteration, the minimum-cost path). This is intentional -- the alternative
is the no-backoff behavior of burning the same cycles hammering the cache
line at full frequency, which wastes power and steals memory bandwidth
from the lock holder. Falling through to futex at the iteration cap
(rather than a wall-clock cap) preserves the no-rdtsc-per-attempt
property from the original design.

The 64-PAUSE cap is a compile-time constant, not configurable. It is a
property of the spin policy, not of any individual lock instance, so it
matches the "single global calibration, not per-lock" decision above.

## Why futex, not just spin-forever

Spinning past a contended lock beyond a short calibrated window wastes CPU
that could go to whatever thread is holding the lock (especially bad under
oversubscription). Falling back to `futex(FUTEX_WAIT)` after the spin window
parks the thread instead, matching the thread-pool use case in the original
notes.

## The 3-state futex mutex

Once the spin window expires the lock transitions from a one-state test-and-
test-and-set spin into the standard three-state Linux futex mutex pattern
(Drepper, "Futexes Are Tricky"). The single `state` word takes three values:

- `0` -- unlocked.
- `1` -- locked, no waiter has parked in `futex(FUTEX_WAIT)` yet.
- `2` -- locked, and one or more threads are parked (or about to park) in
  `FUTEX_WAIT` on this address.

State `2` exists to make wake-loss impossible without an extra word: an
unlocker can decide whether a syscall is needed by reading the value it just
atomically cleared. `futex_wake` is only called when the previous value was
`2`, so an uncontended unlock is a single `xchg` and no syscall. `futex_wait`
is only called after the waiter has confirmed `state == 2` via an
`atomic_load`, so a wake that lands between the waiter's "mark myself pending"
step and its `futex_wait` is observed by that load and the syscall is skipped.

### Lock path

1. Fast path: `cmpxchg(state, 0 -> 1)`. On success the lock is held.
2. Spin window: 1..64 PAUSE hints (exponential backoff, see above) +
   (load; if `state != 0` retry) + `cmpxchg(0 -> 1)`, for a calibrated
   iteration count.
3. Futex loop. Each iteration:
   - `cmpxchg(state, 0 -> new_state)` where `new_state = waited ? 2 : 1`.
     On success the lock is held.
   - Otherwise the observed value is `1` or `2`. If `1`, attempt
     `cmpxchg(state, 1 -> 2)` to mark a waiter present (best-effort; if it
     loses to a concurrent unlock the load below re-checks).
   - `atomic_load(&state)`. If it is not `2` (e.g. the holder just unlocked
     to `0`, or another waiter lost the `1 -> 2` race and is re-trying), do
     not park -- loop.
   - `futex(FUTEX_WAIT|FUTEX_PRIVATE_FLAG, 2, ...)`. The kernel re-checks
     `*addr == 2` under the futex hash bucket lock; if the value changed
     (`-EAGAIN`), or on spurious wake (`0`), or signal (`-EINTR`), loop.
   - Set `waited = 1` so the next successful acquire re-marks `state = 2`,
     keeping the wake chain alive for any remaining parked waiters.

### Unlock path

`atomic_exchange(state, 0)`. If the previous value was `2`, call
`futex(FUTEX_WAKE|FUTEX_PRIVATE_FLAG, 1)` -- wake exactly one waiter.
Waking one (not `INT_MAX`) avoids the thundering-herd pattern: the woken
waiter acquires, runs, and on its next unlock wakes the next waiter, so the
queue drains one at a time. The "waited" flag set by a previously-parked
acquirer ensures that subsequent unlocks in a contention chain continue to
see `prev == 2` and keep calling `futex_wake` until the queue is empty. The
tail unlock (last waiter acquires, no further waiters) does one spurious
`futex_wake(1)` that wakes nobody; this is accepted as a minor cost.

### Race coverage

The two re-checks -- the `atomic_load` before `futex_wait`, and the kernel's
own value check inside `futex_wait` -- close the lost-wakeup window from both
sides. ABA on `2` (a `2 -> 0 -> 2` transition between the load and the wait)
is harmless because the kernel's check is atomic against the same futex hash
bucket: if the value changed at all, `futex_wait` returns `-EAGAIN` and the
loop re-evaluates.

### Non-recursive locking

The protocol assumes one holder at a time. A thread that calls `spinit_lock`
twice without unlocking will, on the second call, enter the futex path and
succeed at `cmpxchg(state, 1 -> 2)` (it itself set `state = 1` on the first
acquire), then park in `futex_wait(2)` forever. The state reads `2` while
self-deadlocked, which is misleading to a debugger but does not corrupt any
other lock. Recursive locking is a documented non-goal; no detection or
recovery is provided.

## Benchmarks

Test host: 2-CPU box, gcc 14.2.0, nasm 2.16.01, `-std=c11 -Wall -Wextra
-Werror -pedantic -O2 -pthread`. `bench_uncontended` is 10M single-thread
lock+unlock cycles; `bench_contended` is 1M cycles per thread on a shared
counter; `bench_backoff` compares spinit against an inline naive
test-and-test-and-set lock (same fast path, same futex fallback, no
backoff, fixed 500-iteration spin).

```
bench_uncontended:        ~15 ns/op
bench_contended (backoff):
  2 threads:              ~36 ns/op
  4 threads:              ~48 ns/op
  8 threads:              ~48 ns/op
```

`bench_backoff` reports a 1.8-2.4x throughput improvement vs the naive
no-backoff lock at 2/4/8 threads. The uncontended fast path is
unaffected (a single `lock cmpxchg`); the improvement is concentrated
under contention, where backoff reduces cache-coherency traffic on the
shared `state` word.

Run `make -C libspinit bench` to reproduce. Contended numbers vary with
host, core count, scheduler placement, and cache topology; the figures
above are from an oversubscribed test box.

## Non-goals

- No per-lock tuning.
- No fairness guarantees (not a ticket lock). The lock is a
  test-and-test-and-set style lock; starvation under heavy contention is a
  known, accepted limitation.
- No Windows/macOS fallback path yet.
