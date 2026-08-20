# libspinit: API (v0.3)

Status: shipped. Implementation in src/spinit.c + src/spinit_x86_64.asm
(x86_64 only). Tests in tests/.

## Overview

TSC-calibrated fixed-window spinlock: spins for a calibrated iteration
count (~500ns target without backoff) then falls back to `futex`
(Linux)/`WaitOnAddress` (Windows, deferred). Calibration happens once
per process, cached in a static, not per-lock. The spin loop is
test-and-test-and-set with exponential backoff (per-iteration PAUSE
count 1 -> 64, doubling per failed iteration, capped at 64): spin on a
relaxed `atomic_load` of `state`, attempt `lock cmpxchg` only when that
load observed 0, so the shared cache line stays shared while N-1
spinners wait. See DESIGN.md. Minimal single-word public type.

## Types

```c
typedef struct {
    _Atomic int state;   /* single word:
                          *   0 = unlocked,
                          *   1 = locked, no parked waiters,
                          *   2 = locked, one or more threads parked in
                          *       futex(FUTEX_WAIT).
                          * state==2 is internal but visible to a debugger. */
} spinit_t;

/* spinit.h applies alignas(64) to state so the struct occupies one cache
 * line, avoiding ping-pong when embedded in caller structs. */

#define SPINIT_INIT { 0 }   // static initializer, e.g. spinit_t lock = SPINIT_INIT;
```

No richer struct, no per-lock tuning knobs, matching "fixed internal
policy" decision. All calibration state lives in a process-wide static,
initialized lazily on first `spinit_lock` call (thread-safe init, e.g. via
`pthread_once` or a C11 atomic flag).

## API

```c
void spinit_init(spinit_t *lock);        // equivalent to SPINIT_INIT, for dynamic alloc

void spinit_lock(spinit_t *lock);        // spins ~500ns (calibrated), then futex-waits
int  spinit_trylock(spinit_t *lock);     // returns 0 on acquired, 1 if already locked
void spinit_unlock(spinit_t *lock);
```

### Lock acquisition

Fast path is a single `lock cmpxchg` (`state` 0 -> 1). On contention the
caller enters a test-and-test-and-set spin loop. Each iteration:

1. Issue 1..64 `PAUSE` hints. The count starts at 1 and doubles per
   failed iteration, capped at 64 (`backoff = backoff < 32 ? backoff*2
   : 64`). The cap exists to keep polling latency bounded: at 64 PAUSEs
   a heavily-contended spinner still polls the cache line roughly every
   ~200ns on modern x86.
2. Relaxed `atomic_load(&state)`. If non-zero, double the backoff
   (subject to the cap) and retry.
3. Only if the load observed 0, attempt `lock cmpxchg(state, 0 -> 1)`.
   On success the lock is held.

Spinning on the read-only load (step 2) keeps the cache line shared
while N-1 spinners wait, avoiding the per-iteration RMW ping-pong that
pure test-and-set would cause under N-way contention. The loop runs for
a calibrated iteration count (`spin_iterations`, ~500ns without backoff;
see DESIGN.md), then falls back to `futex(FUTEX_WAIT|FUTEX_PRIVATE_FLAG)`.
Under heavy contention the wall-clock spin window can exceed 500ns
because backoff issues more PAUSEs per iteration than calibration
assumed; this is intentional. See DESIGN.md.

The three-state `state` word (0/1/2) is what lets `spinit_unlock` decide
whether to issue a `futex_wake` without an extra word of bookkeeping.
See DESIGN.md for the full state machine.

### Unlock

`atomic_exchange(state, 0)`. If the previous value was 2 (a waiter was
parked), call `futex(FUTEX_WAKE|FUTEX_PRIVATE_FLAG, 1)` -- wake exactly
one waiter, not the whole queue. Waking one avoids the thundering-herd
pattern: the woken waiter acquires, runs, and on its next unlock wakes
the next waiter, so the queue drains one at a time.

## Undefined behavior

- **Unlock without lock.** The caller must hold the lock. Unlocking an
  unheld lock is UB and is NOT detected: `atomic_exchange` silently
  writes 0 (a no-op when `state` was already 0), no `futex_wake` is
  issued, and the lock remains usable. Under concurrent use a stray
  unlock can race with a real holder's unlock and permit double-
  acquisition; callers must not rely on detection. `test_edge::test_unlock_without_lock`
  pins the single-threaded observed behavior so a future change to add
  an assert or to corrupt state would be caught.
- **Recursive locking.** Not supported. A thread that calls `spinit_lock`
  twice without unlocking self-deadlocks in `futex_wait` on the second
  call (the second acquire transitions `state` 1 -> 2 and parks). No
  detection, no recovery. The state reads 2 while self-deadlocked,
  which is misleading to a debugger but does not corrupt any other
  lock.

## Non-goals

- No per-lock configuration of spin duration or backoff policy.
- No recursive locking, no priority inheritance, no reader/writer variant; a plain mutual exclusion lock only.
- No Windows fallback in v0.3 (futex path only; `WaitOnAddress` deferred to
  cross-platform expansion).
