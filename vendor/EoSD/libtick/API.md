# libtick: API (v0.3, shipped)

Status: shipped; this document matches the v0.3 heap-based registry
with TSC drift defense.

## Overview

High-resolution sleep-until with deadline/overshoot reporting, plus an optional
multi-timer registry. Single-threaded per context. No pluggable allocator (one
plain libc malloc for the registry array at creation; nothing else allocates).

## Types

```c
typedef struct tick_ctx tick_ctx_t;   // opaque

typedef enum {
    TICK_OK               = 0,
    TICK_ERR_INVALID      = -1,
    TICK_ERR_FULL         = -2,
    TICK_ERR_NOT_FOUND    = -3,
    TICK_ERR_CALIBRATION  = -4,
} tick_err_t;

typedef int tick_timer_id_t;          // index-based handle, -1 = invalid

typedef void (*tick_callback_fn)(tick_ctx_t *ctx, tick_timer_id_t id, void *user_data);

typedef enum {
    TICK_MODE_POLL,      // caller retrieves via tick_wait_next()
    TICK_MODE_CALLBACK,  // fired via tick_run_pending()
} tick_fire_mode_t;
```

## Lifecycle

```c
// Creates a context, immediately runs TSC-frequency calibration (CPUID.15H
// first, ~10ms rdtsc fallback in two 5ms samples) and allocates a fixed-size
// slot array plus two binary min-heaps for `capacity` timers
// (0 = no registry, one-shot sleep_until only).
tick_ctx_t *tick_ctx_create(size_t capacity);

void tick_ctx_destroy(tick_ctx_t *ctx);

// Returns NULL if calibration or allocation failed (e.g. rdtsc unavailable,
// TSC frequency out of the [100 MHz, 10 GHz] bounds, or malloc failure for
// the registry arrays). Call tick_last_error() to find out why. The error
// string is cleared on entry to every public libtick function except
// tick_last_error itself and tick_ctx_destroy, so a successful call leaves
// the buffer empty.
const char *tick_last_error(void);   // thread-local, valid until next libtick call on this thread
```

Thread-safety: not thread-safe. A `tick_ctx_t` must only be used from the
thread that created it. Multi-threaded use requires one ctx per thread.

## One-shot sleep

```c
// Sleeps until `deadline_ns` (monotonic ns epoch, same clock as tick_now()).
// Returns:
//   0  = slept normally, woke at/near the deadline
//   1  = overshoot: deadline had already passed when called
//  <0  = tick_err_t on error
// overshoot_ns (may be NULL): exact ns of lateness (0 if rc == 0)
int tick_sleep_until(tick_ctx_t *ctx, uint64_t deadline_ns, uint64_t *overshoot_ns);

// Current monotonic time in ns, same epoch as deadlines. Performs a low-
// frequency drift check (every 1024 calls or every 5 s, whichever comes
// first) by calling tick_ctx_check_drift internally. That check calls
// clock_gettime, so tick_now is NOT async-signal-safe; it may incur a
// syscall roughly once per 1024 calls. Amortized overhead is < 1 ns per
// call. See tick_ctx_check_drift for details.
uint64_t tick_now(tick_ctx_t *ctx);
```

## Timespec helpers

```c
uint64_t tick_from_timespec(const struct timespec *ts);
void     tick_to_timespec(uint64_t ns, struct timespec *out);
```

## Drift defense

```c
// Re-validates the TSC calibration against clock_gettime(CLOCK_MONOTONIC).
// Reads both clocks close together, computes the TSC-predicted monotonic
// time using the current calibration, and compares it to the actual
// clock_gettime value. If the drift exceeds drift_threshold_ns (default
// 1 ms), recalibrates by updating the (monotonic_base_ns, tsc_base) pair
// to the freshly observed values; tsc_hz and tsc_per_ns are unchanged
// (the TSC frequency does not drift, only the offset).
//
// Returns the observed drift in ns (before any recalibration was applied).
//
// NOT async-signal-safe: clock_gettime may take a syscall. Call this from
// the application's event loop, not from a signal handler. tick_now invokes
// this automatically; explicit calls are for callers that want to control
// the check cadence or inspect the return value.
uint64_t tick_ctx_check_drift(tick_ctx_t *ctx);

// Reports drift-monitoring diagnostics. Any out-pointer may be NULL.
//   max_drift       -- high-water mark of |observed drift| across all checks
//   checks          -- total tick_ctx_check_drift invocations (including
//                      automatic ones from tick_now)
//   recalibrations  -- total recalibrations performed (drift exceeded
//                      threshold)
void tick_ctx_drift_stats(tick_ctx_t *ctx, uint64_t *max_drift,
                          uint64_t *checks, uint64_t *recalibrations);
```

### Automatic check frequency

`tick_now` triggers `tick_ctx_check_drift` every 1024 calls or every 5 s of
wall time, whichever comes first. The 1024-call counter is the primary
trigger (one increment + one compare on the fast path); the 5 s timer is a
fallback for low-frequency callers and only evaluates when the counter has
not yet tripped. The check itself costs ~1 us (one `clock_gettime` syscall +
two `rdtsc`), amortized to < 1 ns per `tick_now` call.

### Recalibration semantics

When `|drift| > drift_threshold_ns`, the (monotonic_base_ns, tsc_base) pair
is replaced with the freshly observed (clock_gettime, rdtsc) pair.
`tsc_hz` and `tsc_per_ns` are never modified by recalibration: the TSC
frequency is constant on `constant_tsc` platforms, only the offset between
TSC ticks and CLOCK_MONOTONIC ns drifts (due to NUMA skew, C-state
transitions, or scheduler migration between cores).

## Multi-timer registry

```c
// Registers a timer for deadline_ns. mode selects poll vs callback delivery.
// cb/user_data only used if mode == TICK_MODE_CALLBACK (cb must be non-NULL then).
// Returns timer id (>=0) on success, negative tick_err_t on error
// (TICK_ERR_FULL if capacity exhausted, including capacity==0).
int tick_register(tick_ctx_t *ctx, uint64_t deadline_ns, tick_fire_mode_t mode,
                   tick_callback_fn cb, void *user_data, tick_timer_id_t *out_id);

// Cancels a pending timer. Returns 0 on success, TICK_ERR_NOT_FOUND if the id
// is unknown or already fired.
int tick_cancel(tick_ctx_t *ctx, tick_timer_id_t id);

// Blocks until at least one poll-mode timer's deadline is reached, or until
// timeout_ns elapses (pass UINT64_MAX for no timeout). Writes fired timer ids
// into fired_ids (caller-provided array of size max_ids), returns count fired
// (>=0), or negative tick_err_t on error. All expired poll-mode timers fire on
// a single call, no batching cap. fired_ids may be NULL, in which case no ids
// are written but the expired timers are still popped and counted. If more
// than max_ids timers expire, they are all fired (marked as no longer pending)
// but only the first max_ids ids are written out; the returned count reflects
// the total. Errors from the internal tick_sleep_until (e.g. clock_nanosleep
// failure) are propagated as negative tick_err_t to the caller.
int tick_wait_next(tick_ctx_t *ctx, tick_timer_id_t *fired_ids, size_t max_ids,
                    uint64_t timeout_ns);

// Invokes callbacks for all expired callback-mode timers, in non-decreasing
// deadline order (the order they pop from cb_heap). Returns count fired,
// or negative tick_err_t on error. Does not block if nothing is due; caller is
// expected to call this from their own event loop tick.
int tick_run_pending(tick_ctx_t *ctx);
```

The registry is a fixed-size slot array plus two binary min-heaps
(poll_heap and cb_heap); see DESIGN.md for the rationale and per-operation
costs.

## Non-goals

- No support for periodic/repeating timers in v0.3, one-shot only. Repeating
  timers are the caller's responsibility and MUST NOT be re-registered from
  inside a callback (the registry is not reentrant; see DESIGN.md). Re-add
  the timer after `tick_run_pending` returns, or use a separate context for
  periodic work.
- No cross-thread timer registration/cancellation.
- No dynamic registry growth; capacity is fixed at creation.
