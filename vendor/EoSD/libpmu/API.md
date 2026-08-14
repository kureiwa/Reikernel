# libpmu: API (v0.3)

Status: v0.3 shipped. `perf_event_open(2)` + `read(2)` path with an internal
`rdpmc` fast path on x86_64, plus graceful degradation when
`perf_event_open` is denied. When `perf_event_open(2)` fails with `EACCES`,
`EPERM`, or `ENOSYS` (containerized environments with restrictive seccomp,
or `perf_event_paranoid >= 3`), `pmu_open` does not return `NULL`. It
returns a heap-allocated dummy context with `*out_err == PMU_ERR_PERM`;
`pmu_start` is a no-op, `pmu_read`/`pmu_stop_and_read` set `*out_value = 0`,
and `pmu_close` just frees. `pmu_is_available(ctx)` lets the caller
distinguish a real fd from a dummy and fall back to an alternative timing
source (e.g. `rdtsc`).

`pmu_read`/`pmu_stop_and_read` transparently use `rdpmc` when the kernel
exposes a non-zero `perf_event_mmap_page->index` (and `cap_user_rdpmc` is
set) and fall back to `read(2)` otherwise. Both paths scale the raw count
by `time_enabled`/`time_running` under multiplexing.

## Overview

Hardware performance counter reads via `perf_event_open(2)`. Fixed small
set of counters matching the original notes' use cases: cycles,
instructions retired, cache misses. Simple open/start/read/close API.

On x86_64, `pmu_read`/`pmu_stop_and_read` mmap the perf event metadata
page and read the counter with `rdpmc` when the kernel allows it
(`perf_event_paranoid <= 1`, or `CAP_PERFMON`/`CAP_SYS_ADMIN`). When
`rdpmc` is unavailable or the event is transiently not scheduled onto a
PMC, they fall back to a `read(2)` syscall. Both paths scale the raw
count by `time_enabled`/`time_running` under multiplexing (see below).

## Types

```c
typedef struct pmu_ctx pmu_ctx_t;   /* opaque, wraps one perf_event_open fd */

typedef enum {
    PMU_CYCLES,             /* unhalted core cycles   (PERF_COUNT_HW_CPU_CYCLES)    */
    PMU_INSTRUCTIONS,       /* instructions retired   (PERF_COUNT_HW_INSTRUCTIONS)  */
    PMU_CACHE_MISSES,       /* last-level cache misses (PERF_COUNT_HW_CACHE_MISSES) */
} pmu_counter_type_t;

typedef enum {
    PMU_OK               = 0,
    PMU_ERR_INVALID      = -1,
    PMU_ERR_PERM         = -2,   /* perf_event_open denied (permissions/sysctl) */
    PMU_ERR_UNAVAILABLE  = -3,   /* counter type not supported, or read/ioctl failure */
} pmu_err_t;
```

## API

```c
/* Opens a counter of the given type for the calling thread (perf_event_open
 * with pid=0, cpu=-1, i.e. follows the calling thread across CPUs). The
 * event is opened disabled; pmu_start enables it. The fd is opened with
 * PERF_FLAG_FD_CLOEXEC so it does not leak across execve. On x86_64 the fd
 * is mmaped and the rdpmc fast path is armed if cap_user_rdpmc is set.
 *
 * Graceful degradation: if perf_event_open(2) fails with EACCES, EPERM, or
 * ENOSYS, pmu_open returns a non-NULL dummy context and writes PMU_ERR_PERM
 * to *out_err. pmu_start on a dummy is a no-op; pmu_read and pmu_stop_and_read
 * set *out_value = 0; pmu_close just frees. Callers that need real counter
 * values should check pmu_is_available(ctx) and fall back to an alternative
 * timing source (e.g. rdtsc) when it returns 0.
 *
 * Returns NULL only for PMU_ERR_INVALID (unknown counter type) or
 * PMU_ERR_UNAVAILABLE (ENODEV/ENOENT/EINVAL -- counter type not supported
 * on this CPU -- or malloc failure). */
pmu_ctx_t *pmu_open(pmu_counter_type_t which, pmu_err_t *out_err);

/* Returns 1 if `ctx` is backed by a real perf_event_open fd (counter reads
 * return real values), 0 if it is a dummy context (counter reads return 0).
 * Returns 0 if `ctx` is NULL. Useful for applications that want to degrade
 * gracefully when perf is denied: call pmu_is_available(ctx) right after
 * pmu_open and pick a fallback path (e.g. rdtsc, clock_gettime) when it
 * returns 0. */
int pmu_is_available(const pmu_ctx_t *ctx);

/* Resets the counter to zero (PERF_EVENT_IOC_RESET) and starts counting
 * (PERF_EVENT_IOC_ENABLE). Returns PMU_OK on success.
 * Calling pmu_start twice in a row RESETs the counter on the second call,
 * discarding the count accumulated since the first call. Callers that
 * want a cumulative count must read before re-starting.
 * On a dummy ctx, pmu_start is a no-op and returns PMU_OK. */
int pmu_start(pmu_ctx_t *ctx);

/* Stops counting (PERF_EVENT_IOC_DISABLE) and reads the current value.
 * On a dummy ctx, sets *out_value = 0 and returns PMU_OK without any ioctl. */
int pmu_stop_and_read(pmu_ctx_t *ctx, uint64_t *out_value);

/* Reads the current value without stopping (for periodic sampling).
 * Uses rdpmc on x86_64 when available, else read(2). Both paths return
 * the multiplexing-scaled count (see "Multiplexing" below).
 * Before pmu_start the event is disabled; pmu_read returns PMU_OK with
 * *out_value == 0 in that case.
 * On a dummy ctx, sets *out_value = 0 and returns PMU_OK without any syscall. */
int pmu_read(pmu_ctx_t *ctx, uint64_t *out_value);

/* Closes the perf fd and frees the context. Safe to call with NULL ctx.
 * On a dummy ctx, just frees the context (no fd to close, no mmap to unmap). */
void pmu_close(pmu_ctx_t *ctx);
```

## Graceful degradation (v0.3)

`perf_event_open(2)` can fail in environments where the kernel denies the
syscall outright:

- `perf_event_paranoid >= 3` (some hardened distros).
- Container runtimes (Docker, Kubernetes) with a restrictive seccomp
  profile that returns `EACCES`, `EPERM`, or `ENOSYS` for the syscall.
- Kernels built without `CONFIG_PERF_EVENTS` (returns `ENOSYS`).

In v0.3, `pmu_open` allocates a **dummy context** with `fd == -1`
and `is_dummy == 1`, sets `*out_err = PMU_ERR_PERM` so the caller can still
detect that degradation happened, and returns the dummy (non-`NULL`). On a
dummy context:

- `pmu_is_available(ctx) == 0`
- `pmu_start(ctx)        == PMU_OK` (no-op, no ioctl)
- `pmu_read(ctx, &v)     == PMU_OK, v == 0` (no syscall, no rdpmc)
- `pmu_stop_and_read     == PMU_OK, v == 0` (no ioctl, no read)
- `pmu_close(ctx)        ` no crash (just `free(ctx)`)

Two usage patterns are supported:

1. **Transparent passthrough.** Callers that only need a non-`NULL` ctx to
   flow through the API (so they do not have to special-case `NULL`
   everywhere) can use the dummy transparently. Counter reads return 0;
   the workload runs normally; metrics just report 0. No code change
   beyond dropping the `NULL` check.

2. **Detect-and-fallback.** Callers that need real counter values call
   `pmu_is_available(ctx)` right after `pmu_open` (or check
   `*out_err == PMU_ERR_PERM`). When it returns 0, they fall back to an
   alternative timing source (`rdtsc`, `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`,
   etc.) and tear the dummy down with `pmu_close`.

`PMU_ERR_PERM` is reported for `EACCES`, `EPERM`, and `ENOSYS`. The other
errnos (`ENODEV`, `ENOENT`, `EINVAL`) still return `NULL` with
`PMU_ERR_UNAVAILABLE` -- those indicate the counter type is not supported on
this CPU, which is a different failure mode and not one the dummy can
usefully paper over (reads would always return 0, indistinguishable from
a real run that genuinely produced 0). `PMU_ERR_INVALID` (unknown counter
type) also still returns `NULL`.

## Multiplexing

When the system has more events than hardware PMCs, the kernel
time-slices them. `perf_event_open` exposes `time_enabled` (wall time
the event was enabled) and `time_running` (time the event was actually
on a PMC). The scaled count is:

    scaled = raw * time_enabled / time_running

libpmu computes this with a 128-bit intermediate (`__int128`, mirroring
the kernel's `mul_u64_u64_div_u64()`) so the multiply does not overflow
`u64` when both `raw` and `time_enabled` are large (e.g. a 2^40-cycle
count under heavy multiplexing with `time_enabled` near 2^63 ns).

Both code paths scale identically:

- **rdpmc path**: reads `time_enabled`/`time_running` from
  `perf_event_mmap_page` and scales `offset + pmc`.
- **read(2) path**: opens the event with
  `attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
  PERF_FORMAT_TOTAL_TIME_RUNNING`, reads 24 bytes
  `{value, time_enabled, time_running}`, and scales `value`.

Without the read_format flag the read(2) path would return the raw
unscaled count and disagree with the rdpmc path under multiplexing.

## rdpmc fast path (x86_64)

When `perf_event_mmap_page->cap_user_rdpmc` is set and `index != 0`,
`pmu_read` reads the counter with `rdpmc` (no syscall). The recipe
follows `Documentation/arch/x86/` and the kernel's
`x86_perf_event_update()`:

1. Read `mmap_page->lock` (seqlock). If odd, retry (writer mid-update).
   Bounded to 64 retries; falls back to `read(2)` if the seqlock never
   settles.
2. Read `index`, `offset`, `time_enabled`, `time_running`, `pmc_width`.
3. If `index == 0`, the event is not currently on a PMC: fall back to
   `read(2)`.
4. `lfence; rdpmc(index - 1); lfence`.
5. Sign-extend per `pmc_width` (kernel idiom: shift left to put the sign
   bit at position 63, then arithmetic shift right).
6. Re-read `lock`; if changed, retry.
7. `count = offset + pmc`, then scale if multiplexed.

The parity check (step 1) runs before any field read so a transiently-
zero `index` (the kernel zeroes `index` on the scheduling path while
holding the seqlock write side) does not cause a spurious fallback.

`rdpmc` availability requires `perf_event_paranoid <= 1` (or
`CAP_PERFMON`/`CAP_SYS_ADMIN`). At the distro default `paranoid=2`,
`cap_user_rdpmc` is 0 and libpmu uses the `read(2)` path. The recipe is
vendor-neutral: the kernel programs the event-select MSR, so
`rdpmc(index-1)` works identically on Intel and AMD. Intel fixed-counter
encoding (`ECX[30]=1`) is handled by the kernel's 1-based `index`
convention (fixed counter N -> `index = 0x40000001 + N`, so `index - 1`
preserves bit 30).

## Edge cases

- **pmu_read before pmu_start**: returns `PMU_OK` with `*out_value == 0`
  (event is opened disabled). Not an error.
- **pmu_start called twice**: the second call's `PERF_EVENT_IOC_RESET`
  zeroes the counter. The count accumulated since the first start is
  lost. Callers wanting a cumulative count must read before re-starting.
- **pmu_close while running**: `close(fd)` disables the event and frees
  resources. The final counter value is lost; call `pmu_stop_and_read`
  first if the value is wanted.
- **pmu_read after pmu_stop_and_read**: the event is disabled; reads
  return the frozen final value until `pmu_start` re-enables it.
- **dummy ctx (v0.3)**: when `perf_event_open` fails with `EACCES`,
  `EPERM`, or `ENOSYS`, `pmu_open` returns a non-`NULL` dummy with
  `*out_err == PMU_ERR_PERM`. `pmu_start`/`pmu_read`/`pmu_stop_and_read`
  all return `PMU_OK` (`pmu_read` and `pmu_stop_and_read` write `0`);
  `pmu_close` just `free()`s. `pmu_is_available(ctx)` returns `0`.
  See "Graceful degradation (v0.3)" above.

## Non-goals (v0.3)

- No generic/arbitrary perf event config: only the three fixed counter
  types. (Generic config can be added later as `pmu_open_raw(struct
  perf_event_attr *)` without breaking the fixed-type API.)
- No Windows (`PDH`/`NtQuerySystemInformation`) support. Linux/
  `perf_event_open` only, matching the toolkit-level platform decision.
- No multi-counter grouped reads (each `pmu_ctx_t` is one counter): if a
  caller needs cycles + instructions simultaneously, open two contexts.
- No raw MSR programming for AMD general PMCs: the kernel programs the
  event-select MSR, so `rdpmc(index-1)` is vendor-neutral and no
  Intel-only caveat applies.
