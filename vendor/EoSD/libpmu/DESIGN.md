# libpmu: Design Notes (v0.3)

## Problem

Measure real cycle/instruction/cache-miss counts for benchmarking other
modules in this toolkit (per original notes: catching a 5-cycle regression
in `libdetour`'s trampoline that `clock_gettime` can't see) and for
adaptive algorithms that want ground-truth CPU behavior rather than wall
clock time.

## Architecture (v0.3)

Three read paths, selected transparently per `pmu_read` call:

1. **rdpmc fast path** (x86_64, when `cap_user_rdpmc` is set): mmap the
   perf fd, read the counter with `rdpmc` via the seqlock protocol on
   `perf_event_mmap_page`. No syscall per read. ~20-40 cycles on modern
   Intel client cores (~6-13 ns at 3 GHz). Not exercised in the build/test
   sandbox (`perf_event_paranoid=2`, no `CAP_PERFMON`); numbers are
   documented from the kernel SDM and `bench_rdpmc.c`'s expected range,
   verified on hosts with `paranoid <= 1`.
2. **read(2) path** (always available, used as fallback): `read(fd, buf,
   24)` returns `{value, time_enabled, time_running}`. A few hundred ns
   per read (one syscall). Also not exercised in the sandbox (the open
   itself is denied); documented from `bench_rdpmc.c`'s expected range.
3. **dummy path** (v0.3, when `perf_event_open` is denied): no fd, no
   syscall, no `rdpmc`. `pmu_read` returns `0`. Selected when
   `ctx->is_dummy == 1`, which is set by `pmu_open` when the underlying
   `perf_event_open(2)` call fails with `EACCES`, `EPERM`, or `ENOSYS`.
   Measured at **2.1 ns/op** for `pmu_read` and **2.1 ns/op** for
   `pmu_stop_and_read` on the build host (`bench/bench_dummy.c`,
   1 000 000 iterations, `clock_gettime(CLOCK_MONOTONIC)`). The cost is
   one branch on `is_dummy`, one store of `0` to `*out_value`, and the
   return.

Both real read paths scale the raw count by `time_enabled / time_running`
under multiplexing using the same overflow-safe formula (see below). The
v0.3 public API only adds `pmu_is_available`; `pmu_open`'s failure-mode
contract changed (it no longer returns `NULL` for `PMU_ERR_PERM`), but
all other signatures are unchanged.

## Why perf_event_open + rdpmc

Raw `rdpmc` reads are faster (~20-40 cycles on modern Intel client cores
(Skylake/Ice Lake/Tiger Lake), ~6-13 ns at 3 GHz, vs. a few hundred ns
for a `read()` syscall on the perf fd) but require the kernel to grant
`rdpmc`-in-userspace access via a mapped perf event page. libpmu uses
the `perf_event_open` + `read()` path as the always-available baseline
and layers the `rdpmc` fast path on top using the exact same public API
(just a faster `pmu_read` internally), falling back to `read(2)` when
`rdpmc` is unavailable.

## Why a fixed small counter set

The original notes' three use cases (CI regression detection, adaptive
spin tuning, cache-miss-driven texture layout debugging) map directly
onto cycles, instructions retired, and cache misses. A fully generic
`perf_event_attr` passthrough API would be more powerful but also exposes
a much larger, harder-to-document surface area. Generic config can be
added later as `pmu_open_raw(struct perf_event_attr *)` without breaking
the fixed-type API.

## Multiplexing and scaling

When the system has more events than hardware PMCs, the kernel
time-slices them. `perf_event_open` exposes:

- `time_enabled`: wall time the event was enabled.
- `time_running`: time the event was actually on a PMC.

The scaled count is `raw * time_enabled / time_running`. The naive u64
multiply overflows silently when both `raw` and `time_enabled` are large
(e.g. `raw ~ 2^40` cycles under heavy multiplexing with `time_enabled ~
2^63` ns). libpmu uses `__int128` for the intermediate, mirroring the
kernel's `mul_u64_u64_div_u64()` in `lib/math/div64.c`:

```c
__extension__ typedef __int128 pmu_i128;
static uint64_t pmu_scale_count(uint64_t count, uint64_t enabled,
                                uint64_t running)
{
    if (enabled == running || running == 0) return count;
    return (uint64_t)((pmu_i128)count * enabled / running);
}
```

`__extension__` suppresses the `-pedantic` warning so the file builds
under `-std=c11 -pedantic -Werror`.

The read(2) path requests
`PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING` at
open time and parses the 24-byte `{value, time_enabled, time_running}`
return. Without this flag, `read(fd, &v, 8)` would return the raw
unscaled count and disagree with the rdpmc path under multiplexing.

## rdpmc seqlock protocol

The recipe follows `Documentation/arch/x86/` and the kernel's
`x86_perf_event_update()` (`arch/x86/events/core.c`):

1. Read `mmap_page->lock` (seqlock). If odd, retry -- the writer is
   mid-update. Bounded to `PMU_RDPMC_MAX_RETRIES = 64` spins; if the
   seqlock never settles, fall back to `read(2)`. (The writer-side
   update is short and rare: scheduler-tick multiplexing.)
2. Read `index`, `offset`, `time_enabled`, `time_running`, `pmc_width`.
3. If `index == 0`, the event is not currently on a PMC: fall back to
   `read(2)` (not a retry -- the event is genuinely de-scheduled).
4. `lfence; rdpmc(index - 1); lfence`. The leading `lfence` orders the
   field loads ahead of the counter read; the trailing `lfence` orders
   the counter read ahead of the `lock` re-read.
5. Sign-extend per `pmc_width` (kernel idiom: `pmc << (64-width)`;
   `(int64_t)shifted >> (64-width)`). `width == 0` means the kernel did
   not report a width; the raw value is used as-is.
6. Re-read `lock`. If changed, retry.
7. `count = offset + pmc`, then `pmu_scale_count(count, enabled, running)`.

The parity check (step 1) runs before any field read. The kernel zeroes
`index` on the scheduling path while holding the seqlock write side, so
a reader that catches an odd seq can transiently see `index == 0` even
when the event is still scheduled. Checking parity first and retrying
avoids a spurious fallback to `read(2)` in that window. Checking
`index == 0` (step 3) is done *after* the parity check has succeeded,
so it acts on a stable snapshot rather than a transient mid-update value.

## Vendor neutrality

`rdpmc(index - 1)` is vendor-neutral because the kernel programs the
event-select MSR for the platform. On Intel, fixed-function counters are
accessed via `ECX[30]=1` + `ECX[0..2]` index; the kernel uses 1-based
indexing for both general and fixed counters (fixed counter N ->
`index = 0x40000001 + N`), so `index - 1` preserves bit 30 and selects
the fixed counter space. On AMD, the kernel programs a general PMC with
the appropriate event-select MSR (e.g. `0xC0` for instructions retired)
and `index - 1` selects it. The asm helper (`pmu_rdpmc` in
`src/pmu_x86_64.asm`) needs no vendor-specific code path. Vendor
detection via CPUID.0H is informational (stored in `ctx->vendor`) and
does not affect the read path.

## Handling permission failures gracefully

`perf_event_open` can fail due to `/proc/sys/kernel/perf_event_paranoid`
restricting unprivileged use. The default value on modern Ubuntu, Debian,
Arch, RHEL, and Fedora is **2**, which disables unprivileged `rdpmc`
access entirely (the kernel reports `mmap_page->cap_user_rdpmc = 0`).
The `read(2)` path still works at `paranoid=2` for the three exposed
counter types. The `rdpmc` fast path requires `paranoid <= 1`, or
`CAP_PERFMON` / `CAP_SYS_ADMIN`, or
`sudo sysctl -w kernel.perf_event_paranoid=1`.

A second failure mode is the kernel denying the syscall outright:

- `perf_event_paranoid >= 3` (some hardened distros) returns `EACCES`.
- Container runtimes (Docker, Kubernetes) with a restrictive seccomp
  profile return `EACCES`, `EPERM`, or `ENOSYS` for the syscall.
- Kernels built without `CONFIG_PERF_EVENTS` return `ENOSYS`.

Both classes of permission failure map to `PMU_ERR_PERM`. Returning
`NULL` on `PMU_ERR_PERM` would force every caller to special-case
`NULL` before calling `pmu_start`/`pmu_read`/`pmu_close` or crash. In
containerized CI -- exactly the environment where a benchmarking toolkit
is most often run -- that is a hard failure.

In v0.3, `pmu_open` instead allocates a **dummy context** with
`fd == -1` and `is_dummy == 1`, sets `*out_err = PMU_ERR_PERM` so the
caller can still tell degradation happened, and returns the dummy
(non-`NULL`). On a dummy context:

- `pmu_start` is a no-op (returns `PMU_OK`, no ioctl).
- `pmu_read` and `pmu_stop_and_read` set `*out_value = 0` and return
  `PMU_OK` (no syscall, no rdpmc).
- `pmu_close` just `free()`s the context (no fd to close, no mmap to
  unmap).
- `pmu_is_available(ctx)` returns `0`.

This lets the two caller patterns keep working without code changes:

1. **Transparent passthrough.** A caller that only needs a non-`NULL`
   ctx to flow through the API can use the dummy transparently; counter
   reads return 0 and the workload runs normally.

2. **Detect-and-fallback.** A caller that needs real counter values calls
   `pmu_is_available(ctx)` right after `pmu_open` (or checks
   `*out_err == PMU_ERR_PERM`) and falls back to `rdtsc` or
   `clock_gettime` when it returns 0, then tears the dummy down with
   `pmu_close`.

`ENODEV`/`ENOENT`/`EINVAL` still return `NULL` with
`PMU_ERR_UNAVAILABLE` -- the counter type is not supported on this CPU,
which the dummy cannot usefully paper over (reads would always return 0,
indistinguishable from a real run that genuinely produced 0). Unknown
counter types return `NULL` with `PMU_ERR_INVALID`. `ENOSYS` is mapped
to `PMU_ERR_PERM` (not `PMU_ERR_UNAVAILABLE`) because it indicates the
syscall itself is unavailable (older kernel or seccomp block), which is
a permission/availability issue the dummy can usefully degrade on.

libpmu stays zero link-time dependency; wiring `pmu_is_available(ctx) == 0`
to an `rdtsc`-based fallback is an application concern (for example
`libspinit`, per the original spec's "adaptive algorithms" use case, if
it later chooses to consume this), not a library-level coupling.

## Why a dummy context and not a separate `pmu_open_dummy`

A separate `pmu_open_dummy` constructor would have forced every caller to
learn a new API entry point and to call it in their own error path. The
dummy is allocated inside `pmu_open` itself so the existing call site
still works: callers that previously did `if (!ctx) { handle_error(); }`
keep working -- they just no longer hit that branch when the failure is a
permission denial, and the workload runs against a ctx that returns 0.
Callers that want to detect the degradation can call `pmu_is_available`;
those that do not can ignore it. The cost is one extra `int` field on
`pmu_ctx_t` and one extra branch at the top of `pmu_start`/`pmu_read`/
`pmu_stop_and_read`/`pmu_close`, which is invisible in any workload that
calls `pmu_read` less than ~10^9 times per second (i.e. all of them).

## Build

The Makefile uses `nasm` (default `NASM ?= nasm`) for
`src/pmu_x86_64.asm`. The asm helpers are x86_64-only; on any other
architecture `PMU_HAVE_RDPMC` is 0 and the rdpmc path is compiled out,
leaving the `read(2)` path as the only one available. The C source
builds warning-clean under `-std=c11 -Wall -Wextra -Werror -pedantic
-O2`.

## Security notes

- `exclude_kernel = 0` (set in `pmu_open`): the counters include
  kernel-mode events. This is the right default for benchmarking (a
  trampoline regression includes the kernel transitions it triggers) but
  means a caller can measure kernel execution time (timing side channel)
  or use cache-miss counters for cache side-channel attacks (Flush+
  Reload, Prime+Probe). `perf_event_paranoid` gates this at the kernel
  level; if the kernel allows the open, the caller already has these
  capabilities and libpmu adds no new surface. Untrusted-caller
  scenarios should set `exclude_kernel = 1` (requires an API addition or
  `pmu_open_raw`).
- The mmap is `PROT_READ` only: libpmu never writes the metadata page,
  so it cannot corrupt kernel state.
- `PERF_FLAG_FD_CLOEXEC` is set at open time so the perf fd does not
  leak across `execve` to a possibly-less-privileged child.
- `rdpmc` with a kernel-supplied `index` could `#GP` if the kernel set a
  bogus index (e.g. with bit 31 set). The kernel is trusted not to do
  this; not a libpmu concern.
- The dummy context holds no kernel resources (no fd, no mmap). It is a
  plain heap allocation; `pmu_close` on a dummy just `free()`s. There is
  no information leak through the dummy because it always returns 0.

## Non-goals (v0.3)

- No generic perf event config.
- No raw MSR programming for AMD general PMCs (the kernel handles event-
  select; libpmu does not need to).
- No multi-counter grouped reads (each `pmu_ctx_t` is one counter): if a
  caller needs cycles + instructions simultaneously, open two contexts.
- No Windows support.
- No caller-configurable degradation policy (e.g. "return `NULL` on
  `PMU_ERR_PERM` instead of a dummy"). The v0.3 contract is that
  `PMU_ERR_PERM` always yields a dummy; callers that want different
  behavior can `pmu_close` the dummy immediately and handle the error
  themselves.
