# libtopo: Design Notes (v0.1)

## Problem

A latency-sensitive application needs to know, once at startup:

- How many CPUs it has, and how they are organized (SMT, cores,
  packages, NUMA nodes).
- Which CPUs share a NUMA node (for memory placement decisions).
- What the cache hierarchy looks like (for sharding decisions: do
  these two CPUs share L3? do they share L2?).
- How to pin a thread to a specific CPU, or to all CPUs in a NUMA
  node.

And, on the hot path of every per-CPU data structure (sharded
counters, per-CPU arenas, MPSC ring buffer slot selection), it needs
to know **which CPU am I running on right now, as cheaply as
possible**. The `getcpu(2)` syscall costs ~96 ns; on a hot path
called millions of times per second, that is the difference between
a per-CPU data structure paying for itself and not.

libtopo answers all five questions. The first four are
initialization-time queries that read CPUID and sysfs; the fifth is
the RDPID fast path.

## Architecture (v0.1)

Five layers, each independently testable:

1. **Topology enumeration** (`topo_probe`):
   CPUID leaf `0x1F` (extended topology v2) when available, else
   leaf `0xB` (x2APIC enumeration). Both expose a subleaf-per-level
   walk: each subleaf reports the level type (SMT, Core, Module,
   Tile, Die, ...) and the number of logical processors at that
   level. The walk terminates when `EAX[4:0] == 0` and
   `EBX[15:0] == 0`. The "highest" level seen (largest
   logical-processor count) is the package level; dividing
   `total_cpus` by it gives `num_packages`.
2. **NUMA discovery** (`topo_numa_nodes`):
   Reads `/sys/devices/system/node/online` for the node list, then
   `/sys/devices/system/node/nodeN/cpulist` for each node's CPU
   mask. On a non-NUMA system the kernel exposes a synthetic
   `node0` containing all online CPUs.
3. **Cache info** (`topo_cache_info`):
   CPUID leaf 4 (deterministic cache parameters). One subleaf per
   cache level (L1D, L1I, L2, L3, ...); each subleaf reports the
   line size, associativity, partitions, sets, and the number of
   logical CPUs sharing an instance. Cache size in bytes = `(ways +
   1) * (partitions + 1) * (line_size + 1) * (sets + 1)`.
4. **Affinity** (`topo_pin`, `topo_pin_node`, `topo_get_affinity`,
   `topo_set_affinity`):
   Wraps `sched_setaffinity(2)` / `sched_getaffinity(2)` with a
   `uint64_t[]` mask layout. `topo_pin` builds a single-bit mask;
   `topo_pin_node` reads the cpulist and builds a multi-bit mask.
5. **Fast getcpu** (`topo_getcpu`):
   RDPID (encoding `F3 0F C7 /F8`) when `CPUID.7.0:ECX[1]` reports
   it available. RDPID reads `IA32_TSC_AUX`, which on Linux holds
   `cpu_number | (node_id << 12)`, so the low 12 bits are the CPU
   number. ~3 cycles. Falls back to `getcpu(2)` syscall (~96 ns in
   the build sandbox) when RDPID is unavailable.

## The ASM boundary

Three NASM routines in `src/topo_x86_64.asm`:

- `topo_cpuid(leaf, subleaf, out[4])` -- CPUID wrapper, identical
  pattern to `libpmu/src/pmu_x86_64.asm` and
  `libspinit/src/spinit_x86_64.asm`. RBX is callee-saved so it is
  pushed; RDX (the `out` pointer) is saved in R8 before CPUID
  because CPUID clobbers EDX (and zero-extends to RDX). Used by
  `topo_probe` (leaves `0x1F` / `0xB`), `topo_cache_info` (leaf
  `4`), and the RDPID-feature check (leaf `7`).
- `topo_rdpid(void)` -- RDPID into EAX. NASM requires an explicit
  32/64-bit GPR operand (`rdpid eax`), unlike RDTSCP which is fixed
  to `EDX:EAX, ECX`. Picks EAX so no extra `mov` is needed for the
  SysV return-value convention. The caller MUST verify
  `CPUID.7.0:ECX[1]` is set before calling -- executing RDPID on an
  unsupported CPU raises `#UD`. `topo_have_rdpid()` does the check.
- `topo_rdtscp_ecx(void)` -- RDTSCP, returns `IA32_TSC_AUX` (ECX) in
  EAX. The TSC value in `EDX:EAX` is discarded. libtopo does not
  call this on the getcpu fast path (RDPID is faster and
  non-serializing); it is exposed for callers that want the
  `IA32_TSC_AUX` value with the ordering guarantee of RDTSCP (e.g.
  pairing with a TSC read for a timestamped CPU report).

## CPUID topology walk

CPUID leaf `0xB` (x2APIC topology) and leaf `0x1F` (extended
topology v2) share the same subleaf format:

| Field        | Bits            | Meaning                                                |
|---|---|---|
| `EAX[4:0]`   | shift right     | Bits to shift the x2APIC ID right to get the next level.|
| `EBX[15:0]`  | logical CPUs    | Number of logical processors at this level (incl. this).|
| `ECX[7:0]`   | level number    | 0 for SMT, 1 for Core, 2 for Module, ...               |
| `ECX[15:8]`  | level type      | 0=Invalid, 1=SMT, 2=Core, 3=Module, 4=Tile, 5=Die.     |
| `EDX[31:0]`  | x2APIC ID       | The current logical processor's x2APIC ID.             |

The terminator subleaf has `EAX[4:0] == 0` and `EBX[15:0] == 0`. We
walk subleaves 0..15 (more than enough; real CPUs stop at subleaf 2
or 3) and record:

- `lp_smt`: the `EBX[15:0]` of the level-type-1 (SMT) subleaf. This
  is `threads_per_core`.
- `lp_core`: the `EBX[15:0]` of the level-type-2 (Core) subleaf.
  This is `cores_per_package * threads_per_core`.
- `lp_high`: the max `EBX[15:0]` seen across all subleaves. This is
  the package-level logical-processor count.

Derived:
- `threads_per_core = lp_smt`
- `cores_per_package = lp_core / lp_smt`
- `num_packages = total_cpus / lp_high`

Leaf `0x1F` supersedes `0xB` and exposes additional levels (Module,
Tile, Die) above Core. libtopo ignores these: the public API only
reports SMT / Core / Package. The higher levels are useful for
things like per-tile cache layout, which `topo_cache_info` exposes
directly via CPUID.4.

`total_cpus` always comes from `sysconf(_SC_NPROCESSORS_ONLN)`, not
from CPUID, because the kernel knows about offlined CPUs, cgroup
cpuset constraints, and CPU hotplug that CPUID cannot see.

## sysfs fallback

If CPUID is uninformative (broken BIOS, certain hypervisors that
don't expose leaf `0xB` / `0x1F` correctly), `topo_probe` falls back
to:

- `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`:
  parse, popcount. Result is `threads_per_core`.
- `/sys/devices/system/cpu/cpu0/topology/core_siblings_list`:
  parse, popcount. Result is `lp_per_package`. Divide by
  `threads_per_core` to get `cores_per_package`.
- `/sys/devices/system/cpu/cpuN/topology/physical_package_id` for
  each online CPU `N`: count unique values. Result is
  `num_packages`.

`num_numa_nodes` is always read from sysfs (`/sys/devices/system/
node/online`), regardless of CPUID.

## RDPID and IA32_TSC_AUX

RDPID reads the `IA32_TSC_AUX` MSR (addr `0xC0000103`) into a
general-purpose register. On Linux the kernel writes
`cpu_number | (node_id << 12)` to this MSR (see `arch/x86/kernel/
tsc.c` and `arch/x86/kernel/cpu/common.c`), so:

- The low 12 bits are the CPU number.
- Bits 12+ hold the NUMA node id.

libtopo masks the RDPID result with `0xFFF` to extract the CPU
number. This handles both modern kernels (which use the
`cpu | (node << 12)` format) and older kernels (which wrote just the
raw CPU number with the high bits zero).

RDTSCP returns the same `IA32_TSC_AUX` value in ECX (alongside the
TSC in `EDX:EAX`). The difference:

- RDPID is non-serializing and writes to any GPR (NASM `rdpid eax`).
  ~3 cycles.
- RDTSCP is partially serializing (it orders prior loads ahead of
  the TSC read, but does not order subsequent loads) and writes to
  fixed registers (`EDX:EAX`, `ECX`). ~30-40 cycles.

libtopo uses RDPID for the getcpu fast path because the serialization
guarantee of RDTSCP is not needed (the kernel updates
`IA32_TSC_AUX` on the scheduling path; reading it without a fence
can return a stale value across a context switch, but the same is
true of the `getcpu(2)` syscall, which also reads the MSR).

## Affinity

`topo_pin` and `topo_pin_node` build a `cpu_set_t` and call
`sched_setaffinity(0, sizeof(set), &set)`. `topo_get_affinity` and
`topo_set_affinity` translate between the `uint64_t[]` mask layout
(LSB = CPU 0) and `cpu_set_t`. The translation is bounded by
`CPU_SETSIZE` (1024 by default), so `mask_words` up to 16 is fully
populated; larger masks work but the high bits are ignored.

The `uint64_t[]` layout is chosen over a direct `cpu_set_t *`
exposure so the public API does not depend on glibc's `cpu_set_t`
layout (which has changed size historically: 1024 CPUs since glibc
2.7, but the kernel accepts masks of any size via the raw syscall).
A caller can pass `mask_words = 16` for the full 1024-CPU range or
`mask_words = 4` for the common 256-CPU case.

## Cache info and per-CPU pinning

CPUID leaf 4 returns the deterministic cache parameters **for the
calling logical CPU**. On a homogeneous package (the common case)
all CPUs in a package return the same parameters. On a heterogeneous
system (e.g. AMD vCache CCDs where one CCD has a 96 MB L3 and the
other has a 32 MB L3), the result depends on which CPU executes the
CPUID.

`topo_cache_info(cpu, level, cache)` handles this by briefly pinning
the calling thread to `cpu` (if `cpu != topo_getcpu()`) via
`sched_setaffinity`, executing CPUID.4, then restoring the original
affinity. The pin-restore pair is two syscalls (~2 us total); on the
hot path this is not a concern because the caller typically caches
the cache-info result at startup.

If the pin fails (e.g. cgroup cpuset excludes the requested CPU),
`topo_cache_info` proceeds with the calling CPU and the result
reflects the calling CPU's caches, not the requested CPU's. This is
documented in the API.

## Bench results

Measured on the build host (`bench/bench_getcpu.c`, 10M iterations,
pinned to CPU 0, TSC at 3.29 GHz):

| path                          | ns/op | cyc/op |
|---|---|---|
| `topo_getcpu` (RDPID)         | 3.3   | 10.6   |
| `sched_getcpu` (libc, inline) | 3.0   | 9.5    |
| `getcpu(2)` syscall           | 95.8  | 306.4  |

The ~3 ns / 10 cycles for `topo_getcpu` breaks down as:

- RDPID instruction: ~3 cycles.
- Branch on the cached `topo_have_rdpid()` flag: ~1 cycle (the
  static variable is in L1, perfectly predicted).
- Function-call overhead of `topo_rdpid` (the asm helper is a
  separate object file; without LTO the call is not inlined): ~5-6
  cycles.

The libc `sched_getcpu` is faster by ~1 ns because glibc inlines the
RDPID intrinsic directly into the caller. The ~1 ns overhead of
`topo_getcpu` is the cost of keeping the RDPID path in NASM (which
makes the asm boundary visible and matches the rest of the toolkit)
rather than using inline asm in the header. Both are ~30x faster
than the syscall; the difference between them is below the noise
floor of most workloads.

If a caller needs the absolute fastest path, calling `topo_rdpid()`
directly (after a one-time `topo_have_rdpid()` check) saves the ~1
ns wrapper overhead. The bench does not do this because the public
API is `topo_getcpu`, and the bench's job is to characterize that
API.

## Zero cross-deps

libtopo links against no other EoSD module. The integration patterns
described in the original design notes -- "pin the libtick thread to
avoid TSC drift", "pin the libflume consumer to a dedicated core",
"shard a libbarrage arena per CPU" -- are documented here, not
implemented in library source:

### Pin the libtick timer thread to avoid TSC drift

`libtick` v0.3 added TSC drift defense. The defense is most effective
when the timer thread is pinned to a single CPU, because then the
TSC it reads is always from the same physical CPU (constant_tsc
guarantees the TSC is consistent across CPUs, but the *rate* at
which it appears to advance relative to wall-clock can vary by a few
ppm between CPUs due to frequency scaling). The pattern:

```c
topo_info_t info;
topo_probe(&info);
topo_pin(0);                 /* pin the timer thread to CPU 0 */
/* ... libtick operations ... */
```

This is the caller's responsibility, not libtick's. libtick does not
link libtopo.

### Pin the libflume consumer to a dedicated core

`libflume`'s single consumer drains the ring buffer. Pinning the
consumer to a dedicated core (no producers on that core) eliminates
scheduler interference and cache pollution from producer threads.
The pattern:

```c
topo_pin(info.total_cpus - 1);   /* last CPU is the consumer */
```

### Sharded libbarrage arena per CPU

A per-CPU `libbarrage` arena gives each thread its own bump
allocator with zero cross-thread cache-line traffic. The arena is
selected by `topo_getcpu()`:

```c
static barrage_arena_t *arenas[256];
unsigned cpu = topo_getcpu();
void *p = barrage_alloc(arenas[cpu], size, align);
```

This is the hot-path use case for `topo_getcpu`'s ~3 ns cost: a
per-CPU arena lookup at every allocation. At 2.3 ns/op for
`barrage_alloc`, the 3 ns `topo_getcpu` overhead is significant
(bumps the alloc to ~5 ns); a caller who knows the CPU at function
entry can hoist the `topo_getcpu` call. The `libbarrage` extreme
test exercises this pattern (see `tests/extreme/`).

### libpmu rdpmc fast path on a pinned CPU

`libpmu`'s rdpmc fast path reads `perf_event_mmap_page->index` to
select the PMC. The index is per-CPU: if the calling thread
migrates between CPUs, the kernel reprograms the PMCs and the index
changes. Pinning the calling thread to a single CPU before opening
the perf event and reading it eliminates the seqlock churn from
cross-CPU migration. The pattern:

```c
topo_pin(target_cpu);
pmu_ctx_t *ctx = pmu_open(PMU_CYCLES, &err);
/* ... pmu_read is now stable, no seqlock retries from migration ... */
```

## Build

The Makefile uses `nasm` (default `NASM ?= nasm`) for
`src/topo_x86_64.asm`. The asm helpers are x86_64-only; on any other
architecture `TOPO_HAVE_X86` is 0 and the CPUID-driven paths are
compiled out, leaving sysfs + `getcpu(2)` as the only paths. The C
source builds warning-clean under `-std=c11 -Wall -Wextra -Werror
-pedantic -O2`.

Tests and benches are linked with `-D_GNU_SOURCE` (for
`sched_getaffinity` / `CPU_SET` / `syscall(SYS_getcpu)`) and `-lrt`
(for `clock_gettime` on older glibc; modern glibc has it in libc, so
`-lrt` is a no-op).

## Security notes

- `topo_pin` and `topo_set_affinity` change the calling thread's
  CPU affinity. They cannot widen the affinity beyond the cgroup
  cpuset (the kernel silently masks the request to the cpuset). They
  can narrow it freely. This is the standard Linux behavior; libtopo
  adds no new surface.
- `topo_getcpu` reads the `IA32_TSC_AUX` MSR via RDPID. The MSR
  holds the CPU number and NUMA node id (no sensitive data). The
  `getcpu(2)` syscall returns the same information.
- `topo_probe` and `topo_numa_nodes` read sysfs. The files
  (`/sys/devices/system/cpu/cpu*/topology/*`,
  `/sys/devices/system/node/nodeN/cpulist`) are world-readable. No
  sensitive information is exposed beyond what the kernel already
  exports.
- `topo_cache_info` briefly pins the calling thread to `cpu` (if
  different from the current CPU) via `sched_setaffinity`, then
  restores the original affinity. The pin-restore pair is two
  syscalls. There is no risk of leaving the thread pinned to a
  different CPU: even if the restore fails (which would only happen
  if the original affinity is no longer valid, e.g. because the
  cgroup cpuset changed underneath us), the worst case is the thread
  is left pinned to `cpu`, which the caller asked about.

## Non-goals (v0.1)

- No per-pair NUMA distance reporting. `/sys/.../nodeN/distance` is
  a per-pair matrix, not a per-node scalar. Callers that need it
  read the sysfs file directly.
- No CPU hotplug notification. `topo_probe` and `topo_numa_nodes`
  reflect the state at call time.
- No IRQ affinity management. Use `/proc/irq/N/smp_affinity`
  directly.
- No memory allocation policy (`set_mempolicy(2)`, `mbind(2)`).
- No Windows / macOS support.
- No support for > 256 CPUs in `topo_numa_node_t.cpu_mask`. The
  `topo_get_affinity` / `topo_set_affinity` pair does not have this
  limit.
