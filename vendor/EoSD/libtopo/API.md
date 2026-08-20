# libtopo: API (v0.1)

Status: v0.1 shipped. CPU topology enumeration, NUMA discovery, cache
info, CPU affinity, and fast `getcpu` (RDPID on x86_64). Implementation
in `src/topo.c` + `src/topo_x86_64.asm` (x86_64 only, System V AMD64
ABI). Tests in `tests/`, benches in `bench/`. All signatures match
`include/topo.h`.

## Overview

libtopo answers four questions a low-level latency-sensitive program
needs answered once at startup, and one it needs answered cheaply on
every call:

1. **How many CPUs do I have, and how are they organized?**
   (`topo_probe`: threads per core, cores per package, packages,
   NUMA nodes, total online CPUs.)
2. **Which CPUs belong to which NUMA node?**
   (`topo_numa_nodes`: per-node CPU bitmap.)
3. **What do the caches look like?**
   (`topo_cache_info`: L1/L2/L3 size, line size, sharing.)
4. **Can I pin a thread to a specific CPU, or to a NUMA node?**
   (`topo_pin`, `topo_pin_node`, `topo_get_affinity`,
   `topo_set_affinity`.)
5. **Which CPU am I running on right now, as cheaply as possible?**
   (`topo_getcpu`: RDPID on x86_64, ~3 ns; getcpu(2) syscall
   fallback, ~96 ns.)

The first four are initialization-time queries that read CPUID and
sysfs. The fifth is on the hot path of any per-CPU data structure
(sharded counters, per-CPU arenas, MPSC ring buffer slot selection)
and is the reason libtopo exists as a separate module rather than a
few inline helpers in libtick or libflume.

Zero cross-module dependencies. The library links against libc only.

## Types

```c
typedef struct {
    unsigned threads_per_core;
    unsigned cores_per_package;
    unsigned num_packages;
    unsigned num_numa_nodes;
    unsigned total_cpus;
} topo_info_t;

typedef struct {
    int      node_id;
    uint64_t cpu_mask[4];   /* up to 256 CPUs */
    unsigned cpu_count;
} topo_numa_node_t;

typedef struct {
    unsigned level;       /* 1=L1, 2=L2, 3=L3 */
    unsigned size_kb;
    unsigned line_size;
    unsigned sharing;     /* number of logical CPUs sharing this cache */
} topo_cache_t;
```

`topo_numa_node_t.cpu_mask` is fixed at 4 `uint64_t` words (256 bits).
For systems with more than 256 CPUs the higher-numbered CPUs are not
represented in this struct; `topo_get_affinity` / `topo_set_affinity`
accept a caller-chosen word count and do not have this limit. The
fixed-width struct is the right shape for the common case (single
socket, <= 256 logical CPUs) and keeps `topo_numa_nodes` simple.

## API

```c
int      topo_probe(topo_info_t *info);
unsigned topo_numa_nodes(topo_numa_node_t *nodes, unsigned max_nodes);
int      topo_cache_info(unsigned cpu, unsigned level, topo_cache_t *cache);
int      topo_pin(unsigned cpu);
int      topo_pin_node(int node_id);
unsigned topo_getcpu(void);
int      topo_get_affinity(uint64_t *mask, unsigned mask_words);
int      topo_set_affinity(const uint64_t *mask, unsigned mask_words);
```

### topo_probe

```c
int topo_probe(topo_info_t *info);
```

Fills `*info` with the system's CPU/NUMA layout. Always returns 0
unless `info` is NULL (`-EINVAL`); every field has a defensive
default of 1 so the caller can proceed even when both CPUID and
sysfs are uninformative.

Source of truth on x86_64: CPUID leaf `0x1F` (extended topology v2)
when available, else leaf `0xB` (x2APIC enumeration). Both expose a
subleaf-per-level walk; each subleaf reports the level type (SMT,
Core, Module, Tile, Die) and the number of logical processors at
that level. The walk terminates when `EAX[4:0] == 0` and
`EBX[15:0] == 0`.

`total_cpus` comes from `sysconf(_SC_NPROCESSORS_ONLN)`.
`num_numa_nodes` comes from `/sys/devices/system/node/online`. On a
non-NUMA system the kernel exposes a single synthetic `node0` and
`num_numa_nodes == 1`.

If CPUID is uninformative (broken BIOS, certain hypervisors), the
fields are filled from
`/sys/devices/system/cpu/cpuN/topology` (`thread_siblings_list`,
`core_siblings_list`, `physical_package_id`).

Thread-safety: safe to call concurrently. The CPUID-feature
detection is memoized in a static variable; the benign race (two
threads filling it in) yields the same value because CPUID output
is constant per CPU.

### topo_numa_nodes

```c
unsigned topo_numa_nodes(topo_numa_node_t *nodes, unsigned max_nodes);
```

Returns the count of online NUMA nodes visible to the kernel (always
`>= 1`: on a non-NUMA system the kernel exposes a synthetic `node0`).
If `nodes` is non-NULL and `max_nodes > 0`, fills `nodes[0..min(count,
max_nodes)-1]` with the descriptor for each node, in ascending
`node_id` order. The `cpu_mask` field is populated from
`/sys/devices/system/node/nodeN/cpulist`; `cpu_count` is the
popcount of `cpu_mask`.

If a node's cpulist cannot be read (e.g. sysfs not mounted), the
corresponding entry's `cpu_mask` is zeroed and `cpu_count` is 0;
`node_id` is still filled. The return value is always the true node
count, even if more than `max_nodes` (so the caller can detect
truncation).

If `/sys/devices/system/node` is entirely missing (chroot without
`/sys` mounted), reports 1 synthetic node with all online CPUs in
its mask.

Thread-safety: safe to call concurrently; reads sysfs only.

### topo_cache_info

```c
int topo_cache_info(unsigned cpu, unsigned level, topo_cache_t *cache);
```

Returns 0 and fills `*cache` on success. `-EINVAL` if `cache` is
NULL or `level` is 0 or > 3. `-ENOENT` if the requested level is
not present (e.g. asking for L3 on a CPU with no L3). `-ENOSYS` on
non-x86_64 hosts (CPUID leaf 4 is x86-specific).

On x86_64 this calls CPUID leaf 4 (deterministic cache parameters)
for the calling logical CPU. If `cpu` is not the current CPU, the
calling thread is briefly pinned to `cpu` via `sched_setaffinity`
and the original affinity is restored afterwards; on homogeneous
systems (the common case) all CPUs in a package return the same
parameters and the pin is a no-op correctness-wise. Per-CPU caches
that differ between cores (e.g. AMD vCache CCDs) get the right
answer for the requested CPU.

Cache size is computed as
`(ways + 1) * (partitions + 1) * (line_size + 1) * (sets + 1)`
from the CPUID.4 fields. `sharing` is `EAX[25:14] + 1` (the number
of logical CPUs sharing an instance of this cache).

The `level` argument selects the cache level (1, 2, or 3). If both
L1D and L1I exist (both at level 1), the first one reported by
CPUID.4 is returned; the caller cannot request L1I specifically
through this API. Add a separate `topo_cache_info_type` API if a
caller needs the L1I parameters.

Thread-safety: safe to call concurrently; the brief affinity pin
affects only the calling thread.

### topo_pin

```c
int topo_pin(unsigned cpu);
```

Pins the calling thread to a single CPU. Builds a single-bit
affinity mask containing only `cpu` and calls `sched_setaffinity(2)`.
Returns 0 on success, `-EINVAL` if `cpu >= CPU_SETSIZE` (1024), or
the negated `errno` from `sched_setaffinity` on failure.

The pin persists until the caller changes affinity again (via
`topo_set_affinity`, `topo_pin_node`, or directly via
`sched_setaffinity`).

Thread-safety: safe to call concurrently; affects only the calling
thread.

### topo_pin_node

```c
int topo_pin_node(int node_id);
```

Pins the calling thread to all CPUs in NUMA node `node_id`. Reads
`/sys/devices/system/node/nodeN/cpulist`, builds the corresponding
affinity mask, and calls `sched_setaffinity(2)`. Returns 0 on
success, `-EINVAL` if `node_id < 0`, `-ENOENT` if the node does not
exist or its cpulist cannot be read, `-EINVAL` if the cpulist is
empty or contains an out-of-range CPU, or the negated `errno` from
`sched_setaffinity` on failure.

Thread-safety: safe to call concurrently; affects only the calling
thread.

### topo_getcpu

```c
unsigned topo_getcpu(void);
```

Returns the current CPU number (0 to `total_cpus - 1`). On x86_64
with RDPID support (`CPUID.7.0:ECX[1]`), uses the RDPID instruction
(~3 cycles) and returns the low 12 bits of `IA32_TSC_AUX`, which on
Linux is set to the CPU number (the high bits hold the NUMA node
id). On CPUs without RDPID, or on non-x86_64 hosts, falls back to
the `getcpu(2)` syscall (~96 ns in the build/test sandbox). Returns
0 if both paths fail (defensive; the kernel always reports a valid
CPU in practice).

The RDPID result can be momentarily stale across a context switch
(the kernel updates `IA32_TSC_AUX` on the scheduling path), but
this is no worse than the `getcpu(2)` syscall, which reads the same
MSR.

Measured on the build host (`bench/bench_getcpu.c`, 10M iterations,
pinned to CPU 0, TSC at 3.29 GHz):

| path                          | ns/op | cyc/op |
|---|---|---|
| `topo_getcpu` (RDPID)         | 3.3   | 10.6   |
| `sched_getcpu` (libc, inline) | 3.0   | 9.5    |
| `getcpu(2)` syscall           | 95.8  | 306.4  |

The ~3 ns / 10 cycles for `topo_getcpu` is the RDPID instruction
itself (~3 cycles) plus a branch on the cached RDPID-availability
flag plus the function-call overhead of the `topo_rdpid` asm helper
(no LTO in the default build). The libc `sched_getcpu` is faster by
~1 ns because glibc inlines the RDPID intrinsic directly. Both are
~30x faster than the syscall.

Thread-safety: safe to call concurrently.

### topo_get_affinity / topo_set_affinity

```c
int topo_get_affinity(uint64_t *mask, unsigned mask_words);
int topo_set_affinity(const uint64_t *mask, unsigned mask_words);
```

`topo_get_affinity` calls `sched_getaffinity(2)` into `mask`,
treated as `mask_words` `uint64_t` words (256 bits at `mask_words =
4`). CPUs beyond `mask_words * 64` are silently truncated (caller
passes a larger mask if needed). Returns 0 on success, `-EINVAL` if
`mask` is NULL or `mask_words` is 0, or the negated `errno` from
`sched_getaffinity`.

`topo_set_affinity` calls `sched_setaffinity(2)` with a mask built
from `mask[0..mask_words-1]`. Returns 0 on success, `-EINVAL` if
`mask` is NULL, `mask_words` is 0, or no bit is set in the mask
(`sched_setaffinity` rejects empty masks with `EINVAL`), or the
negated `errno` from `sched_setaffinity`.

Both translate between the `uint64_t[]` layout (LSB = CPU 0) and the
`cpu_set_t` layout used by the libc wrappers. The translation is
bounded by `CPU_SETSIZE` (1024 by default), so `mask_words` up to 16
is fully populated; larger masks work but the high bits are ignored.

Thread-safety: safe to call concurrently; affects only the calling
thread.

## Minimal usage example

```c
#include <stdio.h>
#include "topo.h"

int main(void) {
    topo_info_t info;
    if (topo_probe(&info) != 0) return 1;
    printf("cpus=%u threads/core=%u cores/pkg=%u pkgs=%u numa=%u\n",
           info.total_cpus, info.threads_per_core,
           info.cores_per_package, info.num_packages,
           info.num_numa_nodes);

    /* Pin to CPU 0 and verify. */
    if (topo_pin(0) == 0) {
        printf("pinned to CPU %u\n", topo_getcpu());
    }

    /* List NUMA nodes. */
    topo_numa_node_t nodes[8];
    unsigned n = topo_numa_nodes(nodes, 8);
    for (unsigned i = 0; i < n; i++) {
        printf("node %d: %u cpus\n", nodes[i].node_id, nodes[i].cpu_count);
    }

    /* Print L1/L2/L3 cache info. */
    topo_cache_t c;
    for (unsigned lvl = 1; lvl <= 3; lvl++) {
        if (topo_cache_info(0, lvl, &c) == 0) {
            printf("L%u: %u KB, line %u, sharing %u\n",
                   c.level, c.size_kb, c.line_size, c.sharing);
        }
    }
    return 0;
}
```

## Edge cases

- **`topo_probe` with NULL `info`**: returns `-EINVAL`.
- **`topo_probe` on a system without `/sys/devices/system/node`**:
  `num_numa_nodes` defaults to 1 (synthetic node0). Every other
  field has a default of 1.
- **`topo_pin(cpu)` where `cpu` is excluded by a cgroup cpuset**:
  `sched_setaffinity` returns `EINVAL`; `topo_pin` returns
  `-EINVAL`. The thread's existing affinity is unchanged.
- **`topo_pin_node(node_id)` for a non-existent node**: returns
  `-ENOENT`.
- **`topo_cache_info(cpu, level, ...)` for an absent level**: returns
  `-ENOENT`. The `cache` struct is left zeroed.
- **`topo_getcpu` on a CPU without RDPID**: falls back to the
  `getcpu(2)` syscall. ~30x slower (~96 ns vs ~3 ns) but correct.
- **`topo_getcpu` if `getcpu(2)` fails**: returns 0 (defensive). The
  kernel always reports a valid CPU in practice.

## Non-goals (v0.1)

- No per-pair NUMA distance reporting. `/sys/.../nodeN/distance` is
  a per-pair matrix, not a per-node scalar, and does not fit the
  `topo_numa_node_t` shape. Callers that need it read the sysfs
  file directly.
- No CPU hotplug notification. `topo_probe` and `topo_numa_nodes`
  reflect the state at call time; a CPU hotplug event between two
  calls can return different results. Register for udev events if
  you need live updates.
- No IRQ affinity management. `topo_pin` only affects the calling
  thread's CPU affinity; it does not move IRQs. Use
  `/proc/irq/N/smp_affinity` directly.
- No memory allocation policy (`set_mempolicy(2)`, `mbind(2)`).
  libtopo exposes *CPU* affinity; NUMA memory policy is a separate
  concern. Callers that want both should call `set_mempolicy` or
  `mbind` directly.
- No Windows (`GetLogicalProcessorInformationEx`) or macOS
  (`thread_policy_set`) support in v0.1. Linux/x86_64 only
  (`nasm -f elf64`).
- No support for > 256 CPUs in the `topo_numa_node_t.cpu_mask`
  struct. `topo_get_affinity` / `topo_set_affinity` accept a
  caller-chosen word count and do not have this limit.
