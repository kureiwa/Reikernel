# EoSD Toolkit

> Embodiment of Sakuya's Asynchronous Kernel-bypass Utility for Yielding & ABI-control
> Deterministic Overhead Optimization for Microtimings

A collection of thirteen independent, low-level C11 systems libraries for
Linux/x86_64. Each module is self-contained: clone one directory, build
it, use it. No cross-module link-time dependencies.

## Modules

| Module | Purpose | Feature |
|---|---|---|
| [`libtick`](./libtick/) | High-res sleep-until with deadline/overshoot reporting | Binary min-heap (30 ns peek), TSC drift defense |
| [`libspoon`](./libspoon/) | Hand-rolled asymmetric coroutine context switch | 15 ns/switch, MXCSR + x87 CW preserved |
| [`libspinit`](./libspinit/) | TSC-calibrated fixed-window spinlock with futex fallback | Exponential backoff, 3-state futex mutex |
| [`libdetour`](./libdetour/) | Inline-hook function interception (no LD_PRELOAD) | int3-brokered patching, PIE-safe trampolines |
| [`libpack`](./libpack/) | Compile-time-described struct serialization, SIMD endianness | ifunc dispatch: SSSE3/AVX2/AVX-512, variable-length |
| [`libcrash`](./libcrash/) | Async-signal-safe crash/minidump writer | ELF core, _Fork out-of-process, user blobs |
| [`libpmu`](./libpmu/) | Hardware performance counter reads | rdpmc fast path, dummy context on perf denial |
| [`libbarrage`](./libbarrage/) | Per-thread bump allocator with batch reset | Alignment parameter, alignas(64), mmap/malloc |
| [`libsva`](./libsva/) | Guarded mmap wrapper, guard pages, TLB flush | Underflow guard, MAP_HUGETLB support |
| [`libflume`](./libflume/) | Wait-free MPSC ring buffer | LOCK XADD enqueue, batched drain, 60 ns/op |
| [`liburing`](./liburing/) | Minimal io_uring wrapper | SQ/CQ ring with shared-memory atomics, 167 ns round-trip |
| [`libtopo`](./libtopo/) | CPU topology and NUMA awareness | RDPID getcpu (3 ns), CPUID 0xB/0x1F, affinity pinning |
| [`libpkey`](./libpkey/) | Intel Memory Protection Keys (MPK) | WRPKRU/RDPKRU (~20 cycles), sub-microsecond page protection |

Each module directory has its own `API.md`, `DESIGN.md`, `include/`,
`src/`, `tests/`, `bench/`, and `Makefile`.

## Tools

| Tool | Purpose |
|---|---|
| [`tools/crashdump`](./tools/) | Standalone CLI to decode libcrash dumps (custom + ELF) to text/JSON/hex |

## Build

Plain Makefiles, NASM for assembly. Targets Linux/x86_64.

    make            # build all 13 modules
    make test       # run all module test suites
    make bench      # run all benchmarks
    make extreme    # run extreme stress tests
    make tools      # build the crashdump CLI
    make clean      # clean everything

Or build a single module:

    cd libtick && make && make test

Requirements: gcc (C11), nasm 2.16+, GNU make.

## Performance

| Module | Metric | Result |
|---|---|---|
| libtick | registry scan @1000 timers | 30 ns (O(1) heap peek) |
| libspoon | context switch | 15 ns |
| libspinit | uncontended lock+unlock | 15 ns |
| libdetour | hook overhead | 4.0 cycles |
| libpack | serialize throughput | 12.7 GB/s at 4 KB (AVX-512 ifunc) |
| libbarrage | allocation latency | 2.3 ns |
| libflume | enqueue latency | 60 ns (wait-free) |
| liburing | submit+reap round-trip | 167 ns |
| libtopo | getcpu via RDPID | 3 ns (29x faster than syscall) |
| libpkey | WRPKRU page protection | ~20 cycles (100x faster than mprotect) |

## Testing

Three tiers:

1. **Unit tests** (`lib*/tests/`): correctness, edge cases, error paths.
2. **Extreme tests** (`tests/extreme/`): max capacity, high contention,
   fuzz round-trips, deep recursion, guard-page faults, crash dumps,
   p50/p99/max latency.
3. **Examples** (`example/`): cross-module integration demos.

## License

MIT. See [`LICENSE`](./LICENSE).
