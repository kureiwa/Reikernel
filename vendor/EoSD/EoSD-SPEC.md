# EoSD Toolkit: Project Specification (v0.3, shipped)

> Embodiment of Sakuya's Asynchronous Kernel-bypass Utility for Yielding & ABI-control
> Deterministic Overhead Optimization for Microtimings

Status: shipped. All ten modules are implemented with tests and benchmarks
at v0.3 (libsva remains at v0.2). This document was originally a
planning artifact from the v0.1 interview; it has been updated in place
to reflect the v0.3 repository layout, module list, and resolved
decisions. Per-module `API.md` and `DESIGN.md` are the source of truth
for function-level detail; this document stays at the architecture /
convention level.

---

## 1. Project Identity

A collection of 10 independent, low-level C11 systems libraries for
Linux/x86_64 (cross-platform expansion later), focused on latency-sensitive
primitives: timing, coroutines, spinlocks, function interception,
serialization, crash handling, hardware performance counters, arena
allocation, virtual memory management, and wait-free message queuing.
The tenth module, `libflume`, was added after the original v0.1 plan as
a wait-free MPSC ring buffer to cover the message-queue primitive called
out in the original design notes but not assigned to a module.

Open-source, public release. The author is an experienced low-level
developer (x86-64 ASM/C since age 13, Arch Linux since 12), so this is not a
beginner-oriented library; explanations in code and docs should assume
systems programming fluency.

This is primarily a mastery/portfolio project. Depth, correctness, and
understanding of OS/ABI internals matter more than shipping speed, but it's
still meant to be genuinely usable once done, not throwaway.

Not modeled on or competing with anything specific. Original design.

---

## 2. Platform & Toolchain

| Aspect | Decision |
|---|---|
| Language standard | C11 |
| Starting arch/OS | x86_64, Linux (Arch) only |
| Future expansion | Windows, macOS, possibly ARM64, after Linux/x86_64 is solid |
| Assembly | NASM, in separate `.asm` files (not inline asm, not GAS) |
| Build system | Plain Makefiles |
| License | MIT |

Implication for Makefiles: each module's build must assemble `.asm` files via
`nasm` to `.o` and link them with the C-compiled `.o` files. Since NASM
output format differs per OS (`elf64` / `win64` / `macho64`), the Makefile
will need an OS-detection or override variable (`FORMAT=elf64` by default),
so the door stays open for the later cross-platform expansion without a
rewrite.

---

## 3. Module Relationship Model

Decision: fully independent, zero cross-deps. Each of the 10 modules
compiles and links standalone. A consumer can take just `libbarrage`
and nothing else.

This directly affects how the use cases described in the original design
notes are treated. Cross-module scenarios described informally there (for
example "libcrash catches libsva's guard-page SIGSEGV," "libspoon uses
libsva for stacks," "libspinit calibrates using libpmu," "libflume is
wrapped by libsva guard pages and drained into libpack") are not
implemented as link-time or compile-time dependencies. They're instead
documented integration patterns: example code showing how a consumer
could wire two modules together in their own application, living in each
module's `DESIGN.md` or an examples folder, never inside the library
source itself.

Tradeoff accepted: slightly less "integrated toolkit" elegance, in exchange
for each module being independently cloneable and usable with no hidden
coupling.

### Pluggable allocator, resolved scope

"Pluggable allocator" doesn't mean a shared central allocator module (that
would violate zero-cross-deps). Instead:

- Only modules that actually own long-lived or bulk memory get a pluggable
  allocator hook, defined locally within that module's own header (a simple
  function-pointer struct, `alloc`/`free`, defaulting to libc malloc/free if
  the caller passes NULL).
- Modules needing this: libspoon (coroutine stacks), libbarrage (arena
  backing store), libsva (mapped region bookkeeping structures, not the
  mmap'd memory itself), libpack (only if the caller wants heap-based
  scratch buffers for variable-length serialization), libflume (the
  `flume_t` bookkeeping handle is malloc'd; the ring itself is mmap'd
  or caller-provided).
- Modules that don't need pluggable allocation: libtick, libpmu, libspinit.
  These are stateless or fixed-size-state and should avoid heap allocation
  entirely.
- libdetour and libcrash get no general allocator hook. libcrash in
  particular must be async-signal-safe and cannot call malloc under any
  configuration; any state it needs must be pre-allocated by the caller at
  init time.

---

## 4. Cross-Module Engineering Conventions

These apply uniformly across all 10 modules so the toolkit feels coherent
even though the code is decoupled.

- **Error handling.** Plain `int` return codes. `0` means success, negative
  values are errno-style error codes (module-specific enums allowed, but
  must stay in negative integer space so callers can uniformly check
  `if (rc < 0)`).
- **Naming.** Per-module prefix matching the module's short name as used in
  the original design doc: `tick_sleep_until()`, `spoon_switch_to()`,
  `spinlock_acquire()` (`libspinit`), `detour_install()`, `pack_serialize()`,
  `crash_install_handler()`, `pmu_read_counter()`, `barrage_alloc()`,
  `sva_map_guarded()`, `flume_enqueue()`. Exact function names get finalized
  per module when each one is designed individually.
- **Thread-safety.** Documented explicitly per function, not per module.
  Every public function's header comment must state whether it's safe to
  call concurrently from multiple threads, and under what conditions (for
  example "safe if each thread uses its own arena handle; NOT safe to share
  one handle across threads without external locking").
- **Versioning.** Independent semver per module (`libtick` can be `v0.3.0`
  while `libspoon` is `v0.1.0`). No single toolkit-wide version number.
- **Performance numbers in the original design doc** (for example
  "50-100ns context switch," "1-2ns bump allocation") are ballpark targets,
  not guarantees. Real numbers get established per module by the
  microbenchmark suite once implemented, and documented in that module's
  `DESIGN.md` with the actual measured hardware and conditions.

---

## 5. Repository Layout

Monorepo, one module per top-level directory, each independently buildable:

```
eosd/
  libtick/
    include/tick.h
    src/tick.c
    src/tick_x86_64.asm
    tests/
    bench/
    API.md
    DESIGN.md
    Makefile
  libspoon/
    include/spoon.h
    src/spoon.c
    src/spoon_x86_64.asm
    tests/
    bench/
    API.md
    DESIGN.md
    Makefile
  libspinit/  ...
  libdetour/  ...
  libpack/    ...
  libcrash/   ...
  libpmu/     ...
  libbarrage/ ...
  libsva/     ...
  libflume/
    include/flume.h
    src/flume.c
    src/flume_x86_64.asm
    tests/
    bench/
    API.md
    DESIGN.md
    Makefile
  tests/extreme/   (cross-module extreme stress tests, shared latency.h)
  example/         (cross-module integration demos)
  tools/           (crashdump CLI for decoding libcrash dumps)
  LICENSE      (MIT)
  README.md    (toolkit overview, links to each module)
  EoSD-SPEC.md (this document)
  Makefile     (top-level: builds all 10 modules + tests/extreme + tools)
```

Each module directory is self-contained. Someone can `cp -r libbarrage/
myproj/` and it still builds.

---

## 6. Documentation

Per module, hand-written Markdown, no Doxygen generation pipeline:

- `API.md`: public function signatures, parameters, return codes,
  thread-safety notes, minimal usage example.
- `DESIGN.md`: the "why." What problem it solves, what it does not do
  (explicit non-goals, since these modules are narrow by design), the ASM
  approach and why, measured benchmark numbers, known limitations.

Top-level `README.md` gives the toolkit pitch and links to each module,
including the documented (non-linked) integration patterns from Section 3.

---

## 7. Testing & Quality Bar

Three tiers, required for every module before it's considered done. All
10 modules meet this bar in v0.3:

1. **Unit tests.** Correctness of the public API, edge cases, error code
   paths. Each module ships at least one test binary; libflume ships
   four (`test_basic`, `test_mpsc`, `test_batch`, `test_stress`).
2. **Microbenchmarks.** Verify actual latency and throughput against the
   ballpark numbers in the design notes; results recorded in that module's
   `DESIGN.md`. libflume ships `bench_enqueue`, `bench_drain`, and
   `bench_mpsc`.
3. **Stress and fuzz tests.** Sustained load (millions of spinlock
   acquisitions, thousands of coroutine switches, repeated crash-handler
   triggers in a subprocess harness, malformed input to `libpack`
   deserialization) to catch corruption, leaks, and crashes under pressure.
   The `tests/extreme/` suite covers 9 of the 10 modules in v0.3 and
   reports p50/p99/max latency via the shared `tests/extreme/latency.h`
   helpers.

Signal-safety-critical code (`libcrash`) additionally has a dedicated test
harness that deliberately triggers real signals (SIGSEGV, SIGABRT) in a
forked child process and inspects the resulting dump, since this can't be
unit-tested in-process safely.

---

## 8. Implementation Order

Left to the author's discretion at build time. No fixed sequence agreed in
this spec. As of v0.3 all ten modules are implemented; the actual order
was libtick -> libspoon -> libspinit -> libdetour -> libpack -> libcrash
-> libpmu -> libbarrage -> libsva -> libflume, with libflume added last
to cover the wait-free MPSC ring-buffer primitive called out in the
original design notes but not assigned to a module.

---

## 9. The 10 Modules: Summary Table

| Module | One-line purpose | Needs pluggable alloc? | Signal-safety constraints? |
|---|---|---|---|
| `libtick` | High-res sleep-until with deadline/overshoot reporting | No | No |
| `libspoon` | Hand-rolled asymmetric coroutine context switch | Yes (stacks) | No |
| `libspinit` | TSC-calibrated adaptive spinlock w/ futex fallback | No | No |
| `libdetour` | Inline-hook function interception (no LD_PRELOAD) | No | No |
| `libpack` | Compile-time-described struct (de)serialization, SIMD endianness | Optional | No |
| `libcrash` | Async-signal-safe crash/minidump writer | No (pre-allocated only) | Yes, core constraint |
| `libpmu` | Hardware performance counter reads (rdpmc/perf_event_open) | No | No |
| `libbarrage` | Thread-local bump allocator with batch reset | Yes (backing store) | No |
| `libsva` | Guarded mmap/VirtualAlloc wrapper, guard pages, TLB flush | Yes (bookkeeping) | Interacts with `libcrash` via documented pattern, not linkage |
| `libflume` | Wait-free MPSC ring buffer (LOCK XADD producers, single consumer) | Yes (handle only; ring is mmap'd or caller-provided) | No |

v0.3 changes per module: `libtick` adds TSC drift defense; `libspinit`
adds exponential backoff; `libpack` adds ifunc dispatch (SSSE3/AVX2/AVX-512);
`libcrash` adds user blobs and the CRASH_AFTER_FORK out-of-process path;
`libpmu` adds graceful degradation via a dummy ctx when `perf_event_open`
is denied; `libsva` (v0.2) adds `SVA_PROT_GUARD_BOTH` and `MAP_HUGETLB`;
`libflume` ships with `LOCK XADD` enqueue, batched drain, and a
multi-producer contention bench. Per-module semver is independent.

Full technical detail for each module (exact API, struct layout, NASM
routine boundaries, benchmark targets) is out of scope for this document.
It is captured in each module's `API.md` and `DESIGN.md`; the v0.3 reality
of all ten modules is reflected there.

---

## 10. Resolved Decisions (formerly Open Items)

- **Windows/macOS NASM object format and ABI differences.** Still deferred.
All v0.3 modules assemble with `nasm -f elf64` unconditionally; no
format-detection logic in any Makefile. The Non-goals section of each
module's `API.md` carries a "No Windows/macOS support in v0.X" line.
- **ARM64 support.** Still deferred; no work planned. The `libspoon`,
`libspinit`, `libdetour`, `libcrash`, `libpmu`, `libsva`, `libtick`, and
`libflume` ASM paths are x86_64-only.
- **CI.** Still deferred; no pipeline is wired. Each module's `make test`
and `make bench` are the verification entry points.
- **Exact function-level API signatures.** Each module's own `API.md` is the
source of truth; this document stays at the architecture/convention
level. All ten modules' `API.md` files are current as of v0.3.
- **v0.3 module additions.** `libflume` was added as the tenth module after
the original v0.1 plan, covering the wait-free MPSC ring-buffer primitive
called out in the original design notes. The public API (`flume_create`,
`flume_attach`, `flume_enqueue`, `flume_dequeue`, `flume_drain`,
`flume_lag`, `flume_ring_bytes`) has been stable since v0.1; the v0.3
bump reflects toolkit-wide version alignment and the addition of
`bench/bench_mpsc.c`. No cross-module link-time dependencies were
introduced: libflume's integrations with libsva (guarded ring), libtick
(lag-time estimation), libspinit (external blocking), and libpack
(zero-copy bulk deserialize) are documented patterns in `libflume/DESIGN.md`,
not library linkage.
- **Extreme test suite.** `tests/extreme/` covers 9 of the 10 modules in
v0.3 (libflume has no `test_flume_extreme` yet; it is exercised by
`tests/test_basic`, `tests/test_mpsc`, `tests/test_batch`, and
`tests/test_stress` plus `bench/bench_enqueue`, `bench/bench_drain`, and
`bench/bench_mpsc`). Each extreme test reports p50/p99/max latency via the
shared `tests/extreme/latency.h` helpers.
- **Tools.** `tools/crashdump` ships as the v0.3 standalone CLI for
decoding libcrash dumps (custom + ELF formats) to text/JSON/hex. Built via
`make tools` at the repo root.
