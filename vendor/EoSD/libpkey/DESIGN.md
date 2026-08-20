# libpkey: Design Notes (v0.1)

## Problem

`mprotect(2)` is the only portable Linux primitive for changing page
permissions, but it is a syscall: ~100-1500 ns per call (depending on
whether the kernel has to split the target VMA and do a TLB shootdown
across cores). For latency-sensitive code that toggles page protection
on a hot path -- a JIT toggling W^X on code pages, a debug allocator
poisoning freed memory, a crash handler guarding its dump buffer -- that
syscall cost dominates.

Intel Memory Protection Keys (MPK) solves this by tagging page-table
entries with a 2-bit key (0-3) at `mprotect` time, then letting
userspace change the *permission* for all pages tagged with a given key
by writing a single 32-bit register (PKRU) with the `WRPKRU` instruction.
No syscall. ~20 cycles. The cost model flips from "syscall per protection
change" to "syscall once to tag, then instruction-per-toggle."

`libpkey` wraps the three MPK syscalls (`pkey_alloc`, `pkey_free`,
`pkey_mprotect`) and exposes the `WRPKRU`/`RDPKRU` instructions as
C functions, with graceful degradation when MPK is unavailable.

## Architecture

```
                  pkey_available()  ---- CPUID 7:0:ECX[4] (OSPKE)
                                        |
              +-------------------------+---------------------+
              | pkey_alloc / pkey_free / pkey_mprotect        |  syscall (one-time)
              +-------------------------+---------------------+
                                        |
              +-------------------------+---------------------+
              | pkey_set_access / pkey_get_access             |  WRPKRU / RDPKRU (hot path)
              | pkey_allow / pkey_deny / pkey_readonly        |  ~20 cycles, no syscall
              +-------------------------+---------------------+
                                        |
                            src/pkey_x86_64.asm
                              pkey_rdpkru  (0F 01 EE)
                              pkey_wrpkru  (0F 01 EF)
                              pkey_cpuid   (CPUID leaf/subleaf)
```

The split is deliberate: the syscall wrappers are called once at setup
time (allocate a key, tag a region), and the PKRU accessors are called
on the hot path (toggle protection). The hot path touches no syscall,
no kernel state, and no shared memory -- only the per-thread PKRU
register.

## OSPKE vs PKU

The task brief specified `pkey_available()` as checking "CPUID 0x7:ECX[3]
(OSPKE -- OS has enabled PKRU)." The bit number and the parenthetical
disagree: CPUID 7:0:ECX[3] is **PKU** (CPU support); CPUID 7:0:ECX[4]
is **OSPKE** (OS has set `CR4.PKE`). This module checks ECX[4] (OSPKE)
because that is the bit that gates userspace usability:

- **PKU=0** (CPU has no MPK support): `WRPKRU`/`RDPKRU` are not
  implemented; they `#UD` (raise `SIGILL`).
- **PKU=1, OSPKE=0** (CPU supports MPK but the OS has not enabled it):
  `WRPKRU`/`RDPKRU` still `#UD` (verified on the build sandbox: an
  Intel Xeon with `pku` in `/proc/cpuinfo` flags but no `ospke`;
  `RDPKRU` raised `SIGILL`). `pkey_alloc(2)` returns `ENOSYS` or
  `EINVAL`.
- **PKU=1, OSPKE=1** (full MPK): instructions execute, PKRU is checked
  on every memory access, `pkey_alloc(2)` works.

Checking PKU ([3]) would report "available" on the build sandbox (PKU=1)
and then `#UD` on the first `WRPKRU` call. Checking OSPKE ([4]) correctly
reports "unavailable" and all functions return `PKEY_ERR_UNSUPPORTED`.
The parenthetical in the brief ("OSPKE -- OS has enabled PKRU") is the
intent; this module implements the intent.

The Linux kernel sets `CR4.PKE` (and thus OSPKE) unconditionally at boot
when `CONFIG_ARCH_HAS_PKEYS=y` and the CPU supports PKU. So in practice
OSPKE=1 on any standard distro kernel on Skylake-X+ hardware. The build
sandbox runs a custom Alibaba Cloud kernel (5.10.134-013.8.3.kangaroo)
that appears to either lack `CONFIG_ARCH_HAS_PKEYS` or not set `CR4.PKE`,
so OSPKE=0 despite PKU=1 -- which is exactly the case graceful degradation
is designed for.

## glibc name clash

glibc 2.27+ declares `pkey_alloc(unsigned int flags, unsigned int
access_rights)`, `pkey_free(int pkey)`, and `pkey_mprotect(void *addr,
size_t len, int prot, int pkey)` in `<sys/mman.h>` under `__USE_GNU`
(i.e. when `_GNU_SOURCE` is defined). These have the same names as this
module's public API but different signatures (e.g. glibc's `pkey_alloc`
takes two arguments; ours takes none).

If a translation unit defines `_GNU_SOURCE` and includes both
`<sys/mman.h>` and `pkey.h`, gcc emits a conflicting-declaration error.
This affects the test and bench files, which need `<sys/mman.h>` for
`mmap`/`PROT_*`/`MAP_ANONYMOUS` and `pkey.h` for the API.

Resolution: `pkey.c` uses `_GNU_SOURCE` (as the task brief instructs)
and calls `syscall(SYS_pkey_*, ...)` directly, but does **not** include
`<sys/mman.h>` -- it forwards the caller's `prot` argument opaquely to
the syscall, so it needs none of the `PROT_*` macros internally. The
test and bench files use `_DEFAULT_SOURCE` instead of `_GNU_SOURCE`.
`_DEFAULT_SOURCE` exposes `MAP_ANONYMOUS` (gated on `__USE_MISC`) and
`syscall()` (also `__USE_MISC`) and `clock_gettime` (via `_POSIX_C_SOURCE`)
without exposing glibc's `pkey_*` wrappers (gated on `__USE_GNU`). No
conflict.

The fallback `SYS_pkey_*` `#define`s in `pkey.c` cover older glibc
headers (< 2.27) that may not define the syscall numbers. The numbers
are stable on x86_64 since Linux 4.6 (`arch/x86/entry/syscalls/
syscall_64.tbl`: 329 = `pkey_mprotect`, 330 = `pkey_alloc`, 331 =
`pkey_free`).

## WRPKRU / RDPKRU instruction encoding

Both instructions are 3-byte opcodes:

| Instruction | Encoding | Operation |
|---|---|---|
| `RDPKRU` | `0F 01 EE` | `EAX = PKRU` |
| `WRPKRU` | `0F 01 EF` | `PKRU = EAX` |

The Intel SDM requires `ECX = 0` and `EDX = 0` for both (future-extension
bits; non-zero raises `#GP`). The asm helpers zero these explicitly
before the instruction. NASM 2.13+ recognizes the `rdpkru`/`wrpkru`
mnemonics; the toolkit standardizes on NASM 2.16.01, which emits the
correct byte sequences (verified via `objdump` on the assembled object).

Both instructions are ~20 cycles on Skylake-X and later. They are
non-privileged (usable from userspace) but `#UD` when `CR4.PKE = 0`.

`WRPKRU` is NOT serialized and NOT fenced. The SDM specifies that PKRU
effects on subsequent memory accesses are observed in program order on
x86_64 (TSO), so no explicit fence is needed between `WRPKRU` and the
next memory access that should be affected. A `pkey_deny(pk)` immediately
before a store to a page tagged with `pk` will fault on that store.

## PKRU bit layout and the read-modify-write in pkey_set_access

PKRU is 32 bits, 2 bits per key:

```
bits [1:0]   = key 0: bit 0 = AD, bit 1 = WD
bits [3:2]   = key 1: bit 2 = AD, bit 3 = WD
bits [5:4]   = key 2: bit 4 = AD, bit 5 = WD
bits [7:6]   = key 3: bit 6 = AD, bit 7 = WD
bits [31:8]  = reserved (must be 0)
```

`pkey_set_access(pkey, ad, wd)` does a read-modify-write:

```c
pkru = pkey_rdpkru();                    /* read current PKRU */
pkru &= ~(3u << (pkey * 2));             /* clear this key's 2 bits */
pkru |= (ad ? 1u : 0u) << (pkey * 2);    /* set AD */
pkru |= (wd ? 2u : 0u) << (pkey * 2);    /* set WD */
pkey_wrpkru(pkru);                       /* write back */
```

This preserves the bits for other keys, so a caller managing multiple
keys does not clobber them. The cost is one `RDPKRU` + one `WRPKRU`
(~40 cycles total). A future optimization for single-key callers could
skip the read and write a constant PKRU, but the current API does not
expose "set absolute PKRU" because that would be unsafe across multiple
keys.

`pkey_get_access(pkey)` reads PKRU and returns:
- 0 (full access) if AD=0 and WD=0
- 1 (read-only) if AD=0 and WD=1
- 2 (no access) if AD=1 (WD is irrelevant; AD dominates)

The convenience wrappers map to `pkey_set_access` with canonical bit
patterns:

| Wrapper | AD | WD | Effect |
|---|---|---|---|
| `pkey_allow(pk)` | 0 | 0 | full access |
| `pkey_deny(pk)` | 1 | 0 | no access |
| `pkey_readonly(pk)` | 0 | 1 | read but not write |

`pkey_deny` clears WD even though AD dominates, so the state is canonical
(`pkey_get_access` returns 2 either way, but reading PKRU directly shows
a clean `10` pair rather than `11`).

## Per-thread PKRU

PKRU is a per-thread register. The kernel saves and restores it across
context switches (in `switch_to()`) and on kernel entry (the kernel
zeroes the AD/WD bits when entering the kernel so userspace restrictions
cannot block kernel accesses to user pages). Consequences:

- `pkey_set_access` on thread A does not affect thread B. If thread A
  denies a key and thread B accesses a page tagged with that key, thread
  B's access uses thread B's PKRU (likely full access) and succeeds.
- A thread that wants to deny access across all threads must call
  `pkey_set_access` on each thread (e.g. via a signal broadcast or by
  having each thread check a flag at a safe point).
- `fork()` copies the parent's PKRU into the child. The child's
  `pkey_alloc` calls return independent keys.

This per-thread property is a feature for some use cases (per-thread
 sandboxes, debug-only poisoning) and a surprise for others (cross-thread
 guard pages). `libsva`'s guard-page model is cross-thread by default
 (it uses `PROT_NONE`, which is a VMA property visible to all threads);
 callers switching to `libpkey` for hot-path toggles must account for the
 per-thread semantics. This is documented in the integration patterns
 below, not enforced by the library.

## Graceful degradation

When `pkey_available()` returns 0 (no OSPKE), all functions return
`PKEY_ERR_UNSUPPORTED` without invoking any syscall or instruction.
This is checked at the top of every public function. The no-syscall
accessors (`pkey_set_access`, `pkey_get_access`, `pkey_allow`,
`pkey_deny`, `pkey_readonly`) do NOT execute `WRPKRU`/`RDPKRU` in this
case -- those instructions `#UD` (raise `SIGILL`) when `CR4.PKE = 0`,
which would crash the process. Callers must check `pkey_available()`
once at startup and branch to a `mprotect`-based fallback.

This mirrors `libpmu`'s graceful-degradation pattern (dummy ctx on
`PMU_ERR_PERM`), but simpler: `libpkey` has no state to allocate, so
there is no "dummy pkey" -- just a uniform `PKEY_ERR_UNSUPPORTED` return.
The caller's fallback path is their own (typically `mprotect`), not
wired by the library.

Tests and benches SKIP cleanly (exit 0) when `pkey_available()` returns
0, matching the toolkit convention in EoSD-SPEC.md §7.

## Build and test in this sandbox

The build sandbox is an Intel Xeon with `pku` in `/proc/cpuinfo` flags
but no `ospke` (CPUID 7:0:ECX[3] = 1, ECX[4] = 0). The kernel is
5.10.134-013.8.3.kangaroo.al8 (Alibaba Cloud), which apparently does not
set `CR4.PKE`. `pkey_alloc(0, 0)` returns `-1` with `errno = ENOSYS`.

Consequence: `pkey_available()` returns 0 and all three test binaries
SKIP:

```
=== tests/test_basic ===
SKIP test_basic: MPK not available (CPUID 7:0:ECX[4] OSPKE not set)
=== tests/test_protect ===
SKIP test_protect: MPK not available (OSPKE not set)
=== tests/test_perf ===
SKIP test_perf: MPK not available (OSPKE not set)
```

`bench/bench_wrpkru` also SKIPs. `bench/bench_mprotect` runs unconditionally
(it has no MPK dependency) and measures the `mprotect` baseline for
comparison:

```
=== bench/bench_mprotect ===
bench_mprotect: real prot change (NONE<->RW): 264.17 ns/op
bench_mprotect: same-prot mprotect (RW->RW)  :  99.26 ns/op
```

On an MPK-capable host (OSPKE=1), `bench_wrpkru` is expected to report
~6-8 ns/op for the bare `WRPKRU` instruction and ~12-20 ns/op for the
`pkey_allow`/`pkey_deny` wrappers (RDPKRU + WRPKRU pair). That is a
~20-100x speedup over `mprotect` for single-page protection toggles.
The ratio depends on whether `mprotect` has to split the target VMA
(higher ratio) or just flip prot bits on an existing VMA (lower ratio,
~20-30x).

## Integration patterns (documented, not linked)

Per EoSD-SPEC.md §3, these are cross-module usage patterns documented
here, NOT link-time dependencies. `libpkey` does not link any other EoSD
module, and no other EoSD module links `libpkey`. The patterns describe
how a consumer could wire `libpkey` into their own application alongside
other EoSD modules.

### Pattern 1: libpkey + libdetour -- fast W^X toggle for patch pages

`libdetour` installs inline hooks by writing into the target function's
code pages. On a W^X-enforcing kernel (or a hardened caller), the patch
path must `mprotect` the page to `PROT_READ|PROT_WRITE`, write the hook,
then `mprotect` back to `PROT_READ|PROT_EXEC`. Two syscalls per patch
installation; ~200-1500 ns each.

With `libpkey`, the caller tags the code page with a pkey at load time
(one `pkey_mprotect`), then toggles write permission with `pkey_allow`/
`pkey_readonly` (no syscall, ~20 cycles) around each patch. The ~24 ms
enable/disable latency floor mentioned in the v0.4 planning notes (from
`mprotect` + TLB shootdown across cores) drops to ~40 cycles.

Caveat: PKRU is per-thread. If the patcher thread and the executor
thread are different, the patcher's `pkey_readonly` does not block the
executor from writing. For a single-threaded patcher + multi-threaded
executors, `pkey_readonly` on the patcher prevents accidental writes
during steady state, and `pkey_allow` is called only during the patch
window. This is weaker than `mprotect` (which is process-wide) but
sufficient for the "protect against accidental writes" use case.

### Pattern 2: libpkey + libcrash -- guard the dump buffer

`libcrash` writes a minidump into a caller-provided buffer from an
async-signal handler. The buffer is large (typically 4-64 KiB) and
lives in the process's address space, where it can be corrupted by a
use-after-free or wild write before the crash. Tagging the buffer with
a pkey and `pkey_deny`-ing it except during the dump write would make
the buffer inaccessible to normal code, reducing the chance of
pre-crash corruption.

Caveat: `libcrash` is async-signal-safe and cannot call `pkey_alloc` /
`pkey_mprotect` (syscalls, not async-signal-safe). The caller must
allocate the key and tag the buffer at init time (before installing the
crash handler), then the handler calls `pkey_allow` (no syscall, just
`WRPKRU`) before writing the dump and `pkey_deny` after. `WRPKRU` is
async-signal-safe (it is a single user-mode instruction with no syscall,
no malloc, no lock). This composes naturally with `libcrash`'s
pre-allocated-state model.

### Pattern 3: libpkey + libsva -- hot-path guard pages

`libsva` installs `PROT_NONE` guard pages around mmap'd regions to
catch overflow. The guard is a VMA property (process-wide, visible to
all threads). For a use case that needs to *temporarily* allow access
to the guard region (e.g. a stack-growth protocol, or a debug probe),
`libsva` would `mprotect` the guard to `PROT_READ|PROT_WRITE` and back
-- two syscalls. With `libpkey`, the caller tags the guard region with
a pkey at `sva_map_guarded` time and toggles access via `pkey_allow`/
`pkey_deny`.

Caveat: same per-thread caveat as Pattern 1. `libsva`'s guard pages are
cross-thread; `libpkey`'s PKRU is per-thread. This pattern is only safe
when the thread that needs to touch the guard is the same thread that
calls `pkey_allow`. For cross-thread guard access, `mprotect` remains
the right tool.

### Pattern 4: libpkey + libbarrage -- poison freed arena memory

`libbarrage` is a bump allocator; freed memory is not returned until a
batch reset. A debug build could tag the arena with a pkey and
`pkey_deny` freed regions (by `pkey_mprotect`-ing the freed range to a
"poison" key) to catch use-after-free in ~20 cycles rather than waiting
for the batch reset. This is a debug-only pattern (the per-thread PKRU
semantics make it incomplete for multi-threaded arenas) and is not
appropriate for production.

## Why raw syscall(SYS_pkey_*) and not glibc wrappers

glibc 2.27+ provides `pkey_alloc`/`pkey_free`/`pkey_mprotect` in
`<sys/mman.h>`. This module's public API uses the same names (per the
task spec) but with different signatures (e.g. `pkey_alloc(void)` vs
glibc's `pkey_alloc(unsigned int, unsigned int)`). Calling the glibc
wrappers from inside `pkey.c` would require `#undef`-ing the macros or
using a different internal name, both of which are fragile. Raw
`syscall(SYS_pkey_*, ...)` is one line, works on any glibc version (the
syscall numbers are in `<sys/syscall.h>` on kernels 4.6+), and avoids
the name clash entirely.

The tradeoff is that `pkey.c` does not benefit from glibc's
async-signal-safety annotations on the wrappers. This is fine: `pkey_alloc`,
`pkey_free`, and `pkey_mprotect` are NOT async-signal-safe (they are
syscalls), and the library documents this. The no-syscall accessors
(`pkey_set_access` etc.) ARE async-signal-safe (they are single
instructions), which matters for Pattern 2 above.

## Security notes

- **PKRU is per-thread, not process-wide.** A `pkey_deny` on one thread
  does not block another thread from accessing the same page. MPK is
  not a security boundary against malicious threads in the same process
  (a malicious thread can `WRPKRU` to clear all restrictions). It is a
  hardening and debugging tool: it catches accidental accesses and
  raises the cost of exploitation (an attacker who hijacks control flow
  must also know to `WRPKRU` before touching the protected region).
- **WRPKRU is non-privileged.** Any userspace code can call it. There
  is no way to "lock" a key's permissions in userspace; the kernel
  could enforce this (via `pkey_alloc` flags) but Linux does not expose
  that in v0.1. A future `pkey_alloc_locked` variant could call
  `pkey_alloc(0, PKEY_DISABLE_ACCESS | PKEY_DISABLE_WRITE)` to
  pre-disable and then never `pkey_allow` -- but that prevents the
  legitimate owner from writing too, so it's only useful for pure-read
  regions.
- **No SIGSEGV handler installed.** A PKRU violation raises `SIGSEGV`
  with `si_code = SEGV_PKUERR` (Linux 4.18+; older kernels report
  `SEGV_ACCERR`). `libpkey` does not install a handler; composability
  with `libcrash` is Pattern 2 above.
- **No information leak.** `RDPKRU` reads only the calling thread's
  PKRU; it cannot read another thread's PKRU or any kernel state. The
  register is zero-initialized by the kernel at `execve` and at
  `clone(CLONE_THREAD)`.

## Non-goals (v0.1)

- No Windows/macOS support. MPK is Linux/x86_64 only in this toolkit.
- No ARM64 support (MTE is a different mechanism, out of scope).
- No pkey_alloc flags passthrough (always `(0, 0)`; post-alloc disabling
  via `pkey_set_access`).
- No multi-key atomic PKRU update.
- No PKRU fault handler integration (documented as a `libcrash` pattern,
  not linked).
- No "locked" key variant (key whose permissions cannot be re-enabled in
  userspace). Would require kernel support beyond what `pkey_alloc`
  flags expose.
