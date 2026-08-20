# libcrash: Design Notes (v0.3, shipped)

## Problem

Capture enough machine state at the moment of a crash to debug it later
(game crash reporters, embedded devices writing to flash before reboot,
test runners that want to survive a segfault in one test), without
doing anything during the signal handler that could itself crash or
deadlock. Async-signal-safety is not optional here; it is the entire
point.

## Why caller-provided buffer + fd, not internal allocation

`malloc` is not guaranteed async-signal-safe (glibc's malloc can
deadlock if the crash happens while the allocator's internal lock is
already held by the crashing thread). `mmap` and `munmap` are also not
on the POSIX async-signal-safe list per `signal-safety(7)`, so libcrash
does not call them from the handler -- the buffer is caller-pre-allocated
at install time. `sigaltstack(2)` (used to set up the alternate signal
stack) is also not on the POSIX async-signal-safe list, but libcrash
calls it only at install time (from `crash_install`), never from the
handler itself. Requiring the caller to pass a pre-allocated buffer and
pre-opened fd at `crash_install()` time, long before any crash happens,
removes every allocation and every non-async-signal-safe call from the
actual signal-handling path. This is the single most important design
constraint in this entire module and overrides convenience
considerations everywhere else.

## Why the classic 5-signal default set

SIGSEGV (bad memory access), SIGABRT (assert failures, `abort()`),
SIGFPE (divide by zero, some FP traps), SIGILL (illegal instruction,
often stack corruption jumping to garbage), SIGBUS (misaligned/invalid
memory access on some platforms). This is the standard "something went
fatally wrong at the machine level" set used by essentially every crash
reporter. Caller can override with a custom signal list.

## v0.1 custom binary format vs. v0.2 ELF core

Writing an actual ELF core file compatible with GDB/WinDbg is
significantly more implementation work (correct program headers, note
sections, memory region enumeration) and most of that work is
redundant with what the OS's own core dump facility already does when
enabled. v0.1 shipped the custom `crash_dump_t` binary format only;
v0.2 adds an optional minimal ELF core path (`CRASH_FORMAT_ELF`) that
writes a 4628-byte core file (Elf64_Ehdr + PT_NOTE + PT_LOAD +
`NT_PRSTATUS` + 4 KiB stack window) readable by `gdb` and `eu-stack`.
The v0.2 ELF core carries the GP register set and a 4 KiB stack
window; it does not carry `NT_X86_XSTATE` (YMM/ZMM) or `NT_AUXV`. The
custom format remains the default and is the only path that captures
YMM. If the caller's buffer is too small to hold the ELF core (it
never is, given the 8 KiB minimum), the handler falls back to the
custom format so a dump is still produced.

## The asm boundary

Two pieces in `crash_x86_64.asm`, written in NASM (elf64):

1. `crash_rdtsc` -- read the timestamp counter (`rdtsc` + shift + or).
   Called early in the handler for a maximally accurate crash-time
   stamp.
2. `crash_copy_4k` -- copy the 4 KiB stack snapshot from `[RSP, RSP+4096)`
   into the dump buffer using `vmovdqu` YMM loads/stores. Avoids
   `memcpy`, which is not on the POSIX async-signal-safe list.

v0.1 also exposed a `crash_capture_ymm` asm helper that read live YMM
registers via `vmovdqu`. v0.2 removed it; see "YMM capture" below.

`vzeroupper` is emitted before each `ret` so callers that do not
otherwise use AVX do not pay an SSE-to-AVX transition penalty on the
next XMM-touching instruction. In the crash path this is moot (we are
about to `_exit`), but the same objects link into test code where the
penalty matters.

## YMM capture (v0.2 redesign)

v0.1 captured YMM0..YMM15 by reading live registers via the
`crash_capture_ymm` NASM helper (16 `vmovdqu ymmN` stores to a caller
buffer). Two independent bugs made the captured values unreliable:

1. **The kernel saves the user's AVX state into the XSAVE area in
   `ucontext.uc_mcontext.fpregs` before delivering the signal and does
   not preserve the live YMM registers across kernel entry.** Reading
   live YMM from the handler returns whatever the kernel left there
   (often zero), not the user's value. Reproduced by loading a known
   pattern into YMM0..YMM3, crashing via NULL deref, and inspecting the
   dump's `ymm` field: the captured values were garbage (mostly zero)
   rather than the loaded pattern.
2. **At `-O2` the compiler vectorizes the 16 GPR stores in
   `crash_write_dump` using legacy SSE `movdqu`/`movups` on XMM0..XMM9,
   clobbering the low 128 bits of YMM0..YMM9 before any live-register
   capture could observe them.** The v0.1 source comment ("the C below
   does not touch vector state") was false at `-O2`. This bug is moot
   once (1) is fixed (we no longer read live registers), but is
   retained as a defensive note: the v0.2 handler is marked
   `__attribute__((target("general-regs-only")))` so future code added
   to the handler cannot accidentally re-introduce vector clobbering.

The v0.2 YMM capture is pure C and parses the XSAVE area directly:

- The FXSAVE area (offset 0..511) is always present. XMM0..XMM15 live
  at offset 160, 16 B each -- the low 128 bits of each YMM. These are
  always copied when `fpregs` is non-NULL.
- If `uc_flags & UC_FP_XSTATE` is set, the XSAVE header follows at
  offset 512. `xstate_bv` (first u64) bit 2 (`XSTATE_YMM`) means the
  YMMH area is populated.
- If `XSTATE_YMM` is set, YMMH0..YMMH15 (the high 128 bits of each
  YMM) live at offset 576, 16 B each. They are copied to the high 128
  bits of each `ymm[i]` slot in the dump.
- Otherwise the high 128 bits are written zero.

This is pure memory-to-memory traffic, so compiler vectorization of
the copy loops cannot corrupt the source data (the source is the
kernel-written XSAVE area in memory, not live registers).

The offsets (160 / 512 / 576) are raw integers rather than `offsetof`
on a struct, because the kernel-guaranteed layout is byte-stable across
glibc versions while glibc's `struct _xstate` is not part of any stable
ABI.

## Post-dump behavior

- `CRASH_AFTER_RERAISE` (default): restore `SIG_DFL` for the signal and
  re-raise via `kill(getpid(), sig)`. Preserves normal OS core-dump
  behavior if the user has that enabled. The two are not mutually
  exclusive.
- `CRASH_AFTER_EXIT`: `_exit(1)` immediately after writing the dump.
  For the embedded-device use case where waiting for the OS is not
  desired.
- `CRASH_AFTER_FORK` (v0.2): `_Fork()` an out-of-process dump writer.
  The child inherits a COW copy of the address space (including
  `g_buf` and the parent's `ucontext`), writes the dump, and `_exit(0)`s.
  The parent `waitpid()`s for the child before re-raising, so the dump
  is on disk before the parent terminates. If the dump writer itself
  faults, the re-entered handler in the child sees `handling == 1`
  (inherited via COW) and `_exit(255)`, leaving the file with whatever
  was written before the fault.

  `waitpid` is not on the POSIX async-signal-safe list. libcrash uses
  it on the `CRASH_AFTER_FORK` parent path after the child has
  `_exit(0)`'d. The child's only libc calls are `write` and `_exit`,
  neither of which holds locks that `waitpid` would deadlock on. This
  is the single intentional deviation from strict POSIX
  async-signal-safety, mandated by the v0.2 spec.

  The `waitpid` loop retries only on `EINTR`; any other error (e.g.
  `ECHILD` if a `SIGCHLD` handler already reaped the child, `EINVAL`)
  breaks out and falls through to `crash_reraise` so the parent cannot
  hang forever. The v0.1-era code did `while (waitpid(...) < 0) { }`
  with no `errno` check and could spin forever on `ECHILD`/`EINVAL`.

## Alternate signal stack sizing

`CRASH_ALTSTACK_SIZE` is 64 KiB in v0.2 (was 32 KiB in v0.1). The
larger stack gives the handler room for: deep nesting (two recursive
handler entries before the `_Atomic` guard kicks in), the
`struct elf_prstatus` (~336 B) and `Elf64_Ehdr`+Phdrs (~176 B) used as
stack-local scaffolding in the ELF path if a future revision chooses to
stack-allocate them, and the getrusage-style frame state. Allocated at
install time via `malloc`, never in the handler.

## Buffer sizing

`crash_min_buffer_size()` returns `8192 + CRASH_MAX_USER_BLOBS *
(CRASH_MAX_BLOB_KEY + 8 + CRASH_MAX_BLOB_SIZE)` = 10560 bytes in v0.3.
The 8 KiB base covers: 4096 for the stack snapshot window, 16*8=128 for
the GPR array, ~512 for vector register storage plus header/misc
fields, padded to 8 KiB for headroom. The v0.2 ELF core needs 4628 B;
the v0.1 custom format needs `sizeof(crash_dump_t)` (~4.7 KiB); both
fit inside 8 KiB. The v0.3 remainder (2368 B) covers the worst-case
user-blob section: 4-byte count + 8 blobs * (32-byte key + 8-byte size
+ 256-byte data). The function is exposed (not a macro) so a later
version can grow the dump struct and bump the number without breaking
callers.

## Recursive crash handling

A static `_Atomic int handling` flag is set to 1 at the top of the
signal handler. If a second signal arrives while `handling == 1` (a
crash inside the crash handler itself, e.g. the stack snapshot logic
touching bad memory), the re-entered handler skips the dump-writing
logic entirely and calls `_exit(255)` immediately. This does not
attempt to capture or diagnose the recursive crash, only to guarantee
the process does not hang or infinite-loop retrying a handler that
itself keeps faulting.

`SA_RESETHAND` (set by `crash_install`) makes the kernel reset the
triggering signal's disposition to `SIG_DFL` on entry, so a same-signal
recursive fault terminates via the OS default directly (the
re-entry guard never even runs). `SA_NODEFER` keeps the triggering
signal unblocked during the handler, so a cross-signal recursive fault
(e.g. SIGSEGV while writing the dump for a SIGABRT) can re-enter the
handler and hit the guard. `SA_NODEFER + SA_RESETHAND` together give
the guard full coverage for cross-signal recursive faults; the
same-signal case is handled by `SA_RESETHAND + SIG_DFL`.

## ucontext NULL-checks

Both the custom-format path and the ELF-format path NULL-check `uc`
before dereferencing `uc->uc_mcontext.gregs`. The kernel never delivers
a `SA_SIGINFO` handler without a `ucontext`, but the check is defensive
and matches the ELF path's behavior; if `uc` is NULL, the register
snapshot fields are zeroed and the consumer can detect the absence via
`rip == 0` / `rsp == 0`.

## User blobs (v0.3)

### Motivation

Register and stack snapshots tell you *where* the crash happened, but
not *what* the application was doing. In a real-time trading engine or
game server, the most valuable debug info is application state: "which
order ID was I processing?", "which player was in the zone?", "what
was the last confirmed sequence number?". The user blob API lets the
caller stash up to 8 small blobs (max 256 B each) of caller-chosen
data in the crash dump, registered before the crash and read at crash
time.

### Design constraints

1. **No allocation on the crash path.** The blob array is a static
   global (`g_user_blobs[CRASH_MAX_USER_BLOBS]`); the blob section is
   built directly in the caller's pre-allocated `g_buf`. No `malloc`,
   no `memcpy`, no `memset` appears on the crash path.

2. **No lock on the crash path.** The handler reads `g_user_blobs`
   lock-free. A spinlock (`g_blob_lock`, `_Atomic int` with
   `atomic_compare_exchange_strong`) serializes `crash_set_user_blob`
   and `crash_clear_user_blob` against each other, but the handler
   never acquires it. A concurrent `set`/`clear` may produce a garbled
   blob entry (torn key, stale data pointer); this is acceptable per
   the spec -- a garbled entry in a crash dump is strictly better than
   a deadlock.

3. **Publication ordering.** The setter writes `data` and `size`
   before writing the key; the key write is what publishes the slot
   (the handler checks `key[0] != '\0'`). On x86-64 TSO, stores are
   not reordered with stores, so `data`/`size` are visible before the
   key. The clearer reverses this: zeros `key[0]` first (unpublishing),
   then clears `data`/`size`. The handler additionally skips slots
   whose `data` is `NULL` or whose `size` is out of range, catching
   partially-updated slots where the publication ordering has not yet
   been observed.

4. **Byte-by-byte copy.** `crash_build_blob_section` copies each
   blob's key and data via explicit load/store loops, not `memcpy`
   (which is not on the POSIX async-signal-safe list). The compiler
   may vectorize these loops; that is fine -- the source data is in
   memory (the `g_user_blobs` array and the caller's `data` buffer),
   not in live YMM registers, so vectorization cannot corrupt the
   captured values (same reasoning as `crash_capture_ymm_from_fpstate`).

5. **Single write(2).** The blob section is appended in `g_buf` right
   after the main dump (custom `crash_dump_t` or ELF core), and the
   whole buffer is written with a single `write(2)`. This required
   refactoring `crash_elf_write` into `crash_elf_build` (build only,
   no write) so the caller can append the blob section before issuing
   the write.

### Dump format change

The dump file layout is now:

```
[main dump: crash_dump_t (custom) or ELF core (ELF)]
[uint32_t blob_count, little-endian]
[for each active blob: key[32] + uint64_t size, LE + data[size]]
```

The blob section is always present (count may be 0). For the ELF
format, the blob section is appended after the 4628-byte ELF core;
`gdb` ignores trailing data beyond the program headers, so the blob
section does not interfere with ELF-core parsing.

### `crash_uninstall` clears blobs

`crash_uninstall` zeroes the entire `g_user_blobs` array under the
spinlock (using `memset`, which is safe at uninstall time -- not on the
crash path). This ensures a `crash_install` → `crash_uninstall` →
`crash_install` cycle starts with a clean blob array.

## Async-signal-safety audit

The crash path (signal entry → dump written → process terminates)
touches only the following primitives. Each is either listed in
`signal-safety(7)` or is explicitly justified below.

**POSIX async-signal-safe libc calls:**

- `write(2)` -- single best-effort `write(g_fd, g_buf, ...)` for the
  whole dump (custom header or ELF core + blob section).
- `sigaction(2)`, `sigemptyset` -- `crash_reraise` restores `SIG_DFL`
  for the triggering signal before re-raising.
- `kill(2)`, `getpid` -- `crash_reraise` re-raises via
  `kill(getpid(), sig)` (`raise(3)` is not async-signal-safe).
- `_exit(2)` -- terminal exit on `CRASH_AFTER_EXIT`, on re-entry guard
  firing (`_exit(255)`), and as a last resort if re-raise fails to
  terminate (`_exit(2)`).
- `getppid`, `getpgrp`, `getsid` -- used by `crash_elf_build` to
  populate `elf_prstatus.pr_ppid`/`pr_pgrp`/`pr_sid`. All listed in
  `signal-safety(7)`.
- `_Fork` (POSIX.1-2024 / Issue 8; glibc 2.36+) -- `CRASH_AFTER_FORK`
  spawns the out-of-process dump writer.

**Intentional deviation:**

- `waitpid(2)` -- NOT on the POSIX async-signal-safe list. Used only on
  the `CRASH_AFTER_FORK` parent path, after the dump-writing child has
  `_exit(0)`'d. glibc's `waitpid` is a thin syscall wrapper with no
  libc locks; the child's only libc calls are `write` and `_exit`,
  neither of which holds locks that `waitpid` would deadlock on. The
  loop retries only on `EINTR`; `ECHILD` (a `SIGCHLD` handler already
  reaped the child) and `EINVAL` break out and fall through to
  `crash_reraise` so the parent cannot hang forever. This is the
  single intentional deviation from strict POSIX async-signal-safety,
  mandated by the v0.2 spec; `test_waitpid_echild` exercises the
  `ECHILD` path.

**Lock-free atomics:**

- `atomic_compare_exchange_strong` on `_Atomic int`
  (`ATOMIC_INT_LOCK_FREE == 2` on x86-64) -- the re-entry guard
  (`g_handling`) and the blob spinlock (`g_blob_lock`). The blob
  spinlock is never acquired on the crash path (the handler reads
  `g_user_blobs` lock-free).

**ASM leaf helpers (`crash_x86_64.asm`):**

- `crash_rdtsc` -- `rdtsc` + shift + or, returns TSC in `rax`.
- `crash_copy_4k` -- 128 `vmovdqu` YMM loads/stores, copies the 4 KiB
  stack snapshot from `[RSP, RSP+4096)` into `g_buf`. Used in place of
  `memcpy` (not async-signal-safe).

**Internal C functions (audited, no libc):**

- `crash_capture_ymm_from_fpstate` -- pure memory-to-memory loads/stores
  parsing the XSAVE area in `uc->uc_mcontext.fpregs`. No libc calls.
- `crash_elf_build` -- explicit field stores into the caller's buffer;
  only libc calls are `getpid`/`getppid`/`getpgrp`/`getsid` (audited
  above). No `malloc`, `memcpy`, `memset`, `printf`.
- `crash_build_blob_section` -- explicit byte load/store loops copying
  blob key/size/data into `g_buf`. No `memcpy`. Compiler vectorization
  is safe (memory-to-memory, no live YMM reads).

**Compiler guarantee:**

- `crash_handler` is declared
  `__attribute__((target("general-regs-only")))`, forbidding the
  compiler from emitting SSE/AVX instructions in the handler or its
  inlined callees. v0.2 reads YMM out of the XSAVE area (no
  live-register reads), so compiler vectorization can no longer corrupt
  captured values; the attribute is retained as a belt-and-braces
  guarantee that no compiler-emitted vector op touches the live AVX
  state on the crash path, and so future code added to the handler
  cannot accidentally re-introduce audit finding C-2 (v0.1's
  `-O2`-vectorized SSE stores clobbering YMM0..YMM9 low bits).

**Excluded by construction:**

No `malloc`, `mmap`, `printf`, `memcpy`, `memset`, `pthread_*`,
`syslog`, `fopen`, `raise`, `signal`, `atexit`, or any other
non-async-signal-safe call appears on the crash path. `malloc` and
`memset` are used only at install/uninstall time (`crash_install`
allocates the altstack; `crash_uninstall` frees it and zeroes the blob
array).

## Benchmarks

Measured on the v0.3 codebase, gcc 14.2.0, nasm 2.16.01, glibc 2.x,
x86-64 Linux, `-O2 -std=c11`. Numbers from `bench/bench_dump` and
`bench/bench_install`:

- **`bench_dump`**: ~218 µs/dump via the `CRASH_AFTER_FORK` path. The
  span covers kernel SIGSEGV delivery + child handler entry + `_Fork`
  + grandchild dump-write (custom format, no blobs) + `_exit` + child
  `waitpid` + re-raise. This is the worst-case post-dump action; the
  in-process `CRASH_AFTER_EXIT` path is faster (no `_Fork`/`waitpid`).
- **`bench_install`**: ~2.2 µs per `crash_install` + `crash_uninstall`
  round-trip. Each round-trip is 10 `sigaction` calls (5 default
  signals, save + restore) + 2 `sigaltstack` calls + 1 `malloc` + 1
  `free`. The already-installed early-exit path is ~2 ns.

The dump-write itself (building the `crash_dump_t` header + 4 KB stack
snapshot + blob section in `g_buf` and issuing one `write(2)`) is a
small fraction of the 218 µs; most of the cost is the `_Fork` +
`waitpid` round-trip and kernel signal delivery. Callers who do not
need out-of-process isolation should prefer `CRASH_AFTER_EXIT`
(in-process, no fork) for lower latency.

## Non-goals

- No network/upload logic.
- No attempt to capture or diagnose a recursive crash (inside the
  handler itself); only to fail safely via `_exit(255)`.
- The v0.2 ELF core does not carry `NT_X86_XSTATE` (YMM/ZMM) or
  `NT_AUXV`; gdb gets the GP set and a 4 KiB stack window only. YMM
  capture is a custom-format-path feature.
- No automatic upload/network reporting.
- Not thread-safe to call `crash_install` / `crash_uninstall`
  concurrently with itself; the caller invokes them single-threaded at
  startup/shutdown. (`crash_set_user_blob` / `crash_clear_user_blob`
  ARE thread-safe via the blob spinlock.)
- The v0.3 blob section appended after the ELF core is not an ELF note;
  `gdb` ignores it. A separate parser is needed to extract blobs from
  an ELF-format dump.
