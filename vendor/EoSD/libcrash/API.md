# libcrash: API (v0.3, shipped)

Async-signal-safe crash/minidump handler. The caller pre-allocates all
memory (a fixed-size scratch buffer) and a pre-opened file descriptor
at install time; nothing is allocated during the actual crash. Handles
the classic crash signal set by default (SIGSEGV, SIGABRT, SIGFPE,
SIGILL, SIGBUS). Writes either a custom minimal binary dump (v0.1) or
a minimal ELF core file (v0.2) into the caller's buffer and out the
caller's fd via a single `write(2)`. Post-dump behavior (re-raise,
`_exit`, or out-of-process `_Fork` + write) is configurable per
install.

**The crash path is async-signal-safe end to end.** Between signal
delivery and the dump being written, the handler touches only
async-signal-safe primitives (`write`, `sigaction`, `sigemptyset`,
`kill`, `getpid`, `getppid`, `getpgrp`, `getsid`, `_exit`, `_Fork`),
the asm helpers in `crash_x86_64.asm` (`crash_rdtsc`, `crash_copy_4k`),
and direct loads/stores. No `malloc`, `printf`, `memcpy`, `memset`,
`pthread_*`, `syslog`, or any other non-async-signal-safe call appears
on that path. `waitpid` is the single intentional deviation, used only
on the `CRASH_AFTER_FORK` parent path after the dump-writing child has
`_exit(0)`'d; see DESIGN.md for the rationale.

## Types

```c
typedef enum {
    CRASH_OK                    =  0,
    CRASH_ERR_INVALID           = -1,
    CRASH_ERR_BUF_TOO_SMALL     = -2,
    CRASH_ERR_ALREADY_INSTALLED = -3,
} crash_err_t;

typedef enum {
    CRASH_AFTER_RERAISE,   /* default: restore SIG_DFL, re-raise; OS core dump runs */
    CRASH_AFTER_EXIT,      /* _exit(1) immediately after writing the dump */
    /*
     * v0.2. _Fork() (async-signal-safe), the child writes the dump and
     * _exit(0)s, the parent waitpid()s for the child before re-raising.
     * Isolates dump-writing from the crashing process: if the dump
     * writer itself faults, the child dies but the parent still
     * terminates cleanly via re-raise.
     */
    CRASH_AFTER_FORK,
} crash_after_action_t;

typedef enum {
    CRASH_FORMAT_CUSTOM = 0,   /* v0.1 fixed-size crash_dump_t, single write(2) */
    CRASH_FORMAT_ELF    = 1,   /* v0.2 minimal ELF core, single write(2) */
} crash_format_t;
```

```c
/* Minimum scratch buffer size. v0.3 returns
 *   8192 + CRASH_MAX_USER_BLOBS * (CRASH_MAX_BLOB_KEY + 8 + CRASH_MAX_BLOB_SIZE)
 *   = 8192 + 8 * (32 + 8 + 256) = 10560.
 * Exposed as a function (not a macro) so callers cannot hardcode the
 * number and a later version can grow it without breaking existing
 * callers. The 8 KiB base covers both dump formats (custom ~4.7 KiB,
 * ELF 4628 B); the remainder is the worst-case user-blob section
 * (count + up to CRASH_MAX_USER_BLOBS blobs at CRASH_MAX_BLOB_SIZE). */
size_t crash_min_buffer_size(void);   /* returns 10560 in v0.3 */
```

## Install / lifecycle

```c
/* v0.1 install. Equivalent to crash_install_elf(..., CRASH_FORMAT_CUSTOM).
 * Existing v0.1 callers recompile and link without source changes. */
int crash_install(void *buf, size_t buf_size, int fd,
                   const int *signals, size_t signal_count,
                   crash_after_action_t after_action);

/* v0.2 install with explicit dump format. */
int crash_install_elf(void *buf, size_t buf_size, int fd,
                      const int *signals, size_t signal_count,
                      crash_after_action_t after_action,
                      crash_format_t format);
```

### Arguments

- `buf` / `buf_size`: caller-owned scratch space. Untouched except during
  a crash. Must be `>= crash_min_buffer_size()`. If the caller does not
  want a forked crash-handler subprocess to inherit this buffer, the
  caller must `madvise(buf, buf_size, MADV_DONTFORK)`; libcrash does not
  touch the buffer's madvise state.
- `fd`: pre-opened writable file descriptor; the dump is written via a
  single `write(2)`. Pre-opening avoids path/permission/exhaustion
  failures at crash time, when the handler has no recovery path.
- `signals` / `signal_count`: which signals to install handlers for.
  `NULL` with `signal_count == 0` selects the default set (SIGSEGV,
  SIGABRT, SIGFPE, SIGILL, SIGBUS). At most 32 signals may be passed.
- `after_action`: what the handler does after writing the dump.
- `format` (`crash_install_elf` only): `CRASH_FORMAT_CUSTOM` (v0.1
  fixed-size struct) or `CRASH_FORMAT_ELF` (v0.2 minimal ELF core).

### Install-time side effects

- Allocates a 64 KiB alternate signal stack via `malloc` and installs it
  with `sigaltstack(2)`. v0.1 used 32 KiB (`4 * SIGSTKSZ`); v0.2 doubles
  it for headroom under deep handler nesting. The altstack is allocated
  here, never on the crash path.
- For each requested signal, installs a `sigaction` with:
  - `SA_ONSTACK` (run on the altstack so a blown main stack does not
    lose the dump),
  - `SA_SIGINFO` (3-arg handler with `ucontext_t`),
  - `SA_NODEFER` (do not block the triggering signal during the
    handler; needed for cross-signal recursive-fault handling via the
    `_Atomic handling` re-entry guard),
  - `SA_RESETHAND` (kernel resets the disposition to `SIG_DFL` on
    entry; same-signal recursive faults terminate via the OS default
    rather than looping in our handler).
- Saves the prior signal dispositions and the prior altstack so
  `crash_uninstall` can restore them.

### Returns

- `CRASH_OK` on success.
- `CRASH_ERR_INVALID` if `buf` is `NULL`, `fd` is negative,
  `signal_count > 32`, or `format` is not a recognized value.
- `CRASH_ERR_BUF_TOO_SMALL` if `buf_size < crash_min_buffer_size()`.
- `CRASH_ERR_ALREADY_INSTALLED` if invoked twice without an
  intervening `crash_uninstall`.

### Re-entry guard

A static `_Atomic int handling` flag is set to 1 at the top of the
signal handler. If a second signal arrives while `handling == 1` (a
crash inside the crash handler itself, e.g. the stack snapshot logic
touching bad memory), the re-entered handler skips the dump-writing
logic entirely and calls `_exit(255)` immediately. This does not
attempt to capture or diagnose the recursive crash, only to guarantee
the process does not hang or infinite-loop retrying a handler that
itself keeps faulting. The flag is reset to 0 by `crash_uninstall`.

```c
/* Restore the signal handlers saved by crash_install, restore the old
 * alternate signal stack, free the altstack buffer, and clear installed
 * state so a subsequent crash_install succeeds. Safe to call any time,
 * including from a context where crash_install failed partway.
 *
 * Not async-signal-safe (calls free). Only invoke during normal
 * operation. */
void crash_uninstall(void);
```

## User blobs (v0.3)

The caller can register up to `CRASH_MAX_USER_BLOBS` (8) application-state
blobs that are written to the dump after the main header. Each blob is
keyed by a short string and points at caller-owned memory that must
remain valid until `crash_uninstall`. At crash time the handler reads
each blob's `data` pointer and copies `size` bytes into the dump. No
allocation happens on the crash path -- the blob section is built
directly in the caller's pre-allocated scratch buffer.

```c
#define CRASH_MAX_USER_BLOBS 8
#define CRASH_MAX_BLOB_KEY   32    /* key buffer size, including NUL */
#define CRASH_MAX_BLOB_SIZE  256   /* max bytes per blob */

typedef struct {
    char        key[CRASH_MAX_BLOB_KEY];
    const void *data;   /* read at crash time; must remain valid until uninstall */
    size_t      size;
} crash_user_blob_t;

/* Register a user blob. key is a NUL-terminated string (max 31 chars
 * before the NUL; empty keys are rejected). data points to caller-owned
 * memory that must remain valid until crash_uninstall. At crash time,
 * the handler reads `size` bytes from `data` and writes them to the
 * dump. Returns 0 on success, -1 if key is NULL or empty, data is NULL,
 * size is 0 or exceeds CRASH_MAX_BLOB_SIZE, the key is too long, or all
 * CRASH_MAX_USER_BLOBS slots are full. If a blob with the same key
 * already exists, its data and size are replaced.
 *
 * Thread-safety: safe to call from any thread. The blob array is
 * protected by a spinlock (atomic CAS, async-signal-safe primitive).
 * Not async-signal-safe in the strict POSIX sense (the compiler may
 * emit a memset call for the key-zeroing loop), but safe to call from
 * any non-crash-path thread context. */
int crash_set_user_blob(const char *key, const void *data, size_t size);

/* Remove a user blob by key. Returns 0 on success, -1 if not found or
 * key is NULL/empty/too long.
 *
 * Thread-safety: safe to call from any thread (same spinlock as
 * crash_set_user_blob). */
int crash_clear_user_blob(const char *key);
```

### Async-signal-safety of the blob path

`crash_set_user_blob` and `crash_clear_user_blob` take a spinlock
(`atomic_compare_exchange_strong` on a `_Atomic int`, lock-free on
x86-64). The crash handler itself does **not** take the lock -- it
reads the blob array lock-free. A concurrent `set`/`clear` may produce
a garbled blob entry in the dump (torn key, stale data pointer), which
is acceptable per the design. The handler defensively skips slots
whose `data` is `NULL` or whose `size` is out of range, catching
partially-updated slots. If a stale `data` pointer faults despite the
checks, the kernel re-enters the handler; `SA_RESETHAND` has already
reset the disposition to `SIG_DFL`, so the process terminates via the
OS default (no recursive handler loop).

### Publication ordering

The setter writes `data` and `size` before writing the key (the key
write is what publishes the slot, since the handler checks
`key[0] != '\0'`). On x86-64 TSO, stores are not reordered with
stores, so the `data`/`size` writes are visible before the key write.
The clearer reverses this: zeros `key[0]` first (unpublishing the
slot), then clears `data`/`size`.

## Dump format: CRASH_FORMAT_CUSTOM (v0.1, v0.3 appends blob section)

Fixed-size struct, written via a single `write(fd, ...)`. Layout:

```c
typedef struct {
    uint64_t magic;             /* CRASH_MAGIC ("CRASHDUM" big-endian) */
    uint64_t timestamp_ns;      /* raw TSC from rdtsc; named _ns for forward compat */
    int      signal_number;     /* signal that triggered the handler */
    int      _pad0;             /* keep gpr 8-aligned; not interpreted */
    uint64_t gpr[16];           /* RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8..R15 */
    uint64_t rip;
    uint64_t rsp;               /* same value as gpr[7] */
    uint64_t eflags;
    uint8_t  ymm[16][32];       /* low 256 bits of YMM0..YMM15, low-then-high byte order */
    uint8_t  stack_snapshot[4096];  /* 4 KiB window at [RSP, RSP+4096) */
} crash_dump_t;
```

The `gpr` array layout, in order: RAX, RBX, RCX, RDX, RSI, RDI, RBP,
RSP, R8..R15. RSP also appears as the dedicated `rsp` field for
convenience; the two values are identical.

### YMM capture (v0.2 fix)

The `ymm[16][32]` field is populated from the XSAVE area saved by the
kernel into `ucontext.uc_mcontext.fpregs` before signal delivery, not
from live YMM registers. v0.1 read the live registers via a NASM
helper; the kernel had already saved them into the XSAVE area and left
the live registers holding whatever it chose (often zero), so the
captured values were unreliable. The v0.2 parser:

- Always copies the low 128 bits of each YMM (== XMM, from the FXSAVE
  area at offset 160) when `fpregs` is non-NULL.
- If `uc_flags & UC_FP_XSTATE` is set, reads `xstate_bv` from the
  XSAVE header (offset 512). If bit 2 (`XSTATE_YMM`) is set, copies
  the high 128 bits of each YMM from the YMMH area at offset 576.
- Otherwise the high 128 bits are written zero.

If the crashing process never used AVX (no YMM in `xstate_bv`), the
high 128 bits are zero; this is the kernel saying "no AVX state to
report," not a capture failure.

### Stack snapshot

A 4 KiB window starting at `RSP` is copied via the asm `crash_copy_4k`
helper (128 `vmovdqu` YMM loads/stores; `memcpy` is not
async-signal-safe). The copy is skipped if `RSP` is 0 or above the
47-bit user address ceiling; in that case `stack_snapshot` retains
whatever bytes the caller's buffer held. The consumer can detect this
by checking the `rsp` field. If the copy faults despite the range
check (e.g. `RSP` points to a mapped but unreadable page), the kernel
re-enters the handler; `SA_RESETHAND` has reset the disposition to
`SIG_DFL`, so the process terminates via the OS default.

### User blob section (v0.3)

After the `crash_dump_t` header, the handler appends a user-blob
section in the same buffer and writes the whole thing with a single
`write(2)`. Layout:

```
offset 0:                   crash_dump_t header (sizeof = ~4784 B)
offset sizeof(crash_dump_t): uint32_t blob_count (little-endian)
offset +4:                   for each blob:
                               char     key[32]   (NUL-padded)
                               uint64_t size      (little-endian)
                               uint8_t  data[size]
```

`blob_count` is the number of active blobs actually written (0 if no
blobs are registered). Each blob's `key` is 32 bytes (NUL-terminated,
zero-padded). `size` is an 8-byte little-endian unsigned integer.
`data` is `size` bytes of the caller's blob data, copied byte-by-byte
(no `memcpy` on the crash path).

The blob section is always present, even when `blob_count` is 0 (in
which case it is just the 4-byte count = 0). A consumer that does not
care about blobs can stop reading after `sizeof(crash_dump_t)`.

## Dump format: CRASH_FORMAT_ELF (v0.2, v0.3 appends blob section)

A minimal ELF core file (4628 B total) is built in the caller's
buffer, then the v0.3 user-blob section is appended after the ELF
core, and the whole buffer is written with a single `write(2)`.
Layout:

```
offset    0:  Elf64_Ehdr             (64 B)
offset   64:  Elf64_Phdr[0] PT_NOTE  (56 B)
offset  120:  Elf64_Phdr[1] PT_LOAD  (56 B)
offset  176:  PT_NOTE payload:
                Elf64_Nhdr           (12 B)
                "CORE\0" + 3 pad     (8 B)
                struct elf_prstatus  (336 B)
                                            = 356 B
offset  532:  PT_LOAD payload: 4 KiB stack snapshot from [RSP, RSP+4096)
offset 4628:  uint32_t blob_count (little-endian, v0.3)
offset 4632:  for each blob: key[32] + uint64_t size + data[size] (v0.3)
```

The v0.3 blob section is appended after the ELF core (same format as
the custom-format blob section above). `gdb` ignores trailing data
beyond the ELF core's program headers, so the blob section does not
interfere with ELF-core parsing. The single `write(2)` covers the
ELF core + blob section.

The PT_LOAD segment covers `[RSP, RSP+4096)` at `p_vaddr = RSP`,
`p_align = 1` (no alignment requirement) so the snapshot can live at
any file offset; `gdb` accepts `p_align = 1` for core PT_LOAD segments.
If `RSP` is NULL or non-canonical, `p_filesz` and `p_memsz` are set to
0 and the 4 KiB at offset 532 is whatever the caller's buffer happened
to hold (`gdb` will not read them, since `PT_LOAD.filesz == 0`).

The `NT_PRSTATUS` note's `prstatus.pr_reg` carries the 27-slot
`user_regs_struct` (R15..GS) translated from `ucontext.uc_mcontext.gregs`
in the kernel's `genregs_get()` order. CS/FS/GS are extracted from
`REG_CSGSFS`. SS, FS_BASE, GS_BASE, ORIG_RAX, and the full AVX state
are not carried in the v0.2 ELF core (no `NT_X86_XSTATE` note); gdb
gets the GP set and the 4 KB stack window only. YMM capture is a
custom-format-path feature in v0.2.

If the caller's buffer is too small to hold 4628 B (it never is, given
`crash_min_buffer_size()` returns 10560 in v0.3), or if the ELF build
fails, the handler silently falls back to `CRASH_FORMAT_CUSTOM` so a
dump is still produced. A dump in the wrong format is strictly better
than no dump.

## Post-dump behavior

- `CRASH_AFTER_RERAISE` (default): restore `SIG_DFL` for the signal
  and re-raise via `kill(getpid(), sig)`. Preserves normal OS core-dump
  behavior if the user also has that enabled; the libcrash dump and the
  OS core are not mutually exclusive. If the re-raise somehow fails to
  terminate, `_exit(2)` as a last resort.
- `CRASH_AFTER_EXIT`: `_exit(1)` immediately after writing the dump.
  For the embedded-device use case (write dump to flash, reboot
  immediately) where waiting for whatever the OS would normally do is
  not desired.
- `CRASH_AFTER_FORK` (v0.2): `_Fork()` (POSIX.1-2024 / glibc 2.36+,
  async-signal-safe). The child inherits a COW copy of the address
  space, writes the dump, and `_exit(0)`s. The parent `waitpid()`s for
  the child before re-raising, so the dump is on disk before the parent
  terminates. If the dump writer itself faults, the re-entered handler
  in the child sees `handling == 1` (inherited via COW) and `_exit(255)`,
  leaving the file with whatever was written before the fault. If
  `_Fork()` fails, the parent falls through to the in-process dump
  path: writing a dump in-process is strictly better than not writing
  one.

  `waitpid` is not on the POSIX async-signal-safe list. libcrash uses
  it on the `CRASH_AFTER_FORK` parent path after the child has
  `_exit(0)`'d. The child's only libc calls are `write` and `_exit`,
  neither of which holds locks that `waitpid` would deadlock on. The
  loop retries only on `EINTR`; any other error (e.g. `ECHILD` if a
  `SIGCHLD` handler already reaped the child, `EINVAL`) breaks out and
  falls through to `crash_reraise` so the parent cannot hang forever.

## Permissions / information disclosure

The dump contains the full GP register set, RIP, RSP, the 4 KiB stack
window at RSP (which may hold secrets, return addresses that defeat
ASLR, or pointers into adjacent mappings), -- for the custom format --
the low 256 bits of YMM0..YMM15, and -- in v0.3 -- any user blobs the
caller registered (which may contain application state such as order
IDs, session tokens, or other sensitive data). The caller is
responsible for opening the dump fd with restrictive permissions
(`0600`); libcrash never opens files itself. Sharing a dump file
across privilege boundaries is a secret leak.

## Non-goals

- No network/upload logic. Writing the dump to the given fd is the full
  extent of this module's responsibility.
- Not thread-safe to call `crash_install` / `crash_uninstall`
  concurrently with itself; the caller invokes them single-threaded at
  startup/shutdown. (`crash_set_user_blob` / `crash_clear_user_blob`
  ARE thread-safe.)
- No attempt to capture or diagnose a recursive crash (inside the
  handler itself); the re-entry guard `_exit(255)`s.
- ELF core (v0.2) does not carry `NT_X86_XSTATE` (YMM/ZMM) or
  `NT_AUXV`; gdb gets the GP set and a 4 KiB stack window only. YMM
  capture is a custom-format-path feature.
- No automatic upload/network reporting.
- The user-blob section appended after the ELF core (v0.3) is not an
  ELF note; gdb ignores it. A separate parser is needed to extract
  blobs from an ELF-format dump.
