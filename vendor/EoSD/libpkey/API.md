# libpkey: API (v0.1)

Status: v0.1 shipped. Wraps Intel Memory Protection Keys (MPK) for
Linux/x86_64. Three syscall wrappers (`pkey_alloc`, `pkey_free`,
`pkey_mprotect`) and four no-syscall PKRU accessors (`pkey_set_access`,
`pkey_get_access`, `pkey_allow`, `pkey_deny`, `pkey_readonly`) backed by
the `WRPKRU`/`RDPKRU` userspace instructions in `src/pkey_x86_64.asm`.

## Overview

MPK is a hardware feature (Intel Skylake-X and later, Linux 4.6+) that
lets userspace change page permissions in ~20 cycles via the `WRPKRU`
instruction, without a syscall. This is ~20-100x faster than `mprotect`
for per-page protection changes. Available when CPUID 7:0:ECX[4] (OSPKE)
is set, which means the OS has set `CR4.PKE` and the PKRU register is
active in userspace.

The PKRU register has 2 bits per key (4 keys total): AD (access disable)
and WD (write disable). AD dominates: when AD=1, both reads and writes
are denied regardless of WD. Key 0 is the default key (pages without an
explicit pkey use key 0); keys 1-3 are available for application use.

Typical workflow:

```c
#include "pkey.h"
#include <sys/mman.h>

if (!pkey_available()) { /* fall back to mprotect */ }

int pk = pkey_alloc();                       /* syscall, one-time */
pkey_mprotect(buf, len, PROT_READ|PROT_WRITE, pk);  /* syscall, one-time */

/* Hot path: toggle protection with NO syscall, ~20 cycles. */
pkey_deny(pk);     /* buf now unreadable/unwritable */
pkey_allow(pk);    /* buf fully accessible again */

pkey_free(pk);     /* syscall, one-time */
```

## Types

```c
typedef enum {
    PKEY_OK              =  0,
    PKEY_ERR_INVALID     = -1,  /* bad argument (pkey out of range, NULL addr) */
    PKEY_ERR_UNSUPPORTED = -2,  /* CPU or OS doesn't support MPK (OSPKE not set) */
    PKEY_ERR_ALLOC       = -3,  /* pkey_alloc failed (no free keys, or EINVAL) */
    PKEY_ERR_MPROTECT    = -4,  /* pkey_mprotect failed */
} pkey_err_t;
```

## API

```c
/* Check if MPK is available. Probes CPUID 7:0:ECX[4] (OSPKE), which is
 * set when the OS has enabled CR4.PKE. CPU-only support (CPUID 7:0:ECX[3]
 * PKU) is not sufficient; WRPKRU/RDPKRU #UD (SIGILL) if CR4.PKE=0, and
 * pkey_alloc(2) returns ENOSYS.
 *
 * Returns 1 if MPK is usable, 0 if not.
 *
 * Thread-safety: safe to call concurrently. CPUID is non-privileged with
 * no shared mutable state. Result is process-wide constant; cacheable. */
int pkey_available(void);

/* Allocate a protection key. Wraps the pkey_alloc(0, 0) syscall.
 *
 * Returns a key number (0-3) on success, negative pkey_err_t on failure:
 *   PKEY_ERR_UNSUPPORTED if pkey_available() returns 0.
 *   PKEY_ERR_ALLOC if the syscall fails (no free keys, or EINVAL).
 *
 * Key 0 is the default key; pkey_alloc typically returns 1, 2, or 3.
 *
 * Thread-safety: safe to call concurrently. The kernel serializes key
 * allocation; each call returns a distinct key. */
int pkey_alloc(void);

/* Free a protection key. Wraps the pkey_free(pkey) syscall. After freeing,
 * the key is returned to the per-process pool and may be reassigned by a
 * subsequent pkey_alloc. Pages still tagged with the freed key retain the
 * tag (the tag is a property of the page-table entry, not the key's
 * allocation state), but the key's PKRU bits become meaningless.
 *
 * Returns PKEY_OK on success, PKEY_ERR_UNSUPPORTED if MPK is unavailable,
 * PKEY_ERR_INVALID if pkey is out of range [0,3] or the syscall fails.
 *
 * Thread-safety: safe concurrently for different keys. Concurrent free of
 * the same key is a caller bug (use-after-free). */
int pkey_free(int pkey);

/* Assign a key to a memory region. Wraps pkey_mprotect(addr, len, prot,
 * pkey). Like mprotect, but also tags the affected pages with `pkey` so
 * subsequent PKRU changes (via pkey_set_access) apply to those pages
 * without further syscalls. `addr` must be page-aligned; `len` is rounded
 * up to a page multiple by the kernel.
 *
 * `prot` is a bitwise-OR of PROT_READ / PROT_WRITE / PROT_EXEC / PROT_NONE
 * from <sys/mman.h>. `pkey` must be a key returned by pkey_alloc (or 0 for
 * the default key).
 *
 * Returns PKEY_OK on success, PKEY_ERR_UNSUPPORTED if MPK is unavailable,
 * PKEY_ERR_INVALID if addr is NULL or pkey is out of range,
 * PKEY_ERR_MPROTECT if the syscall fails.
 *
 * Thread-safety: safe on disjoint regions. Concurrent pkey_mprotect on
 * overlapping regions is a caller bug (kernel VMA splitting is not atomic
 * across overlapping ranges). */
int pkey_mprotect(void *addr, size_t len, int prot, int pkey);

/* Set access permissions for a key via WRPKRU. No syscall, ~20 cycles.
 *
 * access_disable=1: set AD (no read or write).
 * write_disable=1:  set WD (read but not write).
 * Both 0: clear both (full access).
 * Both 1: AD dominates (no access at all).
 *
 * The change takes effect immediately for all pages tagged with `pkey`,
 * on the calling thread only (PKRU is per-thread, saved/restored on
 * context switch by the kernel). Other threads' PKRU is unaffected.
 *
 * Returns PKEY_OK on success, PKEY_ERR_UNSUPPORTED if MPK is unavailable,
 * PKEY_ERR_INVALID if pkey is out of range [0,3].
 *
 * Thread-safety: thread-local. Each thread has its own PKRU; calling
 * pkey_set_access on one thread does not affect another. Safe to call
 * concurrently from multiple threads. */
int pkey_set_access(int pkey, int access_disable, int write_disable);

/* Get current access permissions for a key via RDPKRU. No syscall, ~20
 * cycles.
 *
 * Returns:
 *   0 = full access (AD=0, WD=0)
 *   1 = read-only    (AD=0, WD=1)
 *   2 = no access    (AD=1, WD irrelevant)
 *   PKEY_ERR_UNSUPPORTED if MPK is unavailable.
 *   PKEY_ERR_INVALID if pkey is out of range [0,3].
 *
 * Reads the calling thread's PKRU. Other threads' PKRU is not visible.
 *
 * Thread-safety: thread-local. Safe to call concurrently. */
int pkey_get_access(int pkey);

/* Convenience wrappers around pkey_set_access:
 *   pkey_allow(pkey)    -> clear AD + WD (full access)
 *   pkey_deny(pkey)     -> set AD, clear WD (no access)
 *   pkey_readonly(pkey) -> set WD, clear AD (read but not write)
 *
 * Each is a single RDPKRU + WRPKRU pair (no syscall, ~20-40 cycles).
 * Return values match pkey_set_access. Thread-safety: thread-local. */
int pkey_allow(int pkey);
int pkey_deny(int pkey);
int pkey_readonly(int pkey);
```

## Minimal usage example

```c
#define _DEFAULT_SOURCE
#include "pkey.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

int main(void) {
    if (!pkey_available()) {
        printf("MPK not available; use mprotect fallback\n");
        return 0;
    }

    int pk = pkey_alloc();
    if (pk < 0) { /* handle error */ }

    long ps = sysconf(_SC_PAGESIZE);
    void *buf = mmap(NULL, ps, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    /* Tag the page with the key (one-time syscall). */
    pkey_mprotect(buf, ps, PROT_READ | PROT_WRITE, pk);

    /* Hot path: toggle write permission with no syscall. */
    pkey_readonly(pk);   /* buf is now read-only */
    /* *(char *)buf = 'x';  -- would SIGSEGV */
    pkey_allow(pk);      /* buf is fully writable again */
    *(char *)buf = 'x';  /* succeeds */

    pkey_free(pk);
    munmap(buf, ps);
    return 0;
}
```

## Feature detection and graceful degradation

`pkey_available()` probes CPUID 7:0:ECX[4] (OSPKE). This bit is set by
the Linux kernel at boot when:

1. The CPU supports PKU (CPUID 7:0:ECX[3]).
2. The kernel was built with `CONFIG_ARCH_HAS_PKEYS=y` (x86_64 default
   since 4.6).
3. The kernel chose to set `CR4.PKE` (it does this unconditionally on
   boot when the above hold).

When `pkey_available()` returns 0, all other functions return
`PKEY_ERR_UNSUPPORTED` without invoking any syscall or instruction.
This includes the no-syscall accessors (`pkey_set_access`, `pkey_get_access`,
`pkey_allow`, `pkey_deny`, `pkey_readonly`) -- they do NOT execute
`WRPKRU`/`RDPKRU` because those instructions `#UD` (raise `SIGILL`) when
`CR4.PKE=0`. Callers must check `pkey_available()` once at startup and
branch to a `mprotect`-based fallback path.

MPK availability is process-wide constant; callers may cache the result
of `pkey_available()`.

## PKRU register layout

```
bit  0: key 0 AD     bit  1: key 0 WD
bit  2: key 1 AD     bit  3: key 1 WD
bit  4: key 2 AD     bit  5: key 2 WD
bit  6: key 3 AD     bit  7: key 3 WD
bits [31:8]: reserved (must be 0)
```

- **AD** (Access Disable): when set, both reads and writes to pages
  tagged with this key fault (`SIGSEGV`, `si_code = SEGV_ACCERR`).
- **WD** (Write Disable): when set, writes to pages tagged with this
  key fault; reads are allowed. AD dominates WD.

PKRU is per-thread. The kernel saves and restores it across context
switches and on kernel entry (so userspace AD/WD bits cannot restrict
the kernel). A `pkey_set_access` call on thread A does not affect thread
B's PKRU.

## Error codes

| Code | Meaning |
|---|---|
| `PKEY_OK` (0) | Success |
| `PKEY_ERR_INVALID` (-1) | Bad argument: pkey out of range [0,3], or addr is NULL |
| `PKEY_ERR_UNSUPPORTED` (-2) | MPK not available (OSPKE not set); all functions return this |
| `PKEY_ERR_ALLOC` (-3) | `pkey_alloc` syscall failed (no free keys, or EINVAL) |
| `PKEY_ERR_MPROTECT` (-4) | `pkey_mprotect` syscall failed (bad addr/prot, ENOMEM, etc.) |

## Edge cases

- **pkey_alloc when OSPKE=0**: returns `PKEY_ERR_UNSUPPORTED` without
  calling the syscall. (The kernel would return `ENOSYS` or `EINVAL`.)
- **pkey_set_access on an unallocated key**: succeeds at the PKRU level
  (WRPKRU does not check allocation), but has no effect because no pages
  are tagged with that key. Not an error.
- **pkey_free on a key still tagging pages**: the key is returned to the
  pool, but pages retain the tag. The tag becomes meaningless (the key's
  PKRU bits still apply, but a subsequent `pkey_alloc` may reassign the
  same number and inherit those bits). Callers should `pkey_mprotect`
  the pages back to key 0 before freeing, or accept the stale tag.
- **pkey 0**: the default key. All pages without an explicit pkey use
  key 0. `pkey_deny(0)` denies access to most of the process's memory;
  callers should generally avoid operating on key 0.
- **PKRU after fork**: the child inherits a copy of the parent's PKRU.
  `pkey_alloc` in the child returns independent keys (the kernel
  duplicates the key allocation state). Keys allocated before fork are
  still allocated in the child; freeing them in the child does not
  affect the parent.
- **PKRU across exec**: PKRU is reset to 0 (full access for all keys)
  on `execve`. Key allocations do not survive exec.

## Non-goals (v0.1)

- No Windows/macOS support. MPK on those OSes uses different ABIs
  (`PKEY_*` on Windows is unrelated to Intel MPK; macOS does not expose
  MPK). Linux/x86_64 only, matching the toolkit-level platform decision.
- No ARM64 support. MPK is Intel-only; ARM has a different memory
  tagging mechanism (MTE) that is out of scope.
- No pkey_alloc flags passthrough. The kernel `pkey_alloc(flags,
  access_rights)` accepts `PKEY_DISABLE_ACCESS` / `PKEY_DISABLE_WRITE`
  to pre-disable the key at allocation time. libpkey always passes
  `(0, 0)` and exposes post-allocation disabling via `pkey_set_access`.
  A `pkey_alloc_flags` variant can be added later if needed.
- No multi-key atomic update. `pkey_set_access` updates one key's bits
  via a read-modify-write of PKRU. To change multiple keys atomically,
  a caller would need to read PKRU, mask all target pairs, and write
  once -- this is not exposed in v0.1.
- No PKRU fault handler integration. A `SIGSEGV` from a PKRU violation
  is delivered with `si_code = SEGV_PKUERR` and `si_pkey` set to the
  offending key (Linux 4.18+). libpkey does not install a signal
  handler; composability with `libcrash` is a documented pattern in
  `DESIGN.md`, not a link-time dependency.

## Build

`make` builds `libpkey.a` (static archive). `make test` builds and runs
the three test binaries. `make bench` builds and runs the two bench
binaries. `make clean` removes all generated files.

The asm helpers (`src/pkey_x86_64.asm`) require NASM 2.13+ (for the
`rdpkru`/`wrpkru` mnemonics; the toolkit standardizes on 2.16.01) and
assemble with `-f elf64`. The C source builds warning-clean under
`-std=c11 -Wall -Wextra -Werror -pedantic -O2`.

Tests and benchmarks use `_DEFAULT_SOURCE` (not `_GNU_SOURCE`) to avoid
a signature clash between glibc's `pkey_alloc`/`pkey_free`/`pkey_mprotect`
wrappers (in `<sys/mman.h>` under `__USE_GNU`, since glibc 2.27) and
this module's same-named public API. See `DESIGN.md` "glibc name clash"
for details.
