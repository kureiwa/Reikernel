#ifndef PKEY_H
#define PKEY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* libpkey: Intel Memory Protection Keys (MPK) wrapper.
 *
 * MPK is a hardware feature (Intel Skylake-X and later, Linux 4.6+) that
 * lets userspace change page permissions in ~20 cycles via the WRPKRU
 * instruction, without a syscall. This is ~100x faster than mprotect for
 * per-page protection changes. Available when CPUID 7:0:ECX[4] (OSPKE) is
 * set, which means the OS has set CR4.PKE and the PKRU register is active
 * in userspace.
 *
 * The PKRU register has 2 bits per key (4 keys total): AD (access disable)
 * and WD (write disable). AD dominates: when AD=1, both reads and writes
 * are denied regardless of WD. Key 0 is the default key (pages without an
 * explicit pkey use key 0); keys 1-3 are available for application use.
 *
 * x86_64 only. On any other architecture, or on a CPU/OS without OSPKE,
 * pkey_available() returns 0 and all other functions return
 * PKEY_ERR_UNSUPPORTED. */

typedef enum {
    PKEY_OK              =  0,
    PKEY_ERR_INVALID     = -1,  /* bad argument (pkey out of range, NULL addr) */
    PKEY_ERR_UNSUPPORTED = -2,  /* CPU or OS doesn't support MPK (OSPKE not set) */
    PKEY_ERR_ALLOC       = -3,  /* pkey_alloc failed (no free keys, or EINVAL) */
    PKEY_ERR_MPROTECT    = -4,  /* pkey_mprotect failed */
} pkey_err_t;

/* Check if MPK is available. Probes CPUID 0x7:0:ECX[4] (OSPKE), which is
 * set when the OS has enabled CR4.PKE and the PKRU register is active in
 * userspace. CPU-only support (CPUID 7:0:ECX[3] PKU) is not sufficient;
 * WRPKRU/RDPKRU execute as no-ops (or #UD on older silicon) if CR4.PKE=0.
 *
 * Returns 1 if MPK is usable, 0 if not.
 *
 * Thread-safety: safe to call concurrently. CPUID is a non-privileged
 * instruction with no shared mutable state. The result is process-wide
 * constant; callers may cache it. */
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
 * Thread-safety: safe to call concurrently with respect to other pkey
 * operations on different keys. Concurrent free of the same key is a
 * caller bug (use-after-free). */
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
 * PKEY_ERR_MPROTECT if the syscall fails (errno in EINVAL/ENOMEM/ENOTSUP).
 *
 * Thread-safety: safe to call concurrently on disjoint regions. Concurrent
 * pkey_mprotect on overlapping regions is a caller bug (kernel VMA
 * splitting is not atomic across overlapping ranges). */
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
 * Each is a single WRPKRU (no syscall, ~20 cycles). Return values match
 * pkey_set_access. Thread-safety matches pkey_set_access (thread-local). */
int pkey_allow(int pkey);
int pkey_deny(int pkey);
int pkey_readonly(int pkey);

#ifdef __cplusplus
}
#endif

#endif /* PKEY_H */
