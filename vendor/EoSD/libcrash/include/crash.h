#ifndef CRASH_H
#define CRASH_H

#include <stddef.h>
#include <stdint.h>

/*
 * libcrash: async-signal-safe crash/minidump handler.
 *
 * The caller pre-allocates a fixed-size scratch buffer and pre-opens a
 * writable file descriptor at install time. Nothing is allocated on the
 * crash path: the handler builds the dump directly in the caller's
 * buffer and writes it out with a single write(2). Between signal
 * delivery and the dump being written, the handler touches only
 * async-signal-safe primitives (write, sigaction, kill, getpid, _exit),
 * the asm helpers in crash_x86_64.asm, and direct loads/stores. No
 * malloc, printf, pthread_*, syslog, memcpy, memset, or any other
 * non-async-signal-safe call appears on that path.
 *
 * Thread-safety: crash_install and crash_uninstall are NOT thread-safe;
 * the caller invokes them single-threaded at startup/shutdown. The
 * signal handler itself may run on any thread; a static _Atomic int
 * re-entry guard serializes crash-path execution across threads (the
 * second thread to crash skips the dump and _exit(255)s).
 */

/*
 * Magic written at offset 0 of every dump. ASCII "CRASHDUM" in
 * big-endian byte order (0x43 0x52 0x41 0x53 0x48 0x44 0x55 0x4D).
 */
#define CRASH_MAGIC 0x435241534844554DULL

typedef enum {
    CRASH_OK                    =  0,
    CRASH_ERR_INVALID           = -1,
    CRASH_ERR_BUF_TOO_SMALL     = -2,
    CRASH_ERR_ALREADY_INSTALLED = -3,
} crash_err_t;

typedef enum {
    CRASH_AFTER_RERAISE,   /* restore SIG_DFL and re-raise; OS core dump runs */
    CRASH_AFTER_EXIT,      /* _exit(1) immediately after writing the dump */
    /*
     * v0.2. Spawn a child via _Fork() (POSIX.1-2008 async-signal-safe),
     * have the child write the dump and _exit(0), and have the parent
     * waitpid() for the child before re-raising. Isolates dump-writing
     * from the crashing process: if the dump writer itself faults, the
     * child dies but the parent still terminates cleanly.
     */
    CRASH_AFTER_FORK,
} crash_after_action_t;

/*
 * v0.2 dump format selection.
 *
 * CRASH_FORMAT_CUSTOM: v0.1 behavior. Writes the fixed-size
 * crash_dump_t struct above with a single write(2).
 *
 * CRASH_FORMAT_ELF: writes a minimal ELF core file (Elf64_Ehdr +
 * PT_NOTE/PT_LOAD program headers + NT_PRSTATUS note + 4 KB stack
 * snapshot) into the caller's buffer first, then issues a single
 * write(2). The result is readable by `gdb` and `eu-stack`. If the
 * caller's buffer is too small to hold the ELF core, the handler
 * silently falls back to CRASH_FORMAT_CUSTOM so a dump is still
 * produced.
 */
typedef enum {
    CRASH_FORMAT_CUSTOM = 0,
    CRASH_FORMAT_ELF    = 1,
} crash_format_t;

/*
 * v0.1 dump format. Fixed-size, written with a single write(2). Not
 * ELF-core-compatible; a parser/converter is a possible later addition.
 *
 * The gpr array layout, in order: RAX, RBX, RCX, RDX, RSI, RDI, RBP,
 * RSP, R8..R15. RSP also appears as the dedicated `rsp` field for
 * convenience; the two values are identical.
 */
typedef struct {
    uint64_t magic;             /* CRASH_MAGIC */
    uint64_t timestamp_ns;      /* raw TSC from rdtsc; named _ns for v0.2 compat */
    int      signal_number;     /* signal that triggered the handler */
    int      _pad0;             /* keep gpr 8-aligned; not interpreted */
    uint64_t gpr[16];
    uint64_t rip;
    uint64_t rsp;
    uint64_t eflags;
    uint8_t  ymm[16][32];       /* low 256 bits of YMM0..YMM15 */
    uint8_t  stack_snapshot[4096];
} crash_dump_t;

/*
 * Minimum scratch buffer size. v0.1 returns 8192. Exposed as a function
 * (not a macro) so callers cannot hardcode the value; a later version
 * may grow the dump struct and bump this number without breaking
 * callers who use it.
 *
 * Thread-safety: pure function, safe to call concurrently.
 */
size_t crash_min_buffer_size(void);

/*
 * Install the crash handler.
 *
 * buf/buf_size: caller-owned scratch space. Untouched except during a
 * crash. Must be >= crash_min_buffer_size().
 * fd: pre-opened writable file descriptor; the dump is written via a
 * single write(2). Pre-opening avoids path/permission/exhaustion
 * failures at crash time, when the handler has no recovery path.
 * signals: array of signal numbers, or NULL with signal_count==0 to
 * use the default set (SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS).
 * after_action: what to do after the dump is written.
 *
 * Equivalent to crash_install_elf(buf, buf_size, fd, signals,
 * signal_count, after_action, CRASH_FORMAT_CUSTOM). Provided in v0.2
 * as a backward-compatible wrapper; existing v0.1 callers recompile
 * and link without source changes.
 *
 * Allocates an alternate signal stack (64 KiB, v0.2) via malloc at
 * install time and never in the handler. For each requested signal it
 * installs a sigaction with SA_ONSTACK (run on altstack), SA_SIGINFO
 * (3-arg handler with ucontext), SA_NODEFER (do not block the signal
 * during the handler, so a second crash can re-enter and hit the
 * re-entry guard), and SA_RESETHAND (kernel resets to SIG_DFL after
 * first entry, so a recursive fault on the same signal terminates via
 * the OS default rather than looping in our handler).
 *
 * Return: CRASH_OK on success; CRASH_ERR_INVALID if buf is NULL, fd is
 * negative, or signal_count exceeds 32; CRASH_ERR_BUF_TOO_SMALL if
 * buf_size is below crash_min_buffer_size(); CRASH_ERR_ALREADY_INSTALLED
 * if invoked twice without an intervening crash_uninstall.
 *
 * Thread-safety: not thread-safe. Call once at startup.
 */
int crash_install(void *buf, size_t buf_size, int fd,
                  const int *signals, size_t signal_count,
                  crash_after_action_t after_action);

/*
 * v0.2 install with explicit dump format. See crash_install for the
 * buf/fd/signals/after_action semantics; the additional `format`
 * argument selects between the v0.1 custom binary layout and the
 * v0.2 ELF core layout. crash_install() calls this with
 * CRASH_FORMAT_CUSTOM.
 */
int crash_install_elf(void *buf, size_t buf_size, int fd,
                      const int *signals, size_t signal_count,
                      crash_after_action_t after_action,
                      crash_format_t format);

/*
 * Restore the signal handlers saved by crash_install, restore the old
 * alternate signal stack, free the altstack buffer, clear installed
 * state, and clear all registered user blobs so a subsequent
 * crash_install succeeds with a clean slate. Safe to call any time,
 * including from a context where crash_install failed partway.
 *
 * Not async-signal-safe (calls free). Only invoke during normal
 * operation.
 *
 * Thread-safety: not thread-safe. Call once at shutdown.
 */
void crash_uninstall(void);

/*
 * User blob API (v0.3). Lets the caller stash up to CRASH_MAX_USER_BLOBS
 * application-state blobs (e.g. "the order ID I was processing when I
 * crashed") in the crash dump. Blobs are registered before a crash and
 * read at crash time; no allocation happens on the crash path.
 *
 * The blob's `data` pointer is read at crash time, so it must point to
 * memory that remains valid until crash_uninstall. Static storage or
 * heap memory that is never freed during the process lifetime is
 * appropriate; stack memory of a thread that may exit before a crash
 * is not.
 *
 * Thread-safety: crash_set_user_blob and crash_clear_user_blob are
 * thread-safe (internal spinlock). The crash handler reads the blob
 * array lock-free; a concurrent set/clear may produce a garbled blob
 * entry in the dump, which is acceptable.
 */
#define CRASH_MAX_USER_BLOBS 8
#define CRASH_MAX_BLOB_KEY   32
#define CRASH_MAX_BLOB_SIZE  256

typedef struct {
    char        key[CRASH_MAX_BLOB_KEY];
    const void *data;  /* pointer to caller's data (read at crash time) */
    size_t      size;
} crash_user_blob_t;

/*
 * Register a user blob. key is a NUL-terminated string (max 31 chars
 * before the NUL; empty keys are rejected). data points to caller-owned
 * memory that must remain valid until crash_uninstall. At crash time,
 * the handler reads `size` bytes from `data` and writes them to the
 * dump. Returns 0 on success, -1 if key is NULL or empty, data is NULL,
 * size is 0 or exceeds CRASH_MAX_BLOB_SIZE, the key is too long, or all
 * CRASH_MAX_USER_BLOBS slots are full. If a blob with the same key
 * already exists, its data and size are replaced.
 *
 * Thread-safety: safe to call from any thread. The blob array is
 * protected by a spinlock.
 */
int crash_set_user_blob(const char *key, const void *data, size_t size);

/*
 * Remove a user blob by key. Returns 0 on success, -1 if not found or
 * key is NULL/empty/too long.
 *
 * Thread-safety: safe to call from any thread.
 */
int crash_clear_user_blob(const char *key);

#endif /* CRASH_H */
