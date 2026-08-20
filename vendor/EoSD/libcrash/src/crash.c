/*
 * libcrash: install, uninstall, dump format, signal handler (v0.3).
 *
 * _GNU_SOURCE is supplied by the Makefile (CFLAGS += -D_GNU_SOURCE).
 * It is required because the REG_RAX/REG_RBX/.../REG_RIP/REG_RSP/
 * REG_EFL macros in <sys/ucontext.h> are guarded by __USE_GNU on glibc
 * x86-64, the sa_restorer field in struct sigaction needs the same
 * feature-test macro, and _Fork() (used by CRASH_AFTER_FORK) is
 * declared under __USE_GNU in <unistd.h>.
 *
 * Async-signal-safety audit for the handler (crash_handler below):
 *   - atomic_compare_exchange_strong on _Atomic int (lock-free on
 *     x86-64, ATOMIC_INT_LOCK_FREE == 2): safe.
 *   - _exit, write, kill, getpid, sigaction, sigemptyset: all listed
 *     in signal-safety(7).
 *   - getppid, getpgrp, getsid: also in signal-safety(7); used by the
 *     ELF writer (crash_elf.c) to populate prstatus.
 *   - _Fork (POSIX.1-2024 / Issue 8 async-signal-safe; glibc 2.36+):
 *     used by CRASH_AFTER_FORK.
 *   - waitpid: NOT on the POSIX async-signal-safe list, but glibc's
 *     implementation is a thin syscall wrapper with no libc locks.
 *     Used only on the CRASH_AFTER_FORK parent path, after the child
 *     has _exit(0)'d. The child's only libc calls are write/_exit,
 *     neither of which holds locks that waitpid would deadlock on.
 *     This is the single intentional deviation from strict POSIX
 *     async-signal-safety, and it is mandated by the v0.2 spec.
 *   - crash_rdtsc / crash_copy_4k: hand-written NASM leaf functions,
 *     no libc calls. crash_rdtsc reads the timestamp counter;
 *     crash_copy_4k is used in place of memcpy (which is NOT on the
 *     POSIX async-signal-safe list) for the 4 KB stack snapshot,
 *     both in the custom-format path here and in the ELF-format path
 *     in crash_elf.c. (v0.2 removed the live-register YMM capture
 *     helper: see crash_capture_ymm_from_fpstate below.)
 *   - crash_elf_build: see the file-level audit in crash_elf.c. v0.3
 *     renamed this from crash_elf_write; it now only builds the ELF
 *     core in the buffer and returns the size (no write(2)) so the
 *     caller can append the user-blob section and issue a single
 *     write(2) for the whole dump.
 *   - crash_build_blob_section (v0.3): reads the g_user_blobs array
 *     lock-free (no spinlock on the crash path) and copies each
 *     active blob's key/size/data into g_buf via explicit byte
 *     load/store loops (no memcpy). The compiler may vectorize these
 *     loops; that is fine -- memory-to-memory traffic, no live YMM
 *     reads (same reasoning as crash_capture_ymm_from_fpstate). A
 *     concurrent crash_set_user_blob / crash_clear_user_blob may
 *     produce a garbled entry; the defensive NULL/range checks skip
 *     partially-updated slots. If a stale data pointer faults, the
 *     re-entry guard catches it and _exit(255)s.
 *   - Direct loads/stores on g_buf and the ucontext: safe.
 * No other functions appear on the crash path.
 */

#include <crash.h>
#include "crash_internal.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ucontext.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/wait.h>

/* ASM helpers (crash_x86_64.asm). */
extern uint64_t crash_rdtsc(void);
extern void     crash_copy_4k(void *dst, const void *src);

/* _Fork(): declared under __USE_GNU in <unistd.h>. glibc 2.36+ exposes
 * it; we declare it here unconditionally as well so the link step
 * fails loudly on systems without the symbol rather than silently
 * falling back to fork() (which is NOT async-signal-safe). */
extern pid_t _Fork(void) __THROW;

/* Default signal set: the standard "machine-level fatal" five. */
static const int g_default_signals[5] = {
    SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS,
};

/*
 * Install-time state. Written by crash_install_elf, read-only on the
 * crash path. The handler reads g_buf, g_buf_size, g_fd, g_format,
 * g_after_action, g_handling; the rest is for crash_uninstall.
 */
static void                 *g_buf             = NULL;
static size_t                g_buf_size        = 0;
static int                   g_fd              = -1;
static crash_format_t        g_format          = CRASH_FORMAT_CUSTOM;
static crash_after_action_t  g_after_action    = CRASH_AFTER_RERAISE;
static atomic_int            g_handling        = 0;

/* Saved state for crash_uninstall. */
static stack_t               g_old_altstack;
static void                 *g_altstack_mem    = NULL;
static struct sigaction      g_old_actions[32];
static int                   g_installed_signals[32];
static size_t                g_signal_count    = 0;

/*
 * v0.3 user blob array. Registered by crash_set_user_blob (any thread,
 * any time) and read lock-free by the crash handler. Each slot's
 * key[0] == '\0' means "empty"; the setter publishes a slot by writing
 * the key last, the clearer unpublishes by zeroing key[0] first. On
 * x86-64 TSO, stores are not reordered with stores, so a compiler
 * barrier (provided by the spinlock's atomic ops at set/clear time)
 * is sufficient for cross-thread visibility of the data/size fields
 * once key[0] is observed non-NUL. The handler additionally defends
 * against a partially-updated slot by skipping slots whose data is
 * NULL or whose size is out of range.
 *
 * The spinlock serializes set/clear against each other (not against
 * the handler). It is never acquired on the crash path.
 */
static crash_user_blob_t g_user_blobs[CRASH_MAX_USER_BLOBS];
static _Atomic int       g_blob_lock = 0;

/*
 * Spinlock for the blob array. Acquired only by crash_set_user_blob,
 * crash_clear_user_blob, and crash_uninstall -- never on the crash
 * path. atomic_compare_exchange_strong is async-signal-safe on
 * x86-64 (ATOMIC_INT_LOCK_FREE == 2); the spin is bounded by the
 * brief critical sections in those three callers. If a signal
 * interrupts the lock holder on the same thread and the handler
 * tries to set/clear a blob, it would deadlock -- but the handler
 * never calls these, so the only risk is user code calling set/clear
 * from a signal handler that interrupts another set/clear, which is
 * a caller bug.
 */
static void crash_blob_lock(void)
{
    int expected;
    do {
        expected = 0;
    } while (!atomic_compare_exchange_strong(&g_blob_lock, &expected, 1));
}

static void crash_blob_unlock(void)
{
    atomic_store(&g_blob_lock, 0);
}

/* User-space address ceiling on x86-64 Linux: 47-bit (2^47). Used to
 * reject obvious garbage RSP values before attempting the stack
 * snapshot. Pointers in [0x0000_8000_0000_0000, 0xFFFF_7FFF_FFFF_FFFF]
 * are non-canonical and would fault on access; kernel-space pointers
 * (high bit set) are not for user code either. */
#define CRASH_USER_ADDR_CEILING 0x0000800000000000ULL

/*
 * UC_FP_XSTATE: kernel-set bit in ucontext_t.uc_flags indicating that
 * uc_mcontext.fpregs points to a full XSAVE area (not just a legacy
 * FXSAVE area). Defined by the kernel in <asm/ucontext.h>; not exposed
 * by glibc's <sys/ucontext.h>, so we define it here.
 */
#define CRASH_UC_FP_XSTATE 0x1UL

/* XSAVE state-component bit for AVX YMM high 128 bits (component 2). */
#define CRASH_XSTATE_YMM 0x4ULL

/*
 * XSAVE area offsets, byte-exact. The layout the kernel writes when
 * uc_flags & UC_FP_XSTATE is set:
 *
 *   offset    0: legacy FXSAVE area (512 B). XMM0..XMM15 live at
 *                 offset 160, 16 B each.
 *   offset  512: XSAVE header (64 B). xstate_bv is the first u64;
 *                 bit 2 (CRASH_XSTATE_YMM) means the YMM high-128
 *                 area is populated.
 *   offset  576: YMM high-128 area (256 B). YMMH0..YMMH15 at offset
 *                 576, 16 B each. Each entry holds the high 128 bits
 *                 of YMMi; the low 128 bits live in XMMi in the
 *                 FXSAVE area.
 *
 * These constants are deliberately raw integers rather than offsetof()
 * on a struct, because the kernel-guaranteed layout is byte-stable
 * across glibc versions while glibc's struct _xstate is not part of
 * any stable ABI.
 */
#define CRASH_XSAVE_XMM_OFF    160u   /* XMM0..XMM15 in FXSAVE area */
#define CRASH_XSAVE_HDR_OFF    512u   /* xstate_bv */
#define CRASH_XSAVE_YMMH_OFF   576u   /* YMMH0..YMMH15 */
#define CRASH_XSAVE_MIN_SIZE   576u   /* min bytes needed to read XMM + hdr */
#define CRASH_XSAVE_YMM_END    (CRASH_XSAVE_YMMH_OFF + 16u * 16u)  /* 832 */

/*
 * v0.2 YMM capture. The v0.1 design captured live YMM registers via a
 * NASM helper (crash_capture_ymm). Two independent bugs made that
 * capture unreliable:
 *
 *   (a) The kernel saves the user's AVX state into the XSAVE area
 *       pointed to by uc->uc_mcontext.fpregs before delivering the
 *       signal, and the live YMM registers are not preserved across
 *       the kernel entry. Reading live YMM from the handler returns
 *       whatever the kernel happened to leave there (often zero), not
 *       the user's value. (Audit finding C-1, CRITICAL.)
 *   (b) At -O2 the compiler vectorizes the 16 GPR stores in
 *       crash_write_dump using legacy SSE on XMM0..XMM9, clobbering
 *       the low 128 bits of YMM0..YMM9 before any live-register
 *       capture could observe them. (Audit finding C-2, CRITICAL.)
 *
 * Fix: read the YMM values out of the XSAVE area in uc_mcontext.fpregs.
 * This is pure memory-to-memory traffic, so compiler vectorization of
 * the loop below cannot corrupt the source data (it is in memory, not
 * in registers). The handler is additionally marked
 * target("general-regs-only") as a belt-and-braces guarantee that no
 * compiler-emitted vector op ever touches the live AVX state on the
 * crash path.
 *
 * Output: 512 bytes at `out` (16 YMM registers, 32 B each, low-then-high
 * byte order). If fpregs is NULL, or the XSAVE area is too short to
 * hold YMMH, or the YMM component bit is clear in xstate_bv, the high
 * 128 bits of each register are written zero; the low 128 bits (XMM)
 * are copied from the FXSAVE area whenever fpregs is non-NULL.
 */
static void crash_capture_ymm_from_fpstate(uint8_t out[16][32],
                                           const fpregset_t fpregs,
                                           unsigned long uc_flags)
{
    /* Zero the high 128 bits of every YMM by default; we only fill them
     * in if the XSAVE area actually carries YMM state. */
    for (int i = 0; i < 16; i++) {
        for (int j = 16; j < 32; j++) {
            out[i][j] = 0;
        }
    }

    if (fpregs == NULL) {
        /* No FPU state saved at all: zero the whole output. */
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                out[i][j] = 0;
            }
        }
        return;
    }

    const uint8_t *fp = (const uint8_t *)fpregs;

    /* The FXSAVE area is always present (it is the first 512 B of the
     * fpregs buffer regardless of UC_FP_XSTATE). Copy the low 128 bits
     * of each YMM (== XMMi) from it. */
    for (int i = 0; i < 16; i++) {
        const uint8_t *xmm = fp + CRASH_XSAVE_XMM_OFF + (size_t)i * 16u;
        for (int j = 0; j < 16; j++) {
            out[i][j] = xmm[j];
        }
    }

    /* If the kernel did not save extended state, the high 128 bits stay
     * zero (set above) and we are done. */
    if ((uc_flags & CRASH_UC_FP_XSTATE) == 0) {
        return;
    }

    /* XSAVE header is at offset 512. xstate_bv is the first u64. Read
     * it as a raw little-endian u64; do not assume any particular struct
     * layout in the fpregs buffer. */
    uint64_t xstate_bv = 0;
    const uint8_t *bv_p = fp + CRASH_XSAVE_HDR_OFF;
    for (int b = 0; b < 8; b++) {
        xstate_bv |= ((uint64_t)bv_p[b]) << (8 * b);
    }

    if ((xstate_bv & CRASH_XSTATE_YMM) == 0) {
        /* YMM component not saved (process never touched AVX, or the
         * kernel saved only the legacy area): high bits stay zero. */
        return;
    }

    /* YMMH0..YMMH15 live at offset 576, 16 B each. */
    for (int i = 0; i < 16; i++) {
        const uint8_t *ymmh = fp + CRASH_XSAVE_YMMH_OFF + (size_t)i * 16u;
        for (int j = 0; j < 16; j++) {
            out[i][16 + j] = ymmh[j];
        }
    }
}

/*
 * Alternate signal stack size (v0.2). 64 KiB gives the handler room
 * for: deep nesting (two recursive handler entries before the
 * _Atomic guard kicks in), the struct elf_prstatus (~336 B) and
 * Elf64_Ehdr+Phdrs (~176 B) used as stack-local scaffolding in the
 * ELF path if a future revision chooses to stack-allocate them, and
 * the getrusage-style frame state. v0.1 used 4 * SIGSTKSZ = 32 KiB;
 * v0.2 doubles it per the fact-check recommendation.
 */
#define CRASH_ALTSTACK_SIZE (64u * 1024u)

size_t crash_min_buffer_size(void)
{
    /* Base: 8 KiB for the dump header. The ELF core path needs
     * CRASH_ELF_CORE_SIZE (4628 B), the custom format needs
     * sizeof(crash_dump_t) (~4.7 KiB); both fit inside 8 KiB.
     *
     * v0.3: add space for the user-blob section appended after the
     * header. Worst case is all CRASH_MAX_USER_BLOBS slots occupied,
     * each at CRASH_MAX_BLOB_SIZE: 4-byte count + N * (key + size +
     * data). The blob section is built in g_buf right after the
     * header, so the buffer must hold both. */
    return 8192u
         + CRASH_MAX_USER_BLOBS * (CRASH_MAX_BLOB_KEY + 8u + CRASH_MAX_BLOB_SIZE);
}

/*
 * Build the v0.3 user-blob section in `out` (which points into g_buf
 * right after the main dump -- either sizeof(crash_dump_t) bytes in for
 * the custom format, or CRASH_ELF_CORE_SIZE bytes in for the ELF
 * format). Layout:
 *
 *   [uint32_t count, little-endian]
 *   [for each active blob: key[32] + uint64_t size, little-endian + data[size]]
 *
 * The count is written last, after we know how many active blobs we
 * copied. Returns the total number of bytes written to `out`
 * (including the 4-byte count).
 *
 * Async-signal-safety: NO lock is taken (the handler reads the blob
 * array lock-free). No malloc, no memcpy/memset -- every byte is copied
 * via an explicit load/store loop. The compiler may vectorize the
 * loops; that is fine (memory-to-memory traffic, no live YMM reads,
 * same reasoning as crash_capture_ymm_from_fpstate). A concurrent
 * set/clear may produce a garbled entry (torn key, stale data pointer);
 * the consumer should tolerate this. If a stale data pointer faults
 * despite the NULL/range defensive checks below, the kernel re-enters
 * the handler and the re-entry guard _exit(255)s.
 */
static size_t crash_build_blob_section(uint8_t *out)
{
    uint8_t *const start = out;
    uint8_t       *p     = out + 4u;  /* reserve 4 bytes for count */
    uint32_t       count = 0;

    for (int i = 0; i < CRASH_MAX_USER_BLOBS; i++) {
        /* key[0] == '\0' means the slot is empty. A concurrent clear
         * might set key[0] = '\0' after we read it; we would then
         * copy a partially-cleared slot, which is acceptable. */
        if (g_user_blobs[i].key[0] == '\0') {
            continue;
        }
        const void *data = g_user_blobs[i].data;
        size_t      size = g_user_blobs[i].size;
        /* Defensive: skip if data is NULL or size is out of range.
         * This catches a partially-updated slot where key is published
         * but data/size are not yet visible. */
        if (data == NULL || size == 0 || size > CRASH_MAX_BLOB_SIZE) {
            continue;
        }

        /* Copy key (32 bytes, including the NUL terminator and any
         * trailing pad bytes). */
        for (size_t j = 0; j < CRASH_MAX_BLOB_KEY; j++) {
            p[j] = (uint8_t)g_user_blobs[i].key[j];
        }
        p += CRASH_MAX_BLOB_KEY;

        /* Write size (8 bytes, little-endian). */
        uint64_t sz = (uint64_t)size;
        for (int j = 0; j < 8; j++) {
            p[j] = (uint8_t)((sz >> (8 * j)) & 0xFFu);
        }
        p += 8;

        /* Copy data (size bytes). Byte-by-byte to stay async-signal-
         * safe (no memcpy). */
        const uint8_t *src = (const uint8_t *)data;
        for (size_t j = 0; j < size; j++) {
            p[j] = src[j];
        }
        p += size;

        count++;
    }

    /* Write the count (4 bytes, little-endian) into the reserved slot. */
    start[0] = (uint8_t)(count & 0xFFu);
    start[1] = (uint8_t)((count >> 8)  & 0xFFu);
    start[2] = (uint8_t)((count >> 16) & 0xFFu);
    start[3] = (uint8_t)((count >> 24) & 0xFFu);

    return (size_t)(p - start);
}

/*
 * Write the dump into g_buf and out to g_fd. Called from the handler
 * on the crash path. Async-signal-safe: only write(2), the asm
 * helpers, crash_elf_build (audited in crash_elf.c), and direct
 * loads/stores. If g_format is CRASH_FORMAT_ELF and the buffer is
 * large enough, an ELF core is written; otherwise (buffer too small
 * or g_format == CRASH_FORMAT_CUSTOM) the v0.1 custom format is
 * written. The fallback is silent: a dump in the wrong format is
 * strictly better than no dump.
 *
 * v0.3: after the main dump (custom struct or ELF core), the user-blob
 * section is appended in g_buf and the whole buffer is written with a
 * single write(2). The dump file layout is:
 *
 *   [crash_dump_t | ELF core][uint32_t blob_count][per-blob: key+size+data]
 */
static void crash_write_dump(int sig, const siginfo_t *info,
                             const ucontext_t *uc)
{
    if (g_format == CRASH_FORMAT_ELF) {
        ssize_t elf_size = crash_elf_build(g_buf, g_buf_size,
                                           sig, info, uc);
        if (elf_size >= 0) {
            /* Append the user-blob section after the ELF core, then
             * issue a single write(2) for the whole buffer. */
            uint8_t *blob_start = (uint8_t *)g_buf + (size_t)elf_size;
            size_t   blob_size  = crash_build_blob_section(blob_start);
            (void)write(g_fd, g_buf, (size_t)elf_size + blob_size);
            return;
        }
        /* elf_size == -1: buffer too small for the ELF core. Fall
         * through to the custom format so a dump is still produced. */
    }

    /* CRASH_FORMAT_CUSTOM (or ELF fallback). */
    crash_dump_t *dump = (crash_dump_t *)g_buf;

    dump->magic         = CRASH_MAGIC;
    dump->timestamp_ns  = crash_rdtsc();
    dump->signal_number = sig;
    dump->_pad0         = 0;

    /* uc can be NULL only if the kernel delivered a signal without a
     * ucontext (which it never does for SA_SIGINFO handlers, but the
     * ELF path defends the same way and we mirror it for symmetry).
     * A NULL uc means we have no GPR/RIP/RSP/eflags/YMM data: write
     * zeros into the dump so the consumer can detect the absence via
     * rip==0 and rsp==0. */
    if (uc != NULL) {
        const greg_t *g = uc->uc_mcontext.gregs;

        /*
         * gpr layout: RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8..R15.
         * RSP is also written to the dedicated rsp field below; the two
         * locations hold the same value.
         */
        dump->gpr[0]  = (uint64_t)g[REG_RAX];
        dump->gpr[1]  = (uint64_t)g[REG_RBX];
        dump->gpr[2]  = (uint64_t)g[REG_RCX];
        dump->gpr[3]  = (uint64_t)g[REG_RDX];
        dump->gpr[4]  = (uint64_t)g[REG_RSI];
        dump->gpr[5]  = (uint64_t)g[REG_RDI];
        dump->gpr[6]  = (uint64_t)g[REG_RBP];
        dump->gpr[7]  = (uint64_t)g[REG_RSP];
        dump->gpr[8]  = (uint64_t)g[REG_R8];
        dump->gpr[9]  = (uint64_t)g[REG_R9];
        dump->gpr[10] = (uint64_t)g[REG_R10];
        dump->gpr[11] = (uint64_t)g[REG_R11];
        dump->gpr[12] = (uint64_t)g[REG_R12];
        dump->gpr[13] = (uint64_t)g[REG_R13];
        dump->gpr[14] = (uint64_t)g[REG_R14];
        dump->gpr[15] = (uint64_t)g[REG_R15];

        dump->rip    = (uint64_t)g[REG_RIP];
        dump->rsp    = (uint64_t)g[REG_RSP];
        dump->eflags = (uint64_t)g[REG_EFL];

        /* YMM0..YMM15. v0.2 reads these out of the XSAVE area in
         * uc_mcontext.fpregs (see crash_capture_ymm_from_fpstate
         * for why live-register capture does not work). */
        crash_capture_ymm_from_fpstate(dump->ymm,
                                       uc->uc_mcontext.fpregs,
                                       uc->uc_flags);
    } else {
        /* No ucontext: zero the register snapshot fields. magic /
         * timestamp_ns / signal_number are already populated above. */
        for (int i = 0; i < 16; i++) {
            dump->gpr[i] = 0;
        }
        dump->rip    = 0;
        dump->rsp    = 0;
        dump->eflags = 0;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 32; j++) {
                dump->ymm[i][j] = 0;
            }
        }
    }

    /*
     * Stack snapshot. Skip if RSP is 0 or sits outside the user address
     * range; otherwise copy 4096 bytes from [RSP, RSP+4096) via the
     * asm helper (memcpy is not async-signal-safe). If the read faults
     * despite the range check, the kernel re-enters this handler; the
     * re-entry guard catches it and _exit(255)s. In that case we lose
     * the dump, but the process still terminates safely.
     */
    uint64_t rsp = dump->rsp;
    if (rsp != 0 && rsp < CRASH_USER_ADDR_CEILING) {
        crash_copy_4k(dump->stack_snapshot, (const void *)rsp);
    }
    /* else: leave stack_snapshot as whatever g_buf held. The consumer
     * can check the rsp field to decide whether to trust it. */

    /*
     * Single best-effort write. No retry on short writes: in a crash
     * handler, looping on write() is riskier than accepting a partial
     * dump, and a partial dump (magic + timestamp + signal_number at
     * the head) is still identifiable.
     *
     * v0.3: the user-blob section is appended after the crash_dump_t
     * header in g_buf, so a single write(2) covers both.
     */
    uint8_t *blob_start = (uint8_t *)g_buf + sizeof(*dump);
    size_t   blob_size  = crash_build_blob_section(blob_start);
    (void)write(g_fd, g_buf, sizeof(*dump) + blob_size);
}

/*
 * Restore SIG_DFL for `sig` and re-raise so the OS's default action
 * (typically: core dump + terminate) runs. sigaction and kill are
 * async-signal-safe; raise(3) is not, so we use kill+getpid. If the
 * re-raise somehow fails to terminate us, _exit(2) as a last resort.
 *
 * Called from the in-process path (CRASH_AFTER_RERAISE) and from the
 * CRASH_AFTER_FORK parent path after the child has written the dump.
 */
static void crash_reraise(int sig)
{
    struct sigaction dfl;
    dfl.sa_handler     = SIG_DFL;
    dfl.sa_sigaction   = NULL;
    sigemptyset(&dfl.sa_mask);
    dfl.sa_flags       = 0;
    dfl.sa_restorer    = NULL;
    sigaction(sig, &dfl, NULL);
    kill(getpid(), sig);
    _exit(2);
}

/*
 * The signal handler. Async-signal-safe; see the file-level comment
 * for the audit. Never returns on the crash path: it either _exit()s
 * directly (CRASH_AFTER_EXIT, or recursive crash, or CRASH_AFTER_FORK
 * parent after waitpid if re-raise somehow doesn't terminate) or
 * re-raises the signal to SIG_DFL which terminates the process.
 *
 * target("general-regs-only"): forbids the compiler from emitting
 * SSE/AVX instructions in this function or its callees inlined into
 * it. v0.2 captures YMM out of the XSAVE area (no live-register reads),
 * so compiler vectorization can no longer corrupt captured values;
 * the attribute is retained as a belt-and-braces guarantee that no
 * compiler-emitted vector op touches the live AVX state on the crash
 * path, and so future code added to the handler cannot accidentally
 * re-introduce audit finding C-2.
 */
__attribute__((target("general-regs-only")))
static void crash_handler(int sig, siginfo_t *info, void *ucontext)
{
    /*
     * Re-entry guard. If g_handling is already 1, a previous invocation
     * of this handler is mid-flight (on this or another thread) and the
     * current fault is inside the crash path itself. Skip the dump and
     * _exit(255) per DESIGN.md. atomic_compare_exchange_strong is
     * async-signal-safe (lock-free on x86-64).
     */
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_handling, &expected, 1)) {
        _exit(255);
    }

    const ucontext_t *uc = (const ucontext_t *)ucontext;

    if (g_after_action == CRASH_AFTER_FORK) {
        /*
         * Out-of-process dump via _Fork (POSIX.1-2024 / Issue 8
         * async-signal-safe; glibc 2.36+). The child inherits a COW
         * copy of the address space,
         * including g_buf and the ucontext the parent was passing;
         * the child writes the dump and _exit(0)s. The parent
         * waitpid()s for the child before re-raising, so the dump is
         * on disk before the parent terminates.
         *
         * If _Fork fails (returns -1), we fall through to the
         * in-process dump path below: writing a dump in-process is
         * strictly better than not writing one.
         */
        pid_t child = _Fork();
        if (child == 0) {
            /* Child. The re-entry guard (g_handling == 1) was
             * inherited from the parent via COW; if the dump writer
             * itself faults, the re-entered handler will see
             * g_handling == 1 and _exit(255), leaving the file with
             * whatever was written before the fault. */
            crash_write_dump(sig, info, uc);
            _exit(0);
        } else if (child > 0) {
            /* Parent. waitpid is not strictly POSIX async-signal-safe
             * but is safe in practice here (see file-level audit).
             * Loop on EINTR only; on any other error (ECHILD, EINVAL)
             * fall through to crash_reraise so the parent still
             * terminates instead of hanging forever. Audit finding
             * C-3. */
            int status;
            while (waitpid(child, &status, 0) < 0) {
                if (errno != EINTR) {
                    break;
                }
                /* EINTR: retry. */
            }
            /* The child has written the dump and exited (or waitpid
             * failed unrecoverably). Re-raise so the OS default action
             * runs in the parent (core dump + terminate, if enabled). */
            crash_reraise(sig);
            /* crash_reraise does not return. */
        }
        /* _Fork failed: fall through to in-process dump. */
    }

    /* In-process dump (CRASH_AFTER_RERAISE, CRASH_AFTER_EXIT, or
     * CRASH_AFTER_FORK with _Fork failure). */
    crash_write_dump(sig, info, uc);

    if (g_after_action == CRASH_AFTER_EXIT) {
        _exit(1);
    }

    /* CRASH_AFTER_RERAISE (or CRASH_AFTER_FORK fallback). */
    crash_reraise(sig);
}

int crash_install_elf(void *buf, size_t buf_size, int fd,
                      const int *signals, size_t signal_count,
                      crash_after_action_t after_action,
                      crash_format_t format)
{
    if (buf == NULL || fd < 0) {
        return CRASH_ERR_INVALID;
    }
    if (buf_size < crash_min_buffer_size()) {
        return CRASH_ERR_BUF_TOO_SMALL;
    }
    if (g_buf != NULL || atomic_load(&g_handling) != 0) {
        return CRASH_ERR_ALREADY_INSTALLED;
    }
    /* Validate `format` (forward-compat: reject unknown future values
     * rather than silently treating them as custom). */
    if (format != CRASH_FORMAT_CUSTOM && format != CRASH_FORMAT_ELF) {
        return CRASH_ERR_INVALID;
    }

    const int *sigs;
    size_t     nsig;
    if (signals == NULL || signal_count == 0) {
        sigs = g_default_signals;
        nsig = sizeof(g_default_signals) / sizeof(g_default_signals[0]);
    } else {
        if (signal_count > 32) {
            return CRASH_ERR_INVALID;
        }
        sigs = signals;
        nsig = signal_count;
    }

    /*
     * Alternate signal stack. 64 KiB (v0.2; was 32 KiB in v0.1).
     * Allocated here (install time), never in the handler. See
     * CRASH_ALTSTACK_SIZE comment for sizing rationale.
     */
    size_t altstack_size = CRASH_ALTSTACK_SIZE;
    void  *altstack_mem  = malloc(altstack_size);
    if (altstack_mem == NULL) {
        return CRASH_ERR_INVALID;
    }

    stack_t ss;
    memset(&ss, 0, sizeof(ss));
    ss.ss_sp    = altstack_mem;
    ss.ss_size  = altstack_size;
    ss.ss_flags = 0;
    if (sigaltstack(&ss, &g_old_altstack) != 0) {
        free(altstack_mem);
        return CRASH_ERR_INVALID;
    }

    /*
     * Install handlers. SA_ONSTACK: run on the altstack so we survive
     * a blown main stack. SA_SIGINFO: 3-arg handler with ucontext.
     * SA_NODEFER: do not block the triggering signal during the
     * handler, so a second crash can re-enter and the _Atomic handling
     * guard can catch it (rather than the signal being deferred until
     * after we _exit, which would lose the re-entry signal entirely).
     * SA_RESETHAND: kernel resets the handler to SIG_DFL on entry, so
     * a recursive fault on the same signal terminates via the OS
     * default rather than looping in our handler. SA_NODEFER +
     * SA_RESETHAND together give the re-entry guard full coverage for
     * cross-signal recursive faults (the same-signal case is handled
     * by SA_RESETHAND + SIG_DFL).
     */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = crash_handler;
    sa.sa_flags     = SA_ONSTACK | SA_SIGINFO | SA_NODEFER | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);

    size_t installed = 0;
    for (size_t i = 0; i < nsig; i++) {
        if (sigaction(sigs[i], &sa, &g_old_actions[i]) != 0) {
            /* Roll back the signals we already installed. */
            for (size_t j = 0; j < installed; j++) {
                sigaction(g_installed_signals[j], &g_old_actions[j], NULL);
                g_installed_signals[j] = 0;
            }
            sigaltstack(&g_old_altstack, NULL);
            free(altstack_mem);
            return CRASH_ERR_INVALID;
        }
        g_installed_signals[i] = sigs[i];
        installed++;
    }

    /*
     * Publish the crash-path state LAST. Before this point, a signal
     * arriving in this thread would run our handler with g_buf == NULL
     * and crash on the first store; with g_buf still NULL, that crash
     * re-enters the handler, hits the guard, and _exit(255)s. After
     * this point, the handler has a valid buffer to write into.
     */
    g_buf             = buf;
    g_buf_size        = buf_size;
    g_fd              = fd;
    g_format          = format;
    g_after_action    = after_action;
    g_altstack_mem    = altstack_mem;
    g_signal_count    = nsig;

    return CRASH_OK;
}

int crash_install(void *buf, size_t buf_size, int fd,
                  const int *signals, size_t signal_count,
                  crash_after_action_t after_action)
{
    /* v0.2 backward-compatibility shim: v0.1 callers see the same
     * signature and behavior (custom binary format). */
    return crash_install_elf(buf, buf_size, fd, signals, signal_count,
                             after_action, CRASH_FORMAT_CUSTOM);
}

void crash_uninstall(void)
{
    if (g_buf == NULL && g_altstack_mem == NULL) {
        return;
    }

    for (size_t i = 0; i < g_signal_count; i++) {
        sigaction(g_installed_signals[i], &g_old_actions[i], NULL);
        g_installed_signals[i] = 0;
    }
    g_signal_count = 0;

    sigaltstack(&g_old_altstack, NULL);
    if (g_altstack_mem != NULL) {
        free(g_altstack_mem);
        g_altstack_mem = NULL;
    }

    g_buf          = NULL;
    g_buf_size     = 0;
    g_fd           = -1;
    g_format       = CRASH_FORMAT_CUSTOM;
    g_after_action = CRASH_AFTER_RERAISE;
    atomic_store(&g_handling, 0);

    /* v0.3: clear all registered user blobs so a subsequent
     * crash_install starts with a clean blob array. memset is safe
     * here (crash_uninstall is not on the crash path). */
    crash_blob_lock();
    memset(g_user_blobs, 0, sizeof(g_user_blobs));
    crash_blob_unlock();
}

/* ---- v0.3 user blob API ---- */

/*
 * Validate and measure a blob key. Returns the key length (excluding
 * NUL) on success, or -1 if key is NULL, empty, or longer than
 * CRASH_MAX_BLOB_KEY-1 chars.
 */
static int crash_blob_key_len(const char *key)
{
    if (key == NULL) {
        return -1;
    }
    size_t klen = 0;
    while (klen < CRASH_MAX_BLOB_KEY - 1u && key[klen] != '\0') {
        klen++;
    }
    if (key[klen] != '\0' || klen == 0) {
        /* Too long, or empty (empty keys are rejected because
         * key[0] == '\0' is the "empty slot" sentinel used by the
         * lock-free handler). */
        return -1;
    }
    return (int)klen;
}

/*
 * Compare a slot's key against a NUL-terminated key of known length.
 * Both sides are guaranteed to have a NUL within CRASH_MAX_BLOB_KEY
 * bytes. Returns 1 on match, 0 otherwise.
 */
static int crash_blob_key_match(const crash_user_blob_t *slot,
                                const char *key, size_t klen)
{
    if (slot->key[0] == '\0') {
        return 0;
    }
    for (size_t j = 0; j <= klen; j++) {
        if (slot->key[j] != key[j]) {
            return 0;
        }
    }
    return 1;
}

int crash_set_user_blob(const char *key, const void *data, size_t size)
{
    int klen = crash_blob_key_len(key);
    if (klen < 0) {
        return -1;
    }
    if (data == NULL || size == 0 || size > CRASH_MAX_BLOB_SIZE) {
        return -1;
    }

    crash_blob_lock();

    /* If the key already exists, replace its data and size. */
    int free_slot = -1;
    for (int i = 0; i < CRASH_MAX_USER_BLOBS; i++) {
        if (g_user_blobs[i].key[0] != '\0') {
            if (crash_blob_key_match(&g_user_blobs[i], key, (size_t)klen)) {
                /* Replace in place. Order: write data and size first,
                 * then the key is already published (non-NUL key[0]
                 * is already there), so a concurrent handler read
                 * sees either the old data or the new data, both
                 * valid pointers. */
                g_user_blobs[i].data = data;
                g_user_blobs[i].size = size;
                crash_blob_unlock();
                return 0;
            }
        } else if (free_slot < 0) {
            free_slot = i;
        }
    }

    if (free_slot < 0) {
        crash_blob_unlock();
        return -1;  /* all slots full */
    }

    /* Publish a new slot. Order: write data and size first, then the
     * key last (the key write is what makes the slot visible to the
     * lock-free handler, which checks key[0] != '\0'). The spinlock's
     * release semantics (via the unlock's atomic_store) ensure the
     * data/size writes are visible to any thread that later observes
     * the key -- but the handler does not take the lock, so we also
     * rely on x86-64 TSO (stores are not reordered with stores) to
     * make the data/size visible before the key[0] store. */
    g_user_blobs[free_slot].data = data;
    g_user_blobs[free_slot].size = size;
    /* Copy the key including the NUL. */
    for (int j = 0; j <= klen; j++) {
        g_user_blobs[free_slot].key[j] = key[j];
    }
    /* Zero the rest of the key buffer so the handler copies clean
     * pad bytes (not stale heap contents) into the dump. */
    for (int j = klen + 1; j < CRASH_MAX_BLOB_KEY; j++) {
        g_user_blobs[free_slot].key[j] = '\0';
    }

    crash_blob_unlock();
    return 0;
}

int crash_clear_user_blob(const char *key)
{
    int klen = crash_blob_key_len(key);
    if (klen < 0) {
        return -1;
    }

    crash_blob_lock();

    int found = -1;
    for (int i = 0; i < CRASH_MAX_USER_BLOBS; i++) {
        if (crash_blob_key_match(&g_user_blobs[i], key, (size_t)klen)) {
            found = i;
            break;
        }
    }

    if (found < 0) {
        crash_blob_unlock();
        return -1;
    }

    /* Unpublish: zero key[0] first so the lock-free handler stops
     * reading this slot, then clear data and size. A concurrent
     * handler read that already observed key[0] != '\0' may still
     * read the old (still-valid) data pointer; that is acceptable
     * per the design. */
    g_user_blobs[found].key[0] = '\0';
    for (int j = 1; j < CRASH_MAX_BLOB_KEY; j++) {
        g_user_blobs[found].key[j] = '\0';
    }
    g_user_blobs[found].data = NULL;
    g_user_blobs[found].size = 0;

    crash_blob_unlock();
    return 0;
}
