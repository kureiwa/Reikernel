/* libpmu v0.3: v0.2's perf_event_open(2) + read(2) path with an internal
 * rdpmc fast path, plus graceful degradation when perf_event_open is denied.
 *
 * v0.3 changes the failure mode of pmu_open when perf_event_open(2) returns
 * EACCES, EPERM, or ENOSYS (containerized environments with restrictive
 * seccomp, or perf_event_paranoid >= 3). Instead of returning NULL and
 * forcing every caller to handle a NULL ctx, pmu_open now allocates a
 * dummy context with fd == -1 and is_dummy == 1, sets *out_err =
 * PMU_ERR_PERM so the caller can tell degradation happened, and returns
 * the dummy (non-NULL). On a dummy ctx: pmu_start is a no-op, pmu_read
 * and pmu_stop_and_read set *out_value = 0, pmu_close just frees. The
 * caller can check pmu_is_available(ctx) to distinguish a real fd from a
 * dummy and fall back to an alternative timing source (e.g. rdtsc).
 *
 * v0.2 internals are unchanged: perf_event_open(2) + read(2) path with an
 * internal rdpmc fast path. pmu_read/pmu_stop_and_read transparently use
 * rdpmc when the kernel exposes a non-zero perf_event_mmap_page->index
 * (paranoid <= 1 or CAP_PERFMON) and fall back to the read(2) path
 * otherwise.
 *
 * Both paths scale the raw count by time_enabled/time_running under
 * multiplexing. The scaling uses __int128 (mirrors the kernel's
 * mul_u64_u64_div_u64()) to avoid u64 overflow when count and enabled are
 * both large. The read(2) path requests PERF_FORMAT_TOTAL_TIME_ENABLED |
 * PERF_FORMAT_TOTAL_TIME_RUNNING and parses the 24-byte
 * {value, time_enabled, time_running} return; the rdpmc path reads the
 * same fields from the mmap metadata page.
 *
 * The rdpmc sequence follows Documentation/arch/x86/ and the kernel's own
 * x86_perf_event_update() (arch/x86/events/core.c): read seqlock, retry
 * if odd, read index/offset/time_enabled/time_running/pmc_width, lfence;
 * rdpmc(index-1); lfence, sign-extend per pmc_width, re-check seqlock,
 * scale if multiplexed. */

#define _GNU_SOURCE
#include "pmu.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

/* x86_64-only asm helpers. On any other architecture the rdpmc path is
 * compiled out and the v0.1 read(2) path is the only one available. */
#if defined(__x86_64__)
extern uint64_t pmu_rdpmc(uint32_t index);
extern void     pmu_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);
#define PMU_HAVE_RDPMC 1
#else
#define PMU_HAVE_RDPMC 0
#endif

/* One metadata page plus one data page. The metadata page (page 0) holds
 * struct perf_event_mmap_page; the data page is unused by libpmu but the
 * kernel requires the mmap to cover at least one data page when samples
 * are not consumed. */
#define PMU_MMAP_PAGES_DATA  1
#define PMU_MMAP_PAGES_TOTAL (1 + PMU_MMAP_PAGES_DATA)

/* Bounded seqlock retry. The writer-side update is short and rare
 * (scheduler tick multiplexing); 64 spins is far beyond anything observed
 * in practice. If exceeded, we drop to the read(2) path. */
#define PMU_RDPMC_MAX_RETRIES 64

enum pmu_vendor {
    PMU_VENDOR_UNKNOWN = 0,
    PMU_VENDOR_INTEL,
    PMU_VENDOR_AMD,
};

struct pmu_ctx {
    int                 fd;
    pmu_counter_type_t type;
    /* 1 if perf_event_open failed with EACCES/EPERM/ENOSYS and pmu_open
     * degraded to a dummy context (fd == -1, mmap_base == NULL). On a
     * dummy ctx, pmu_start is a no-op, pmu_read/pmu_stop_and_read set
     * *out_value = 0, and pmu_is_available returns 0. */
    int                                is_dummy;
    /* rdpmc fast-path state. mmap_base == NULL means mmap failed or was
     * never attempted; in that case use_rdpmc is 0 and pmu_read uses the
     * read(2) path. */
    void                              *mmap_base;
    size_t                             mmap_size;
    struct perf_event_mmap_page       *pc;
    int                                use_rdpmc;
    enum pmu_vendor                    vendor;
};

/* Maps a public counter type to a PERF_COUNT_HW_* config value. Returns
 * PMU_OK / PMU_ERR_INVALID via *err. */
static pmu_err_t pmu_config_for_type(pmu_counter_type_t which, __u64 *out_config)
{
    switch (which) {
        case PMU_CYCLES:       *out_config = PERF_COUNT_HW_CPU_CYCLES;    return PMU_OK;
        case PMU_INSTRUCTIONS: *out_config = PERF_COUNT_HW_INSTRUCTIONS;  return PMU_OK;
        case PMU_CACHE_MISSES: *out_config = PERF_COUNT_HW_CACHE_MISSES;  return PMU_OK;
        default:               *out_config = 0;                          return PMU_ERR_INVALID;
    }
}

/* Maps perf_event_open(2) errno to a public pmu_err_t. EPERM/EACCES/ENOSYS
 * -> PMU_ERR_PERM (paranoid sysctl, missing CAP_PERFMON, seccomp block, or
 * kernel built without perf support). ENODEV/ENOENT/EINVAL ->
 * PMU_ERR_UNAVAILABLE (counter type not supported on this CPU). Anything
 * else also collapses to PMU_ERR_UNAVAILABLE; the v0.1 API has no generic
 * errno-passthrough code.
 *
 * v0.3: ENOSYS moved here from the PMU_ERR_UNAVAILABLE arm. ENOSYS means
 * the syscall itself is unavailable (older kernel, or seccomp returning
 * ENOSYS to mask the call). That is a permission/availability issue that
 * pmu_open now degrades to a dummy ctx on, so it belongs in PMU_ERR_PERM. */
static pmu_err_t pmu_err_from_errno(int e)
{
    switch (e) {
        case EPERM:
        case EACCES:
        case ENOSYS:
            return PMU_ERR_PERM;
        case ENODEV:
        case ENOENT:
        case EINVAL:
            return PMU_ERR_UNAVAILABLE;
        default:
            return PMU_ERR_UNAVAILABLE;
    }
}

/* Scales a raw counter count by enabled/running to account for
 * multiplexing. Uses a 128-bit intermediate to avoid u64 overflow when
 * both count and enabled are large (e.g. count ~ 2^40 cycles under heavy
 * multiplexing with enabled ~ 2^63 ns; the naive (count * enabled) /
 * running overflows silently and returns wrong values). Mirrors the
 * kernel's mul_u64_u64_div_u64() in lib/math/. running == 0 returns
 * count unchanged (defensive; should not occur post-enable).
 *
 * __int128 is a GCC/Clang extension; __extension__ suppresses the
 * -pedantic warning so the file still builds under -std=c11 -pedantic
 * -Werror. */
__extension__ typedef __int128 pmu_i128;
static uint64_t pmu_scale_count(uint64_t count, uint64_t enabled,
                                uint64_t running)
{
    if (enabled == running || running == 0) return count;
    return (uint64_t)((pmu_i128)count * enabled / running);
}

#if PMU_HAVE_RDPMC
/* Detects the CPU vendor via CPUID.0H. The 12-byte vendor string is laid
 * out as EBX, EDX, ECX in Intel's documented order: "GenuineIntel" or
 * "AuthenticAMD". Vendor is informational for v0.2 (the rdpmc recipe
 * itself is vendor-neutral because the kernel programs the event-select
 * MSR); it determines whether Intel fixed-counter encoding
 * (ECX[30]=1 + ECX[0..2] index) is in play, which affects documentation
 * and v0.3 raw-MSR-programming work. */
static enum pmu_vendor pmu_detect_vendor(void)
{
    uint32_t regs[4] = {0, 0, 0, 0};
    pmu_cpuid(0x0, 0x0, regs);
    /* EBX:EDX:ECX in Intel order forms the vendor string. */
    char vendor[13];
    memcpy(vendor + 0, &regs[1], 4);   /* EBX */
    memcpy(vendor + 4, &regs[3], 4);   /* EDX */
    memcpy(vendor + 8, &regs[2], 4);   /* ECX */
    vendor[12] = '\0';

    if (memcmp(vendor, "GenuineIntel", 12) == 0) return PMU_VENDOR_INTEL;
    if (memcmp(vendor, "AuthenticAMD", 12) == 0) return PMU_VENDOR_AMD;
    return PMU_VENDOR_UNKNOWN;
}
#endif

/* Attempts to mmap the perf_event fd and set up the rdpmc fast path.
 * On any failure, leaves ctx->mmap_base == NULL && ctx->use_rdpmc == 0
 * so pmu_read falls back to the v0.1 read(2) path. Never fails the open
 * itself: a missing rdpmc path is not an error condition for the public
 * API, only a performance degradation. */
static void pmu_setup_mmap(pmu_ctx_t *ctx)
{
    ctx->mmap_base = NULL;
    ctx->mmap_size = 0;
    ctx->pc        = NULL;
    ctx->use_rdpmc = 0;

#if PMU_HAVE_RDPMC
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return;
    size_t map_len = (size_t)ps * PMU_MMAP_PAGES_TOTAL;

    /* PROT_READ is sufficient: libpmu never writes samples into the data
     * page. MAP_SHARED is required so the kernel's updates to the
     * metadata page are visible to us. */
    void *base = mmap(NULL, map_len, PROT_READ, MAP_SHARED, ctx->fd, 0);
    if (base == MAP_FAILED) return;

    struct perf_event_mmap_page *pc =
        (struct perf_event_mmap_page *)base;

    /* cap_user_rdpmc is set by the kernel at open time based on
     * perf_event_paranoid and CR4.PCE. index is set when the event is
     * actually scheduled onto a PMC (typically at ENABLE time); we
     * tolerate index == 0 here and re-check on every pmu_read so a
     * counter that gets de-scheduled at runtime falls back gracefully. */
    if (pc->cap_user_rdpmc) {
        ctx->mmap_base = base;
        ctx->mmap_size = map_len;
        ctx->pc        = pc;
        ctx->use_rdpmc = 1;
    } else {
        /* rdpmc denied (paranoid >= 2 and no CAP_PERFMON). Unmap and
         * use the read(2) path. */
        munmap(base, map_len);
    }
#endif
}

#if PMU_HAVE_RDPMC
/* The rdpmc fast-path read. Returns PMU_OK and writes *out_value on
 * success. Returns PMU_ERR_UNAVAILABLE if rdpmc is not usable right now
 * (index == 0, counter currently not scheduled, or seqlock never
 * converged); the caller falls back to read(2). */
static int pmu_read_rdpmc(struct perf_event_mmap_page *pc, uint64_t *out_value)
{
    if (!pc->cap_user_rdpmc) return PMU_ERR_UNAVAILABLE;

    for (int retry = 0; retry < PMU_RDPMC_MAX_RETRIES; retry++) {
        uint32_t seq = pc->lock;
        __asm__ volatile("" ::: "memory");   /* barrier() */

        /* Retry if the writer is mid-update. Checking parity BEFORE
         * reading any field avoids acting on a transiently-zero index
         * (or any other field) the writer has not yet finished updating.
         * The kernel zeroes index on the scheduling path while holding
         * the seqlock write side, so a reader that catches an odd seq
         * can see index == 0 even when the event is still scheduled. */
        if (seq & 1u) continue;

        uint32_t index   = pc->index;
        uint64_t offset  = pc->offset;
        uint64_t enabled = pc->time_enabled;
        uint64_t running = pc->time_running;
        uint16_t width   = pc->pmc_width;

        __asm__ volatile("" ::: "memory");   /* barrier() */

        /* index == 0 means the kernel has not scheduled this event onto a
         * PMC (or has de-scheduled it). No rdpmc value is meaningful. */
        if (index == 0) return PMU_ERR_UNAVAILABLE;

        uint64_t pmc = pmu_rdpmc(index - 1);

        /* Sign-extend the rdpmc result per pmc_width using the kernel's
         * documented idiom (left-shift to put the sign bit at position
         * 63, then arithmetic right-shift back). width == 0 means the
         * kernel did not report a width; treat the raw value as
         * already-correct (no extension). */
        if (width != 0 && width < 64) {
            uint64_t shifted = pmc << (64 - width);
            pmc = (uint64_t)((int64_t)shifted >> (64 - width));
        }

        __asm__ volatile("" ::: "memory");   /* barrier() */
        uint32_t seq2 = pc->lock;

        /* Retry if the seqlock moved (writer updated fields while we
         * were reading). seq2 parity is covered: seq was verified even
         * above, so if seq2 is odd then seq != seq2. */
        if (seq != seq2) continue;

        uint64_t count = offset + pmc;
        *out_value = pmu_scale_count(count, enabled, running);
        return PMU_OK;
    }

    /* Seqlock never settled. Fall back to read(2). */
    return PMU_ERR_UNAVAILABLE;
}
#endif

pmu_ctx_t *pmu_open(pmu_counter_type_t which, pmu_err_t *out_err)
{
    __u64 config = 0;
    pmu_err_t cfg_err = pmu_config_for_type(which, &config);
    if (cfg_err != PMU_OK) {
        if (out_err) *out_err = cfg_err;
        return NULL;
    }

    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type   = PERF_TYPE_HARDWARE;
    attr.size   = sizeof(attr);
    attr.config = config;
    attr.disabled       = 1;     /* do not start counting until pmu_start */
    attr.exclude_user   = 0;     /* count user-mode events */
    attr.exclude_kernel = 0;     /* count kernel-mode events */
    attr.exclude_hv     = 0;     /* count hypervisor events (no-op on x86_64) */
    /* Request time_enabled/time_running in the read(2) return so the
     * read(2) path can scale the raw count under multiplexing the same
     * way the rdpmc path does. Without this, read(fd, &v, 8) returns
     * the unscaled raw count and the two paths disagree when the event
     * is multiplexed. */
    attr.read_format    = PERF_FORMAT_TOTAL_TIME_ENABLED
                        | PERF_FORMAT_TOTAL_TIME_RUNNING;

    /* PERF_FLAG_FD_CLOEXEC so the perf fd does not leak across execve. */
    int fd = (int)syscall(SYS_perf_event_open, &attr,
                          /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1,
                          PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) {
        pmu_err_t e = pmu_err_from_errno(errno);
        if (e == PMU_ERR_PERM) {
            /* Graceful degradation: EACCES/EPERM/ENOSYS. Allocate a
             * dummy context so callers do not have to special-case NULL.
             * pmu_start/pmu_read/pmu_stop_and_read/pmu_close all handle
             * is_dummy == 1 without touching the perf fd. *out_err is
             * still PMU_ERR_PERM so callers that need real counters can
             * detect the degradation via pmu_is_available(ctx) or by
             * checking *out_err, and fall back to an alternative timing
             * source. */
            pmu_ctx_t *d = malloc(sizeof(*d));
            if (!d) {
                if (out_err) *out_err = PMU_ERR_UNAVAILABLE;
                return NULL;
            }
            d->fd        = -1;
            d->type      = which;
            d->is_dummy  = 1;
            d->mmap_base = NULL;
            d->mmap_size = 0;
            d->pc        = NULL;
            d->use_rdpmc = 0;
            d->vendor    = PMU_VENDOR_UNKNOWN;
            if (out_err) *out_err = PMU_ERR_PERM;
            return d;
        }
        if (out_err) *out_err = e;
        return NULL;
    }

    pmu_ctx_t *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        close(fd);
        if (out_err) *out_err = PMU_ERR_UNAVAILABLE;
        return NULL;
    }
    ctx->fd        = fd;
    ctx->type      = which;
    ctx->is_dummy  = 0;
    ctx->vendor    = PMU_VENDOR_UNKNOWN;

    pmu_setup_mmap(ctx);
#if PMU_HAVE_RDPMC
    ctx->vendor = pmu_detect_vendor();
#endif

    if (out_err) *out_err = PMU_OK;
    return ctx;
}

int pmu_is_available(const pmu_ctx_t *ctx)
{
    if (!ctx) return 0;
    return ctx->is_dummy ? 0 : 1;
}

int pmu_start(pmu_ctx_t *ctx)
{
    if (!ctx) return PMU_ERR_INVALID;
    if (ctx->is_dummy) return PMU_OK;   /* no-op on dummy ctx */
    if (ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0) < 0) {
        return PMU_ERR_UNAVAILABLE;
    }
    if (ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
        return PMU_ERR_UNAVAILABLE;
    }
    return PMU_OK;
}

int pmu_read(pmu_ctx_t *ctx, uint64_t *out_value)
{
    if (!ctx || !out_value) return PMU_ERR_INVALID;

    if (ctx->is_dummy) {
        /* No fd, no syscall, no rdpmc. Reads on a dummy ctx always
         * return 0 so callers can run the same code path as on a real
         * ctx without crashing. */
        *out_value = 0;
        return PMU_OK;
    }

#if PMU_HAVE_RDPMC
    if (ctx->use_rdpmc && ctx->pc) {
        int rc = pmu_read_rdpmc(ctx->pc, out_value);
        if (rc == PMU_OK) return PMU_OK;
        /* rdpmc not usable right now (index == 0, seqlock churn, etc.).
         * Fall through to the read(2) path. */
    }
#endif

    /* read(2) returns {value, time_enabled, time_running} because
     * attr.read_format requests TOTAL_TIME_ENABLED|TOTAL_TIME_RUNNING.
     * Scale the raw value with the same overflow-safe formula as the
     * rdpmc path so the two paths agree under multiplexing. */
    struct {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    } buf;
    ssize_t n = read(ctx->fd, &buf, sizeof(buf));
    if (n != (ssize_t)sizeof(buf)) {
        return PMU_ERR_UNAVAILABLE;
    }
    *out_value = pmu_scale_count(buf.value, buf.time_enabled, buf.time_running);
    return PMU_OK;
}

int pmu_stop_and_read(pmu_ctx_t *ctx, uint64_t *out_value)
{
    if (!ctx || !out_value) return PMU_ERR_INVALID;
    if (ctx->is_dummy) {
        /* No ioctl (fd == -1) and no read. Reads on a dummy ctx always
         * return 0. */
        *out_value = 0;
        return PMU_OK;
    }
    if (ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
        return PMU_ERR_UNAVAILABLE;
    }
    /* After DISABLE the counter value is frozen; the mmap_page->offset is
     * updated by the kernel and rdpmc reads return the final PMC value.
     * pmu_read picks rdpmc or read(2) as usual. */
    return pmu_read(ctx, out_value);
}

void pmu_close(pmu_ctx_t *ctx)
{
    if (!ctx) return;
    if (ctx->is_dummy) {
        /* No fd to close, no mmap to unmap. Just free. */
        free(ctx);
        return;
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    if (ctx->mmap_base) {
        munmap(ctx->mmap_base, ctx->mmap_size);
    }
    free(ctx);
}
