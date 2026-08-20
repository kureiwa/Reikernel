/* Bench: rdpmc latency vs read(2) latency.
 *
 * Three measurements on a cycles counter:
 *
 *   1. libpmu pmu_read(): uses the v0.2 rdpmc fast path if available
 *      (paranoid <= 1 or CAP_PERFMON), otherwise falls back to read(2).
 *      This is what real callers pay.
 *
 *   2. raw read(fd, &v, 8) on the same kind of perf fd: the v0.1 syscall
 *      round-trip cost, independent of libpmu.
 *
 *   3. raw pmu_rdpmc(index - 1) on the mmap_page->index from a self-opened
 *      perf fd: the bare asm helper cost (lfence;rdpmc;lfence plus the
 *      C-side seqlock overhead is intentionally excluded here to isolate
 *      instruction-level latency). Skipped if index == 0.
 *
 * Expected: when rdpmc is available, (3) shows ~20-40 cycles (~6-13 ns at
 * 3 GHz) and (1) should be close to (3) plus a small constant for the
 * seqlock bookkeeping. When rdpmc is denied (paranoid=2, the sandbox
 * default), (1) and (2) are both ~100-300 ns/op and (3) is skipped. */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 199309L

#include "pmu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <linux/perf_event.h>

/* The asm helper is part of libpmu.a; declare it here so the bench can
 * time the bare rdpmc instruction sequence directly. */
extern uint64_t pmu_rdpmc(uint32_t index);

#define ITERS 1000000ULL

static double now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1.0;
    }
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Open a cycles perf fd for the calling thread and mmap it. On success
 * returns 0 and writes *out_fd, *out_base, *out_pc. On failure returns -1. */
static int open_self_cycles(int *out_fd, void **out_base,
                            struct perf_event_mmap_page **out_pc)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type           = PERF_TYPE_HARDWARE;
    attr.size           = sizeof(attr);
    attr.config         = PERF_COUNT_HW_CPU_CYCLES;
    attr.disabled       = 1;
    attr.exclude_user   = 0;
    attr.exclude_kernel = 0;
    attr.exclude_hv     = 0;

    int fd = (int)syscall(SYS_perf_event_open, &attr,
                          0, -1, -1, PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) return -1;

    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) { close(fd); return -1; }

    void *base = mmap(NULL, (size_t)ps * 2, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { close(fd); return -1; }

    *out_fd   = fd;
    *out_base = base;
    *out_pc   = (struct perf_event_mmap_page *)base;
    return 0;
}

/* A direct read(2) on the perf fd. */
static uint64_t read_via_syscall(int fd)
{
    uint64_t v = 0;
    ssize_t n = read(fd, &v, sizeof(v));
    (void)n;
    return v;
}

/* A direct pmu_rdpmc on the mmap_page->index, with the seqlock fields
 * snapshotted once (cheap path: no retry loop, just one consistent
 * snapshot). For latency isolation this is intentional. */
static uint64_t read_via_rdpmc(struct perf_event_mmap_page *pc)
{
    uint32_t index = pc->index;
    if (index == 0) return 0;
    return pmu_rdpmc(index - 1);
}

int main(void)
{
    /* --- measurement 1: libpmu pmu_read() --------------------------- */
    pmu_err_t err = PMU_OK;
    pmu_ctx_t *ctx = pmu_open(PMU_CYCLES, &err);
    if (!ctx) {
        if (err == PMU_ERR_PERM) {
            printf("SKIP bench_rdpmc: pmu_open PMU_ERR_PERM\n");
            printf("  Try: sudo sysctl -w kernel.perf_event_paranoid=1\n");
            return 0;
        }
        printf("FAIL bench_rdpmc: pmu_open err=%d\n", (int)err);
        return 1;
    }
    if (pmu_start(ctx) != PMU_OK) {
        printf("FAIL bench_rdpmc: pmu_start\n");
        pmu_close(ctx);
        return 1;
    }

    uint64_t v_lib = 0;
    for (int i = 0; i < 1024; i++) pmu_read(ctx, &v_lib);  /* warm */

    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) pmu_read(ctx, &v_lib);
    double t1 = now_ns();
    double ns_lib = (t1 - t0) / (double)ITERS;

    /* --- measurement 2 & 3: raw syscall vs raw rdpmc ---------------- */
    int raw_fd = -1;
    void *raw_base = NULL;
    struct perf_event_mmap_page *raw_pc = NULL;
    if (open_self_cycles(&raw_fd, &raw_base, &raw_pc) != 0) {
        printf("note: raw perf_event_open+mmap failed; "
               "skipping raw syscall and rdpmc measurements\n");
    } else {
        ioctl(raw_fd, PERF_EVENT_IOC_ENABLE, 0);
        uint32_t idx = raw_pc->index;

        /* raw read(2) */
        uint64_t v_sys = 0;
        for (int i = 0; i < 1024; i++) v_sys = read_via_syscall(raw_fd);

        t0 = now_ns();
        for (uint64_t i = 0; i < ITERS; i++) v_sys = read_via_syscall(raw_fd);
        t1 = now_ns();
        double ns_sys = (t1 - t0) / (double)ITERS;

        printf("bench_rdpmc: read(2) syscall  : %.2f ns/op (final=%llu)\n",
               ns_sys, (unsigned long long)v_sys);
        printf("bench_rdpmc:   expected ~100-300 ns/op for the syscall "
               "round-trip\n");

        /* raw rdpmc, only if index != 0 */
        if (idx != 0) {
            uint64_t v_rdp = 0;
            for (int i = 0; i < 1024; i++) v_rdp = read_via_rdpmc(raw_pc);

            t0 = now_ns();
            for (uint64_t i = 0; i < ITERS; i++) v_rdp = read_via_rdpmc(raw_pc);
            t1 = now_ns();
            double ns_rdp = (t1 - t0) / (double)ITERS;

            printf("bench_rdpmc: pmu_rdpmc bare   : %.2f ns/op (final=%llu, "
                   "index=%u)\n",
                   ns_rdp, (unsigned long long)v_rdp, idx);
            printf("bench_rdpmc:   expected ~20-40 cycles (~6-13 ns at 3GHz) "
                   "for the bare asm helper\n");
        } else {
            printf("bench_rdpmc: pmu_rdpmc bare   : SKIP (index=0, "
                   "perf_event_paranoid>=2)\n");
        }

        ioctl(raw_fd, PERF_EVENT_IOC_DISABLE, 0);
        long ps = sysconf(_SC_PAGESIZE);
        if (ps > 0) munmap(raw_base, (size_t)ps * 2);
        close(raw_fd);
    }

    printf("bench_rdpmc: libpmu pmu_read() : %.2f ns/op (final=%llu)\n",
           ns_lib, (unsigned long long)v_lib);
    if (!pmu_is_available(ctx)) {
        printf("bench_rdpmc:   ctx is DUMMY (perf_event_open denied, "
               "PMU_ERR_PERM=%d)\n", err == PMU_ERR_PERM);
        printf("bench_rdpmc:   dummy path = one branch + store 0 + return; "
               "see bench_dummy\n");
        printf("bench_rdpmc:   try: sudo sysctl -w kernel.perf_event_paranoid=1\n");
    } else if (raw_fd < 0) {
        printf("bench_rdpmc:   (raw measurements skipped; pmu_read falls "
               "back to read(2) so this is the syscall cost)\n");
    }

    pmu_close(ctx);
    return 0;
}
