/* bench_dummy: measure the overhead of the v0.3 dummy context.
 *
 * The dummy context is returned by pmu_open when perf_event_open(2)
 * fails with EACCES / EPERM / ENOSYS (e.g. perf_event_paranoid >= 2
 * for an unprivileged user, or a seccomp filter blocking the syscall).
 * On a dummy ctx:
 *
 *   pmu_start         -> no-op, returns PMU_OK
 *   pmu_read          -> *out_value = 0, returns PMU_OK (no syscall, no rdpmc)
 *   pmu_stop_and_read -> *out_value = 0, returns PMU_OK (no syscall)
 *   pmu_close         -> free(ctx)
 *
 * This bench measures the cost of one pmu_read on a dummy ctx, so a
 * caller who calls pmu_read in a hot loop can see the per-call cost
 * of the degradation path. Expected: single-digit nanoseconds (one
 * branch on is_dummy, one store to *out_value, one return).
 *
 * If pmu_open returns a real fd (perf_event_paranoid <= 1 or root),
 * the bench SKIPs the dummy measurement and notes that the real
 * pmu_read cost is reported by bench_read / bench_rdpmc.
 */

#define _POSIX_C_SOURCE 199309L

#include "pmu.h"

#include <stdio.h>
#include <time.h>

#define ITERS 1000000ULL

static double now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1.0;
    }
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(void)
{
    pmu_err_t err = PMU_OK;
    pmu_ctx_t *ctx = pmu_open(PMU_CYCLES, &err);
    if (!ctx) {
        printf("FAIL bench_dummy: pmu_open err=%d\n", (int)err);
        return 1;
    }

    if (pmu_is_available(ctx)) {
        printf("bench_dummy: SKIP (host has real perf access; "
               "use bench_read / bench_rdpmc for real-counter numbers)\n");
        pmu_close(ctx);
        return 0;
    }

    if (err != PMU_ERR_PERM) {
        printf("bench_dummy: unexpected err=%d on a dummy ctx\n", (int)err);
        pmu_close(ctx);
        return 1;
    }

    if (pmu_start(ctx) != PMU_OK) {
        printf("FAIL bench_dummy: pmu_start on dummy\n");
        pmu_close(ctx);
        return 1;
    }

    /* Warm up. */
    uint64_t v = 0xDEADBEEFCAFEBABEULL;
    for (int i = 0; i < 1024; i++) {
        pmu_read(ctx, &v);
    }
    if (v != 0) {
        printf("FAIL bench_dummy: dummy pmu_read returned v=%llu (expected 0)\n",
               (unsigned long long)v);
        pmu_close(ctx);
        return 1;
    }

    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        pmu_read(ctx, &v);
    }
    double t1 = now_ns();
    double ns_per = (t1 - t0) / (double)ITERS;

    /* Also measure pmu_stop_and_read on the dummy path. The counter
     * was started above; stop_and_read disables (no-op) and reads (0). */
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        pmu_stop_and_read(ctx, &v);
    }
    t1 = now_ns();
    double stop_ns = (t1 - t0) / (double)ITERS;

    pmu_close(ctx);

    printf("bench_dummy: %llu iterations (dummy ctx, is_available=0)\n",
           (unsigned long long)ITERS);
    printf("bench_dummy: pmu_read          : %.2f ns/op (expected ~0, "
           "single-digit ns)\n", ns_per);
    printf("bench_dummy: pmu_stop_and_read : %.2f ns/op\n", stop_ns);
    printf("bench_dummy:   dummy path = one branch on is_dummy + store 0 + "
           "return\n");
    return 0;
}
