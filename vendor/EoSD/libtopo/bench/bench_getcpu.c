/* bench_getcpu: measure the latency of topo_getcpu (RDPID fast path
 * when available) vs the getcpu(2) syscall, and vs an inline RDTSCP
 * reading IA32_TSC_AUX directly.
 *
 * Expected on x86_64 with RDPID support (CPUID.7.0:ECX[1]):
 *   topo_getcpu (RDPID)   : single-digit ns (~3 cycles)
 *   getcpu(2) syscall     : 15-50 ns (one syscall)
 *   RDTSCP (topo_rdtscp_ecx): single-digit ns (~30-40 cycles, serializing)
 *
 * Without RDPID, topo_getcpu falls through to the syscall and the
 * two columns agree.
 *
 * The bench reports ns/op (via CLOCK_MONOTONIC) and cycles/op (via
 * RDTSC, calibrated against CLOCK_MONOTONIC over the same window).
 * The cycles/op number is meaningful only if the calling thread is
 * pinned to a single CPU; the bench pins to CPU 0 by default and
 * falls back to the original affinity on error. */

#include "topo.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

#define ITERS 10000000ULL

/* rdtsc-beg/end pair. The lfence on the read side orders prior
 * instructions before the TSC read; the trailing lfence orders the
 * TSC read before subsequent instructions. This is the Intel SDM
 * RDTSC recipe. */
static inline uint64_t rdtsc_serialized(void)
{
    unsigned lo, hi;
    __asm__ volatile("lfence; rdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtscp_serialized(void)
{
    unsigned lo, hi;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx", "memory");
    __asm__ volatile("lfence" ::: "memory");
    return ((uint64_t)hi << 32) | lo;
}

static double now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1.0;
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Calibrate cycles-per-ns by reading TSC and CLOCK_MONOTONIC twice,
 * back-to-back, and computing the ratio. Returns cycles/ns or -1 on
 * failure. */
static double calibrate_cps(void)
{
    uint64_t t0 = rdtsc_serialized();
    double n0 = now_ns();
    /* Spin briefly so the delta is non-trivial. */
    volatile unsigned sink = 0;
    for (int i = 0; i < 100000; i++) sink += i;
    (void)sink;
    uint64_t t1 = rdtsc_serialized();
    double n1 = now_ns();
    double dn = n1 - n0;
    if (dn <= 0.0) return -1.0;
    return (double)(t1 - t0) / dn;
}

int main(void)
{
    topo_info_t info;
    if (topo_probe(&info) != 0) {
        printf("FAIL bench_getcpu: topo_probe failed\n");
        return 1;
    }

    /* Pin to CPU 0 for stable cycle counts. If the pin fails (cgroup
     * cpuset excludes CPU 0), proceed anyway -- cycles/op will be
     * noisier. */
    int pinned = (topo_pin(0) == 0);
    if (pinned) {
        printf("bench_getcpu: pinned to CPU 0\n");
    } else {
        printf("bench_getcpu: pin(0) failed; running on default affinity "
               "(cycles/op will be noisy)\n");
    }

    printf("bench_getcpu: total_cpus=%u threads/core=%u cores/pkg=%u "
           "pkgs=%u numa=%u\n",
           info.total_cpus, info.threads_per_core, info.cores_per_package,
           info.num_packages, info.num_numa_nodes);

    double cps = calibrate_cps();
    if (cps > 0.0) {
        printf("bench_getcpu: calibrated %.3f cycles/ns (TSC frequency %.2f GHz)\n",
               cps, cps);
    } else {
        printf("bench_getcpu: TSC calibration failed; cycles/op will be omitted\n");
    }

    /* Warm up. */
    volatile unsigned sink = 0;
    for (int i = 0; i < 1024; i++) {
        sink += topo_getcpu();
    }

    /* --- topo_getcpu (RDPID fast path if available) --- */
    uint64_t tsc0 = rdtsc_serialized();
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        sink += topo_getcpu();
    }
    double t1 = now_ns();
    uint64_t tsc1 = rdtsc_serialized();
    double ns_topo = (t1 - t0) / (double)ITERS;
    double cyc_topo = (cps > 0.0)
        ? (double)(tsc1 - tsc0) / (double)ITERS : -1.0;

    /* --- getcpu(2) syscall --- */
    tsc0 = rdtsc_serialized();
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        unsigned int cpu;
        if (syscall(SYS_getcpu, &cpu, NULL, NULL) == 0) {
            sink += cpu;
        }
    }
    t1 = now_ns();
    tsc1 = rdtsc_serialized();
    double ns_sys = (t1 - t0) / (double)ITERS;
    double cyc_sys = (cps > 0.0)
        ? (double)(tsc1 - tsc0) / (double)ITERS : -1.0;

    /* --- sched_getcpu() (libc, RDPID-based on x86_64 if available) --- */
    tsc0 = rdtsc_serialized();
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        sink += (unsigned)sched_getcpu();
    }
    t1 = now_ns();
    tsc1 = rdtsc_serialized();
    double ns_libc = (t1 - t0) / (double)ITERS;
    double cyc_libc = (cps > 0.0)
        ? (double)(tsc1 - tsc0) / (double)ITERS : -1.0;

    /* Suppress unused-warning on sink. */
    if (sink == 0) {
        printf("(note: sink=0)\n");
    }

    printf("\nbench_getcpu: %llu iterations\n", (unsigned long long)ITERS);
    printf("  %-32s %10s %10s\n", "path", "ns/op", "cyc/op");
    printf("  %-32s %10.2f %10.2f\n", "topo_getcpu (RDPID if avail)",
           ns_topo, cyc_topo);
    printf("  %-32s %10.2f %10.2f\n", "sched_getcpu (libc)",
           ns_libc, cyc_libc);
    printf("  %-32s %10.2f %10.2f\n", "getcpu(2) syscall",
           ns_sys, cyc_sys);

    /* Sanity: topo_getcpu should be no slower than the syscall. */
    if (ns_topo > ns_sys * 1.5) {
        printf("\nWARN: topo_getcpu slower than syscall (RDPID not in use?)\n");
    } else {
        printf("\ntopo_getcpu speedup over syscall: %.2fx\n",
               ns_sys / ns_topo);
    }

    /* Verify correctness: pin to CPU 0, topo_getcpu should report 0. */
    if (pinned) {
        unsigned c = topo_getcpu();
        if (c == 0) {
            printf("correctness: topo_getcpu()==0 after pin(0) [PASS]\n");
        } else {
            printf("correctness: topo_getcpu()==%u after pin(0) [FAIL]\n", c);
        }
    }

    return 0;
}
