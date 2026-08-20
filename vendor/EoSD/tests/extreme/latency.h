#ifndef EOSD_EXTREME_LATENCY_H
#define EOSD_EXTREME_LATENCY_H

/*
 * Shared latency measurement helpers for the extreme test suite.
 *
 * Each extreme test reports p50/p99/max latency for its core operation
 * in addition to the existing correctness checks. Timing uses
 * clock_gettime(CLOCK_MONOTONIC), which is vDSO-accelerated on
 * x86-64 Linux (~20-30 ns per call). Operations faster than the clock
 * resolution will be measured at the clock floor; this is documented
 * behavior, not a bug.
 *
 * All functions are static inline so each translation unit gets its
 * own copy and there is no linkage burden on the module Makefiles.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Returns current monotonic time in nanoseconds. */
static inline uint64_t latency_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Comparator for uint64_t ascending (used by qsort). */
static inline int latency_cmp_u64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* Sorts `samples` in place (ascending) and writes the nearest-rank
 * p50, p99, and max. n must be > 0.
 *
 *   p50 = samples[(n - 1) * 50 / 100]
 *   p99 = samples[(n - 1) * 99 / 100]
 *   max = samples[n - 1]
 */
static inline void latency_stats(uint64_t *samples, size_t n,
                                  uint64_t *out_p50, uint64_t *out_p99,
                                  uint64_t *out_max)
{
    qsort(samples, n, sizeof(uint64_t), latency_cmp_u64);
    *out_p50 = samples[(n - 1) * 50 / 100];
    *out_p99 = samples[(n - 1) * 99 / 100];
    *out_max = samples[n - 1];
}

/* Prints a latency summary line in ns:
 *   "  <name>: p50=X ns, p99=Y ns, max=Z ns (N=<n>)" */
static inline void latency_print_ns(const char *name, uint64_t p50,
                                     uint64_t p99, uint64_t max, size_t n)
{
    printf("  %s: p50=%llu ns, p99=%llu ns, max=%llu ns (N=%zu)\n",
           name, (unsigned long long)p50, (unsigned long long)p99,
           (unsigned long long)max, n);
}

/* Same, but reports values in milliseconds (for coarse-grained ops
 * like fork+waitpid). */
static inline void latency_print_ms(const char *name, uint64_t p50_ns,
                                     uint64_t p99_ns, uint64_t max_ns,
                                     size_t n)
{
    printf("  %s: p50=%.3f ms, p99=%.3f ms, max=%.3f ms (N=%zu)\n",
           name, (double)p50_ns / 1e6, (double)p99_ns / 1e6,
           (double)max_ns / 1e6, n);
}

#endif /* EOSD_EXTREME_LATENCY_H */
