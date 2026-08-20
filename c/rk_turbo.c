/* rk_turbo.c -- C-side glue for the turbo() context manager.
 * Probes libtopo, builds a physical-core mask (skips hyperthreads),
 * pins the calling thread + inherited OMP workers, sets OMP_* env vars
 * and omp_set_num_threads. Does NOT swap PyTorch's allocator (needs a
 * C++ extension to hook c10::Allocator). Nested enter returns -1.
 */

#define _GNU_SOURCE
#include "rk_api.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <omp.h>

#include "topo.h"

/* Mask width matching topo_numa_node_t.cpu_mask (4 * 64 = 256 CPUs).
 * The sandbox has 2 CPUs; a 1024-CPU box would need 16 words, but libtopo's
 * topo_pin accepts up to CPU_SETSIZE (1024) and topo_get_affinity accepts
 * arbitrary mask_words, so 4 words is plenty for any realistic box and
 * matches libtopo's own struct width. */
#define RK_TURBO_MASK_WORDS 4u

/* Turbo state. Saved on enter; restored on exit. The g_active flag also
 * serves as the nested-call guard (a second rk_turbo_enter while
 * g_active == 1 returns -1). */
static int           g_active    = 0;
static topo_info_t   g_topo;
static uint64_t      g_orig_mask[RK_TURBO_MASK_WORDS];
static int           g_have_orig = 0;

/* --- sysfs helpers (small enough to keep local rather than pulling in
 *     a separate header; libtopo has these as statics but doesn't expose
 *     them) --- */

/* Read up to bufsz-1 bytes from `path` into `buf`, NUL-terminate, trim
 * trailing whitespace. Returns bytes read (>= 0) on success, -1 on
 * open/read error or empty result. */
static ssize_t rk_read_sysfs(const char *path, char *buf, size_t bufsz)
{
    if (bufsz == 0) return -1;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, bufsz - 1);
    close(fd);
    if (n < 0) return -1;
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == ' '
                  || buf[n - 1] == '\t' || buf[n - 1] == '\r')) {
        n--;
    }
    buf[n] = '\0';
    return n;
}

/* Parse a cpulist string like "0-1,3,5-7" into `out` (an array of CPU IDs).
 * Writes at most max_out IDs. Returns the number of IDs parsed (may exceed
 * max_out; in that case the caller knows truncation occurred). Returns -1
 * on parse error. */
static int rk_parse_cpulist(const char *s, unsigned *out, unsigned max_out)
{
    unsigned count = 0;
    const char *p = s;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (*p < '0' || *p > '9') return -1;

        char *end;
        unsigned long a = strtoul(p, &end, 10);
        if (end == p) return -1;
        p = end;

        unsigned long b = a;
        if (*p == '-') {
            p++;
            if (*p < '0' || *p > '9') return -1;
            b = strtoul(p, &end, 10);
            if (end == p) return -1;
            p = end;
        }

        if (b < a) return -1;
        for (unsigned long c = a; c <= b; c++) {
            if (count < max_out && out != NULL) {
                out[count] = (unsigned)c;
            }
            count++;
        }

        while (*p == ' ' || *p == '\t') p++;
    }
    return (int)count;
}

/* Build a CPU affinity mask containing exactly one CPU per physical core.
 *
 * For each online CPU c, read thread_siblings_list. Find the minimum CPU
 * ID in the list (the physical-core representative). If c is the minimum,
 * set bit c; otherwise skip it (hyperthread sibling of an already-covered
 * core). On a non-SMT sandbox, each CPU's list contains only itself, so
 * the mask is {0, 1, ...} and the count equals total_cpus.
 *
 * Returns the popcount of the mask on success, -1 on hard failure. Soft
 * failure (sysfs unmounted) is handled by treating each CPU as its own
 * representative. */
static int rk_build_physical_mask(uint64_t *mask, unsigned words)
{
    for (unsigned i = 0; i < words; i++) mask[i] = 0;

    topo_info_t info;
    if (topo_probe(&info) != 0) return -1;

    unsigned total = info.total_cpus;
    if (total == 0) return -1;
    if (total > words * 64) total = words * 64;

    unsigned count = 0;
    char path[160];
    char buf[256];
    unsigned siblings[256];

    for (unsigned c = 0; c < total; c++) {
        int pn = snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list",
            c);
        if (pn <= 0 || (size_t)pn >= sizeof(path)) {
            /* Path too long for buffer (shouldn't happen for c < 1024).
             * Treat c as its own representative. */
            mask[c / 64] |= (UINT64_C(1) << (c % 64));
            count++;
            continue;
        }

        ssize_t k = rk_read_sysfs(path, buf, sizeof(buf));
        if (k <= 0) {
            /* sysfs unreadable. Treat c as its own representative (the
             * non-SMT sandbox default). */
            mask[c / 64] |= (UINT64_C(1) << (c % 64));
            count++;
            continue;
        }

        int n = rk_parse_cpulist(buf, siblings, 256);
        if (n <= 0) {
            mask[c / 64] |= (UINT64_C(1) << (c % 64));
            count++;
            continue;
        }

        /* Find the minimum CPU ID in this sibling list (the representative). */
        unsigned min_cpu = siblings[0];
        for (int i = 1; i < n; i++) {
            if (siblings[i] < min_cpu) min_cpu = siblings[i];
        }

        /* Add c iff it is the minimum (one bit per physical core). */
        if (c == min_cpu) {
            mask[c / 64] |= (UINT64_C(1) << (c % 64));
            count++;
        }
    }

    return (int)count;
}

int rk_turbo_enter(void)
{
    /* Nested-call guard. The Python wrapper also tracks this so the user
     * gets a clear RuntimeError on `with rk.turbo(): with rk.turbo():`. */
    if (g_active) return -1;

    if (topo_probe(&g_topo) != 0) return -1;

    /* Save the calling thread's current affinity so we can restore it on
     * exit. topo_get_affinity wraps sched_getaffinity. */
    if (topo_get_affinity(g_orig_mask, RK_TURBO_MASK_WORDS) != 0) {
        return -1;
    }
    g_have_orig = 1;

    /* Build the physical-core mask (one CPU per physical core, skipping
     * hyperthreads). */
    uint64_t phys_mask[RK_TURBO_MASK_WORDS];
    for (unsigned i = 0; i < RK_TURBO_MASK_WORDS; i++) phys_mask[i] = 0;
    int n_phys = rk_build_physical_mask(phys_mask, RK_TURBO_MASK_WORDS);
    if (n_phys <= 0) {
        g_have_orig = 0;
        return -1;
    }

    /* Pin the calling thread to the physical-core mask. On Linux, threads
     * spawned from this thread (including libgomp's worker pool, if it
     * lazily creates workers from inside the with-turbo region) inherit
     * this mask, so they get constrained to physical cores too. */
    if (topo_set_affinity(phys_mask, RK_TURBO_MASK_WORDS) != 0) {
        g_have_orig = 0;
        return -1;
    }

    /* Hint the OpenMP runtime. setenv affects future parallel-region
     * creations. libgomp reads OMP_NUM_THREADS the first time it builds
     * a thread pool (per-process, lazily), so this may be a no-op if a
     * previous parallel region already initialised the pool. The
     * omp_set_num_threads() call below is the reliable mechanism. */
    char nbuf[16];
    snprintf(nbuf, sizeof(nbuf), "%d", n_phys);
    setenv("OMP_NUM_THREADS", nbuf, /*overwrite=*/1);
    setenv("OMP_PROC_BIND",   "close", /*overwrite=*/1);
    setenv("OMP_PLACES",      "cores", /*overwrite=*/1);

    /* Directly update the libgomp nthreads-var ICV. This works even if
     * the pool was already created. Affects the libreikernel.so OMP
     * runtime (which is the same libgomp as torch's, when both are built
     * with gcc). */
    omp_set_num_threads(n_phys);

    g_active = 1;
    return n_phys;
}

int rk_turbo_exit(void)
{
    if (!g_active) return 0;

    int rc = 0;
    if (g_have_orig) {
        /* Restore the original affinity mask. Even if this fails we still
         * clear g_active so the user can retry the with-block. */
        rc = topo_set_affinity(g_orig_mask, RK_TURBO_MASK_WORDS);
    }

    /* We do NOT unset the OMP_* env vars or reset omp_set_num_threads:
     * we didn't save the previous value, and leaving them set matches the
     * user's likely intent for repeated turbo() calls. The torch thread
     * count restore is handled on the Python side. */

    g_active    = 0;
    g_have_orig = 0;
    return rc;
}

int rk_topo_detect(unsigned *threads_per_core,
                   unsigned *cores_per_package,
                   unsigned *num_packages,
                   unsigned *num_numa_nodes,
                   unsigned *total_cpus)
{
    topo_info_t info;
    if (topo_probe(&info) != 0) return -1;
    if (threads_per_core)  *threads_per_core  = info.threads_per_core;
    if (cores_per_package) *cores_per_package = info.cores_per_package;
    if (num_packages)      *num_packages      = info.num_packages;
    if (num_numa_nodes)    *num_numa_nodes    = info.num_numa_nodes;
    if (total_cpus)        *total_cpus        = info.total_cpus;
    return 0;
}
