#ifndef TOPO_H
#define TOPO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU topology summary. Filled by topo_probe. The struct is small
 * (20 bytes) on purpose so it can live on the stack without guilt.
 *
 *   threads_per_core    - logical CPUs per physical core (>=1; 1 means
 *                         no SMT / hyperthreading).
 *   cores_per_package   - physical cores per socket/package (>=1).
 *   num_packages        - sockets (>=1).
 *   num_numa_nodes      - NUMA nodes visible to the kernel (>=1; on a
 *                         non-NUMA system the kernel exposes a single
 *                         synthetic node0).
 *   total_cpus          - online logical CPUs (== sum over packages of
 *                         cores_per_package * threads_per_core on a
 *                         homogeneous system).
 *
 * Source of truth on x86_64: CPUID leaf 0x1F (extended topology) when
 * the CPU supports it, else leaf 0xB (x2APIC enumeration). sysfs
 * /sys/devices/system/cpu/cpuN/topology (thread_siblings_list,
 * core_siblings_list, physical_package_id) is the fallback when CPUID
 * is uninformative (broken BIOS, certain hypervisors). total_cpus and
 * num_numa_nodes always come from the kernel (sysconf and sysfs). */
typedef struct {
    unsigned threads_per_core;
    unsigned cores_per_package;
    unsigned num_packages;
    unsigned num_numa_nodes;
    unsigned total_cpus;
} topo_info_t;

/* NUMA node descriptor. cpu_mask is a 256-bit bitmap (4 * uint64_t);
 * bit i is set iff logical CPU i belongs to this node. cpu_count is
 * the popcount of cpu_mask. memory_distance is intentionally not
 * exposed here: /sys/.../nodeN/distance is a per-pair matrix, not a
 * per-node scalar, and does not fit this struct's shape. Callers that
 * need it can read the sysfs file directly. */
typedef struct {
    int      node_id;
    uint64_t cpu_mask[4];   /* up to 256 CPUs */
    unsigned cpu_count;
} topo_numa_node_t;

/* Cache descriptor for one cache level on one logical CPU. level is
 * 1/2/3 (matching the caller's request); size_kb is the total cache
 * size in KiB; line_size is the coherency line size in bytes; sharing
 * is the number of logical CPUs that share an instance of this cache
 * (1 = private to the core; >1 = shared, e.g. L3). */
typedef struct {
    unsigned level;       /* 1=L1, 2=L2, 3=L3 */
    unsigned size_kb;
    unsigned line_size;
    unsigned sharing;     /* number of logical CPUs sharing this cache */
} topo_cache_t;

/* Probe CPU topology. Fills *info with the system's CPU/NUMA layout.
 * On success returns 0 and *info is fully populated (all fields >= 1,
 * total_cpus >= 1). On failure returns a negative errno-style code
 * (-EINVAL if info is NULL; no other failure modes are currently
 * defined -- every field has a defensive default that lets the caller
 * proceed even if CPUID and sysfs were both uninformative).
 *
 * On x86_64 reads CPUID leaf 0x1F (or 0xB as fallback) for the SMT /
 * Core / Package hierarchy, then reads /sys/devices/system/node/online
 * for num_numa_nodes and sysconf(_SC_NPROCESSORS_ONLN) for total_cpus.
 * On a non-x86_64 host only the sysfs path is used.
 *
 * Thread-safety: safe to call concurrently from multiple threads.
 * The CPUID-feature detection is memoized in a static variable; the
 * benign race (two threads racing to fill it in) yields the same
 * value because CPUID output is constant per CPU. */
int topo_probe(topo_info_t *info);

/* Enumerate NUMA nodes. Returns the count of online NUMA nodes
 * visible to the kernel (always >= 1: on a non-NUMA system the kernel
 * exposes a synthetic node0). If `nodes` is non-NULL and `max_nodes`
 * is > 0, fills nodes[0..min(count, max_nodes)-1] with the descriptor
 * for each node, in ascending node_id order. The cpu_mask field of
 * each filled entry is populated from
 * /sys/devices/system/node/nodeN/cpulist; cpu_count is the popcount.
 *
 * If a node's cpulist cannot be read (e.g. sysfs not mounted), the
 * corresponding entry's cpu_mask is zeroed and cpu_count is 0; the
 * node_id is still filled. The return value is always the true node
 * count, even if more than max_nodes (so the caller can detect
 * truncation).
 *
 * Thread-safety: safe to call concurrently; reads sysfs only. */
unsigned topo_numa_nodes(topo_numa_node_t *nodes, unsigned max_nodes);

/* Get cache info for a given logical CPU and cache level (1, 2, or 3).
 * On success returns 0 and fills *cache. Returns -EINVAL if cache or
 * level is NULL/0/>3. Returns -ENOENT if the requested level is not
 * present (e.g. asking for L3 on a CPU with no L3). Returns -ENOSYS
 * on non-x86_64 hosts (CPUID leaf 4 is x86-specific).
 *
 * On x86_64 this calls CPUID leaf 4 (deterministic cache parameters)
 * for the calling logical CPU. If `cpu` is not the current CPU, the
 * calling thread is briefly pinned to `cpu` (via sched_setaffinity)
 * and the original affinity is restored afterwards; on homogeneous
 * systems (the common case) all CPUs in a package return the same
 * parameters and the pin is a no-op correctness-wise. Per-CPU caches
 * that differ between cores (e.g. AMD vCache CCDs) get the right
 * answer for the requested CPU.
 *
 * Thread-safety: safe to call concurrently; the brief affinity pin
 * affects only the calling thread. */
int topo_cache_info(unsigned cpu, unsigned level, topo_cache_t *cache);

/* Pin the calling thread to a specific CPU. Sets a single-bit
 * affinity mask containing only `cpu` and calls sched_setaffinity(2).
 * Returns 0 on success, -EINVAL if cpu >= 256, or the negated errno
 * from sched_setaffinity on failure (e.g. -EPERM if a cgroup cpuset
 * excludes `cpu`, though for the calling thread's own affinity this
 * is rare). The pin persists until the caller changes affinity again
 * (e.g. via topo_set_affinity, topo_pin_node, or directly via
 * sched_setaffinity).
 *
 * Thread-safety: safe to call concurrently; affects only the calling
 * thread. */
int topo_pin(unsigned cpu);

/* Pin the calling thread to all CPUs in a NUMA node. Reads
 * /sys/devices/system/node/nodeN/cpulist, builds the corresponding
 * affinity mask, and calls sched_setaffinity(2). Returns 0 on
 * success, -EINVAL if node_id < 0, -ENOENT if the node does not exist
 * or its cpulist cannot be read, or the negated errno from
 * sched_setaffinity on failure. */
int topo_pin_node(int node_id);

/* Get the current CPU number (0 to N-1). On x86_64 with RDPID support
 * (CPUID.7.0:ECX[1]), uses the RDPID instruction (~3 cycles) and
 * returns the low 12 bits of IA32_TSC_AUX, which on Linux is set to
 * the CPU number (the high bits hold the NUMA node id). On CPUs
 * without RDPID, or on non-x86_64 hosts, falls back to the getcpu(2)
 * syscall (~20-50 ns). Returns 0 if both paths fail (defensive; the
 * kernel always reports a valid CPU in practice).
 *
 * The RDPID result can be momentarily stale across a context switch
 * (the kernel updates IA32_TSC_AUX on the scheduling path), but this
 * is no worse than the getcpu(2) syscall, which reads the same MSR.
 *
 * Thread-safety: safe to call concurrently. */
unsigned topo_getcpu(void);

/* Get the calling thread's CPU affinity mask. sched_getaffinity(2)
 * into `mask`, treated as `mask_words` uint64_t words (256 bits at
 * mask_words=4). CPUs beyond mask_words * 64 are silently truncated
 * (caller passes a larger mask if needed). Returns 0 on success,
 * -EINVAL if mask is NULL or mask_words is 0, or the negated errno
 * from sched_getaffinity.
 *
 * Thread-safety: safe to call concurrently. */
int topo_get_affinity(uint64_t *mask, unsigned mask_words);

/* Set the calling thread's CPU affinity mask. Calls sched_setaffinity
 * with a mask built from `mask[0..mask_words-1]`. Returns 0 on
 * success, -EINVAL if mask is NULL or mask_words is 0 (or if no bit
 * is set in the mask -- sched_setaffinity rejects empty masks with
 * EINVAL), or the negated errno from sched_setaffinity.
 *
 * Thread-safety: safe to call concurrently; affects only the calling
 * thread. */
int topo_set_affinity(const uint64_t *mask, unsigned mask_words);

#ifdef __cplusplus
}
#endif

#endif /* TOPO_H */
