/* libtopo: CPU topology, NUMA, cache, and affinity primitives.
 *
 * On x86_64 the topology hierarchy (threads/core, cores/package,
 * num_packages) comes from CPUID leaf 0x1F (extended topology) when
 * available, else leaf 0xB (x2APIC enumeration). Both leaves expose a
 * subleaf-per-level walk: each subleaf reports the level type (SMT,
 * Core, Module, Tile, Die, ...) and the number of logical processors
 * at that level (i.e. sharing that level). The walk terminates when
 * EAX[4:0] == 0 && EBX[15:0] == 0.
 *
 * total_cpus and num_numa_nodes always come from the kernel
 * (sysconf(_SC_NPROCESSORS_ONLN) and /sys/devices/system/node/online),
 * because the kernel knows about offlined CPUs, cgroup cpuset
 * constraints, and NUMA emulation that CPUID cannot see.
 *
 * If CPUID is uninformative (broken BIOS, certain hypervisors), the
 * fields are filled from /sys/devices/system/cpu/cpuN/topology
 * (thread_siblings_list, core_siblings_list, physical_package_id).
 * Every field has a defensive default of 1 (or 0 for total_cpus
 * before the sysconf call) so callers can proceed even when both
 * CPUID and sysfs are unhelpful.
 *
 * CPU affinity wraps sched_setaffinity(2) / sched_getaffinity(2)
 * with a uint64_t[] mask layout (caller picks the word count). The
 * topo_pin and topo_pin_node helpers build single-CPU and node-wide
 * masks respectively.
 *
 * The fast getcpu path uses the RDPID instruction (F3 0F C7 /F8) on
 * x86_64 when CPUID.7.0:ECX[1] reports it available. RDPID reads
 * IA32_TSC_AUX, which on Linux holds "cpu_number | (node_id << 12)",
 * so the low 12 bits are the CPU number. RDPID is non-serializing
 * and costs ~3 cycles, vs. ~20-50 ns for the getcpu(2) syscall. The
 * fallback path uses syscall(SYS_getcpu, ...).
 *
 * Cache info comes from CPUID leaf 4 (deterministic cache
 * parameters): one subleaf per cache level (L1D, L1I, L2, L3, ...).
 * Each subleaf reports the line size, associativity, partitions,
 * sets, and the number of logical CPUs sharing an instance. Cache
 * size in bytes = (ways+1) * (partitions+1) * (line_size+1) *
 * (sets+1). The result reflects the calling CPU's caches; if the
 * caller asks about a different CPU we briefly pin to it via
 * sched_setaffinity so the CPUID executes there. */

#define _GNU_SOURCE
#include "topo.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#if defined(__x86_64__)
extern void     topo_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);
extern uint32_t topo_rdpid(void);
extern uint32_t topo_rdtscp_ecx(void);
#define TOPO_HAVE_X86 1
#else
#define TOPO_HAVE_X86 0
/* On non-x86_64 hosts there is no .asm helper. CPUID-driven paths are
 * compiled out and the library falls back to sysfs + getcpu(2). */
#endif

/* Width of the in-struct NUMA mask (matches topo_numa_node_t). */
#define TOPO_CPU_MASK_WORDS 4   /* 4 * 64 = 256 CPUs */

/* sysfs helpers */

/* Read up to bufsz-1 bytes from `path` into `buf`, NUL-terminate,
 * and trim trailing whitespace. Returns bytes read (>= 0) on success,
 * -1 on open/read error or empty file. */
static ssize_t topo_read_sysfs(const char *path, char *buf, size_t bufsz)
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

/* Parse a cpulist string like "0-1,3,5-7" into a bitmap of `words`
 * uint64_t words. Bit i is set iff CPU i is in the list. Returns 0 on
 * success, -1 on parse error or a CPU index >= words * 64. */
static int topo_parse_cpulist(const char *s, uint64_t *mask, unsigned words)
{
    for (unsigned i = 0; i < words; i++) mask[i] = 0;
    if (!s) return 0;

    const char *p = s;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (!isdigit((unsigned char)*p)) return -1;

        char *end;
        unsigned long a = strtoul(p, &end, 10);
        if (end == p) return -1;
        p = end;

        unsigned long b = a;
        if (*p == '-') {
            p++;
            if (!isdigit((unsigned char)*p)) return -1;
            b = strtoul(p, &end, 10);
            if (end == p) return -1;
            p = end;
        }

        if (b < a) return -1;
        for (unsigned long c = a; c <= b; c++) {
            if (c >= (unsigned long)words * 64) return -1;
            mask[c / 64] |= (UINT64_C(1) << (c % 64));
        }

        while (*p == ' ' || *p == '\t') p++;
    }
    return 0;
}

/* Popcount over `words` uint64_t words. */
static unsigned topo_popcount(const uint64_t *mask, unsigned words)
{
    unsigned n = 0;
    for (unsigned i = 0; i < words; i++) {
        n += (unsigned)__builtin_popcountll(mask[i]);
    }
    return n;
}

/* Parse a sysfs "list" file (e.g. /sys/devices/system/node/online,
 * which looks like "0" or "0-3" or "0,2-3,5") into an array of
 * unsigned IDs. Writes up to `max_ids` IDs into `out` and returns
 * the count that *would* have been written (so callers can detect
 * truncation). */
static unsigned topo_parse_id_list(const char *s, unsigned *out, unsigned max_ids)
{
    unsigned count = 0;
    const char *p = s;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (!isdigit((unsigned char)*p)) break;

        char *end;
        unsigned long a = strtoul(p, &end, 10);
        p = end;
        unsigned long b = a;
        if (*p == '-') {
            p++;
            if (!isdigit((unsigned char)*p)) break;
            b = strtoul(p, &end, 10);
            p = end;
        }

        if (b < a) break;
        for (unsigned long id = a; id <= b; id++) {
            if (count < max_ids && out) {
                out[count] = (unsigned)id;
            }
            count++;
        }
    }
    return count;
}

/* CPUID feature detection */

#if TOPO_HAVE_X86
/* Detect RDPID availability once and memoize. CPUID.(7,0):ECX[1]
 * indicates RDPID support. The result is constant per CPU so the
 * benign race (two threads racing to fill the cache) is harmless. */
static int topo_have_rdpid(void)
{
    static int cached = -1;
    if (cached >= 0) return cached;

    uint32_t regs[4] = {0, 0, 0, 0};
    topo_cpuid(0x7, 0x0, regs);
    int have = (regs[2] & (1u << 1)) ? 1 : 0;
    cached = have;
    return have;
}
#endif

/* topo_probe */

#if TOPO_HAVE_X86
/* Walk CPUID leaf 0x1F (preferred) or 0xB (fallback) for the SMT /
 * Core / Package hierarchy. Writes the parsed values into *info
 * (threads_per_core, cores_per_package, num_packages). total_cpus
 * must already be set by the caller. Returns 0 if at least the SMT
 * level was found, -1 if CPUID was uninformative. */
static int topo_probe_cpuid(topo_info_t *info)
{
    uint32_t regs[4] = {0, 0, 0, 0};
    topo_cpuid(0x0, 0x0, regs);
    uint32_t max_leaf = regs[0];
    if (max_leaf < 0xB) return -1;

    /* Leaf 0x1F (extended topology v2) supersedes 0xB and exposes
     * Module/Tile/Die levels. Use it when available; it has the same
     * subleaf format as 0xB. */
    uint32_t leaf = (max_leaf >= 0x1F) ? 0x1F : 0xB;

    unsigned lp_smt  = 0;   /* logical processors at the SMT level */
    unsigned lp_core = 0;   /* logical processors at the Core level */
    unsigned lp_high = 0;   /* max lp_at_level seen (== package LP count) */

    for (uint32_t sub = 0; sub < 16; sub++) {
        topo_cpuid(leaf, sub, regs);
        uint32_t eax_shift = regs[0] & 0x1Fu;
        uint32_t ebx_lp    = regs[1] & 0xFFFFu;
        /* ECX[7:0] = level number, ECX[15:8] = level type:
         *   0 = Invalid, 1 = SMT, 2 = Core, 3 = Module,
         *   4 = Tile, 5 = Die (Intel SDM Vol 4, Table 2-32). */
        uint32_t ecx_type  = (regs[2] >> 8) & 0xFFu;

        /* Terminator subleaf. */
        if (eax_shift == 0 && ebx_lp == 0) break;

        if (ebx_lp > lp_high) lp_high = ebx_lp;
        if (ecx_type == 1) lp_smt = ebx_lp;            /* SMT */
        else if (ecx_type == 2) lp_core = ebx_lp;      /* Core */
        /* Higher levels (Module/Tile/Die) are ignored: we only
         * report threads/core, cores/package, and num_packages. */
    }

    if (lp_smt > 0) {
        info->threads_per_core = lp_smt;
    }
    if (lp_core > 0 && lp_smt > 0) {
        info->cores_per_package = lp_core / lp_smt;
    } else if (lp_high > 0 && lp_smt > 0) {
        /* No Core level reported (single-core-per-package systems
         * sometimes omit it). Cores per package = 1. */
        info->cores_per_package = 1;
    }
    if (lp_high > 0 && info->total_cpus > 0) {
        info->num_packages = info->total_cpus / lp_high;
        if (info->num_packages == 0) info->num_packages = 1;
    }
    return (lp_smt > 0) ? 0 : -1;
}
#endif

/* Count distinct physical_package_id values across online CPUs by
 * reading /sys/devices/system/cpu/cpuN/topology/physical_package_id.
 * Used as the sysfs fallback for num_packages when CPUID is missing
 * or disagrees with sysfs. */
static unsigned topo_count_packages_sysfs(unsigned total_cpus)
{
    unsigned char seen[256];
    memset(seen, 0, sizeof(seen));
    unsigned count = 0;
    char path[160];
    char buf[32];

    for (unsigned c = 0; c < total_cpus && c < 1024; c++) {
        int n = snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/physical_package_id", c);
        if (n <= 0 || (size_t)n >= sizeof(path)) continue;
        ssize_t k = topo_read_sysfs(path, buf, sizeof(buf));
        if (k <= 0) continue;
        char *end;
        unsigned long pid = strtoul(buf, &end, 10);
        if (end == buf) continue;
        if (pid >= 256) continue;
        if (!seen[pid]) { seen[pid] = 1; count++; }
    }
    return count;
}

/* sysfs fallback for threads_per_core and cores_per_package, using
 * cpu0's topology/thread_siblings_list and topology/core_siblings_list.
 * Only fills fields that are still 0 in *info. */
static void topo_probe_sysfs(topo_info_t *info)
{
    char buf[1024];

    if (info->threads_per_core == 0) {
        ssize_t n = topo_read_sysfs(
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
            buf, sizeof(buf));
        if (n > 0) {
            uint64_t m[TOPO_CPU_MASK_WORDS];
            memset(m, 0, sizeof(m));
            if (topo_parse_cpulist(buf, m, TOPO_CPU_MASK_WORDS) == 0) {
                info->threads_per_core = topo_popcount(m, TOPO_CPU_MASK_WORDS);
            }
        }
    }

    if (info->cores_per_package == 0) {
        ssize_t n = topo_read_sysfs(
            "/sys/devices/system/cpu/cpu0/topology/core_siblings_list",
            buf, sizeof(buf));
        if (n > 0) {
            uint64_t m[TOPO_CPU_MASK_WORDS];
            memset(m, 0, sizeof(m));
            if (topo_parse_cpulist(buf, m, TOPO_CPU_MASK_WORDS) == 0) {
                unsigned lp_per_pkg =
                    topo_popcount(m, TOPO_CPU_MASK_WORDS);
                if (lp_per_pkg > 0 && info->threads_per_core > 0) {
                    info->cores_per_package = lp_per_pkg / info->threads_per_core;
                }
            }
        }
    }

    if (info->num_packages == 0) {
        info->num_packages = topo_count_packages_sysfs(info->total_cpus);
    }
}

int topo_probe(topo_info_t *info)
{
    if (!info) return -EINVAL;
    memset(info, 0, sizeof(*info));

    /* total_cpus from sysconf. On failure, leave at 0; the
     * topology-derived fallback below fills it in. */
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu > 0) info->total_cpus = (unsigned)ncpu;

    /* num_numa_nodes from /sys/devices/system/node/online. On a
     * non-NUMA system the file may be missing; we then probe for
     * /sys/devices/system/node/node0 (synthetic node0). */
    {
        char buf[256];
        ssize_t n = topo_read_sysfs("/sys/devices/system/node/online",
            buf, sizeof(buf));
        if (n > 0) {
            unsigned ids[64];
            unsigned cnt = topo_parse_id_list(buf, ids, 64);
            info->num_numa_nodes = cnt;
        }
        if (info->num_numa_nodes == 0) {
            if (access("/sys/devices/system/node/node0", F_OK) == 0) {
                info->num_numa_nodes = 1;
            }
        }
    }

    /* CPUID-driven topology (x86_64 only). */
#if TOPO_HAVE_X86
    topo_probe_cpuid(info);
#endif

    /* sysfs fallback for any field CPUID left at 0. */
    topo_probe_sysfs(info);

    /* Defensive defaults: every field >= 1, total_cpus >= 1. */
    if (info->threads_per_core  == 0) info->threads_per_core  = 1;
    if (info->cores_per_package == 0) info->cores_per_package = 1;
    if (info->num_packages      == 0) info->num_packages      = 1;
    if (info->num_numa_nodes    == 0) info->num_numa_nodes    = 1;
    if (info->total_cpus        == 0) {
        info->total_cpus = info->threads_per_core
                         * info->cores_per_package
                         * info->num_packages;
    }
    return 0;
}

/* topo_numa_nodes */

unsigned topo_numa_nodes(topo_numa_node_t *nodes, unsigned max_nodes)
{
    char buf[1024];
    ssize_t n = topo_read_sysfs("/sys/devices/system/node/online",
        buf, sizeof(buf));

    unsigned ids[256];
    unsigned count = 0;
    if (n > 0) {
        count = topo_parse_id_list(buf, ids, 256);
    }
    if (count == 0) {
        /* No online file. Probe /sys/devices/system/node/node0. */
        if (access("/sys/devices/system/node/node0", F_OK) == 0) {
            ids[0] = 0;
            count = 1;
        }
    }

    /* Fill caller's array up to max_nodes. */
    for (unsigned i = 0; i < count && i < max_nodes; i++) {
        topo_numa_node_t *nd = nodes ? &nodes[i] : NULL;
        if (!nd) continue;

        nd->node_id = (int)ids[i];
        memset(nd->cpu_mask, 0, sizeof(nd->cpu_mask));
        nd->cpu_count = 0;

        char path[160];
        int pn = snprintf(path, sizeof(path),
            "/sys/devices/system/node/node%u/cpulist", ids[i]);
        if (pn <= 0 || (size_t)pn >= sizeof(path)) continue;

        char b2[1024];
        ssize_t k = topo_read_sysfs(path, b2, sizeof(b2));
        if (k <= 0) continue;

        uint64_t mask[TOPO_CPU_MASK_WORDS];
        memset(mask, 0, sizeof(mask));
        if (topo_parse_cpulist(b2, mask, TOPO_CPU_MASK_WORDS) != 0) continue;
        memcpy(nd->cpu_mask, mask, sizeof(nd->cpu_mask));
        nd->cpu_count = topo_popcount(mask, TOPO_CPU_MASK_WORDS);
    }

    /* Synthetic single-node case when sysfs is entirely missing
     * (e.g. chroot without /sys mounted). Report one node with all
     * online CPUs in its mask so the caller's "pin to node 0" can
     * still work. */
    if (count == 0 && nodes && max_nodes > 0) {
        topo_numa_node_t *nd = &nodes[0];
        nd->node_id = 0;
        memset(nd->cpu_mask, 0, sizeof(nd->cpu_mask));
        nd->cpu_count = 0;

        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        unsigned total = (ncpu > 0) ? (unsigned)ncpu : 0;
        if (total > TOPO_CPU_MASK_WORDS * 64) {
            total = TOPO_CPU_MASK_WORDS * 64;
        }
        for (unsigned c = 0; c < total; c++) {
            nd->cpu_mask[c / 64] |= (UINT64_C(1) << (c % 64));
        }
        nd->cpu_count = total;
        count = 1;
    }

    return count;
}

/* topo_cache_info */

#if TOPO_HAVE_X86
/* Walk CPUID leaf 4 subleaves until one matches the requested level
 * (1/2/3) or the terminator subleaf (EAX[4:0] == 0) is reached.
 * Returns 0 and fills *cache on match, -ENOENT if no subleaf matches.
 * Reflects the calling CPU's caches. */
static int topo_cache_info_cpuid(unsigned level, topo_cache_t *cache)
{
    for (uint32_t sub = 0; sub < 32; sub++) {
        uint32_t regs[4] = {0, 0, 0, 0};
        topo_cpuid(0x4, sub, regs);

        uint32_t type = regs[0] & 0x1Fu;
        if (type == 0) break;   /* terminator subleaf */

        uint32_t lvl = (regs[0] >> 5) & 0x7u;
        if (lvl != level) continue;

        /* CPUID.4 EAX[25:14]: max number of addressable IDs for
         * logical processors sharing this cache, minus 1.
         * EBX[11:0]: system coherency line size minus 1 (bytes).
         * EBX[20:12]: physical line partitions minus 1.
         * EBX[31:22]: associativity minus 1.
         * ECX: number of sets minus 1. */
        unsigned line_size  = (regs[1] & 0xFFFu) + 1;
        unsigned partitions = ((regs[1] >> 12) & 0x3FFu) + 1;
        unsigned ways       = ((regs[1] >> 22) & 0x3FFu) + 1;
        unsigned sets       = regs[2] + 1;
        unsigned sharing    = ((regs[0] >> 14) & 0xFFFu) + 1;

        /* Total cache size in bytes = ways * partitions * line * sets.
         * Use 64-bit intermediate: a 64 MB L3 with 64-byte lines, 16
         * ways, 1 partition, 65536 sets fits in u64. */
        uint64_t size_bytes = (uint64_t)ways * partitions * line_size * sets;
        unsigned size_kb = (unsigned)(size_bytes / 1024u);

        cache->level     = lvl;
        cache->size_kb   = size_kb;
        cache->line_size = line_size;
        cache->sharing   = sharing;
        return 0;
    }
    return -ENOENT;
}
#endif

int topo_cache_info(unsigned cpu, unsigned level, topo_cache_t *cache)
{
    if (!cache) return -EINVAL;
    if (level == 0 || level > 3) return -EINVAL;
    memset(cache, 0, sizeof(*cache));

    /* Save the calling thread's current affinity so we can restore it.
     * If sched_getaffinity fails we proceed without the save/restore
     * (the pin is skipped in that case). */
    cpu_set_t orig_set;
    CPU_ZERO(&orig_set);
    int have_orig = (sched_getaffinity(0, sizeof(orig_set), &orig_set) == 0);

    /* If the caller asked about a CPU other than the current one,
     * briefly pin to it. On a homogeneous package this is a no-op
     * correctness-wise; on a heterogeneous one (e.g. big.LITTLE-style
     * x86 with different L3 per CCD) it gives the right answer. */
    unsigned cur = topo_getcpu();
    int pinned = 0;
    if (have_orig && cpu != cur && cpu < (unsigned)CPU_SETSIZE) {
        cpu_set_t pin;
        CPU_ZERO(&pin);
        CPU_SET((int)cpu, &pin);
        if (sched_setaffinity(0, sizeof(pin), &pin) == 0) {
            pinned = 1;
        }
        /* If pinning failed (e.g. cgroup cpuset excludes the CPU),
         * proceed with the calling CPU. The result will reflect the
         * calling CPU, not `cpu`. We document this in the API. */
    }

    int rc;
#if TOPO_HAVE_X86
    rc = topo_cache_info_cpuid(level, cache);
#else
    (void)cpu;
    rc = -ENOSYS;
#endif

    /* Restore the original affinity before returning. */
    if (pinned && have_orig) {
        (void)sched_setaffinity(0, sizeof(orig_set), &orig_set);
    }
    return rc;
}

/* topo_pin / topo_pin_node / topo_get_affinity / topo_set_affinity */

int topo_pin(unsigned cpu)
{
    if (cpu >= (unsigned)CPU_SETSIZE) return -EINVAL;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((int)cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) < 0) {
        return -errno;
    }
    return 0;
}

int topo_pin_node(int node_id)
{
    if (node_id < 0) return -EINVAL;

    char path[160];
    int pn = snprintf(path, sizeof(path),
        "/sys/devices/system/node/node%d/cpulist", node_id);
    if (pn <= 0 || (size_t)pn >= sizeof(path)) return -EINVAL;

    char buf[1024];
    ssize_t n = topo_read_sysfs(path, buf, sizeof(buf));
    if (n < 0) return -ENOENT;

    uint64_t mask[TOPO_CPU_MASK_WORDS];
    memset(mask, 0, sizeof(mask));
    if (topo_parse_cpulist(buf, mask, TOPO_CPU_MASK_WORDS) != 0) {
        return -EINVAL;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    int any = 0;
    for (unsigned c = 0; c < TOPO_CPU_MASK_WORDS * 64
                       && c < (unsigned)CPU_SETSIZE; c++) {
        if (mask[c / 64] & (UINT64_C(1) << (c % 64))) {
            CPU_SET((int)c, &set);
            any = 1;
        }
    }
    if (!any) return -EINVAL;

    if (sched_setaffinity(0, sizeof(set), &set) < 0) {
        return -errno;
    }
    return 0;
}

int topo_get_affinity(uint64_t *mask, unsigned mask_words)
{
    if (!mask || mask_words == 0) return -EINVAL;
    for (unsigned i = 0; i < mask_words; i++) mask[i] = 0;

    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) < 0) {
        return -errno;
    }
    for (unsigned c = 0; c < mask_words * 64
                       && c < (unsigned)CPU_SETSIZE; c++) {
        if (CPU_ISSET((int)c, &set)) {
            mask[c / 64] |= (UINT64_C(1) << (c % 64));
        }
    }
    return 0;
}

int topo_set_affinity(const uint64_t *mask, unsigned mask_words)
{
    if (!mask || mask_words == 0) return -EINVAL;

    cpu_set_t set;
    CPU_ZERO(&set);
    int any = 0;
    for (unsigned c = 0; c < mask_words * 64
                       && c < (unsigned)CPU_SETSIZE; c++) {
        if (mask[c / 64] & (UINT64_C(1) << (c % 64))) {
            CPU_SET((int)c, &set);
            any = 1;
        }
    }
    if (!any) return -EINVAL;   /* sched_setaffinity would return EINVAL */

    if (sched_setaffinity(0, sizeof(set), &set) < 0) {
        return -errno;
    }
    return 0;
}

/* topo_getcpu */

unsigned topo_getcpu(void)
{
#if TOPO_HAVE_X86
    if (topo_have_rdpid()) {
        /* RDPID returns IA32_TSC_AUX in ECX. On Linux the kernel
         * writes "cpu_number | (node_id << 12)" to IA32_TSC_AUX
         * (arch/x86/kernel/tsc.c), so the low 12 bits are the CPU
         * number. Masking also handles older kernels that wrote the
         * raw CPU number (with the high bits zero). */
        return topo_rdpid() & 0xFFFu;
    }
#endif
    /* Fallback: getcpu(2) syscall. glibc does not expose a wrapper
     * for getcpu on all builds; the raw syscall is portable. */
    unsigned int cpu = 0;
    if (syscall(SYS_getcpu, &cpu, NULL, NULL) == 0) {
        return cpu;
    }
    return 0;
}
