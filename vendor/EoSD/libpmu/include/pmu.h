#ifndef PMU_H
#define PMU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque. Wraps one perf_event_open fd plus the counter type it was opened
 * with. One context == one counter; if a caller needs cycles + instructions
 * simultaneously, open two contexts.
 *
 * On x86_64 the context also holds an mmap of the perf event metadata
 * page used by the rdpmc fast path (see pmu_read). */
typedef struct pmu_ctx pmu_ctx_t;

typedef enum {
    PMU_CYCLES,         /* unhalted core cycles (PERF_COUNT_HW_CPU_CYCLES)  */
    PMU_INSTRUCTIONS,   /* instructions retired  (PERF_COUNT_HW_INSTRUCTIONS) */
    PMU_CACHE_MISSES,   /* last-level cache misses (PERF_COUNT_HW_CACHE_MISSES) */
} pmu_counter_type_t;

typedef enum {
    PMU_OK              =  0,
    PMU_ERR_INVALID     = -1,   /* NULL ctx, NULL out_value, or unknown type */
    PMU_ERR_PERM        = -2,   /* perf_event_open denied (EPERM/EACCES/ENOSYS) */
    PMU_ERR_UNAVAILABLE = -3,   /* counter type not supported (ENODEV/ENOENT/EINVAL) */
} pmu_err_t;

/* Opens a counter of the given type for the calling thread. Internally calls
 * perf_event_open(2) with pid=0 (calling thread), cpu=-1 (follows the thread
 * across CPUs), PERF_TYPE_HARDWARE, and the config value mapped from `which`.
 * The event is opened disabled; pmu_start enables it.
 *
 * Returns a heap-allocated context on success (caller frees with pmu_close).
 *
 * Graceful degradation: if perf_event_open(2) fails with EACCES, EPERM, or
 * ENOSYS (containerized environments with restrictive seccomp, or
 * perf_event_paranoid >= 3), pmu_open does NOT return NULL. Instead it
 * returns a heap-allocated dummy context and, if out_err is non-NULL, writes
 * PMU_ERR_PERM to *out_err so the caller can tell degradation happened. A
 * dummy context has fd == -1; pmu_start is a no-op, pmu_read and
 * pmu_stop_and_read set *out_value = 0, and pmu_is_available returns 0.
 * Callers that need real counter values should check pmu_is_available(ctx)
 * after pmu_open and fall back to an alternative timing source (e.g. rdtsc)
 * when it returns 0. Callers that only need a non-NULL ctx to flow through
 * the API (so they do not have to special-case NULL everywhere) can use the
 * dummy transparently.
 *
 * On other failures pmu_open returns NULL and, if out_err is non-NULL,
 * writes one of PMU_ERR_INVALID (unknown counter type) or PMU_ERR_UNAVAILABLE
 * (ENODEV/ENOENT/EINVAL -- counter type not supported on this CPU -- or malloc
 * failure) to *out_err.
 *
 * Thread-safety: safe to call from multiple threads concurrently; each call
 * produces an independent fd and context. */
pmu_ctx_t *pmu_open(pmu_counter_type_t which, pmu_err_t *out_err);

/* Returns 1 if the PMU context is backed by a real perf_event_open fd
 * (counter reads return real values), 0 if it is a dummy context (counter
 * reads return 0). Useful for applications that want to degrade gracefully
 * when perf is denied. Returns 0 if ctx is NULL. */
int pmu_is_available(const pmu_ctx_t *ctx);

/* Resets the counter to zero (PERF_EVENT_IOC_RESET) then enables it
 * (PERF_EVENT_IOC_ENABLE). Returns PMU_OK on success, PMU_ERR_INVALID if
 * ctx is NULL, PMU_ERR_UNAVAILABLE on ioctl failure.
 *
 * If ctx is a dummy context (pmu_is_available(ctx) == 0), pmu_start is a
 * no-op and returns PMU_OK without issuing any ioctl.
 *
 * Calling pmu_start twice in a row RESETs the counter on the second call:
 * the count accumulated since the first pmu_start is lost. Callers that
 * want a cumulative count must read before re-starting.
 *
 * Thread-safety: not safe to call concurrently with itself or pmu_stop_and_read
 * on the same ctx without external coordination; the two ioctls are not
 * atomic with respect to each other. */
int pmu_start(pmu_ctx_t *ctx);

/* Disables the counter (PERF_EVENT_IOC_DISABLE) and reads the current
 * value via pmu_read. Returns PMU_OK on success. After DISABLE the
 * counter is frozen; subsequent pmu_read calls return the same final
 * value until pmu_start re-enables it.
 *
 * If ctx is a dummy context, pmu_stop_and_read sets *out_value = 0 and
 * returns PMU_OK without issuing any ioctl or syscall.
 *
 * Thread-safety: not safe to call concurrently with pmu_start or itself on
 * the same ctx. */
int pmu_stop_and_read(pmu_ctx_t *ctx, uint64_t *out_value);

/* Reads the current value without stopping. For periodic sampling of a
 * long-running counter.
 *
 * On x86_64, uses the rdpmc fast path (no syscall) when the kernel exposes
 * a non-zero perf_event_mmap_page->index (perf_event_paranoid <= 1 or
 * CAP_PERFMON); otherwise falls back to read(2). Both paths scale the raw
 * count by time_enabled/time_running under multiplexing so they agree.
 *
 * Before pmu_start the event is disabled; pmu_read returns PMU_OK with
 * *out_value == 0 in that case (not an error).
 *
 * If ctx is a dummy context, pmu_read sets *out_value = 0 and returns
 * PMU_OK without issuing any syscall or rdpmc instruction.
 *
 * Returns PMU_OK on success, PMU_ERR_INVALID if ctx or out_value is NULL,
 * PMU_ERR_UNAVAILABLE if the read(2) fallback fails.
 *
 * Thread-safety: safe to call concurrently with itself on the same ctx; the
 * kernel serializes reads of the counter value. Not safe to call concurrently
 * with pmu_start (which RESETs) or pmu_close on the same ctx. */
int pmu_read(pmu_ctx_t *ctx, uint64_t *out_value);

/* Closes the perf fd and frees the context. Safe to call with NULL ctx
 * (no-op). If the counter is running, close(fd) disables it and the
 * final value is lost; call pmu_stop_and_read first if the value is
 * wanted. If ctx is a dummy context, pmu_close just frees the context
 * (there is no fd to close and no mmap to unmap).
 * Thread-safety: not safe to call concurrently with any other
 * operation on the same ctx. */
void pmu_close(pmu_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PMU_H */
