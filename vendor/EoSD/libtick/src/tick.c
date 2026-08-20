/*
 * libtick implementation. See include/tick.h for the public contract and
 * DESIGN.md for the rationale.
 *
 * Calibration: CPUID.15H first (TSC frequency leaf, zero-cost); fallback to
 * a ~10ms rdtsc-vs-clock_gettime busy wait. The result is stored as tsc_hz
 * (ticks per second) and tsc_per_ns (Q20 fixed-point, ticks per ns * 2^20).
 *
 * tick_now() converts (rdtsc - tsc_base) to ns and adds monotonic_base_ns,
 * producing a CLOCK_MONOTONIC-aligned value. The fast path uses the Q20
 * reciprocal for deltas < 2^40 ticks (~97 min at 3 GHz); the slow path uses
 * a two-step divide that is overflow-safe for any realistic ctx lifetime.
 *
 * HZ detection (/proc/interrupts LOC line, 100ms window, fallback 1000)
 * informs the spin-vs-syscall threshold in tick_sleep_until: clock_nanosleep
 * gets close, then a brief TSC spin tightens the last sub-jiffy gap.
 *
 * Registry (v0.2): two binary min-heaps keyed by deadline_ns, one for
 * TICK_MODE_POLL timers and one for TICK_MODE_CALLBACK timers. The public
 * tick_timer_id_t remains a stable slot index into a fixed-size slot array;
 * each slot carries its current heap_pos so cancel is O(log n) without a
 * linear search. Free slots are tracked via a singly-linked free list stored
 * in the slot's link field (reused as next_free when not in_use).
 *
 *   - tick_register:  pop a free slot, fill it, heap_push (sift up). O(log n).
 *   - tick_cancel:    heap_remove_at(slot.heap_pos), push slot to free list. O(log n).
 *   - tick_wait_next: peek poll_heap root (min deadline), sleep until it, pop
 *                     all expired. O(k log n) where k = expired count.
 *   - tick_run_pending: pop all expired from cb_heap into a malloc'd array of
 *                     popped slot indices (in deadline order), then dispatch
 *                     by iterating that array. O(k log n + k) where k = fired
 *                     count. The dispatch visits only the k popped slots, not
 *                     the full capacity; callbacks fire in deadline order.
 *
 * Not reentrant: tick_register / tick_cancel from inside a tick_run_pending
 * callback is undefined behavior (see DESIGN.md). The snapshot taken in
 * pass 1 of tick_run_pending (move matched slots from in_use=1 to firing=1
 * and pop them from cb_heap, recording each into a dispatch array) keeps
 * the dispatch set stable for the duration of pass 2; a callback that
 * (incorrectly) mutates the registry cannot affect which slots pass 2
 * visits or the order in which they are visited.
 *
 * Drift defense (v0.3): tick_now triggers a low-frequency recalibration
 * check every 1024 calls or every 5 s, whichever comes first. The check
 * reads clock_gettime(CLOCK_MONOTONIC) and rdtsc close together, computes
 * the TSC-predicted monotonic time from the current calibration, and
 * compares it to the actual clock_gettime value. If |drift| exceeds
 * drift_threshold_ns (default 1 ms), the (monotonic_base_ns, tsc_base)
 * pair is replaced with the freshly observed pair; tsc_hz and tsc_per_ns
 * are unchanged (TSC frequency does not drift, only the offset). This
 * covers multi-socket NUMA skew and C-state power-saving scenarios where
 * a thread that migrates between cores may see the TSC shift slightly.
 * The check calls clock_gettime (a syscall) and is therefore NOT
 * async-signal-safe; tick_now inherits this property on the ~1-in-1024
 * call that triggers the check.
 */

#define _GNU_SOURCE
#include <tick.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>

/* Assembly helpers in tick_x86_64.asm. */
extern uint64_t tick_rdtsc(void);
extern uint64_t tick_rdtscp(void);
extern void     tick_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);

/* 128-bit type for the magic-number divide in tick_now's fast path.
 * __extension__ silences -pedantic on the GCC __int128 extension. */
__extension__ typedef unsigned __int128 tick_u128;

/* Granlund-Montgomery magic for an exact 64-bit-by-32-bit division.
 * For divisor d in (2^L, 2^(L+1)] (i.e. L = floor(log2(d)) and d not a
 * power of two), m = ceil(2^(64+L) / d) fits in 64 bits (in [2^63, 2^64))
 * and floor(n / d) = floor(n * m / 2^(64+L)) for all 64-bit n, where the
 * multiplication is 128-bit. For d = 2^L (power of two), m = 1 and
 * shift = L gives floor(n / d) = n >> L.
 *
 * Replaces the 20-30-cycle DIVQ in tick_now's fast path with a 3-cycle
 * MULQ + 1-cycle SHR. Computed once at calibration; the hot path is a
 * single inline multiply+shift. */
struct tick_div_magic {
    uint64_t mul;
    int      shift;
};

static struct tick_div_magic compute_div_magic(uint32_t d) {
    struct tick_div_magic r = { 0, 0 };
    if (d == 0) {
        return r;
    }
    int L = 31 - __builtin_clz(d);
    if ((d & (d - 1)) == 0) {
        /* power of two: m=1, shift=L => n/d = n>>L */
        r.mul = 1;
        r.shift = L;
        return r;
    }
    int s = 64 + L;
    tick_u128 pow_s = (tick_u128)1 << s;
    /* ceil(2^s / d) = (2^s + d - 1) / d, using 128-bit math. */
    r.mul = (uint64_t)((pow_s + d - 1) / d);
    r.shift = s;
    return r;
}

static inline uint64_t div_via_magic(uint64_t n, struct tick_div_magic m) {
    return (uint64_t)(((tick_u128)n * m.mul) >> m.shift);
}

/* ---- Thread-local error reporting ---- */

static _Thread_local char last_error_buf[160];

static void set_last_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(last_error_buf, sizeof(last_error_buf), fmt, ap);
    va_end(ap);
}

/* ---- Internal types ---- */

/* Timer slot. The link field is overloaded:
 *   - in_use=1: link is the slot's index in its mode's heap (heap_pos).
 *   - in_use=0, firing=0: link is the next free slot index in the free list
 *     (SIZE_MAX terminates the list).
 *   - in_use=0, firing=1: transient state during tick_run_pending dispatch;
 *     link is not meaningful (the slot has been popped from cb_heap but not
 *     yet returned to the free list).
 *
 * firing=1 is never visible to the caller; it is only set between the two
 * passes of tick_run_pending to keep the dispatch snapshot stable. */
struct tick_timer {
    uint64_t         deadline_ns;
    tick_fire_mode_t mode;
    tick_callback_fn cb;
    void            *user_data;
    size_t           link;     /* heap_pos when in_use; next_free when free */
    int              in_use;
    int              firing;
};

struct tick_ctx {
    uint64_t tsc_hz;            /* ticks per second */
    uint64_t tsc_per_ns;        /* Q20 fixed-point: (tsc_hz << 20) / 1e9 */
    struct tick_div_magic tsc_per_ns_magic;  /* magic for fast divide */
    uint64_t monotonic_base_ns; /* CLOCK_MONOTONIC ns at calibration */
    uint64_t tsc_base;          /* TSC ticks at calibration */
    int      runtime_hz;        /* detected kernel HZ (spin-vs-syscall input) */
    uint64_t (*read_tsc)(void); /* tick_rdtsc or tick_rdtscp */
    size_t   capacity;
    struct tick_timer *timers;  /* slot storage, indexed by tick_timer_id_t */
    size_t  *poll_heap;         /* min-heap of POLL-mode slot indices */
    size_t   poll_heap_size;
    size_t  *cb_heap;           /* min-heap of CALLBACK-mode slot indices */
    size_t   cb_heap_size;
    size_t   free_head;         /* head of free-slot list, or SIZE_MAX */
    size_t   timer_count;       /* poll_heap_size + cb_heap_size (for diagnostics) */

    /* Drift defense (v0.3). last_drift_check_ns is the monotonic ns recorded
     * by the most recent tick_ctx_check_drift call; tick_now compares its own
     * result against this to decide whether the 5 s timer has elapsed.
     * drift_threshold_ns defaults to 1 ms. tick_now_call_count is reset to 0
     * on every check (automatic or manual). */
    uint64_t last_drift_check_ns;     /* monotonic ns at last drift check */
    uint64_t drift_threshold_ns;      /* recalibrate if |drift| exceeds this */
    uint64_t max_observed_drift_ns;   /* high-water mark of |drift| */
    uint64_t drift_check_count;       /* total tick_ctx_check_drift calls */
    uint64_t drift_recalibration_count; /* total recalibrations performed */
    uint64_t tick_now_call_count;     /* tick_now calls since last auto-check */
};

/* ---- Feature detection ---- */

static int has_rdtscp(void) {
    uint32_t out[4];
    tick_cpuid(0x80000000U, 0, out);
    if (out[0] < 0x80000001U) {
        return 0;
    }
    tick_cpuid(0x80000001U, 0, out);
    return (int)((out[3] >> 27) & 1u);
}

/* CPUID leaf 15H: TSC frequency = ECX * EBX / EAX, where ECX is the core
 * crystal clock in Hz. Returns 0 if the leaf is absent, EAX is zero, or
 * ECX is zero (ratio known but absolute frequency not enumerated). */
static uint64_t cpuid_15h_tsc_hz(void) {
    uint32_t max_leaf[4];
    tick_cpuid(0x0U, 0, max_leaf);
    if (max_leaf[0] < 0x15U) {
        return 0;
    }
    uint32_t out[4];
    tick_cpuid(0x15U, 0, out);
    uint32_t eax = out[0];
    uint32_t ebx = out[1];
    uint32_t ecx = out[2];
    if (eax == 0 || ecx == 0) {
        return 0;
    }
    return (uint64_t)ecx * (uint64_t)ebx / (uint64_t)eax;
}

/* Fallback: measure TSC frequency by pairing rdtsc with clock_gettime over a
 * ~5ms busy wait, taking the minimum of two samples (10ms total) to suppress
 * preemption noise. */
static uint64_t measure_tsc_hz_via_clock(uint64_t (*read_tsc)(void)) {
    const long target_ns = 5 * 1000 * 1000;  /* 5ms per sample */
    uint64_t best_hz = 0;
    for (int sample = 0; sample < 2; sample++) {
        struct timespec t0, t1;
        uint64_t c0, c1;
        long ns;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        c0 = read_tsc();
        for (;;) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ns = (long)((t1.tv_sec - t0.tv_sec) * 1000000000L
                      + (t1.tv_nsec - t0.tv_nsec));
            if (ns >= target_ns) {
                break;
            }
        }
        c1 = read_tsc();
        ns = (long)((t1.tv_sec - t0.tv_sec) * 1000000000L
                  + (t1.tv_nsec - t0.tv_nsec));
        if (ns <= 0) {
            continue;
        }
        uint64_t hz = (c1 - c0) * 1000000000ULL / (uint64_t)ns;
        if (best_hz == 0 || hz < best_hz) {
            best_hz = hz;
        }
    }
    return best_hz;
}

/* ---- HZ detection ---- */

/* Parse /proc/interrupts and return the sum of per-CPU counts on the LOC
 * (local timer interrupts) line. Returns -1 if not found or unparseable. */
static long read_loc_sum(FILE *f) {
    char line[1024];
    rewind(f);
    /* First line is the CPU header; skip it. */
    if (!fgets(line, sizeof(line), f)) {
        return -1;
    }
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "LOC:", 4) != 0) {
            continue;
        }
        long sum = 0;
        const char *p = line + 4;
        int parsed_any = 0;
        while (*p) {
            while (*p == ' ' || *p == '\t') {
                p++;
            }
            if (!isdigit((unsigned char)*p)) {
                break;
            }
            char *end;
            long n = strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            sum += n;
            parsed_any = 1;
            p = end;
        }
        if (!parsed_any) {
            return -1;
        }
        return sum;
    }
    return -1;
}

/* Detect runtime HZ by counting LOC interrupts over a 100ms window.
 * Fallback to 1000 if /proc/interrupts is absent, unparseable, or the
 * measured value is outside [100, 10000]. */
static int detect_runtime_hz(void) {
    FILE *f = fopen("/proc/interrupts", "r");
    if (!f) {
        return 1000;
    }
    long count1 = read_loc_sum(f);
    if (count1 < 0) {
        fclose(f);
        return 1000;
    }

    struct timespec ts = { 0, 100 * 1000 * 1000L };  /* 100ms */
    nanosleep(&ts, NULL);

    long count2 = read_loc_sum(f);
    fclose(f);
    if (count2 < 0 || count2 < count1) {
        return 1000;
    }
    long delta = count2 - count1;
    if (delta <= 0) {
        return 1000;
    }
    /* delta counts over 100ms; *10 for per-second. */
    int hz = (int)(delta * 10);
    if (hz < 100 || hz > 10000) {
        return 1000;
    }
    return hz;
}

/* ---- Free-slot list ---- */

/* Initializes the free list as 0 -> 1 -> ... -> (capacity-1) -> SIZE_MAX.
 * The first tick_register calls therefore return ids 0, 1, 2, ... in
 * registration order, matching v0.1's flat-array behavior for the common
 * fill-from-empty case. */
static void free_list_init(struct tick_ctx *ctx) {
    ctx->free_head = 0;
    for (size_t i = 0; i < ctx->capacity; i++) {
        ctx->timers[i].link    = (i + 1 < ctx->capacity) ? (i + 1) : SIZE_MAX;
        ctx->timers[i].in_use  = 0;
        ctx->timers[i].firing  = 0;
    }
}

static size_t free_list_pop(struct tick_ctx *ctx) {
    if (ctx->free_head == SIZE_MAX) {
        return SIZE_MAX;
    }
    size_t slot = ctx->free_head;
    ctx->free_head = ctx->timers[slot].link;
    return slot;
}

static void free_list_push(struct tick_ctx *ctx, size_t slot) {
    ctx->timers[slot].link = ctx->free_head;
    ctx->free_head = slot;
}

/* ---- Binary min-heap operations ---- */
/* Each heap is an array of slot indices, ordered by the deadline_ns of the
 * slot each index points at. The heap is keyed by deadline; mode is fixed
 * per-heap (poll_heap holds only POLL-mode slots, cb_heap only CALLBACK). */

static int heap_less(struct tick_ctx *ctx, const size_t *heap,
                     size_t a, size_t b) {
    return ctx->timers[heap[a]].deadline_ns < ctx->timers[heap[b]].deadline_ns;
}

static void heap_swap(struct tick_ctx *ctx, size_t *heap, size_t a, size_t b) {
    size_t sa = heap[a];
    size_t sb = heap[b];
    heap[a] = sb;   /* sb now lives at index a */
    heap[b] = sa;   /* sa now lives at index b */
    ctx->timers[sb].link = a;
    ctx->timers[sa].link = b;
}

static void heap_sift_up(struct tick_ctx *ctx, size_t *heap, size_t pos) {
    while (pos > 0) {
        size_t parent = (pos - 1) / 2;
        if (heap_less(ctx, heap, pos, parent)) {
            heap_swap(ctx, heap, pos, parent);
            pos = parent;
        } else {
            break;
        }
    }
}

static void heap_sift_down(struct tick_ctx *ctx, size_t *heap,
                           size_t size, size_t pos) {
    for (;;) {
        size_t left  = 2 * pos + 1;
        size_t right = 2 * pos + 2;
        size_t smallest = pos;
        if (left < size && heap_less(ctx, heap, left, smallest)) {
            smallest = left;
        }
        if (right < size && heap_less(ctx, heap, right, smallest)) {
            smallest = right;
        }
        if (smallest == pos) {
            break;
        }
        heap_swap(ctx, heap, pos, smallest);
        pos = smallest;
    }
}

static void heap_push(struct tick_ctx *ctx, size_t *heap, size_t *size,
                      size_t slot) {
    heap[*size] = slot;
    ctx->timers[slot].link = *size;
    (*size)++;
    heap_sift_up(ctx, heap, *size - 1);
}

static size_t heap_pop_root(struct tick_ctx *ctx, size_t *heap, size_t *size) {
    size_t root = heap[0];
    (*size)--;
    if (*size > 0) {
        heap[0] = heap[*size];
        ctx->timers[heap[0]].link = 0;
        heap_sift_down(ctx, heap, *size, 0);
    }
    return root;
}

/* Removes the element at pos by moving the last element into its place and
 * restoring the heap property. The moved element may need to sift up (if it
 * is smaller than its new parent) or down (if it is larger than a child);
 * it cannot need both. Caller must ensure pos < *size. */
static void heap_remove_at(struct tick_ctx *ctx, size_t *heap, size_t *size,
                           size_t pos) {
    (*size)--;
    if (pos == *size) {
        /* Removed the last element; nothing to fix up. */
        return;
    }
    heap[pos] = heap[*size];
    ctx->timers[heap[pos]].link = pos;
    if (pos > 0 && heap_less(ctx, heap, pos, (pos - 1) / 2)) {
        heap_sift_up(ctx, heap, pos);
    } else {
        heap_sift_down(ctx, heap, *size, pos);
    }
}

/* ---- Lifecycle ---- */

tick_ctx_t *tick_ctx_create(size_t capacity) {
    last_error_buf[0] = '\0';

    tick_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_last_error("calloc failed for tick_ctx");
        return NULL;
    }

    /* Select the TSC read primitive. rdtscp is preferred when available;
     * it is partially serializing and avoids the lfence-before-rdtsc that
     * the lfence;rdtsc path needs. */
    ctx->read_tsc = has_rdtscp() ? tick_rdtscp : tick_rdtsc;

    /* 1. Probe CPUID.15H for a zero-cost TSC frequency. */
    ctx->tsc_hz = cpuid_15h_tsc_hz();

    /* 2. Fallback: ~10ms rdtsc-vs-clock_gettime calibration. */
    if (ctx->tsc_hz == 0) {
        ctx->tsc_hz = measure_tsc_hz_via_clock(ctx->read_tsc);
    }

    if (ctx->tsc_hz == 0) {
        set_last_error("calibration failed: could not determine TSC frequency");
        free(ctx);
        return NULL;
    }

    /* Sanity bounds: 100 MHz to 10 GHz. Rejects wildly inconsistent reads. */
    if (ctx->tsc_hz < 100000000ULL || ctx->tsc_hz > 10000000000ULL) {
        set_last_error("calibration failed: TSC frequency %lu Hz out of bounds",
                       (unsigned long)ctx->tsc_hz);
        free(ctx);
        return NULL;
    }

    /* Q20 fixed-point: ticks per ns * 2^20. Used by the tick_now fast path. */
    ctx->tsc_per_ns = (ctx->tsc_hz << 20) / 1000000000ULL;

    /* Precompute the magic multiplier+shift for the fast-path divide
     * (delta << 20) / tsc_per_ns. The Q20 divisor always fits in 32 bits
     * (max ~10.5M for a 10 GHz TSC), so the magic fits in 64 bits. */
    ctx->tsc_per_ns_magic = compute_div_magic((uint32_t)ctx->tsc_per_ns);

    /* 3. Capture the (CLOCK_MONOTONIC ns, TSC ticks) base pair. Interleave
     *    the two rdtsc reads around clock_gettime and average to estimate
     *    the TSC at the instant clock_gettime's internal read occurred. */
    struct timespec ts;
    uint64_t tsc_before = ctx->read_tsc();
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t tsc_after  = ctx->read_tsc();
    ctx->tsc_base          = (tsc_before + tsc_after) / 2;
    ctx->monotonic_base_ns = (uint64_t)ts.tv_sec * 1000000000ULL
                           + (uint64_t)ts.tv_nsec;

    /* Drift defense defaults. last_drift_check_ns is seeded from the same
     * clock_gettime value as monotonic_base_ns so the 5 s timer in tick_now
     * starts counting from calibration time. */
    ctx->last_drift_check_ns        = ctx->monotonic_base_ns;
    ctx->drift_threshold_ns         = 1000000ULL;  /* 1 ms */
    ctx->max_observed_drift_ns      = 0;
    ctx->drift_check_count          = 0;
    ctx->drift_recalibration_count  = 0;
    ctx->tick_now_call_count        = 0;

    /* 4. Detect runtime HZ (spin-vs-syscall input). */
    ctx->runtime_hz = detect_runtime_hz();

    /* 5. Allocate the registry: slot array + two heaps. */
    ctx->capacity       = capacity;
    ctx->poll_heap_size = 0;
    ctx->cb_heap_size   = 0;
    ctx->timer_count    = 0;
    ctx->free_head      = SIZE_MAX;
    if (capacity > 0) {
        ctx->timers    = calloc(capacity, sizeof(struct tick_timer));
        ctx->poll_heap = calloc(capacity, sizeof(size_t));
        ctx->cb_heap   = calloc(capacity, sizeof(size_t));
        if (!ctx->timers || !ctx->poll_heap || !ctx->cb_heap) {
            set_last_error("calloc failed for timer registry (%zu slots)", capacity);
            free(ctx->timers);
            free(ctx->poll_heap);
            free(ctx->cb_heap);
            free(ctx);
            return NULL;
        }
        free_list_init(ctx);
    }

    return ctx;
}

void tick_ctx_destroy(tick_ctx_t *ctx) {
    /* Does not clear last_error_buf: the caller may have just had a
     * tick_ctx_create failure and still wants to read tick_last_error()
     * after calling tick_ctx_destroy(NULL). */
    if (!ctx) {
        return;
    }
    free(ctx->timers);
    free(ctx->poll_heap);
    free(ctx->cb_heap);
    free(ctx);
}

const char *tick_last_error(void) {
    return last_error_buf;
}

/* ---- Time conversion ---- */

uint64_t tick_now(tick_ctx_t *ctx) {
    last_error_buf[0] = '\0';
    if (!ctx) {
        return 0;
    }
    uint64_t tsc   = ctx->read_tsc();
    uint64_t delta = tsc - ctx->tsc_base;
    uint64_t ns;
    if (delta < (1ULL << 40)) {
        /* Fast path: Q20 reciprocal via magic-number divide. Replaces the
         * ~20-cycle DIVQ with a 3-cycle MULQ + 1-cycle SHR. Overflow-safe
         * for deltas < 2^40 ticks (~97 min at 3 GHz, ~58 min at 5 GHz). */
        ns = div_via_magic(delta << 20, ctx->tsc_per_ns_magic);
    } else {
        /* Slow path: two-step divide. Overflow-safe for any realistic ctx
         * lifetime (delta * 1e9 < 2^64 for delta < ~580 years at 3 GHz). */
        uint64_t s = delta / ctx->tsc_hz;
        uint64_t r = delta % ctx->tsc_hz;
        ns = s * 1000000000ULL + (r * 1000000000ULL) / ctx->tsc_hz;
    }
    uint64_t result = ns + ctx->monotonic_base_ns;

    /* Drift auto-check: every 1024 calls or every 5 s, whichever comes
     * first. The counter is the primary trigger (one increment, one
     * compare); the 5 s timer is a fallback for low-frequency callers and
     * only evaluates when the counter has not yet tripped. Both branches
     * resolve in a handful of cycles on the fast path; the actual
     * recalibration work (clock_gettime + rdtsc) runs at most once per
     * 1024 calls, so the amortized cost is < 1 ns per tick_now. */
    ctx->tick_now_call_count++;
    if (__builtin_expect(ctx->tick_now_call_count >= 1024, 0) ||
        (result >= ctx->last_drift_check_ns &&
         (result - ctx->last_drift_check_ns) >= 5000000000ULL)) {
        ctx->tick_now_call_count = 0;
        (void)tick_ctx_check_drift(ctx);
    }

    return result;
}

uint64_t tick_from_timespec(const struct timespec *ts) {
    last_error_buf[0] = '\0';
    if (!ts) {
        return 0;
    }
    return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

void tick_to_timespec(uint64_t ns, struct timespec *out) {
    last_error_buf[0] = '\0';
    if (!out) {
        return;
    }
    out->tv_sec  = (time_t)(ns / 1000000000ULL);
    out->tv_nsec = (long)(ns % 1000000000ULL);
}

/* ---- Drift defense ---- */

uint64_t tick_ctx_check_drift(tick_ctx_t *ctx) {
    last_error_buf[0] = '\0';
    if (!ctx) {
        return 0;
    }

    /* Read clock_gettime and rdtsc as close together as possible. The same
     * interleave-and-average trick as calibration estimates the TSC at the
     * instant clock_gettime's internal read occurred, so the (actual_ns,
     * tsc_now) pair refers to a single instant. */
    uint64_t tsc_before = ctx->read_tsc();
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t tsc_after = ctx->read_tsc();
    uint64_t tsc_now   = (tsc_before + tsc_after) / 2;
    uint64_t actual_ns = (uint64_t)ts.tv_sec * 1000000000ULL
                       + (uint64_t)ts.tv_nsec;

    /* TSC-predicted monotonic time using the current calibration. Mirrors
     * tick_now's conversion so the two stay byte-for-byte consistent. */
    uint64_t delta = tsc_now - ctx->tsc_base;
    uint64_t predicted_ns;
    if (delta < (1ULL << 40)) {
        predicted_ns = div_via_magic(delta << 20, ctx->tsc_per_ns_magic);
    } else {
        uint64_t s = delta / ctx->tsc_hz;
        uint64_t r = delta % ctx->tsc_hz;
        predicted_ns = s * 1000000000ULL + (r * 1000000000ULL) / ctx->tsc_hz;
    }
    predicted_ns += ctx->monotonic_base_ns;

    /* Drift = |actual - predicted|. */
    uint64_t drift;
    if (actual_ns >= predicted_ns) {
        drift = actual_ns - predicted_ns;
    } else {
        drift = predicted_ns - actual_ns;
    }

    ctx->drift_check_count++;
    if (drift > ctx->max_observed_drift_ns) {
        ctx->max_observed_drift_ns = drift;
    }
    ctx->last_drift_check_ns = actual_ns;

    /* Recalibrate if drift exceeds threshold. Replaces the base pair with
     * the freshly observed (clock_gettime, rdtsc) pair; tsc_hz and
     * tsc_per_ns are unchanged (TSC frequency does not drift, only the
     * offset between TSC ticks and CLOCK_MONOTONIC ns). */
    if (drift > ctx->drift_threshold_ns) {
        ctx->monotonic_base_ns = actual_ns;
        ctx->tsc_base          = tsc_now;
        ctx->drift_recalibration_count++;
    }

    return drift;
}

void tick_ctx_drift_stats(tick_ctx_t *ctx, uint64_t *max_drift,
                          uint64_t *checks, uint64_t *recalibrations) {
    last_error_buf[0] = '\0';
    if (!ctx) {
        if (max_drift)       *max_drift       = 0;
        if (checks)          *checks          = 0;
        if (recalibrations)  *recalibrations  = 0;
        return;
    }
    if (max_drift)       *max_drift      = ctx->max_observed_drift_ns;
    if (checks)          *checks         = ctx->drift_check_count;
    if (recalibrations)  *recalibrations = ctx->drift_recalibration_count;
}

/* Test-only hook: not declared in tick.h. Tests link against this symbol
 * to corrupt the calibration and exercise the drift-check / recalibration
 * path. Shifts monotonic_base_ns by drift_ns, simulating TSC skew without
 * touching tsc_hz / tsc_per_ns (the frequency is correct, only the offset
 * is wrong). A subsequent tick_ctx_check_drift observes |drift_ns| of drift
 * and, if it exceeds the threshold, recalibrates. */
void tick_test_inject_drift_ns(tick_ctx_t *ctx, int64_t drift_ns) {
    if (!ctx) {
        return;
    }
    if (drift_ns >= 0) {
        ctx->monotonic_base_ns += (uint64_t)drift_ns;
    } else {
        uint64_t mag = (uint64_t)(-drift_ns);
        ctx->monotonic_base_ns = (ctx->monotonic_base_ns > mag)
                               ? (ctx->monotonic_base_ns - mag)
                               : 0;
    }
}

/* ---- One-shot sleep ---- */

int tick_sleep_until(tick_ctx_t *ctx, uint64_t deadline_ns, uint64_t *overshoot_ns) {
    last_error_buf[0] = '\0';
    if (!ctx) {
        if (overshoot_ns) *overshoot_ns = 0;
        return -TICK_ERR_INVALID;
    }

    uint64_t now = tick_now(ctx);
    if (deadline_ns <= now) {
        if (overshoot_ns) {
            *overshoot_ns = now - deadline_ns;
        }
        return 1;
    }

    /* Spin cap: hybrid sleep-then-spin. clock_nanosleep gets within ~1 jiffy;
     * a TSC spin tightens the last sub-jiffy gap. The cap is derived from the
     * detected HZ (quarter jiffy on low-HZ kernels, 20us on high-HZ kernels
     * where high-res timers are likely), with a 200us ceiling to bound CPU
     * cost. */
    uint64_t spin_cap;
    if (ctx->runtime_hz >= 1000) {
        spin_cap = 20000;   /* 20us: high-HZ kernel, likely high-res timers */
    } else {
        uint64_t jiffy = 1000000000ULL / (uint64_t)ctx->runtime_hz;
        spin_cap = jiffy / 4;
        if (spin_cap > 200000ULL) {
            spin_cap = 200000ULL;
        }
    }

    uint64_t remaining = deadline_ns - now;
    if (remaining <= spin_cap) {
        /* Deadline is within the spin cap; spin directly. */
        while (tick_now(ctx) < deadline_ns) {
            __builtin_ia32_pause();
        }
    } else {
        /* Sleep until (deadline - spin_cap), then spin the rest. */
        uint64_t sleep_target = deadline_ns - spin_cap;
        struct timespec ts;
        tick_to_timespec(sleep_target, &ts);
        int rc;
        do {
            rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
        } while (rc == EINTR);
        if (rc != 0) {
            set_last_error("clock_nanosleep failed: %d", rc);
            if (overshoot_ns) *overshoot_ns = 0;
            return -TICK_ERR_INVALID;
        }
        while (tick_now(ctx) < deadline_ns) {
            __builtin_ia32_pause();
        }
    }

    now = tick_now(ctx);
    if (overshoot_ns) {
        *overshoot_ns = (now > deadline_ns) ? (now - deadline_ns) : 0;
    }
    return 0;
}

/* ---- Registry ---- */

int tick_register(tick_ctx_t *ctx, uint64_t deadline_ns, tick_fire_mode_t mode,
                  tick_callback_fn cb, void *user_data, tick_timer_id_t *out_id) {
    last_error_buf[0] = '\0';
    if (!ctx) {
        return -TICK_ERR_INVALID;
    }
    if (mode != TICK_MODE_POLL && mode != TICK_MODE_CALLBACK) {
        set_last_error("tick_register: invalid mode %d", (int)mode);
        return -TICK_ERR_INVALID;
    }
    if (mode == TICK_MODE_CALLBACK && !cb) {
        set_last_error("tick_register: callback mode requires non-NULL cb");
        return -TICK_ERR_INVALID;
    }
    if (ctx->capacity == 0 || !ctx->timers) {
        set_last_error("tick_register: registry capacity is 0");
        return -TICK_ERR_FULL;
    }

    size_t slot = free_list_pop(ctx);
    if (slot == SIZE_MAX) {
        set_last_error("tick_register: registry full (%zu slots)", ctx->capacity);
        return -TICK_ERR_FULL;
    }

    struct tick_timer *t = &ctx->timers[slot];
    t->in_use      = 1;
    t->firing      = 0;
    t->deadline_ns = deadline_ns;
    t->mode        = mode;
    t->cb          = cb;
    t->user_data   = user_data;

    if (mode == TICK_MODE_POLL) {
        heap_push(ctx, ctx->poll_heap, &ctx->poll_heap_size, slot);
    } else {
        heap_push(ctx, ctx->cb_heap, &ctx->cb_heap_size, slot);
    }

    ctx->timer_count++;
    tick_timer_id_t id = (tick_timer_id_t)slot;
    if (out_id) {
        *out_id = id;
    }
    return id;
}

int tick_cancel(tick_ctx_t *ctx, tick_timer_id_t id) {
    last_error_buf[0] = '\0';
    if (!ctx || id < 0 || (size_t)id >= ctx->capacity) {
        return -TICK_ERR_NOT_FOUND;
    }
    if (!ctx->timers[id].in_use) {
        /* Already free, already fired, or in the transient firing state. */
        return -TICK_ERR_NOT_FOUND;
    }

    struct tick_timer *t = &ctx->timers[id];
    size_t pos = t->link;  /* heap_pos */
    if (t->mode == TICK_MODE_POLL) {
        heap_remove_at(ctx, ctx->poll_heap, &ctx->poll_heap_size, pos);
    } else {
        heap_remove_at(ctx, ctx->cb_heap, &ctx->cb_heap_size, pos);
    }

    t->in_use = 0;
    free_list_push(ctx, (size_t)id);
    ctx->timer_count--;
    return 0;
}

int tick_wait_next(tick_ctx_t *ctx, tick_timer_id_t *fired_ids, size_t max_ids,
                   uint64_t timeout_ns) {
    last_error_buf[0] = '\0';
    if (!ctx) {
        return -TICK_ERR_INVALID;
    }

    uint64_t now = tick_now(ctx);
    uint64_t deadline_cap;
    if (timeout_ns == UINT64_MAX) {
        deadline_cap = UINT64_MAX;
    } else if (timeout_ns > UINT64_MAX - now) {
        deadline_cap = UINT64_MAX;  /* overflow guard */
    } else {
        deadline_cap = now + timeout_ns;
    }

    if (ctx->poll_heap_size == 0) {
        /* No poll-mode timers. Block until the timeout cap if finite;
         * return immediately (rather than hang) if infinite. */
        if (deadline_cap == UINT64_MAX || deadline_cap <= now) {
            return 0;
        }
        uint64_t ov;
        int rc = tick_sleep_until(ctx, deadline_cap, &ov);
        if (rc < 0) {
            return rc;
        }
        return 0;
    }

    uint64_t nearest = ctx->timers[ctx->poll_heap[0]].deadline_ns;

    if (nearest > now) {
        /* Nearest poll timer is not yet due. Decide whether to wait, time
         * out, or sleep until the cap. */
        if (deadline_cap <= now) {
            /* timeout_ns == 0 (or already expired): do not sleep, do not fire. */
            return 0;
        }
        if (nearest > deadline_cap) {
            /* Nearest deadline exceeds the timeout cap; sleep until cap. */
            uint64_t ov;
            int rc = tick_sleep_until(ctx, deadline_cap, &ov);
            if (rc < 0) {
                return rc;
            }
            return 0;
        }
        /* Sleep until the nearest deadline. The spin inside tick_sleep_until
         * guarantees now >= nearest on return, so the collect pass below will
         * fire at least the root. */
        uint64_t ov;
        int rc = tick_sleep_until(ctx, nearest, &ov);
        if (rc < 0) {
            return rc;
        }
    }

    /* Collect all expired poll-mode timers from the heap. Each pop is
     * O(log n); the loop runs k times where k = expired count. Writes up to
     * max_ids into fired_ids; the returned count reflects the total popped
     * (may exceed max_ids). */
    size_t count = 0;
    now = tick_now(ctx);
    while (ctx->poll_heap_size > 0
           && ctx->timers[ctx->poll_heap[0]].deadline_ns <= now) {
        size_t slot = heap_pop_root(ctx, ctx->poll_heap, &ctx->poll_heap_size);
        if (fired_ids && count < max_ids) {
            fired_ids[count] = (tick_timer_id_t)slot;
        }
        ctx->timers[slot].in_use = 0;
        free_list_push(ctx, slot);
        ctx->timer_count--;
        count++;
    }
    return (int)count;
}

int tick_run_pending(tick_ctx_t *ctx) {
    if (!ctx) {
        return -TICK_ERR_INVALID;
    }
    last_error_buf[0] = '\0';
    if (ctx->cb_heap_size == 0) {
        return 0;
    }

    uint64_t now = tick_now(ctx);

    /* Pass 1: pop all expired callback-mode timers from cb_heap, recording
     * the popped slot indices into fired_slots in the order they were popped
     * (which is non-decreasing deadline order, since cb_heap is a min-heap
     * keyed by deadline_ns). Each popped slot transitions from in_use=1 to
     * firing=1 (not yet free). This is the snapshot: the set of slots that
     * pass 2 will dispatch, in the order they will be dispatched.
     *
     * A callback that (incorrectly) registers a new timer cannot add to
     * this set, because the new timer enters in_use=1 and pass 2 only
     * visits slots recorded here. The dispatch array is sized to the
     * pre-pop cb_heap_size, which is the upper bound on the fired count. */
    size_t max_count = ctx->cb_heap_size;
    size_t *fired_slots = malloc(max_count * sizeof(size_t));
    if (!fired_slots) {
        set_last_error("tick_run_pending: out of memory for dispatch set (%zu slots)",
                       max_count);
        return -TICK_ERR_INVALID;
    }

    size_t count = 0;
    while (ctx->cb_heap_size > 0
           && ctx->timers[ctx->cb_heap[0]].deadline_ns <= now) {
        size_t slot = heap_pop_root(ctx, ctx->cb_heap, &ctx->cb_heap_size);
        ctx->timers[slot].in_use = 0;
        ctx->timers[slot].firing = 1;
        ctx->timer_count--;
        fired_slots[count++] = slot;
    }

    if (count == 0) {
        free(fired_slots);
        return 0;
    }

    /* Pass 2: invoke callbacks in the order recorded in pass 1 (non-
     * decreasing deadline order). Reads cb/user_data into locals before
     * freeing the slot, so even a callback that (incorrectly) re-registers
     * into this same slot cannot corrupt the dispatch we are about to do.
     * The slot is returned to the free list before the callback runs; a
     * re-register from the callback may consume it, but pass 2 has already
     * advanced past this index in the dispatch array. */
    for (size_t i = 0; i < count; i++) {
        size_t slot = fired_slots[i];
        tick_callback_fn cb = ctx->timers[slot].cb;
        void *ud = ctx->timers[slot].user_data;
        tick_timer_id_t id = (tick_timer_id_t)slot;
        ctx->timers[slot].firing = 0;
        free_list_push(ctx, slot);
        if (cb) {
            cb(ctx, id, ud);
        }
    }

    free(fired_slots);
    return (int)count;
}
