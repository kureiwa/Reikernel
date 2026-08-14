/*
 * libspinit implementation. See include/spinit.h for the public contract
 * and DESIGN.md for the rationale. The spin is a calibrated iteration
 * count (~500ns without backoff); within that window the per-iteration
 * PAUSE count adapts via exponential backoff to reduce cache-line
 * traffic under contention.
 *
 * Spin loop strategy: a calibrated iteration count, not a per-attempt
 * RDTSC deadline. DESIGN.md is explicit ("no rdtsc per attempt"). Each
 * iteration issues between 1 and 64 PAUSE hints (doubling each failed
 * iteration, capped at 64), one relaxed atomic_load of state, and (only
 * if the load observed 0) one lock cmpxchg. This is the test-and-test-
 * and-set pattern with exponential backoff: spinning on a read-only
 * shared cache line avoids the per-iter RMW ping-pong that pure
 * test-and-set would cause under N-way contention, and the backoff
 * reduces load frequency when the lock is clearly held. Calibration
 * measures ticks-per-iteration (one PAUSE + one cmpxchg) and derives
 * iterations-per-500ns from the TSC frequency; under backoff the
 * wall-clock spin window can exceed 500ns because each iteration may
 * issue more PAUSEs than calibration assumed. This is intentional --
 * the alternative is burning the same cycles hammering the cache line.
 *
 * Futex state machine (the standard three-state Linux futex mutex):
 *   state == 0: unlocked
 *   state == 1: locked, no parked waiters
 *   state == 2: locked, one or more threads parked in FUTEX_WAIT
 *
 * Unlock uses atomic_exchange(0) and wakes one waiter only if the
 * previous value was 2. A thread that has parked in futex at least once
 * re-marks state = 2 on its next acquire, so subsequent unlocks in a
 * contention chain keep waking waiters until the queue drains. The tail
 * unlock does one spurious futex_wake(1); accepted cost.
 */

#define _GNU_SOURCE
#include <spinit.h>

#include <stdint.h>
#include <stdatomic.h>
#include <stdalign.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <errno.h>

/* Assembly helpers in spinit_x86_64.asm. */
extern uint64_t spinit_rdtsc(void);
extern void     spinit_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);

#define SPINIT_SPIN_NS_TARGET      500   /* ~500ns fixed spin window */
#define SPINIT_FALLBACK_ITERATIONS 500   /* used when constant_tsc absent */
#define SPINIT_CALIBRATE_PROBE_N   256   /* iterations for ticks-per-spin probe */

/* Process-wide calibration. spin_iterations is 0 until pthread_once
 * runs calibrate(); the lock path calls ensure_calibrated() first. */
static pthread_once_t calibrate_once = PTHREAD_ONCE_INIT;
static uint64_t spin_iterations = 0;

static int has_constant_tsc(void) {
    uint32_t out[4];
    spinit_cpuid(0x80000000U, 0, out);
    if (out[0] < 0x80000007U) {
        return 0;
    }
    spinit_cpuid(0x80000007U, 0, out);
    return (out[3] >> 8) & 1u;
}

/* CPUID leaf 15H: TSC frequency = ECX * EBX / EAX, where ECX is the
 * core crystal clock in Hz. Returns 0 if the leaf is not populated
 * (ECX == 0) or the ratio is degenerate (EAX == 0). */
static uint64_t cpuid_15h_tsc_hz(void) {
    uint32_t out[4];
    spinit_cpuid(0x15U, 0, out);
    uint32_t eax = out[0];
    uint32_t ebx = out[1];
    uint32_t ecx = out[2];
    if (ecx == 0 || eax == 0) {
        return 0;
    }
    return (uint64_t)ecx * (uint64_t)ebx / (uint64_t)eax;
}

/* Estimate TSC frequency by pairing rdtsc with clock_gettime(CLOCK_MONOTONIC)
 * over a ~1ms busy wait. Takes the minimum of two samples to reduce
 * preemption noise. */
static uint64_t measure_tsc_hz_via_clock(void) {
    const long target_ns = 1000 * 1000;
    uint64_t best_hz = 0;
    for (int sample = 0; sample < 2; sample++) {
        struct timespec t0, t1;
        uint64_t c0, c1;
        long ns;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        c0 = spinit_rdtsc();
        for (;;) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
               + (t1.tv_nsec - t0.tv_nsec);
            if (ns >= target_ns) {
                break;
            }
        }
        c1 = spinit_rdtsc();
        ns = (t1.tv_sec - t0.tv_sec) * 1000000000L
           + (t1.tv_nsec - t0.tv_nsec);
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

/* Measure TSC ticks per spin-loop iteration by running a fixed probe
 * loop that matches the minimum-cost hot-path spin body: one PAUSE +
 * a cmpxchg that fails (state stays 1). The hot path may issue more
 * PAUSEs per iteration under exponential backoff (up to 64); this
 * probe measures the floor, not the contended average. The iteration
 * cap (spin_iterations) is therefore an upper bound on attempts, not
 * a strict wall-clock guarantee. Returns >= 1. */
static uint64_t measure_ticks_per_spin_iter(void) {
    _Atomic int probe = 1;  /* contended so cmpxchg(0->1) fails */
    const int N = SPINIT_CALIBRATE_PROBE_N;
    uint64_t t0 = spinit_rdtsc();
    for (int i = 0; i < N; i++) {
        __builtin_ia32_pause();
        int expected = 0;
        atomic_compare_exchange_strong(&probe, &expected, 1);
    }
    uint64_t t1 = spinit_rdtsc();
    uint64_t total = t1 - t0;
    uint64_t per_iter = total / (uint64_t)N;
    return per_iter == 0 ? 1 : per_iter;
}

static void calibrate(void) {
    uint64_t iters = SPINIT_FALLBACK_ITERATIONS;

    if (has_constant_tsc()) {
        uint64_t hz = cpuid_15h_tsc_hz();
        if (hz == 0) {
            hz = measure_tsc_hz_via_clock();
        }
        if (hz > 0) {
            /* Ticks in 500ns = hz * 500 / 1e9 = hz / 2e6. */
            uint64_t target_ticks = hz / 2000000ULL;
            uint64_t ticks_per_iter = measure_ticks_per_spin_iter();
            uint64_t computed = target_ticks / ticks_per_iter;
            if (computed > 0) {
                iters = computed;
            }
        }
    }

    if (iters < 1) {
        iters = 1;
    }
    spin_iterations = iters;
}

static inline void ensure_calibrated(void) {
    pthread_once(&calibrate_once, calibrate);
}

void spinit_init(spinit_t *lock) {
    atomic_store(&lock->state, 0);
}

void spinit_lock(spinit_t *lock) {
    ensure_calibrated();

    /* Fast path: single lock cmpxchg, state 0 -> 1. */
    int expected = 0;
    if (atomic_compare_exchange_strong(&lock->state, &expected, 1)) {
        return;
    }

    /* Spin window: fixed iteration count, no rdtsc per attempt.
     * Test-and-test-and-set with exponential backoff: spin on a read-only
     * load of state and only attempt the lock cmpxchg when state == 0 is
     * observed. The per-iteration PAUSE count starts at 1 and doubles each
     * failed iteration up to 64, reducing load frequency when the lock is
     * clearly held. The iteration cap (spin_iterations) is unchanged -- the
     * backoff stays inside the existing spin window and falls through to
     * futex when it is exhausted. backoff is local to this call, so a
     * successful acquire implicitly resets it to 1 on the next call. */
    uint32_t backoff = 1;
    for (uint64_t i = 0; i < spin_iterations; i++) {
        for (uint32_t p = 0; p < backoff; p++) {
            __builtin_ia32_pause();
        }
        if (atomic_load(&lock->state) != 0) {
            backoff = backoff < 32 ? backoff * 2 : 64;
            continue;
        }
        expected = 0;
        if (atomic_compare_exchange_strong(&lock->state, &expected, 1)) {
            return;
        }
        backoff = backoff < 32 ? backoff * 2 : 64;
    }

    /* Futex fallback. State is 1 or 2 (locked). A thread that has parked
     * at least once re-marks state = 2 on acquire so the next unlock
     * wakes any remaining waiters. */
    int waited = 0;
    for (;;) {
        int new_state = waited ? 2 : 1;
        expected = 0;
        if (atomic_compare_exchange_strong(&lock->state, &expected, new_state)) {
            return;
        }
        /* expected now holds the value seen: 1 or 2. */
        if (expected == 1) {
            int e = 1;
            /* Best-effort: mark that a waiter is present. If this fails
             * because state changed, the load below re-checks. */
            atomic_compare_exchange_strong(&lock->state, &e, 2);
        }
        if (atomic_load(&lock->state) != 2) {
            continue;
        }
        /* FUTEX_WAIT returns 0 on wake (including spurious), -EINTR on
         * signal, -EAGAIN on value mismatch. All three loop back. */
        long rc = syscall(SYS_futex, &lock->state,
                          FUTEX_WAIT | FUTEX_PRIVATE_FLAG, 2,
                          NULL, NULL, 0);
        (void)rc;
        waited = 1;
    }
}

int spinit_trylock(spinit_t *lock) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&lock->state, &expected, 1)) {
        return 0;
    }
    return 1;
}

void spinit_unlock(spinit_t *lock) {
    int prev = atomic_exchange(&lock->state, 0);
    if (prev == 2) {
        long rc = syscall(SYS_futex, &lock->state,
                          FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1,
                          NULL, NULL, 0);
        (void)rc;
    }
}
