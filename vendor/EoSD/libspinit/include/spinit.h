#ifndef SPINIT_H
#define SPINIT_H

/*
 * libspinit: TSC-calibrated fixed-window spinlock with futex fallback.
 * See API.md and DESIGN.md for the design rationale and the v0.1 non-goals
 * (no per-lock tuning, no fairness, no recursive locking, no PI).
 */

#include <stdatomic.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lock state values. 0 = unlocked, 1 = locked with no parked waiters,
 * 2 = locked with one or more threads parked in futex(FUTEX_WAIT). The
 * 2 sentinel is internal but documented here because it is visible in the
 * public struct and a debugger may inspect it.
 *
 * The state word is alignas(64) so that the struct occupies one cache
 * line. Without this, embedding a spinit_t next to other read/write
 * fields in a caller struct (thread-pool task queues, log buffers) would
 * ping-pong the line under contention.
 */
typedef struct {
    alignas(64) _Atomic int state;
} spinit_t;

/* Static initializer. Equivalent to spinit_init at runtime. */
#define SPINIT_INIT { 0 }

/*
 * Initialize a dynamically-allocated lock. Writes state = 0. Redundant
 * after SPINIT_INIT; provided for symmetry with malloc-then-init patterns.
 *
 * Thread-safety: not concurrent with another lock/unlock on the same lock.
 */
void spinit_init(spinit_t *lock);

/*
 * Acquire the lock. The fast path is a single lock cmpxchg (state 0 -> 1).
 * On contention the thread spins for a calibrated ~500ns window with a
 * PAUSE hint between attempts, then parks in futex(FUTEX_WAIT|FUTEX_PRIVATE_FLAG).
 *
 * Calibration runs once per process on the first spinit_lock call
 * (pthread_once). It reads CPUID.80000007H:EDX[8] (constant_tsc); if set
 * it queries CPUID.15H or measures TSC frequency via clock_gettime+rdtsc
 * and derives an iteration count for ~500ns. Without constant_tsc it
 * falls back to a fixed iteration count.
 *
 * Returns with the lock held. Never returns an error.
 *
 * Thread-safety: yes; designed for concurrent callers.
 */
void spinit_lock(spinit_t *lock);

/*
 * Try to acquire the lock without spinning. One cmpxchg (0 -> 1).
 *
 * Returns 0 if acquired, 1 if already locked. (Non-idiomatic vs POSIX
 * convention; matches API.md.)
 *
 * Thread-safety: yes.
 */
int spinit_trylock(spinit_t *lock);

/*
 * Release the lock. atomic_exchange(state, 0); if the previous value was
 * 2 (a waiter was parked), call futex(FUTEX_WAKE|FUTEX_PRIVATE_FLAG, 1).
 *
 * Caller must hold the lock. Unlocking an unheld lock is undefined
 * behavior and is NOT detected: the atomic_exchange silently writes 0
 * (a no-op when state was already 0) and no futex_wake is issued. Under
 * concurrent use a stray unlock can race with a real holder's unlock and
 * permit double-acquisition; callers must not rely on detection.
 *
 * Thread-safety: yes.
 */
void spinit_unlock(spinit_t *lock);

#ifdef __cplusplus
}
#endif

#endif /* SPINIT_H */
