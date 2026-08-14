/*
 * bench_sleep: sleep_until accuracy vs target.
 *
 * Sleeps 1ms (now + 1ms) 1000 times. Reports average, min, and max
 * overshoot in ns. The hybrid sleep-then-spin in tick_sleep_until
 * should keep the average in the low hundreds of ns on an unloaded
 * machine.
 */
#include <tick.h>

#include <stdio.h>
#include <stdint.h>

#define N_SLEEPS 1000
#define SLEEP_NS 1000000ULL  /* 1ms */

int main(void) {
    tick_ctx_t *ctx = tick_ctx_create(0);
    if (!ctx) {
        fprintf(stderr, "bench_sleep: ctx_create failed: %s\n", tick_last_error());
        return 1;
    }

    /* Warm up: one sleep to prime calibration and the scheduler. */
    uint64_t ov;
    tick_sleep_until(ctx, tick_now(ctx) + SLEEP_NS, &ov);

    uint64_t total_overshoot = 0;
    uint64_t max_overshoot = 0;
    uint64_t min_overshoot = UINT64_MAX;

    for (int i = 0; i < N_SLEEPS; i++) {
        uint64_t deadline = tick_now(ctx) + SLEEP_NS;
        int rc = tick_sleep_until(ctx, deadline, &ov);
        if (rc < 0) {
            fprintf(stderr, "bench_sleep: sleep error: %s\n", tick_last_error());
            tick_ctx_destroy(ctx);
            return 1;
        }
        total_overshoot += ov;
        if (ov > max_overshoot) max_overshoot = ov;
        if (ov < min_overshoot) min_overshoot = ov;
    }

    printf("bench_sleep: %d sleeps of %llu ns\n", N_SLEEPS,
           (unsigned long long)SLEEP_NS);
    printf("  avg overshoot: %llu ns\n",
           (unsigned long long)(total_overshoot / N_SLEEPS));
    printf("  min overshoot: %llu ns\n", (unsigned long long)min_overshoot);
    printf("  max overshoot: %llu ns\n", (unsigned long long)max_overshoot);

    tick_ctx_destroy(ctx);
    return 0;
}
