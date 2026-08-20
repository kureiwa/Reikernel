/* rk_norm.c -- RMSNorm, FP32, (B,T,C).
 * y = x / sqrt(mean(x^2) + eps) * weight.
 * OpenMP collapse(2) over (B,T); per-thread libbarrage arena holds the
 * per-row sum-of-squares scratch. Math matches PyTorch's fp32 CPU path
 * (true divide, no -ffast-math, no inv_rms precompute).
 */

#include "rk_api.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "barrage.h"

/* Per-thread libbarrage arena. 4 MiB, mmap-backed, lazily created, never
 * destroyed. Falls back to stack scratch if barrage_create fails. */
#define RK_ARENA_BYTES (4u * 1024u * 1024u)

static barrage_arena_t *rk_get_thread_arena(void)
{
    static __thread barrage_arena_t *arena = NULL;
    if (arena == NULL) {
        arena = barrage_create(RK_ARENA_BYTES, NULL);
        /* If barrage_create fails (out of address space / mmap denied) the
         * caller falls back to a stack scratch. Don't crash the kernel. */
    }
    return arena;
}

/* Fallback scratch on the stack if the arena is unavailable. Tiny, only
 * used to hold the per-row scalar sum. */
static float rk_stack_scratch(void)
{
    return 0.0f;
}

int rk_rms_norm(const float *x, const float *weight, float *y,
                int B, int T, int C, float eps)
{
    /* input validation */
    if (x == NULL || weight == NULL || y == NULL) return -1;
    if (B <= 0 || T <= 0 || C <= 0) return -1;
    if (eps < 0.0f) return -1;

    /* Cap B*T*C at INT32_MAX; we use int indices in the inner loops. */
    if ((int64_t)B * (int64_t)T > (int64_t)INT32_MAX / (int64_t)C) return -1;

    const int C_f = C;                 /* C as int (already validated) */
    const long bt = (long)B * (long)T; /* row count */

    /* Prime the calling thread's arena TLS slot. */
    (void)rk_get_thread_arena();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *xb = x + (((long)b * (long)T + (long)t) * (long)C_f);
            float       *yb = y + (((long)b * (long)T + (long)t) * (long)C_f);

            /* Per-row scratch from the arena; falls back to stack if NULL. */
            barrage_arena_t *arena = rk_get_thread_arena();
            float *pscratch = NULL;
            if (arena != NULL) {
                barrage_err_t err = BARRAGE_OK;
                pscratch = (float *)barrage_alloc(arena, sizeof(float),
                                                 _Alignof(float), &err);
            }
            float stack_v = rk_stack_scratch();
            if (pscratch == NULL) pscratch = &stack_v;
            *pscratch = 0.0f;

            /* Sum of x^2 across the row. omp simd reduction vectorises
             * along C; PyTorch does the same. */
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < C_f; ++i) {
                const float xi = xb[i];
                sum += xi * xi;
            }
            *pscratch = sum;

            /* PyTorch fp32 path: mean = sum/C, rms = sqrtf(mean+eps),
             * y = (x*weight)/rms. No inv_rms precompute, no -ffast-math. */
            const float mean = (*pscratch) / (float)C_f;
            const float rms  = sqrtf(mean + eps);

            /* FMA for (x*weight) only adds precision to the numerator. */
            #pragma omp simd
            for (int i = 0; i < C_f; ++i) {
                yb[i] = (xb[i] * weight[i]) / rms;
            }
        }
    }

    /* Don't reset; keep the bump pointer warm for the next call. */
    (void)bt;

    return 0;
}
