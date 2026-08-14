/* rk_layernorm.c -- LayerNorm, FP32, (B,T,C).
 * Two-pass per row: sum -> mean, sum((x-mean)^2) -> var -> rstd, then
 * normalize+scale+shift. rstd is computed once per row and multiplied
 * in (matches PyTorch's CPU kernel ordering, not per-element divide like
 * rk_rms_norm). Per-thread libbarrage arena for the per-row mean+rstd
 * scratch. No -ffast-math; the (x-mean)*rstd*weight+bias chain fuses to
 * 2 FMAs at -O3.
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
        /* If barrage_create fails, caller falls back to a stack scratch. */
    }
    return arena;
}

int rk_layer_norm(const float *x, const float *weight, const float *bias,
                  float *y, int B, int T, int C, float eps)
{
    /* input validation. bias may be NULL (matches F.layer_norm with
     * bias=None). */
    if (x == NULL || weight == NULL || y == NULL) return -1;
    if (B <= 0 || T <= 0 || C <= 0) return -1;
    if (eps < 0.0f) return -1;

    /* Cap B*T*C at INT32_MAX; we use int indices in the inner loops. */
    if ((int64_t)B * (int64_t)T > (int64_t)INT32_MAX / (int64_t)C) return -1;

    const int C_f = C;                 /* C as int (already validated) */

    /* Prime the calling thread's arena TLS slot. */
    (void)rk_get_thread_arena();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *xb = x + (((long)b * (long)T + (long)t) * (long)C_f);
            float       *yb = y + (((long)b * (long)T + (long)t) * (long)C_f);

            /* Per-row mean+rstd scratch from the arena; falls back to stack. */
            barrage_arena_t *arena = rk_get_thread_arena();
            float *pscratch = NULL;
            if (arena != NULL) {
                barrage_err_t err = BARRAGE_OK;
                pscratch = (float *)barrage_alloc(arena,
                                                  2u * sizeof(float),
                                                  _Alignof(float), &err);
            }
            float stack_scratch[2] = { 0.0f, 0.0f };
            if (pscratch == NULL) pscratch = stack_scratch;

            /* --- Pass 1: per-row sum -> mean --- */
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < C_f; ++i) {
                sum += xb[i];
            }
            const float mean = sum / (float)C_f;
            pscratch[0] = mean;

            /* --- Pass 2: per-row sum of (x - mean)^2 -> var -> rstd.
             * Row is hot in L1 after pass 1, so the second read is cheap. */
            float sum_sq = 0.0f;
            #pragma omp simd reduction(+:sum_sq)
            for (int i = 0; i < C_f; ++i) {
                const float d = xb[i] - mean;
                sum_sq += d * d;
            }
            const float var  = sum_sq / (float)C_f;
            /* rstd = 1 / sqrt(var + eps). Computed once per row and
             * multiplied in (matches PyTorch's CPU kernel ordering). */
            const float rstd = 1.0f / sqrtf(var + eps);
            pscratch[1] = rstd;

            /* --- Pass 3: per-element normalize + scale + shift.
             * y[i] = (x[i] - mean) * rstd * weight[i] + bias[i]
             * Fuses to 2 FMAs at -O3. When bias is NULL, drop the +bias
             * term (matches F.layer_norm with bias=None). */
            if (bias != NULL) {
                #pragma omp simd
                for (int i = 0; i < C_f; ++i) {
                    yb[i] = (xb[i] - mean) * rstd * weight[i] + bias[i];
                }
            } else {
                #pragma omp simd
                for (int i = 0; i < C_f; ++i) {
                    yb[i] = (xb[i] - mean) * rstd * weight[i];
                }
            }
        }
    }

    /* Don't reset; keep the bump pointer warm for the next call. */
    return 0;
}
