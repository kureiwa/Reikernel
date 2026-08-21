/* rk_layernorm.c -- LayerNorm, FP32, (B,T,C).
 * Two-pass per row: sum -> mean, sum((x-mean)^2) -> var -> rstd, then
 * normalize+scale+shift. rstd is computed once per row and multiplied
 * in (matches PyTorch's CPU kernel ordering, not per-element divide like
 * rk_rms_norm). Per-row mean+rstd lives on the stack (8 bytes, always
 * L1). No -ffast-math; the (x-mean)*rstd*weight+bias chain fuses to
 * 2 FMAs at -O3.
 *
 * v0.6: removed the per-thread libbarrage arena. The arena was
 * bump-allocating 8 bytes per row and never reset, so it filled after
 * ~500 calls and fell back to the stack scratch anyway. The matmul op
 * still uses the arena (large packed panels); the norm ops don't.
 */

#include "rk_api.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

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

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *xb = x + (((long)b * (long)T + (long)t) * (long)C_f);
            float       *yb = y + (((long)b * (long)T + (long)t) * (long)C_f);

            /* Per-row mean+rstd scratch on the stack. 8 bytes, always L1.
             * Replaces the v0.5 barrage_alloc + TLS lookup + fallback
             * path that the audit (IMPROVEMENTS.md B2) flagged as net
             * negative. Holds [mean, rstd]. */
            float row_scratch[2] = { 0.0f, 0.0f };

            /* --- Pass 1: per-row sum -> mean --- */
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < C_f; ++i) {
                sum += xb[i];
            }
            const float mean = sum / (float)C_f;
            row_scratch[0] = mean;

            /* --- Pass 2: per-row sum of (x - mean)^2 -> var -> rstd.
             * Row is hot in L1 after pass 1, so the second read is cheap. */
            const float row_mean = row_scratch[0];
            float sum_sq = 0.0f;
            #pragma omp simd reduction(+:sum_sq)
            for (int i = 0; i < C_f; ++i) {
                const float d = xb[i] - row_mean;
                sum_sq += d * d;
            }
            const float var  = sum_sq / (float)C_f;
            /* rstd = 1 / sqrt(var + eps). Computed once per row and
             * multiplied in (matches PyTorch's CPU kernel ordering). */
            const float rstd = 1.0f / sqrtf(var + eps);
            row_scratch[1] = rstd;

            /* --- Pass 3: per-element normalize + scale + shift.
             * y[i] = (x[i] - mean) * rstd * weight[i] + bias[i]
             * Fuses to 2 FMAs at -O3. When bias is NULL, drop the +bias
             * term (matches F.layer_norm with bias=None). */
            const float row_rstd = row_scratch[1];
            if (bias != NULL) {
                #pragma omp simd
                for (int i = 0; i < C_f; ++i) {
                    yb[i] = (xb[i] - row_mean) * row_rstd * weight[i] + bias[i];
                }
            } else {
                #pragma omp simd
                for (int i = 0; i < C_f; ++i) {
                    yb[i] = (xb[i] - row_mean) * row_rstd * weight[i];
                }
            }
        }
    }

    return 0;
}
