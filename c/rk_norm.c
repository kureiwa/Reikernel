/* rk_norm.c -- RMSNorm, FP32, (B,T,C).
 * y = x / sqrt(mean(x^2) + eps) * weight.
 * OpenMP collapse(2) over (B,T); per-row sum-of-squares lives on the
 * stack (8 bytes, always L1). Math matches PyTorch's fp32 CPU path
 * (true divide, no -ffast-math, no inv_rms precompute).
 *
 * v0.6: removed the per-thread libbarrage arena. The arena was
 * bump-allocating 4 bytes per row and never reset (the "keep the bump
 * pointer warm" comment was inverted), so it filled after ~1024 calls
 * and silently fell back to the stack scratch anyway. The arena added
 * TLS lookup + barrage_alloc + error-out-param overhead per row for
 * zero benefit. The matmul op still uses the arena (large packed
 * panels); the norm ops don't.
 */

#include "rk_api.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

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

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *xb = x + (((long)b * (long)T + (long)t) * (long)C_f);
            float       *yb = y + (((long)b * (long)T + (long)t) * (long)C_f);

            /* Per-row sum-of-squares on the stack. 4 bytes, always L1.
             * Replaces the v0.5 barrage_alloc + TLS lookup + fallback
             * path that the audit (IMPROVEMENTS.md B2) flagged as net
             * negative. */
            float row_scratch[1] = { 0.0f };

            /* Sum of x^2 across the row. omp simd reduction vectorises
             * along C; PyTorch does the same. */
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < C_f; ++i) {
                const float xi = xb[i];
                sum += xi * xi;
            }
            row_scratch[0] = sum;

            /* PyTorch fp32 path: mean = sum/C, rms = sqrtf(mean+eps),
             * y = (x*weight)/rms. No inv_rms precompute, no -ffast-math. */
            const float mean = row_scratch[0] / (float)C_f;
            const float rms  = sqrtf(mean + eps);

            /* FMA for (x*weight) only adds precision to the numerator. */
            #pragma omp simd
            for (int i = 0; i < C_f; ++i) {
                yb[i] = (xb[i] * weight[i]) / rms;
            }
        }
    }

    return 0;
}
