/* rk_adamw.c -- fused FP32 AdamW optimizer step.
 *
 *   m = beta1 * m + (1 - beta1) * grad
 *   v = beta2 * v + (1 - beta2) * grad^2
 *   denom = sqrt(v) / sqrt(1 - beta2^step) + eps
 *   p = p * (1 - lr * wd) - (lr / (1 - beta1^step)) * m / denom
 *
 * Single pass over the param vector. AVX-512 vectorised when
 * __AVX512F__ is set; scalar fallback otherwise. Math matches
 * torch.optim.AdamW (decoupled weight decay, biased first/second
 * moments, bias-corrected step size).
 *
 * The caller owns params/grads/exp_avg/exp_avg_sq and must keep them
 * alive for the duration of the call. All four tensors must be the same
 * length n and 64-byte aligned for the AVX-512 path (the Python wrapper
 * enforces contiguous float32 tensors, which gives at least 16-byte
 * alignment; the C side uses unaligned loads so 16-byte is fine).
 */

#include "rk_api.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __AVX512F__
#include <immintrin.h>
#define RK_HAS_AVX512 1
#else
#define RK_HAS_AVX512 0
#endif

int rk_adamw_step(float *params, const float *grads,
                  float *exp_avg, float *exp_avg_sq,
                  int n, float lr, float beta1, float beta2,
                  float eps, float weight_decay, int step)
{
    /* --- input validation --- */
    if (params == NULL || grads == NULL
        || exp_avg == NULL || exp_avg_sq == NULL) return -1;
    if (n <= 0) return -1;
    if (step <= 0) return -1;
    if (lr < 0.0f) return -1;
    if (beta1 < 0.0f || beta1 >= 1.0f) return -1;
    if (beta2 < 0.0f || beta2 >= 1.0f) return -1;
    if (eps < 0.0f) return -1;
    if (weight_decay < 0.0f) return -1;

    /* Cap n at INT32_MAX; we use int indices in the inner loops. */
    if (n > INT32_MAX) return -1;

    /* Bias-correction factors. Computed once, scalar (not per-element). */
    const float bc1 = 1.0f - powf(beta1, (float)step);   /* 1 - beta1^step */
    const float bc2 = 1.0f - powf(beta2, (float)step);   /* 1 - beta2^step */
    const float bc2_sqrt = sqrtf(bc2);                   /* sqrt(1 - beta2^step) */
    const float step_size = lr / bc1;                    /* lr / (1 - beta1^step) */
    const float decay_mul = 1.0f - lr * weight_decay;    /* (1 - lr * wd) */

    /* Per-element constants for the m/v update. */
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

#if RK_HAS_AVX512
    /* AVX-512 vectorised path. 16 floats per iter. */
    const __m512 beta1_v   = _mm512_set1_ps(beta1);
    const __m512 beta2_v   = _mm512_set1_ps(beta2);
    const __m512 omb1_v    = _mm512_set1_ps(one_minus_b1);
    const __m512 omb2_v    = _mm512_set1_ps(one_minus_b2);
    const __m512 eps_v     = _mm512_set1_ps(eps);
    const __m512 bc2sq_v   = _mm512_set1_ps(bc2_sqrt);
    const __m512 ss_v      = _mm512_set1_ps(step_size);
    const __m512 decay_v   = _mm512_set1_ps(decay_mul);

    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 p  = _mm512_loadu_ps(params + i);
        const __m512 g  = _mm512_loadu_ps(grads + i);
        __m512 m  = _mm512_loadu_ps(exp_avg + i);
        __m512 v  = _mm512_loadu_ps(exp_avg_sq + i);

        /* m = beta1 * m + (1 - beta1) * g */
        m = _mm512_fmadd_ps(beta1_v, m, _mm512_mul_ps(omb1_v, g));
        /* v = beta2 * v + (1 - beta2) * g * g */
        const __m512 g_sq = _mm512_mul_ps(g, g);
        v = _mm512_fmadd_ps(beta2_v, v, _mm512_mul_ps(omb2_v, g_sq));

        /* denom = sqrt(v) / sqrt(bc2) + eps */
        const __m512 sqrt_v   = _mm512_sqrt_ps(v);
        const __m512 denom    = _mm512_add_ps(
            _mm512_div_ps(sqrt_v, bc2sq_v), eps_v);

        /* p = p * (1 - lr*wd) - step_size * m / denom
         *   = decay_mul * p - step_size * (m / denom)
         * Fused as fmsub: decay_mul * p - step_size * (m / denom). */
        const __m512 m_over_denom = _mm512_div_ps(m, denom);
        p = _mm512_fnmadd_ps(ss_v, m_over_denom,
                             _mm512_mul_ps(decay_v, p));

        _mm512_storeu_ps(params + i, p);
        _mm512_storeu_ps(exp_avg + i, m);
        _mm512_storeu_ps(exp_avg_sq + i, v);
    }

    /* Scalar tail. */
    for (; i < n; ++i) {
        float p  = params[i];
        const float g  = grads[i];
        float m  = exp_avg[i];
        float v  = exp_avg_sq[i];

        m = beta1 * m + one_minus_b1 * g;
        v = beta2 * v + one_minus_b2 * g * g;

        const float denom = sqrtf(v) / bc2_sqrt + eps;
        p = decay_mul * p - step_size * (m / denom);

        params[i]    = p;
        exp_avg[i]   = m;
        exp_avg_sq[i] = v;
    }
#else
    /* Scalar path. OpenMP omp simd vectorises with AVX2 at -O3
     * -march=native. */
    #pragma omp simd
    for (int i = 0; i < n; ++i) {
        float p  = params[i];
        const float g  = grads[i];
        float m  = exp_avg[i];
        float v  = exp_avg_sq[i];

        m = beta1 * m + one_minus_b1 * g;
        v = beta2 * v + one_minus_b2 * g * g;

        const float denom = sqrtf(v) / bc2_sqrt + eps;
        p = decay_mul * p - step_size * (m / denom);

        params[i]    = p;
        exp_avg[i]   = m;
        exp_avg_sq[i] = v;
    }
#endif

    return 0;
}
