/* rk_softmax.c -- softmax over last dim, FP32, (B,T,V).
 * Three passes per row: max, exp+sum, divide. AVX-512 polynomial exp
 * (Cody-Waite + degree-6 Taylor) when __AVX512F__ is set; scalar expf
 * otherwise. Math matches PyTorch's softmax_lastdim CPU kernel.
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

#ifdef __AVX512F__
#include <immintrin.h>
#define RK_HAS_AVX512 1
#else
#define RK_HAS_AVX512 0
#endif

#if RK_HAS_AVX512
/* rk_exp512_ps: AVX-512 polynomial exp for 16 FP32 lanes.
 * Cody-Waite range reduction + degree-6 Taylor + bit-cast 2^n.
 * ~1 ULP error for |x| < ~88; passes F.softmax at atol=1e-6. */
static inline __m512 rk_exp512_ps(__m512 xv)
{
    const __m512 inv_log2_v = _mm512_set1_ps(1.4426950408889634f);
    const __m512 log2_hi_v  = _mm512_set1_ps(0.693145751953125f);
    const __m512 log2_lo_v  = _mm512_set1_ps(1.4286067653301870e-6f);

    /* n = round(x * inv_log2) via add-copysign(0.5)-then-truncate (gcc's
     * <immintrin.h> doesn't expose _mm512_round_ps directly). The Cody-
     * Waite reduction below must use the truncated integer n, not the +0.5
     * intermediate nv. */
    const __m512 nx = _mm512_mul_ps(xv, inv_log2_v);
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    const __m512 nx_sign    = _mm512_and_ps(nx, sign_mask);
    const __m512 offset     = _mm512_or_ps(_mm512_set1_ps(0.5f), nx_sign);
    const __m512 nv         = _mm512_add_ps(nx, offset);
    const __m512i n_int     = _mm512_cvttps_epi32(nv);
    const __m512  n_float   = _mm512_cvtepi32_ps(n_int);

    /* r = x - n * log2_hi - n * log2_lo (Cody-Waite split-constant reduction).
     * _mm512_fnmadd_ps(a, b, c) = c - a*b. */
    __m512 r = _mm512_fnmadd_ps(n_float, log2_hi_v, xv);  /* r = xv - n*log2_hi */
    r = _mm512_fnmadd_ps(n_float, log2_lo_v, r);          /* r = r  - n*log2_lo */

    /* Degree-6 Taylor polynomial via Horner's scheme:
     *   poly = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r*1/720))))) */
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.1666666716337204f);   /* 1/6  */
    const __m512 c4 = _mm512_set1_ps(0.0416666679084301f);  /* 1/24 */
    const __m512 c5 = _mm512_set1_ps(0.0083333337679505f);  /* 1/120 */
    const __m512 c6 = _mm512_set1_ps(0.0013888889225196f);  /* 1/720 */

    __m512 poly = c6;
    poly = _mm512_fmadd_ps(r, poly, c5);   /* poly = c6*r + c5 */
    poly = _mm512_fmadd_ps(r, poly, c4);   /* poly = (c6*r + c5)*r + c4 */
    poly = _mm512_fmadd_ps(r, poly, c3);
    poly = _mm512_fmadd_ps(r, poly, c2);
    poly = _mm512_fmadd_ps(r, poly, c1);
    poly = _mm512_fmadd_ps(r, poly, c0);   /* poly = 1 + r*(... + 1) */

    /* 2^n via float bit manipulation: a float with exponent n + bias is
     *   bit_pattern = (n + 127) << 23
     * (n_int was computed above from the truncated nv.) */
    const __m512i bias    = _mm512_set1_epi32(127);
    const __m512i exp_bits = _mm512_add_epi32(n_int, bias);
    const __m512i shifted  = _mm512_slli_epi32(exp_bits, 23);
    const __m512  pow2n   = _mm512_castsi512_ps(shifted);

    /* exp(x) = poly * 2^n */
    return _mm512_mul_ps(poly, pow2n);
}
#endif  /* RK_HAS_AVX512 */

int rk_softmax(const float *x, float *y, int B, int T, int V)
{
    /* input validation */
    if (x == NULL || y == NULL) return -1;
    if (B <= 0 || T <= 0 || V <= 0) return -1;

    /* Cap B*T*V at INT32_MAX; we use int indices in the inner loops. */
    if ((int64_t)B * (int64_t)T > (int64_t)INT32_MAX / (int64_t)V) return -1;

    const int V_f = V;                 /* V as int (already validated) */

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *xb = x + (((long)b * (long)T + (long)t) * (long)V_f);
            float       *yb = y + (((long)b * (long)T + (long)t) * (long)V_f);

            /* Per-row max+sum scratch on the stack. 8 bytes, always L1.
             * Replaces the v0.5 barrage_alloc + TLS lookup + fallback
             * path that the audit (IMPROVEMENTS.md B2) flagged as net
             * negative. Holds [max, sum]. */
            float row_scratch[2] = { 0.0f, 0.0f };

            /* --- Pass 1: per-row max (numerically-stable anchor m) --- */
            float m;
#if RK_HAS_AVX512
            {
                /* AVX-512 vectorised max reduction. */
                __m512 maxv = _mm512_set1_ps(-INFINITY);
                int i = 0;
                for (; i + 16 <= V_f; i += 16) {
                    const __m512 xv = _mm512_loadu_ps(xb + i);
                    maxv = _mm512_max_ps(maxv, xv);
                }
                m = _mm512_reduce_max_ps(maxv);
                for (; i < V_f; ++i) {
                    if (xb[i] > m) m = xb[i];
                }
            }
#else
            /* Scalar fallback path. OpenMP omp simd reduction(max:m) lets
             * -O3 -march=native emit AVX2 vmaxps ymm. */
            m = -INFINITY;
            #pragma omp simd reduction(max:m)
            for (int i = 0; i < V_f; ++i) {
                m = (xb[i] > m) ? xb[i] : m;
            }
#endif
            row_scratch[0] = m;

            /* --- Pass 2: compute y[i] = exp(x[i] - m), accumulate sum s. --- */
            const float row_max = row_scratch[0];
            float s = 0.0f;
            int i = 0;
#if RK_HAS_AVX512
            {
                /* AVX-512 vectorised path: 16 floats per iter via the
                 * rk_exp512_ps polynomial approximation. */
                const __m512 mv = _mm512_set1_ps(row_max);
                __m512 sv = _mm512_setzero_ps();
                for (; i + 16 <= V_f; i += 16) {
                    const __m512 xv      = _mm512_loadu_ps(xb + i);
                    const __m512 xv_sub  = _mm512_sub_ps(xv, mv);
                    const __m512 ev      = rk_exp512_ps(xv_sub);
                    _mm512_storeu_ps(yb + i, ev);
                    sv = _mm512_add_ps(sv, ev);
                }
                s = _mm512_reduce_add_ps(sv);
            }
#endif
            /* Scalar tail (AVX-512 path) or full loop (AVX2 path).
             * Uses libm expf. */
            #if !RK_HAS_AVX512
            #pragma omp simd reduction(+:s)
            #endif
            for (; i < V_f; ++i) {
                const float e = expf(xb[i] - row_max);
                yb[i] = e;
                s += e;
            }
            row_scratch[1] = s;

            /* --- Pass 3: normalise by sum (per-element divide). --- */
            const float row_sum = row_scratch[1];
#if RK_HAS_AVX512
            {
                const __m512 sv_v = _mm512_set1_ps(row_sum);
                int j = 0;
                for (; j + 16 <= V_f; j += 16) {
                    const __m512 yv = _mm512_loadu_ps(yb + j);
                    _mm512_storeu_ps(yb + j, _mm512_div_ps(yv, sv_v));
                }
                for (; j < V_f; ++j) {
                    yb[j] = yb[j] / row_sum;
                }
            }
#else
            /* Scalar fallback path. OpenMP omp simd vectorises with AVX2. */
            #pragma omp simd
            for (int j = 0; j < V_f; ++j) {
                yb[j] = yb[j] / row_sum;
            }
#endif
        }
    }

    return 0;
}
