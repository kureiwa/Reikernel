/* rk_qkv.c -- fused QKV projection.
 *
 * Given input x of shape (M, C) and pre-stacked weights W_qkv of shape
 * (3, C, C), compute Q = x @ W_q, K = x @ W_k, V = x @ W_v in a single
 * C call. Saves two Python -> C dispatch round-trips per layer vs three
 * separate rk.mm calls. The underlying matmul reuses rk_mm (blis or
 * cblas backend, selected at compile time).
 *
 * W_qkv layout (row-major): [W_q (C x C) | W_k (C x C) | W_v (C x C)],
 * i.e. W_qkv[0] = W_q, W_qkv[1] = W_k, W_qkv[2] = W_v. The user stacks
 * the three weight matrices once at model init.
 *
 * Returns 0 on success, -1 on bad input or if any of the three rk_mm
 * calls fails.
 */

#include "rk_api.h"

#include <stddef.h>
#include <stdint.h>

int rk_mm_qkv(const float *x, const float *W_qkv,
              float *Q, float *K, float *V,
              int M, int C)
{
    /* --- input validation --- */
    if (x == NULL || W_qkv == NULL
        || Q == NULL || K == NULL || V == NULL) return -1;
    if (M < 0 || C < 0) return -1;

    /* Degenerate shapes (M=0): no work. Matches torch.mm. */
    if (M == 0) {
        return 0;
    }
    if (C == 0) {
        return 0;
    }

    /* Cap M*C at INT32_MAX; rk_mm also checks but fail fast here. */
    if ((int64_t)M * (int64_t)C > (int64_t)INT32_MAX) return -1;
    /* Cap 3*C*C (the W_qkv size) at INT32_MAX to avoid overflow when
     * computing the per-matrix offset below. */
    if ((int64_t)3 * (int64_t)C * (int64_t)C > (int64_t)INT32_MAX) return -1;

    /* W_qkv is (3, C, C) row-major. The three weight matrices are
     * contiguous: W_q at offset 0, W_k at offset C*C, W_v at 2*C*C. */
    const float *W_q = W_qkv;
    const float *W_k = W_qkv + (size_t)C * (size_t)C;
    const float *W_v = W_qkv + (size_t)2 * (size_t)C * (size_t)C;

    /* Q = x @ W_q, K = x @ W_k, V = x @ W_v. Each is (M, C) = (M, C) @ (C, C). */
    int rc;
    rc = rk_mm(x, W_q, Q, M, C, C);
    if (rc != 0) return rc;
    rc = rk_mm(x, W_k, K, M, C, C);
    if (rc != 0) return rc;
    rc = rk_mm(x, W_v, V, M, C, C);
    if (rc != 0) return rc;

    return 0;
}
