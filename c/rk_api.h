#ifndef RK_API_H
#define RK_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* rk_rms_norm: y = x / sqrt(mean(x^2) + eps) * weight.
 * FP32 contiguous. x shape (B,T,C), weight shape (C,), y shape (B,T,C).
 * Math matches PyTorch's fp32 CPU path: mean = sum(x^2)/C, rms = sqrtf,
 * y = (x*weight)/rms. No -ffast-math.
 * Returns 0 on success, -1 on bad input. */
int rk_rms_norm(const float *x, const float *weight, float *y,
                int B, int T, int C, float eps);

/* rk_mm: C = A @ B, FP32 matmul (drop-in for torch.mm).
 * A shape (M,K), B shape (K,N), C shape (M,N). Two compile-time backends:
 *
 *   default (blis): BLIS-style MC=64/KC=256/NC=128 blocking,
 *                   MR=8/NR=16 AVX-512 FMA microkernel, OpenMP
 *                   collapse(2) over (MC,NC). Per-thread libbarrage
 *                   arena for packed panels. Tail handling zero-pads
 *                   rows/cols.
 *   cblas:          selected by -DRK_MM_CBLAS. Calls cblas_sgemm
 *                   (MKL / OpenBLAS / netlib BLAS); the BLAS vendor
 *                   manages threading and scratch. Bit-exact with
 *                   torch.mm when PyTorch links the same BLAS.
 *
 * Returns 0 on success, -1 on bad input. */
int rk_mm(const float *A, const float *B, float *C, int M, int K, int N);

/* rk_softmax: numerically-stable softmax along the last dim.
 * FP32 contiguous. x shape (B,T,V), y shape (B,T,V). Three passes per
 * row: max, exp+sum, divide. AVX-512 polynomial exp when __AVX512F__;
 * scalar expf otherwise. Only dim=-1 is supported.
 * Returns 0 on success, -1 on bad input. */
int rk_softmax(const float *x, float *y, int B, int T, int V);

/* rk_layer_norm: y = (x - mean) / sqrt(var + eps) * weight + bias.
 * FP32 contiguous. x shape (B,T,C), weight and bias shape (C,). bias may
 * be NULL (treated as zero, matches F.layer_norm with bias=None).
 * Two-pass: sum -> mean, sum((x-mean)^2) -> var -> rstd = 1/sqrtf(var+eps),
 * then normalize+scale+shift. rstd is computed once per row and multiplied
 * in (matches PyTorch's CPU kernel ordering, unlike rk_rms_norm which uses
 * per-element divide). Per-thread libbarrage arena for per-row scratch.
 * Returns 0 on success, -1 on bad input. */
int rk_layer_norm(const float *x, const float *weight, const float *bias,
                  float *y, int B, int T, int C, float eps);

/* rk_turbo_enter: activate "turbo" mode for the calling thread.
 * Probes libtopo, saves the current affinity mask, builds a physical-core
 * mask (skips hyperthreads), pins the calling thread to it, sets
 * OMP_NUM_THREADS / OMP_PROC_BIND=close / OMP_PLACES=cores, and calls
 * omp_set_num_threads(N). Does NOT swap PyTorch's allocator (needs a C++
 * extension to hook c10::Allocator).
 * Returns the physical core count on success (>= 1), -1 on failure
 * (libtopo init failure, affinity set failure, or nested call). */
int rk_turbo_enter(void);

/* rk_turbo_exit: deactivate "turbo" mode.
 * Restores the calling thread's CPU affinity mask. Does NOT unset the
 * OMP_* env vars or reset omp_set_num_threads (matches the user's likely
 * intent for repeated turbo() calls). The torch thread count restore is
 * the Python wrapper's responsibility.
 * Returns 0 on success (or if turbo wasn't active, idempotent), or a
 * negative errno-style code from topo_set_affinity on failure. */
int rk_turbo_exit(void);

/* rk_topo_detect: query the system's CPU/NUMA layout via EoSD libtopo.
 * Fills any non-NULL out-param with the corresponding field from
 * topo_info_t (see vendor/EoSD/libtopo/include/topo.h). Defensive
 * defaults of 1 per field if topo_probe fails.
 * Returns 0 on success, -1 on failure. */
int rk_topo_detect(unsigned *threads_per_core,
                   unsigned *cores_per_package,
                   unsigned *num_packages,
                   unsigned *num_numa_nodes,
                   unsigned *total_cpus);

/* rk_adamw_step: fused FP32 AdamW optimizer step (drop-in for one
 * torch.optim.AdamW step on a flat param vector).
 *
 *   m = beta1 * m + (1 - beta1) * grad
 *   v = beta2 * v + (1 - beta2) * grad^2
 *   denom = sqrt(v) / sqrt(1 - beta2^step) + eps
 *   p = p * (1 - lr * wd) - (lr / (1 - beta1^step)) * m / denom
 *
 * Single pass over the param vector, AVX-512 vectorised when
 * __AVX512F__. Decoupled weight decay, biased first/second moments,
 * bias-corrected step size. Matches torch.optim.AdamW numerics.
 *
 * Args:
 *   params, grads, exp_avg, exp_avg_sq: FP32 contiguous, length n.
 *     params, exp_avg, exp_avg_sq are updated in place.
 *   n:           element count (>= 1).
 *   lr:          learning rate (>= 0).
 *   beta1, beta2: Adam moment decay rates in [0, 1).
 *   eps:         denom stabiliser (>= 0).
 *   weight_decay: decoupled weight decay (>= 0).
 *   step:        1-indexed step number used for bias correction (>= 1).
 *
 * Returns 0 on success, -1 on bad input. */
int rk_adamw_step(float *params, const float *grads,
                  float *exp_avg, float *exp_avg_sq,
                  int n, float lr, float beta1, float beta2,
                  float eps, float weight_decay, int step);

/* rk_mm_qkv: fused QKV projection.
 *   Q = x @ W_q, K = x @ W_k, V = x @ W_v
 * where W_qkv is a pre-stacked (3, C, C) tensor with W_q at [0],
 * W_k at [1], W_v at [2]. Single C call saves two Python -> C dispatch
 * round-trips per layer vs three separate rk_mm calls. The underlying
 * matmul reuses rk_mm (blis or cblas backend, compile-time selected).
 *
 * Args:
 *   x:      FP32 contiguous, shape (M, C).
 *   W_qkv:  FP32 contiguous, shape (3, C, C). Row-major.
 *   Q, K, V: FP32 contiguous output buffers, each shape (M, C). May be
 *            NULL only when M == 0 or C == 0 (no-op).
 *   M, C:   dimensions (>= 0).
 *
 * Returns 0 on success, -1 on bad input or if any of the three rk_mm
 * calls fails. */
int rk_mm_qkv(const float *x, const float *W_qkv,
              float *Q, float *K, float *V,
              int M, int C);

#ifdef __cplusplus
}
#endif

#endif /* RK_API_H */
