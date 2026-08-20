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

#ifdef __cplusplus
}
#endif

#endif /* RK_API_H */
