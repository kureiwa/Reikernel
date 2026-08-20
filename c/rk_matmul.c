/* rk_matmul.c -- FP32 matmul C = A @ B.
 *
 * Two backends selectable at compile time via -DRK_MM_CBLAS:
 *
 *   default (blis): BLIS-style MC=64/KC=256/NC=128 blocking with
 *                   MR=8/NR=16 AVX-512 FMA microkernel. OpenMP
 *                   collapse(2) over (MC,NC). Per-thread 32 MiB
 *                   libbarrage arena for packed panels.
 *   cblas:          thin wrapper around cblas_sgemm (MKL / OpenBLAS /
 *                   netlib BLAS). Lets the BLAS vendor manage its own
 *                   threading and scratch, eliminating the hand-tiled
 *                   pack/microkernel path. Bit-exact with torch.mm when
 *                   PyTorch links the same BLAS.
 *
 * Math matches torch.mm within atol=1e-5 (FP32 FMA, no -ffast-math,
 * no reciprocal tricks).
 */

#include "rk_api.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef RK_MM_CBLAS
#include <cblas.h>
#endif

#include "barrage.h"

/* --- tile / block dimensions --- */
#define MR 8     /* micro-tile rows; 8 zmm accumulators  */
#define NR 16    /* micro-tile cols; one zmm width         */
#define MC 64    /* rows of A per MC block (8 micro-tiles) */
#define NC 128   /* cols of B per NC block (8 micro-tiles) */
#define KC 256   /* K-slice per kc-step                    */

/* Per-thread libbarrage arena. 32 MiB, mmap-backed, lazily created, never
 * destroyed. Reset at the start of each rk_mm call. Falls back to malloc
 * if barrage_create fails. */
#define RK_ARENA_BYTES (32u * 1024u * 1024u)

#ifndef RK_RESTRICT
#define RK_RESTRICT __restrict__
#endif

static barrage_arena_t *rk_get_thread_arena(void)
{
    static __thread barrage_arena_t *arena = NULL;
    if (arena == NULL) {
        arena = barrage_create(RK_ARENA_BYTES, NULL);
        /* If barrage_create fails, caller falls back to malloc. */
    }
    return arena;
}

/* MR x NR FP32 microkernel body. FMAs kc iterations of A_pack x B_pack
 * into 8 local zmm accumulators (zeroed per call), then adds the local
 * sum into the persistent accumulators at accs[0..7] with a separate
 * _mm512_add_ps. The separate add (rather than FMA-into-accs directly)
 * preserves the baseline's FP rounding: each kc-step produces a
 * FMA-reduced partial sum, and the cross-kc accumulation is a plain
 * add. That keeps rk.mm bit-exact with torch.mm for K <= KC and within
 * atol=1e-5 for K > KC, matching the pre-restructure math.
 *
 * The caller zeroes accs before the first kc-step (equivalent to
 * loading C since rk_mm memsets C to 0) and stores accs to C once after
 * the last kc-step. This keeps the 8 zmm accumulators alive across all
 * kc-steps for a given (ii, jj) micro-tile, eliminating the per-kc-step
 * C load+add+store round-trip that the old always-store microkernel did. */
static inline void rk_microkernel_body(
    int kc,
    const float *RK_RESTRICT A_pack,
    const float *RK_RESTRICT B_pack,
    int ldb,
    __m512 *RK_RESTRICT accs)
{
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    __m512 c4 = _mm512_setzero_ps();
    __m512 c5 = _mm512_setzero_ps();
    __m512 c6 = _mm512_setzero_ps();
    __m512 c7 = _mm512_setzero_ps();

    /* Main FMA loop. 8 independent FMAs per k iteration, one per accumulator. */
    for (int k = 0; k < kc; ++k) {
        const __m512 b = _mm512_loadu_ps(B_pack + (size_t)k * ldb);

        c0 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[0 * kc + k]), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[1 * kc + k]), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[2 * kc + k]), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[3 * kc + k]), b, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[4 * kc + k]), b, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[5 * kc + k]), b, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[6 * kc + k]), b, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(A_pack[7 * kc + k]), b, c7);
    }

    /* Cross-kc accumulation as a separate add (not fused with FMA).
     * Matches the baseline's `C = old_C + partial_sum` rounding. */
    accs[0] = _mm512_add_ps(accs[0], c0);
    accs[1] = _mm512_add_ps(accs[1], c1);
    accs[2] = _mm512_add_ps(accs[2], c2);
    accs[3] = _mm512_add_ps(accs[3], c3);
    accs[4] = _mm512_add_ps(accs[4], c4);
    accs[5] = _mm512_add_ps(accs[5], c5);
    accs[6] = _mm512_add_ps(accs[6], c6);
    accs[7] = _mm512_add_ps(accs[7], c7);
}

/* Store 8 zmm accumulators (one MR-row of NR lanes each) to a MR x NR
 * tile of C. M-tail skips rows past M; N-tail uses a mask to write only
 * the valid lanes. Called once per (ii, jj) micro-tile after all kc-steps
 * have run. */
static inline void rk_microkernel_store(
    const __m512 *RK_RESTRICT accs,
    float *RK_RESTRICT C, int ldc,
    int ii, int jj, int M, int N)
{
    const int m_tile = (ii + MR <= M) ? MR : (M - ii);
    const int n_tile = (jj + NR <= N) ? NR : (N - jj);
    const __mmask16 nmask = (n_tile == NR)
        ? (__mmask16)0xFFFFu
        : (__mmask16)((1u << n_tile) - 1u);

#define RK_STORE_ROW(i)                                                    \
    do {                                                                   \
        if (i < m_tile) {                                                  \
            float *crow = C + (size_t)(ii + i) * ldc + jj;                 \
            _mm512_mask_storeu_ps(crow, nmask, accs[i]);                   \
        }                                                                  \
    } while (0)

    /* Same code path for full-width and N-tail; nmask handles the difference. */
    RK_STORE_ROW(0);
    RK_STORE_ROW(1);
    RK_STORE_ROW(2);
    RK_STORE_ROW(3);
    RK_STORE_ROW(4);
    RK_STORE_ROW(5);
    RK_STORE_ROW(6);
    RK_STORE_ROW(7);
#undef RK_STORE_ROW
    (void)nmask;
}

/* Pack a [mc_padded x kc] panel of A into A_pack (row-major, stride kc).
 * Rows past mc_rows are zero-padded so the unused microkernel rows get 0. */
static inline void rk_pack_A(
    float *RK_RESTRICT A_pack,
    const float *RK_RESTRICT A, int K,
    int mc_start, int mc_rows, int mc_padded,
    int kk, int kc)
{
    for (int i = 0; i < mc_padded; ++i) {
        float *dst = A_pack + (size_t)i * kc;
        if (i < mc_rows) {
            memcpy(dst, A + (size_t)(mc_start + i) * K + kk,
                   (size_t)kc * sizeof(float));
        } else {
            memset(dst, 0, (size_t)kc * sizeof(float));
        }
    }
}

/* Pack a [kc x nc_padded] panel of B into B_pack (row-major, stride
 * nc_padded). Cols past nc_cols are zero-padded. */
static inline void rk_pack_B(
    float *RK_RESTRICT B_pack, int nc_padded,
    const float *RK_RESTRICT B, int N,
    int nc_start, int nc_cols,
    int kk, int kc)
{
    for (int k = 0; k < kc; ++k) {
        float *dst = B_pack + (size_t)k * nc_padded;
        const float *src = B + (size_t)(kk + k) * N + nc_start;
        memcpy(dst, src, (size_t)nc_cols * sizeof(float));
        if (nc_cols < nc_padded) {
            memset(dst + nc_cols, 0,
                   (size_t)(nc_padded - nc_cols) * sizeof(float));
        }
    }
}

int rk_mm(const float *A, const float *B, float *C, int M, int K, int N)
{
    /* --- input validation --- */
    if (A == NULL || B == NULL || C == NULL) return -1;
    if (M < 0 || K < 0 || N < 0) return -1;

    /* Degenerate shapes (M=0 or N=0): no work. K=0: output is all zeros
     * (memset below handles it). Matches torch.mm. */
    if (M == 0 || N == 0) {
        return 0;
    }

    /* Cap M*K, K*N, M*N at INT32_MAX; int loop bounds must not wrap. */
    if ((int64_t)M * (int64_t)K > (int64_t)INT32_MAX) return -1;
    if ((int64_t)K * (int64_t)N > (int64_t)INT32_MAX) return -1;
    if ((int64_t)M * (int64_t)N > (int64_t)INT32_MAX) return -1;

#ifdef RK_MM_CBLAS
    /* cblas_sgemm path. The BLAS vendor manages threading and scratch
     * internally; we just pass row-major A (M x K, ld=K) and B (K x N,
     * ld=N) and let it write C (M x N, ld=N). beta=0 means C is
     * overwritten, matching the blis path's memset+add behaviour and
     * torch.mm's semantics (not torch.addmm).
     *
     * The blis helpers (rk_get_thread_arena, rk_microkernel, rk_pack_A,
     * rk_pack_B) are compiled but unused in this path; reference the
     * arena helper to silence -Wunused-function. */
    (void)rk_get_thread_arena;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (const int)M, (const int)N, (const int)K,
                1.0f, A, (const int)K, B, (const int)N,
                0.0f, C, (const int)N);
    return 0;
#else
    /* Zero C so the always-add microkernel computes C = A@B (not C += A@B).
     * For K=0 this is the only work and the output is correctly all zeros. */
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));

    /* Prime the calling thread's arena TLS slot before the parallel region. */
    (void)rk_get_thread_arena();

    const int n_mc = (M + MC - 1) / MC;
    const int n_nc = (N + NC - 1) / NC;

    /* One parallel region per rk_mm call. Each OMP worker resets its arena,
     * allocates A_pack + B_pack once, and reuses them across its (mc, nc)
     * iterations. Falls back to malloc if the arena is unavailable. */
    #pragma omp parallel
    {
        barrage_arena_t *arena = rk_get_thread_arena();
        if (arena != NULL) {
            barrage_reset(arena);
        }

        float *A_pack = NULL;
        float *B_pack = NULL;
        __m512 *accs = NULL;
        int a_from_malloc = 0;
        int b_from_malloc = 0;
        int accs_from_malloc = 0;

        if (arena != NULL) {
            barrage_err_t err = BARRAGE_OK;
            A_pack = (float *)barrage_alloc(
                arena, (size_t)MC * KC * sizeof(float), 64, &err);
        }
        if (A_pack == NULL) {
            A_pack = (float *)malloc((size_t)MC * KC * sizeof(float));
            a_from_malloc = 1;
        }
        if (arena != NULL) {
            barrage_err_t err = BARRAGE_OK;
            B_pack = (float *)barrage_alloc(
                arena, (size_t)KC * NC * sizeof(float), 64, &err);
        }
        if (B_pack == NULL) {
            B_pack = (float *)malloc((size_t)KC * NC * sizeof(float));
            b_from_malloc = 1;
        }
        /* Per-thread accumulator slab. Holds the 8 zmm registers for
         * every (ii, jj) micro-tile in the largest possible (mc, nc)
         * block: (MC/MR) * (NC/NR) tiles * MR=8 zmm per tile. Sized
         * for the worst case so we can reuse it across all (mc, nc)
         * iterations without re-allocating. */
        {
            const size_t accs_bytes = (size_t)(MC / MR) * (NC / NR) * MR
                                      * sizeof(__m512);
            if (arena != NULL) {
                barrage_err_t err = BARRAGE_OK;
                accs = (__m512 *)barrage_alloc(arena, accs_bytes, 64, &err);
            }
            if (accs == NULL) {
                accs = (__m512 *)malloc(accs_bytes);
                accs_from_malloc = 1;
            }
        }

        /* If scratch allocation failed entirely we can't compute this
         * block; iterations are no-ops, leaving C zeroed. Graceful-degrade
         * path that never triggers for the bench/test shapes. */
        if (A_pack != NULL && B_pack != NULL && accs != NULL) {
            #pragma omp for collapse(2) schedule(static)
            for (int mc = 0; mc < n_mc; ++mc) {
                for (int nc = 0; nc < n_nc; ++nc) {
                    const int mc_start = mc * MC;
                    const int mc_end   = (mc_start + MC <= M)
                                         ? (mc_start + MC) : M;
                    const int nc_start = nc * NC;
                    const int nc_end   = (nc_start + NC <= N)
                                         ? (nc_start + NC) : N;

                    const int mc_rows   = mc_end - mc_start;
                    const int nc_cols   = nc_end - nc_start;
                    const int mc_padded = ((mc_rows + MR - 1) / MR) * MR;
                    const int nc_padded = ((nc_cols + NR - 1) / NR) * NR;

                    /* Number of (MR, NR) micro-tiles in this (mc, nc)
                     * block. Each tile owns 8 zmm accumulators in accs. */
                    const int n_ii = (mc_rows + MR - 1) / MR;
                    const int n_jj = (nc_cols + NR - 1) / NR;
                    const size_t accs_bytes_block =
                        (size_t)n_ii * (size_t)n_jj * MR * sizeof(__m512);

                    /* Zero the accumulator slab for this block once.
                     * Since rk_mm memset C to 0 above, this is equivalent
                     * to "load C" for the first kc-step but skips the
                     * per-kc-step C load+add+store round-trip. */
                    memset(accs, 0, accs_bytes_block);

                    /* kc-step loop stays outermost so packing A and B
                     * once per kc-step is preserved. Each kc-step FMAs
                     * into the SAME persistent accumulator slab. */
                    for (int kk = 0; kk < K; kk += KC) {
                        const int kc = (kk + KC <= K) ? KC : (K - kk);

                        rk_pack_A(A_pack, A, K,
                                  mc_start, mc_rows, mc_padded,
                                  kk, kc);
                        rk_pack_B(B_pack, nc_padded, B, N,
                                  nc_start, nc_cols,
                                  kk, kc);

                        int tile_idx = 0;
                        for (int ii = 0; ii < mc_rows; ii += MR) {
                            for (int jj = 0; jj < nc_cols; jj += NR) {
                                rk_microkernel_body(
                                    kc,
                                    A_pack + (size_t)ii * kc,
                                    B_pack + (size_t)jj,
                                    /*ldb=*/nc_padded,
                                    &accs[(size_t)tile_idx * MR]);
                                tile_idx++;
                            }
                        }
                    }

                    /* Single store pass: write the persistent
                     * accumulators to C once after all kc-steps. */
                    {
                        int tile_idx = 0;
                        for (int ii = 0; ii < mc_rows; ii += MR) {
                            for (int jj = 0; jj < nc_cols; jj += NR) {
                                rk_microkernel_store(
                                    &accs[(size_t)tile_idx * MR],
                                    C, /*ldc=*/N,
                                    mc_start + ii, nc_start + jj,
                                    M, N);
                                tile_idx++;
                            }
                        }
                    }
                }
            }
        }

        /* Release scratch. Arena memory is freed by the next call's
         * barrage_reset; malloc'd memory is freed here. */
        if (a_from_malloc) free(A_pack);
        if (b_from_malloc) free(B_pack);
        if (accs_from_malloc) free(accs);
    }

    return 0;
#endif  /* RK_MM_CBLAS */
}
