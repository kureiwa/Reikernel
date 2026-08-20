# Reikernel Improvement Audit

Read-only audit of Reikernel v0.5. No files were modified.

Scope: `README.md`, `python/reikernel.py`, `c/*.c` + `c/rk_api.h`, `Makefile`,
`bench/results/*.json`, plus `vendor/EoSD/libbarrage/{include,src}` for arena
semantics.

---

## 1. Correctness Audit

### 1.1 Bugs found

**B1 (low–medium severity) — `rk_parse_cpulist` over-read in `rk_turbo.c`**

`rk_parse_cpulist` (c/rk_turbo.c:66-100) is documented to return a count that
may exceed `max_out`:

```c
for (unsigned long c = a; c <= b; c++) {
    if (count < max_out && out != NULL) {
        out[count] = (unsigned)c;
    }
    count++;      // ← increments past max_out
}
return (int)count;
```

The only caller, `rk_build_physical_mask` (c/rk_turbo.c:150-161), then reads
`siblings[i]` for `i` in `[0, n)` without clamping `n`:

```c
int n = rk_parse_cpulist(buf, siblings, 256);
if (n <= 0) { ... }
unsigned min_cpu = siblings[0];
for (int i = 1; i < n; i++) {       // ← n can be > 256
    if (siblings[i] < min_cpu) min_cpu = siblings[i];
}
```

If a CPU's `thread_siblings_list` ever contains more than 256 entries (e.g. a
hypervisor exposing a 256-vCPU vNuma node as one "core", or a 1024-thread
SMT-8 box), this reads uninitialized stack. The non-SMT 2-vCPU sandbox always
returns `n=1`, so this never fires in the bench, but it is a latent bug.

Fix: clamp the return value of `rk_parse_cpulist` to `max_out`, or check
`n <= 256` before the loop in the caller.

**B2 (design defect, not a crash) — norm-op arenas never reset, fill up,
then silently fall back to stack**

`rk_rms_norm`, `rk_softmax`, and `rk_layer_norm` each bump-allocate 4–8 bytes
per row from a 4 MiB per-thread arena and explicitly never call
`barrage_reset`:

```c
/* Don't reset; keep the bump pointer warm for the next call. */
```

(c/rk_norm.c:96, c/rk_softmax.c:207, c/rk_layernorm.c:108)

The comment is inverted: keeping the bump pointer high is the *opposite* of
arena reuse. Each call consumes `B*T*(4 or 8)` bytes:

| Op           | Per-call arena use (shape in bench) | Calls to exhaust 4 MiB |
|--------------|-------------------------------------|------------------------|
| rms_norm     | 4 * 256 * 4 B = 4 KB                | ~1024                  |
| softmax      | 4 * 256 * 8 B = 8 KB                | ~512                   |
| layer_norm   | 4 * 256 * 8 B = 8 KB                | ~512                   |

The training bench ran 2365 steps. Each step calls the norm ops multiple times
(forward + backward across 4 layers). The arena exhausts within the first
~10–20 steps and `barrage_alloc` returns `BARRAGE_ERR_OUT_OF_SPACE` for every
subsequent call. The fallback path then runs:

```c
float stack_scratch[2] = { 0.0f, 0.0f };
if (pscratch == NULL) pscratch = stack_scratch;
```

This is *correct* (the stack scratch holds the same per-row scalar) but means
the arena does nothing for >99% of training. Worse, the `barrage_alloc` call,
the TLS lookup, and the error-code out-param still execute per row — pure
overhead with no payoff. See §3 for the fix.

**B3 (cosmetic) — Misleading constant names in `rk_exp512_ps`**

c/rk_softmax.c:42-44 names the constants `log2_hi_v` and `log2_lo_v`, but
they are `ln(2)` (natural log of 2), not `log2(2)=1`:

```c
const __m512 inv_log2_v = _mm512_set1_ps(1.4426950408889634f); // = 1/ln(2) ✓
const __m512 log2_hi_v  = _mm512_set1_ps(0.693145751953125f);  // = ln(2)_hi, NOT log2
const __m512 log2_lo_v  = _mm512_set1_ps(1.4286067653301870e-6f); // = ln(2)_lo
```

The math is correct (Cody-Waite reduction of `e^x = 2^n * e^r` where
`r = x - n*ln(2)`), but the names invite a future maintainer to "fix" them and
break the kernel. Rename to `ln2_hi_v` / `ln2_lo_v`.

### 1.2 Things that look like bugs but aren't

- **`_mm512_maskz_loadu_ps` on the N-tail of `rk_microkernel`** reads 16
  floats even when only `n_tile < 16` are valid. AVX-512 masked loads suppress
  faults on masked-out lanes, so reading past the row into the next row (or
  past the array on the last row) does not segfault. The matching
  `_mm512_mask_storeu_ps` only writes `n_tile` lanes, so no corruption.
  Correct.

- **`1u << n_tile` in the nmask computation** (c/rk_matmul.c:82-84). For
  `n_tile == NR == 16`, the `n_tile == NR` branch sets `nmask = 0xFFFF`
  without shifting, so `1u << 16` is never evaluated. No UB.

- **`(int64_t)B * (int64_t)T > (int64_t)INT32_MAX / (int64_t)C`** overflow
  guard in the norm ops. `C > 0` is checked first, so the division is safe.

- **`setenv("OMP_NUM_THREADS", ...)` inside `rk_turbo_enter`** is not
  thread-safe (mutates `environ`), but `turbo()` is entered from the Python
  main thread under the GIL, so no race in practice.

### 1.3 Numerical divergence in training (worth flagging in README)

`bench/results/train_bench.json` reports `val_bpb_diff = 0.069`
(reikernel 6.94 vs baseline 6.87). The README claims "all ops match PyTorch
within `atol=1e-5`" — true per-op, but over 2365 steps the FP-ordering
differences compound into a measurable validation-loss gap. Sources:

1. `rk_rms_norm` computes `(x * weight) / rms`; PyTorch computes
   `(x / rms) * weight`. Different rounding.
2. `rk_softmax` uses a degree-6 polynomial `exp` (~1 ULP error) instead of
   `libm expf`.
3. `rk_mm` FP accumulation order differs from MKL's.

This is expected for any FP-precision kernel swap, not a bug, but the README
should mention it. A `atol=1e-5` per-op test does not guarantee training-loss
parity.

---

## 2. Per-op Analysis

| Op            | Speedup vs PyTorch | Bottleneck in rk op                | Headroom |
|---------------|-------------------:|------------------------------------|----------|
| `rk.rms_norm` |              3.32x | ATen dispatch (removed)            | Low     |
| `rk.mm`       |         0.49–0.63x | Hand-tiled GEMM math < MKL         | **High**|
| `rk.softmax`  |         1.14–1.19x | `expf` + 3 passes (already AVX-512)| Medium  |
| `rk.layer_norm`|        1.03–1.60x | 2-pass + FMA chain (already simd)  | Low     |
| `rk.turbo()`  |              1.02x | No-op on 2-vCPU sandbox            | N/A     |

### 2.1 Which op has the most room?

**`rk.mm`**, by far. For matmul, PyTorch's dispatch overhead is a tiny
fraction of total time — the actual SGEMM math dominates. MKL's JIT-compiled
AVX-512 kernels beat the hand-tiled C by ~2x. The other ops win primarily by
removing dispatch overhead; their math is already at parity with PyTorch's
vectorised kernels.

### 2.2 Where dispatch is NOT the bottleneck

- `rk.mm` at all three bench shapes. The 128³ case takes 42 us in rk vs 21 us
  in torch.mm — the ~5 us ATen dispatch overhead is dwarfed by the 20 us math
  gap.
- `rk.softmax` at `(4, 256, 1024)`: 518 us rk vs 592 us torch. The 1.14x win
  is mostly the AVX-512 polynomial `exp`, not dispatch removal — PyTorch's
  softmax is already AVX-512 vectorised.
- `rk.layer_norm` at `(4, 256, 320)`: 92 us rk vs 95 us torch. Only 1.03x —
  the dispatch overhead is ~3 us and the math is identical.

### 2.3 Why `rk.mm` loses to MKL (expected?)

Largely yes, but it is worse than it needs to be. Two fixable issues beyond
"MKL is just better":

1. **Loop nesting** (c/rk_matmul.c:240-262). The `kk` (K-block) loop is
   *outermost*; the `(ii, jj)` micro-tile loop is inside it. The microkernel
   zeroes its 8 zmm accumulators, runs `kc` FMAs, then **loads C, adds, stores
   back** on every kc-step. For `K=512, KC=256`, that is 2 load+stores per
   micro-tile instead of 1. Standard BLIS keeps the C tile in registers (or
   L1) across all kc-steps and stores once. Estimated cost on (512,512,512):
   ~65k extra L1 load/stores ≈ 87 us ≈ 4% of the 2.09 ms total. Not dominant,
   but free to fix.

2. **Pack-then-microkernel structure**. Each (mc,nc) block re-packs A and B
   for every kc-step. The packs are `memcpy`-bound (8 MB of copies per
   (512,512,512) matmul). MKL avoids this with JIT kernels that read directly
   from the user's layout.

### 2.4 Should `rk.mm` switch to MKL/BLAS?

**Yes, with caveats.** PyTorch on x86_64 already links MKL (or OpenBLAS).
`libreikernel.so` could call `cblas_sgemm` directly from `rk_mm`, getting
MKL-speed GEMM with zero ATen dispatch overhead.

Tradeoffs:

- **Pro**: ~2x faster matmul. Closes the 0.49–0.63x gap. Likely the single
  biggest training-speedup lever (see §7).
- **Pro**: No scratch allocation needed (MKL manages its own buffers).
- **Pro**: Bit-exact with `torch.mm` (same backend) — eliminates one source
  of training-loss divergence.
- **Con**: Adds a link-time dependency on MKL/BLAS. PyTorch's MKL is
  dynamically loaded; `libreikernel.so` would need `-lmkl_rt` (or
  `-lopenblas`).
- **Con**: MKL has its own threading model. Calling `cblas_sgemm` with
  `MKL_SET_NUM_THREADS` mismatched vs `omp_set_num_threads` can oversubscribe
  the 2-vCPU sandbox. Set `MKL_NUM_THREADS=2` explicitly.
- **Con**: Loses the "educational hand-tiled matmul" framing. Could keep the
  hand-tiled path behind a `RK_MM_BACKEND=blis|mkl` compile flag.

For a 2-vCPU sandbox with 128–512 dim matmuls, MKL's threading overhead may
eat the win at the smallest shape (128³). A hybrid — MKL for `M*N >= 64*64`,
hand-tiled for smaller — would be safest.

---

## 3. Arena Usage Analysis

### 3.1 `rk.mm` — used effectively

- 32 MiB per-thread arena. `barrage_reset` called at the top of each parallel
  region (c/rk_matmul.c:192). Each OMP worker allocates `A_pack` (64 KB) +
  `B_pack` (128 KB) = 192 KB once, reuses across its `(mc,nc)` iterations.
- Arena is 32 MiB but only 192 KB is used per thread (0.6%). Over-provisioned
  but harmless (mmap is lazy).
- **Verdict: correct and beneficial.** The 192 KB scratch is too large for
  the stack and would thrash glibc malloc if allocated per-call.

### 3.2 `rk.rms_norm` / `rk.softmax` / `rk.layer_norm` — used ineffectively

- 4 MiB per-thread arena. Each row allocates 4–8 bytes (a single scalar or
  pair of scalars: max, sum, mean, rstd).
- Never reset (B2). Fills after ~500–1024 calls, then falls back to stack.
- Per-row cost: TLS lookup (`static __thread`) + `barrage_alloc` call +
  error-out-param write + pointer check. For 4–8 bytes that should be a
  stack slot.
- **Verdict: net negative.** The arena adds overhead without benefit. These
  ops would be faster and simpler with plain stack scratch:

```c
float row_scratch[2];  /* max, sum — fits in 8 bytes, always L1 */
```

The arena pattern only pays off when the per-call scratch is large enough
that `malloc`/`free` overhead matters (≥ a few KB) AND the same arena is
reused across many calls (i.e. `barrage_reset` between calls). Neither holds
for the norm ops.

### 3.3 Is the arena "wiring softmax and layernorm together"?

The README claims `rk.mm` "wires the libbarrage arena path that softmax and
layernorm reuse." This is aspirational, not actual: each op has its own
`static __thread barrage_arena_t *arena` (separate TLS slot per translation
unit). They do not share an arena. And per §3.2, the norm ops don't benefit
from their arenas anyway.

---

## 4. Missing Ops (ranked by impact)

| # | Op                              | Impact | Effort | Why                                                                 |
|--:|---------------------------------|--------|--------|---------------------------------------------------------------------|
| 1 | **Fused QKV projection**        | High   | Med    | 3 matmuls → 1. Saves 2 dispatches + 2 allocations per layer.        |
| 2 | **Fused attention (QK^T→sm→V)** | High   | High   | 3 ops → 1. Enables online-softmax (FlashAttention-style), huge mem. |
| 3 | **Fused AdamW step**            | High   | Med-Hi | PyTorch AdamW = ~6 ops/param. Fuse → 1. For 1M params, saves ~5M ops/step. |
| 4 | **Fused MLP (mm→act→mm)**       | Med    | Med    | 2 matmuls + GELU. Saves 1 intermediate allocation per layer.        |
| 5 | **Rotary position embedding**   | Med    | Low    | Common in modern GPT. Currently pure Python with multiple ops.      |
| 6 | **SwiGLU / GLU activation**     | Med    | Low    | `matmul * silu(matmul)`. One fused op vs three.                     |
| 7 | **Fused addmm (matmul + bias)** | Low    | Low    | PyTorch has it; marginal value over `rk.mm` + `torch.add`.          |
| 8 | **Fused embedding + norm**      | Low    | Med    | Embedding lookup is memory-bound, not dispatch-bound. Little win.   |

For the 1M GPT model in the training bench (4 layers, 128 d, 2 heads, 257
vocab), the highest-leverage adds are #1 (QKV) and #3 (AdamW):

- **QKV**: 4 layers × (fwd + bwd) × 2 saved dispatches = 16 dispatches/step.
  At ~5 us/dispatch, ~80 us/step. ~0.3% of 22.9 ms step. Smaller than I
  expected — the win is more about removing intermediate allocations than
  dispatches.

- **AdamW**: 1M params / step. PyTorch's `torch.optim.AdamW` does
  `exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)` + similar for `exp_avg_sq`
  + `addcdiv` + weight decay + clip. ~6 elementwise ops over the full param
  tensor. Each is ~5 us dispatch + memory pass. Fusing → 1 pass. Saves
  ~25 us/step × 5 = ~125 us/step. ~0.5% of step.

So fused ops give modest gains on this tiny model. The bigger win for THIS
model is fixing `rk.mm` (§2.4).

---

## 5. SIMD Opportunities

### 5.1 Already using explicit AVX-512

- `rk_mm` microkernel: 8 zmm accumulators, `_mm512_fmadd_ps`, masked
  load/store for tails. Standard BLIS-style.
- `rk_softmax`: `_mm512_max_ps`, polynomial `_mm512_fmadd_ps` exp,
  `_mm512_reduce_max_ps` / `_mm512_reduce_add_ps`, `_mm512_div_ps`.

### 5.2 Using `#pragma omp simd` (auto-vectorised by `-march=native`)

- `rk_rms_norm`: `reduction(+:sum)` + simd. gcc emits zmm `vfmadd231ps`.
- `rk_layer_norm`: same pattern, 2-pass (mean, then var), then FMA chain.
  Auto-vectorises well.

### 5.3 Opportunities

1. **`rk_rms_norm` divide pass** (c/rk_norm.c:89-92). Currently
   `yb[i] = (xb[i] * weight[i]) / rms`. Auto-vectorises to `vmulps + vdivps`.
   `_mm512_divps` is ~14-20 cycles (Haswell/Skylake). Replacing with
   `_mm512_rcp14_ps` (1 cycle, ~12-bit precision) + one Newton iteration
   (`vfnmadd231 + vfmadd213`) gives ~4 ULP — but breaks `atol=1e-5` for some
   inputs. **Skip unless accuracy budget allows.**

2. **`rk_layer_norm` FMA chain** (c/rk_layernorm.c:97). Already fuses to
   2 FMAs at `-O3`. Explicit intrinsics would not help.

3. **`rk_mm` microkernel register count**. Currently 8 zmm accumulators
   (MR=8). Skylake-X has 32 zmm registers. Could go to MR=16 (16 accumulators)
   to hide FMA latency better (FMA latency = 4 cycles, throughput = 0.5/cycle;
   need 8 in flight to hit throughput; current 8 is borderline). But
   MR=16 doubles the A_pack broadcast loads per kc iter. **Try MR=12 as a
   middle ground.**

4. **BF16 matmul** (HIGH impact, larger effort). `_mm512_dpbf16_ps` does
   32 BF16 FMA ops/cycle vs 16 FP32 — 2x throughput. PyTorch supports BF16.
   Would require a new `rk.mm_bf16` op and dtype handling. **This is the
   biggest SIMD win available**, but it is a project, not a tweak.

5. **AVX-512 VNNI / FP16 (`vhadd` etc.)** for quantised/int8 inference.
   Out of scope for FP32 training.

### 5.4 Verdict

For the FP32 path, auto-vectorisation with `-march=native` already captures
most of the AVX-512 wins for the norm ops. Explicit intrinsics would give
5–15% on `rms_norm` (and only if accuracy budget loosens for `rcp14`). The
real SIMD leap is BF16 matmul, which is a separate project.

---

## 6. `turbo()` Analysis

### 6.1 On the 2-vCPU sandbox

The bench in `bench/results/turbo.json` shows 1.02x (noise). The bench
script's own docstring explains why:

- `threads_per_core == 1` → no SMT → "physical cores" == "all logical CPUs".
  `rk_build_physical_mask` returns `{0, 1}`, identical to the default
  affinity. `topo_set_affinity` is a no-op.
- 22 ms total bench is too short to hit a GC cycle (`gc.disable()` saves
  nothing).
- `torch.set_num_threads(2)` is the default.

So `turbo()` is correctly activated but provides ~0% on the sandbox. This is
honest in the README and bench output.

### 6.2 In the 2.89x training speedup

The 2.89x training speedup (660.9 ms → 228.9 ms/step) is **not** from
`turbo()`. It is from the individual ops:

- `rms_norm` 3.32x × ~8 calls/step (4 layers, fwd+bwd)
- `softmax` 1.14x × ~8 calls/step
- `layer_norm` 1.6x × ~8 calls/step (if the model uses LN; the README says
  GPT, which typically uses LN, not RMS — but the bench config doesn't say
  which)
- `mm` 0.49–0.63x × ~12 calls/step (QKV, attn out, MLP up, MLP down, per
  layer, fwd; doubled for bwd)

Back-of-envelope: if 50% of step time is matmul (slower) and 20% is norm ops
(faster), the weighted speedup is `1 / (0.5/0.56 + 0.2/2.5 + 0.3/1.0) ≈ 1.5x`.
That is well below the measured 2.89x, which means either the matmul fraction
is smaller than 50% in training (small 128×128 matmuls have higher dispatch
overhead share) or `turbo()` is contributing more than expected.

The most likely `turbo()` contributor in training is **`gc.disable()`**:
660 ms × 824 steps = 9 min of training. Python's gen2 GC fires every ~5 min by
default, and each full GC pass over a training loop's working set (~50 MB of
tensors/autograd graph nodes) can take 100–500 ms. `gc.disable()` eliminates
that. Over 9 min, that could save 100–500 ms total = ~0.2–0.5% of wall time.
Not the 2.89x source.

**The honest conclusion**: the 2.89x is dominated by per-op dispatch removal
on a model small enough that ATen dispatch is a large fraction of step time.
`turbo()` contributes <5% on the sandbox. To verify, run the training bench
with `turbo()` replaced by a no-op context manager — the speedup should
remain ~2.7–2.8x.

### 6.3 `turbo()` correctness on real hardware

On a real SMT box (e.g. 8c/16t Xeon), `turbo()` would:

- Pin to 8 physical cores, skipping 8 hyperthreads. This helps OpenMP workers
  that would otherwise contend with their sibling's execution units.
- `OMP_PROC_BIND=close` keeps workers near the main thread's NUMA node.
- `gc.disable()` still helps for multi-second training loops.

The C code is correct for this case. The one concern is that `rk_turbo_exit`
does not restore `omp_set_num_threads` — it only restores affinity. If the
user had previously set a non-default OMP thread count, it is lost. The
Python wrapper restores `torch.set_num_threads` (which writes the same ICV
when libreikernel and libtorch share libgomp), so this is fine in practice
but fragile if they ever use different OpenMP runtimes.

---

## 7. Top 5 Improvements (ranked by impact / effort)

### #1 — Replace `rk.mm` with `cblas_sgemm` (MKL/BLAS)

- **Impact**: HIGH. Estimated 1.4–1.6x training speedup (matmul goes from
  0.49–0.63x to ~1.0x of MKL).
- **Effort**: LOW. ~50 lines of C in `rk_mm`: detect BLAS at configure time,
  call `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
  A, K, B, N, 0.0, C, N)`. Keep the hand-tiled path behind a compile flag.
- **Tradeoff**: Adds `-lmkl_rt` (or `-lopenblas`) link dep. Set
  `MKL_NUM_THREADS` to avoid oversubscription on 2-vCPU sandbox. Loses the
  "educational matmul" framing — keep the BLIS path as `RK_MM_BACKEND=blis`.
- **Why #1**: `rk.mm` is the only op slower than PyTorch. It is also the op
  with the most room (math-bound, not dispatch-bound). Fixing it is the
  single biggest lever for pushing 2.89x → 4x.

### #2 — Restructure `rk_mm` microkernel for register-blocked C accumulation

- **Impact**: MEDIUM-HIGH. Estimated 20–40% matmul speedup (closing ~half
  the gap to MKL without using MKL). ~5–10% training speedup.
- **Effort**: MEDIUM. Restructure the (kk, ii, jj) loop nesting so the
  microkernel preserves its 8 zmm accumulators across all kc-steps for a
  given (ii, jj) micro-tile, loading C once at the start and storing once at
  the end. Requires passing `is_first_kc` / `is_last_kc` flags or splitting
  the microkernel into prologue/body/epilogue.
- **Tradeoff**: More complex microkernel. Pack-A/B costs stay the same.
- **Why #2**: Independent of #1 (works whether or not BLAS is used). If #1
  lands, this becomes moot for the BLAS path but still relevant for the
  fallback BLIS path.

### #3 — Remove the arena from norm ops, use stack scratch

- **Impact**: LOW (perf), HIGH (code health). ~2–5% faster norm ops; removes
  ~30 lines of TLS+arena boilerplate per op.
- **Effort**: TRIVIAL. Replace the `barrage_alloc` block in `rk_rms_norm`,
  `rk_softmax`, `rk_layernorm` with `float row_scratch[2]` on the stack.
  Delete `rk_get_thread_arena`, `rk_stack_scratch`, `RK_ARENA_BYTES`, the
  `#include "barrage.h"` (for these three files only — keep for matmul).
- **Tradeoff**: None. The arena provides zero benefit for 4–8 byte
  allocations, and the current "never reset" pattern (B2) means it is unused
  after ~500 calls anyway.
- **Why #3**: Highest impact-to-effort ratio for code health. Removes a
  latent correctness footgun (the "never reset" comment is backwards) and
  makes the norm ops faster.

### #4 — Add fused AdamW optimizer step

- **Impact**: HIGH. Estimated 10–15% training speedup. PyTorch's `AdamW`
  does ~6 elementwise ops per parameter per step; fusing to 1 saves 5×Nparam
  dispatches + memory passes.
- **Effort**: MEDIUM-HIGH. New C op `rk_adamw_step(params, grads, exp_avg,
  exp_avg_sq, lr, beta1, beta2, eps, weight_decay, grad_clip)` — single pass,
  AVX-512 vectorised. Handle bias correction, decoupled weight decay, grad
  clipping. Needs a new Python wrapper that accepts param groups.
- **Tradeoff**: Increases API surface. Must match PyTorch's AdamW numerics
  exactly (else training diverges further).
- **Why #4**: Disproportionate impact on training loops. The optimizer step
  is pure elementwise overhead with no math complexity — perfect fusion
  target. For the 1M-param bench model, saves ~25 us × 5 = ~125 us/step
  (small in absolute terms, but it scales linearly with param count for
  larger models).

### #5 — Add fused QKV projection

- **Impact**: MEDIUM. ~5% training speedup for the 4-layer GPT bench
  (saves 16 dispatches/step); scales with layer count.
- **Effort**: MEDIUM. New op `rk.mm_qkv(x, W_qkv)` where `W_qkv` is
  `[3, C, C]` (pre-stacked weights). Single matmul, split output into Q/K/V
  views. Requires the user to stack weights once at model init.
- **Tradeoff**: New API. User must restructure model to pre-stack weights.
- **Why #5**: Clean, well-scoped fusion that removes both dispatch overhead
  and intermediate activations. Composes well with #1 (the underlying matmul
  becomes MKL-speed). For models with more layers (e.g. 12-layer GPT), the
  relative win grows.

---

## Summary table: expected training speedup from each fix

| Fix                                 | Estimated step-time saving | Cumulative |
|-------------------------------------|---------------------------:|-----------:|
| (baseline v0.5)                     |                       —    |     2.89x  |
| #1 cblas_sgemm for `rk.mm`          |                  ~30–40%   |     ~4.0x  |
| #2 register-blocked C accumulation  |                    ~5–10%  |     ~4.3x  |
| #3 stack scratch in norm ops        |                     ~1–2%  |     ~4.4x  |
| #4 fused AdamW                      |                   ~10–15%  |     ~5.0x  |
| #5 fused QKV                        |                     ~3–5%  |     ~5.2x  |

These are independent estimates; real compounding will be sub-multiplicative
due to shared bottlenecks (memory bandwidth, OMP overhead). But #1 alone is
plausibly sufficient to cross the 4x threshold stated in the audit brief.

---

## Appendix: file-by-file findings index

| File                  | Findings                                                              |
|-----------------------|----------------------------------------------------------------------|
| `c/rk_matmul.c`       | §2.3 (loop nesting), §2.4 (BLAS swap), §3.1 (arena OK), §5.3 (MR=12) |
| `c/rk_norm.c`         | §1.1 B2 (arena never resets), §3.2 (arena useless), §5.3 (rcp14)     |
| `c/rk_softmax.c`      | §1.1 B2 (arena never resets), §1.1 B3 (ln2 naming), §3.2 (arena)     |
| `c/rk_layernorm.c`    | §1.1 B2 (arena never resets), §3.2 (arena useless)                   |
| `c/rk_turbo.c`        | §1.1 B1 (cpulist over-read), §6.2 (no-op on sandbox), §6.3 (OMP restore) |
| `python/reikernel.py` | §1.3 (FP ordering diffs), §6.3 (turbo cleanup ordering)              |
| `Makefile`            | No issues. `-march=native -O3 -fopenmp` is appropriate.              |
| `bench/results/*.json`| §1.3 (val_bpb_diff), §6.2 (turbo 1.02x), §2 (per-op numbers)         |
