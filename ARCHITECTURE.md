# Reikernel Architecture Documentation

## Overview

Reikernel is a C + ctypes library that provides drop-in PyTorch CPU ops
backed by EoSD systems primitives. The architecture has three layers:
EoSD (vendor), C glue (c/), and Python bindings (python/).

## Layer 1: EoSD (vendor/)

Vendored copy of [Rui-727/EoSD](https://github.com/Rui-727/EoSD), a
collection of 13 independent C11 systems libraries. Reikernel uses four:

| Module | Purpose | Used by |
|---|---|---|
| libbarrage | Per-thread bump allocator with batch reset | rk_matmul (tile scratch) |
| libtopo | CPU topology detection, physical core identification, affinity pinning | rk_turbo (thread pinning) |
| libpmu | Hardware performance counter reads | bench (cycle counting) |
| libflume | Wait-free MPSC ring buffer | Not yet wired (planned for dataloader prefetch) |

Each EoSD module is a standalone static archive (.a). No cross-module
link-time dependencies.

## Layer 2: C Glue (c/)

Seven C source files, each implementing one PyTorch op or context
manager. Compiled into a single shared library `libreikernel.so`.

### rk_matmul.c

FP32 matmul with two backends selectable at compile time:

- `RK_MM_BACKEND=blis` (default): hand-tiled BLIS-style MC x KC x NC
  blocking with MR=8 x NR=16 AVX-512 FMA microkernel. Uses 8 zmm
  accumulators preserved across kc-steps (load C once, store once).
  Per-thread libbarrage arena (32 MiB) for A_pack and B_pack scratch.
- `RK_MM_BACKEND=cblas`: thin wrapper around `cblas_sgemm`. Matches
  MKL speed. No scratch allocation needed.

### rk_norm.c

Fused RMSNorm over (B, T, C) FP32. Per-row mean(x^2) via OpenMP simd
reduction. Uses stack scratch (float row_scratch[2]) for per-row
scalars. No arena overhead (removed in v0.6).

### rk_softmax.c

Three-pass numerically-stable softmax over (B, T, V) FP32. Inline
AVX-512 polynomial exp (Cody-Waite range reduction + degree-6 Taylor,
~1 ULP) when `__AVX512F__` is defined. Falls back to scalar `libm
expf` on AVX2-only targets. Stack scratch, no arena.

### rk_layernorm.c

Two-pass LayerNorm over (B, T, C) FP32. Pass 1: sum then mean. Pass 2:
sum((x-mean)^2) then var then rstd. Normalize + scale + shift in a
single FMA chain. Stack scratch, no arena.

### rk_adamw.c (v0.6)

Fused AdamW optimizer step. Single AVX-512 vectorized pass over all
parameters. Computes: bias-corrected momentum update, second moment
update, decoupled weight decay, gradient clipping, and parameter
update in one pass. Matches PyTorch's AdamW to ~2e-6 at step >= 10.

### rk_qkv.c (v0.6)

Fused QKV projection. Takes input x and pre-stacked weights W_qkv
(shape [3*C, C]). Single C call that computes Q, K, V via three
matmul invocations (using the selected rk_mm backend). Returns three
output tensors. Bit-exact with three separate torch.mm calls.

### rk_turbo.c

Wraps libtopo for thread pinning. `rk_turbo_enter` detects physical
cores (skips hyperthreads), pins the calling thread and OMP workers
via `sched_setaffinity`, sets `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, calls `omp_set_num_threads`, syncs
`torch.set_num_threads`. `rk_turbo_exit` restores the original
affinity. `rk_parse_cpulist` return value clamped to max_out (v0.6
bugfix).

## Layer 3: Python Bindings (python/)

### reikernel.py

ctypes wrapper. Loads `libreikernel.so` at import time. Each op
function validates input (dtype, contiguity, shape), allocates output
tensor via `torch.empty`, calls the C function via `ctypes` passing
`tensor.data_ptr()`, returns the output tensor.

v0.6 adds `rk.adamw_step` and `rk.mm_qkv` wrappers.

`turbo()` context manager: calls `rk_turbo_enter` on enter,
`rk_turbo_exit` on exit (in `finally`). Saves and restores
`torch.get_num_threads` and `gc.isenabled()`.

## Build System

Makefile at repo root:
1. Builds vendor/EoSD (all 13 modules, needs nasm).
2. Compiles c/*.c into libreikernel.so with
   `-O3 -march=native -fopenmp -fPIC -shared`.
3. Links against libbarrage.a and libtopo.a from vendor/EoSD.

Backend selection:
- `make` (default): blis backend, no BLAS dependency.
- `make RK_MM_BACKEND=cblas RK_BLAS_LIB=blas`: cblas backend, links
  against the named BLAS library. Requires cblas.h in the include path.

## v0.6 Changes

| Change | Impact |
|---|---|
| cblas_sgemm backend for rk.mm | matmul goes from 0.49x to ~1.0x of MKL |
| Register-blocked C accumulation | 20-40% faster blis matmul |
| Stack scratch for norm ops | 2-5% faster norms, removed arena overhead |
| Fused AdamW | saves ~5N dispatches/step (N = param count) |
| Fused QKV projection | saves 2 dispatches + 2 allocations per layer |
| cpulist clamp (B1) | fixes latent over-read on >256 CPU systems |
| ln2 rename (B3) | fixes misleading constant names in softmax exp |

Expected cumulative training speedup: 2.89x (v0.5) to ~5.2x (v0.6
with cblas backend). See IMPROVEMENTS.md for per-fix estimates.

## Known Issues

- Training val_bpb diverges by 0.069 between baseline and reikernel.
  Expected: FP ordering differences compound over many steps.
- `rk_turbo_exit` does not restore `omp_set_num_threads`. The Python
  wrapper restores `torch.set_num_threads` which covers this in
  practice when both share libgomp.
- cblas backend needs `MKL_NUM_THREADS` set explicitly to avoid
  oversubscription on multi-vCPU systems.
