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
| libbarrage | Per-thread bump allocator with batch reset | rk_matmul (scratch), rk_norm/softmax/layernorm (per-row scalars) |
| libtopo | CPU topology detection, physical core identification, affinity pinning | rk_turbo (thread pinning) |
| libpmu | Hardware performance counter reads | bench (cycle counting) |
| libflume | Wait-free MPSC ring buffer | Not yet wired (planned for dataloader prefetch) |

Each EoSD module is a standalone static archive (.a). No cross-module
link-time dependencies. Build with `make` from `vendor/EoSD/`.

## Layer 2: C Glue (c/)

Five C source files, each implementing one PyTorch op or context manager.
Compiled into a single shared library `libreikernel.so`.

### rk_matmul.c

Hand-tiled FP32 matmul. BLIS-style MC x KC x NC blocking with
MR=8 x NR=16 AVX-512 FMA microkernel. Uses 8 zmm accumulators.
Per-thread libbarrage arena (32 MiB) for A_pack and B_pack scratch.

Linking: `-O3 -march=native -fopenmp -fPIC -shared`.

Known issue: kc-step loop is outermost, causing redundant C load/store
per micro-tile. See IMPROVEMENTS.md #2.

### rk_norm.c

Fused RMSNorm over (B, T, C) FP32. Per-row mean(x^2) via OpenMP simd
reduction. Per-thread libbarrage arena (4 MiB) for per-row scalar
scratch. Falls back to stack scratch when arena is exhausted.

Known issue: arena never resets, fills after ~1024 calls, falls back
to stack for the rest of training. See IMPROVEMENTS.md #3.

### rk_softmax.c

Three-pass numerically-stable softmax over (B, T, V) FP32. Pass 1: per-
row max. Pass 2: exp + sum. Pass 3: divide. Uses inline AVX-512
polynomial exp (Cody-Waite range reduction + degree-6 Taylor, ~1 ULP)
when `__AVX512F__` is defined. Falls back to scalar `libm expf` on
AVX2-only targets.

### rk_layernorm.c

Two-pass LayerNorm over (B, T, C) FP32. Pass 1: sum -> mean. Pass 2:
sum((x-mean)^2) -> var -> rstd. Normalize + scale + shift in a single
FMA chain. rstd computed once per row and multiplied in (matches
PyTorch's CPU kernel ordering).

### rk_turbo.c

Wraps libtopo for thread pinning. `rk_turbo_enter` detects physical
cores (skips hyperthreads), pins the calling thread + OMP workers via
`sched_setaffinity`, sets `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, calls `omp_set_num_threads`, syncs
`torch.set_num_threads`. `rk_turbo_exit` restores the original affinity.

Known issue: `rk_parse_cpulist` can over-read on systems with >256
CPUs per sibling group. See IMPROVEMENTS.md B1.

## Layer 3: Python Bindings (python/)

### reikernel.py

ctypes wrapper. Loads `libreikernel.so` at import time. Each op function:
1. Validates input (dtype, contiguity, shape).
2. Allocates output tensor via `torch.empty`.
3. Calls the C function via `ctypes`, passing `tensor.data_ptr()`.
4. Returns the output tensor.

`turbo()` context manager: calls `rk_turbo_enter` on enter, `rk_turbo_exit`
on exit (in `finally`). Also saves/restores `torch.get_num_threads` and
`gc.isenabled()`.

### _rk_autograd.py (Xenon-Camellia side)

Autograd wrappers for `rk.softmax` and `rk.rms_norm`. The C ops are
opaque to PyTorch autograd, so these wrappers implement the backward
pass in Python. Falls back to `F.softmax` / `F.rms_norm` when
libreikernel.so is not available.

## Data Flow

```
Python: rk.rms_norm(x, weight, eps)
  |
  v
reikernel.py: validate, alloc y, call C
  |
  v
rk_norm.c: barrage_alloc scratch, OpenMP parallel for over (B,T),
            per-row: sum(x^2), rms=sqrt(mean+eps), y=(x*w)/rms
  |
  v
libbarrage: bump-allocate 4 bytes/row from 4 MiB arena
  |
  v
returns y tensor to Python
```

## Build System

Makefile at repo root:
1. Builds vendor/EoSD (all 13 modules, needs nasm).
2. Compiles c/*.c into libreikernel.so with `-O3 -march=native -fopenmp -fPIC -shared`.
3. Links against libbarrage.a and libtopo.a from vendor/EoSD.

## Known Issues

See [IMPROVEMENTS.md](./IMPROVEMENTS.md) for the full audit. Summary:

1. `rk.mm` slower than MKL (0.49-0.63x). Plan: switch to `cblas_sgemm`.
2. `rk_mm` loop nesting causes redundant C load/store. Plan: restructure.
3. Norm ops use arena for 4-8 bytes, never reset. Plan: use stack scratch.
4. `rk_parse_cpulist` over-read on >256 CPU systems. Plan: clamp return.
5. Training val_bpb diverges by 0.069 between baseline and reikernel.
   Expected: FP ordering differences compound over 2365 steps.

## Planned Improvements

| Priority | Improvement | Expected training speedup |
|---|---|---:|
| 1 | cblas_sgemm for rk.mm | 2.89x -> ~4.0x |
| 2 | Register-blocked C accumulation | +5-10% |
| 3 | Stack scratch in norm ops | +1-2% |
| 4 | Fused AdamW step | +10-15% |
| 5 | Fused QKV projection | +3-5% |

Cumulative estimate: 2.89x -> ~5.2x (sub-multiplicative due to shared
bottlenecks).
