# Reikernel

A PyTorch CPU backend extension. Built for a 2-vCPU cloud sandbox
(AVX-512 + FMA, 4 GB RAM). Uses [Rui-727/EoSD](https://github.com/Rui-727/EoSD)
for arena allocation, CPU topology, thread pinning, and perf counters.

## Motivation

PyTorch's CPU backend has per-op dispatch overhead that dominates on
small models (under 1M params). Every `torch.mm` round-trips through
ATen dispatch and allocates a fresh tensor. Threads are not pinned to
physical cores. There is no arena to pool intermediates from a single
optimizer step.

EoSD solves the systems side. Reikernel wraps EoSD's modules into
PyTorch-shaped ops: drop-in replacement functions plus a
`with reikernel.turbo()` context manager that pins threads and disables
Python GC for the hot loop.

## Usage

```python
import torch
import reikernel as rk

C = rk.mm(A, B)                          # matmul (blis or cblas backend)
y = rk.rms_norm(x, weight, eps)          # RMSNorm
y = rk.layer_norm(x, weight, bias, eps)  # LayerNorm
p = rk.softmax(logits)                   # softmax

# Fused ops (v0.6)
rk.adamw_step(params, grads, m, v, lr, b1, b2, eps, wd, step)  # fused AdamW
Q, K, V = rk.mm_qkv(x, W_qkv)           # fused QKV projection

with rk.turbo() as n_physical:
    out = model(input)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
```

v0.6 adds: cblas_sgemm backend for rk.mm, register-blocked C
accumulation in the blis microkernel, stack scratch for norm ops
(removed arena overhead), fused AdamW, fused QKV projection.
v0.5 shipped: thread pinning, OMP control, GC disable.

## Build

    make                                # default: blis backend
    make RK_MM_BACKEND=cblas RK_BLAS_LIB=blas  # cblas backend
    make bench                          # benchmarks vs PyTorch
    make test                           # run all test suites

Requirements: gcc (C11), nasm (for EoSD assembly), Python 3.10+,
PyTorch (any version that exposes `torch.Tensor.data_ptr`).
For cblas backend: a BLAS library with cblas.h (MKL, OpenBLAS, or BLAS).

## Performance

Median of 5 trials, 1000 iters/trial, interleaved. 2-vCPU sandbox Xeon
with AVX-512, `-march=native -O3 -fopenmp`, `OMP_NUM_THREADS=2`.

| Op | Shape | PyTorch (us) | Reikernel (us) | Speedup |
|---|---|---:|---:|---:|
| `rk.rms_norm` | (4,256,128) | 74.19 | 22.34 | 3.32x |
| `rk.mm` (blis) | (128,128,128) | 20.51 | 41.87 | 0.49x |
| `rk.mm` (cblas) | (128,128,128) | 20.51 | ~21 | ~1.0x |
| `rk.softmax` | (4,256,257) | 127.96 | 107.40 | 1.19x |
| `rk.layer_norm` | (4,256,128) | 47.91 | 29.93 | 1.60x |
| `rk.turbo()` | (4,256,128) | 22.55 | 22.07 | 1.02x |

v0.6 changes: norm ops are ~2-5% faster (removed arena overhead).
rk.mm cblas backend matches MKL speed. Fused adamw and qkv reduce
dispatch overhead in training loops.

Training bench (v0.5): 2.89x speedup over plain PyTorch on a 1M GPT.
v0.6 expected: ~4-5x with cblas backend + fused ops. See
[ARCHITECTURE.md](./ARCHITECTURE.md) for details and
[IMPROVEMENTS.md](./IMPROVEMENTS.md) for the full audit.

Correctness: all ops match PyTorch within `atol=1e-5` (softmax within
`1e-6`). Fused adamw matches to ~2e-6 at step >= 10 (FP ordering).

## License

MIT.
