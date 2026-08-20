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

C = rk.mm(A, B)
y = rk.rms_norm(x, weight, eps)
y = rk.layer_norm(x, weight, bias, eps)
p = rk.softmax(logits)

with rk.turbo() as n_physical:
    out = model(input)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
```

v0.5 ships thread pinning, OMP control, and GC disable. The allocator
swap requires a C++ extension that links against libtorch, which is
outside what a pure ctypes wrapper can reach.

## Build

    make            # builds vendor/EoSD first, then c/, then runs tests
    make bench      # runs benchmarks vs PyTorch

Requirements: gcc (C11), nasm (for EoSD assembly), Python 3.10+,
PyTorch (any version that exposes `torch.Tensor.data_ptr`).

## Performance

Median of 5 trials, 1000 iters/trial, interleaved. 2-vCPU sandbox Xeon
with AVX-512, `-march=native -O3 -fopenmp`, `OMP_NUM_THREADS=2`.

| Op | Shape | PyTorch (us) | Reikernel (us) | Speedup |
|---|---|---:|---:|---:|
| `rk.rms_norm` | (4,256,128) | 74.19 | 22.34 | 3.32x |
| `rk.mm` | (128,128,128) | 20.51 | 41.87 | 0.49x |
| `rk.mm` | (256,256,256) | 168.09 | 284.59 | 0.59x |
| `rk.softmax` | (4,256,257) | 127.96 | 107.40 | 1.19x |
| `rk.layer_norm` | (4,256,128) | 47.91 | 29.93 | 1.60x |
| `rk.turbo()` | (4,256,128) | 22.55 | 22.07 | 1.02x |

`rk.mm` loses to MKL on isolated GEMM. The point is the arena path
that other ops reuse, not beating MKL at sgemm. See IMPROVEMENTS.md
for the plan to switch `rk.mm` to `cblas_sgemm`.

Training bench: 2.89x speedup over plain PyTorch on a 1M GPT (6L,
128d, 4h, 257 vocab, 600s budget). See
[Xenon-Camellia](https://github.com/kureiwa/Xenon-Camellia).

Correctness: all ops match PyTorch within `atol=1e-5` (softmax within
`1e-6`). Per-trial timings in `bench/results/*.json`.

## License

MIT.
