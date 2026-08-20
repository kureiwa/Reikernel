# Reikernel

A PyTorch CPU backend extension. Built for a 2-vCPU cloud sandbox
(AVX-512 + FMA, 4 GB RAM). Uses [Rui-727/EoSD](https://github.com/Rui-727/EoSD)
under the hood for the systems primitives: arena allocator, CPU topology,
NUMA pinning, wait-free queues, perf counters.

## Motivation

PyTorch's CPU backend is good, but for tiny models (under 1M params) its
per-op overhead dominates. Every `torch.mm` round-trips through ATen
dispatch and allocates a fresh tensor; every `F.rms_norm` pays the same
dispatch cost; threads aren't pinned to physical cores; there's no arena
to pool the intermediates from a single optimizer step. EoSD already
solves the systems side. Reikernel wraps EoSD's modules into
PyTorch-shaped ops: drop-in replacement functions plus a
`with reikernel.turbo()` context manager that pins threads and disables
Python GC for the hot loop.

## Usage

### Drop-in PyTorch ops

```python
import torch
import reikernel as rk

C = rk.mm(A, B)                          # instead of torch.mm
y = rk.rms_norm(x, weight, eps)          # instead of F.rms_norm
y = rk.layer_norm(x, weight, bias, eps)  # instead of F.layer_norm
p = rk.softmax(logits)                   # instead of F.softmax
```

These ops are functionally identical to PyTorch's (same math, same FP32).
They don't go through ATen dispatch. They allocate via an EoSD
`libbarrage` arena reset per call and pin the calling thread to a
physical core via EoSD's `libtopo` on the first call.

### `turbo()` context manager

```python
with rk.turbo() as n_physical:
    # Calling thread + OMP worker pool pinned to physical cores
    # (hyperthreads skipped) via EoSD libtopo.
    # OMP_NUM_THREADS / OMP_PROC_BIND=close / OMP_PLACES=cores set.
    # torch.set_num_threads(n_physical) synced.
    # Python gc.disable()'d for the duration.
    out = model(input)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
# Outside: threads unpinned, torch thread count restored, gc re-enabled
# (and gc.collect() runs once to catch anything that piled up).
```

v0.5 ships thread pinning, OMP control, and GC disable. The allocator
swap promised in v0.1 is deferred: hooking ATen's `c10::Allocator`
requires a C++ extension that links against libtorch, which is outside
what a pure ctypes wrapper can reach.

You can also query the detected topology directly:

```python
>>> rk.topo_detect()
{'threads_per_core': 1, 'cores_per_package': 2, 'num_packages': 1,
 'num_numa_nodes': 1, 'total_cpus': 2, 'physical_cores': 2}
```

### Direct EoSD bindings (advanced)

```python
from reikernel import eosd

eosd.pmu.cycles_start()
# ... work ...
stats = eosd.pmu.cycles_stop()

eosd.topo.pin_to_physical(0)

topo = eosd.topo.detect()
print(topo.numa_nodes)
```

## Architecture

```
Reikernel/
├── README.md
├── Makefile                      # top-level build, dispatches to subdirs
├── vendor/EoSD/                  # git subtree of Rui-727/EoSD (pinned)
├── c/                            # C glue between EoSD and PyTorch tensors
│   ├── rk_matmul.c               # tiled FP32 matmul
│   ├── rk_norm.c                 # fused RMSNorm
│   ├── rk_softmax.c              # numerically-stable softmax
│   ├── rk_layernorm.c            # LayerNorm with mean+var fused
│   ├── rk_turbo.c                # wraps libtopo for turbo() thread pinning
│   └── rk_api.h                  # public C API
├── python/
│   ├── reikernel.py             # ctypes bindings to rk_*.c
│   └── eosd/                    # direct Python bindings to EoSD modules
├── tests/                       # correctness tests vs PyTorch
└── bench/                       # microbenchmarks (uses EoSD libpmu for cycles)
```

## Build

    make            # builds vendor/EoSD first, then c/, then runs tests
    make bench      # runs benchmarks vs PyTorch
    make install    # installs libreikernel.so + Python module

Requirements: gcc (C11), nasm (for EoSD's assembly), Python 3.10+,
PyTorch (any version that exposes `torch.Tensor.data_ptr`).

## Performance

All numbers are median of 5 trials, 1000 iters/trial, interleaved. 2-vCPU
sandbox Xeon with AVX-512, `-march=native -O3 -fopenmp`,
`OMP_NUM_THREADS=2`, no `-ffast-math`.

| Op              | Shape            | PyTorch (us) | Reikernel (us) | Speedup |
|-----------------|------------------|-------------:|----------------:|--------:|
| `rk.rms_norm`   | `(4,256,128)`    |        74.19 |           22.34 |   3.32x |
| `rk.mm`         | `(128,128,128)`  |        20.51 |           41.87 |   0.49x |
| `rk.mm`         | `(256,256,256)`  |       168.09 |          284.59 |   0.59x |
| `rk.mm`         | `(512,512,512)`  |      1313.29 |         2086.53 |   0.63x |
| `rk.softmax`    | `(4,256,257)`    |       127.96 |          107.40 |   1.19x |
| `rk.softmax`    | `(4,256,1024)`   |       591.60 |          518.48 |   1.14x |
| `rk.layer_norm` | `(4,256,128)`    |        47.91 |           29.93 |   1.60x |
| `rk.layer_norm` | `(4,256,320)`    |        94.58 |           92.14 |   1.03x |
| `rk.turbo()`    | `(4,256,128)`    |        22.55 |           22.07 |   1.02x |

The `turbo()` row is `rk.rms_norm` with the context manager active vs
inactive. `rk.rms_norm` beats PyTorch by 3.32x because PyTorch's CPU
RMSNorm path goes through ATen dispatch and allocates per call.
`rk.softmax` and `rk.layer_norm` win 1.1-1.6x: PyTorch's kernels are
already AVX-512 vectorised, so the win is dispatch overhead only.
`rk.mm` loses to MKL on isolated GEMM as expected; the point is wiring
the libbarrage arena path that softmax and layernorm reuse, not beating
MKL at sgemm. `rk.turbo()` is structural: on the 2-vCPU sandbox pinning
is a no-op (no SMT to skip) and the bench is too short to hit a GC cycle,
so the speedup sits at ~1.0x within noise.

Correctness: all ops match PyTorch within `atol=1e-5` (softmax within
`1e-6`). Max abs diffs: rms_norm `9.5e-7`, mm bit-exact on 9/10 shapes
(`1.91e-6` on `(1,128,128)`), softmax `7.45e-9`, layer_norm `1.43e-6`.
Per-trial timings: `bench/results/*.json`.

## License

MIT.

## Acknowledgments

Built on [Rui-727/EoSD](https://github.com/Rui-727/EoSD). Its arena,
topology, and pinning primitives are what this library wraps for
PyTorch use.
