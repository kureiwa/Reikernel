"""Microbenchmark: reikernel.softmax vs torch.nn.functional.softmax.

Shapes:
  - (4, 256, 257)   -- the model's final logits shape (vocab=257)
  - (4, 256, 1024)  -- wider vocab case

Method:
  - 1000 iterations per trial
  - 5 trials per impl, interleaved (so thermal/cache state hits both equally)
  - median of the 5 trials reported

Saves JSON to bench/results/softmax.json with per-shape, per-trial timings
and the median + speedup. Prints PyTorch us, Reikernel us, speedup ratio
for each shape.

Correctness check before the benchmark: torch.allclose(rk.softmax(x),
F.softmax(x, dim=-1), atol=1e-6, rtol=1e-6). Tighter than matmul (1e-5)
because softmax is more numerically sensitive.
"""

import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

import reikernel as rk  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

SHAPES = [
    (4, 256, 257),    # final logits shape (vocab=257)
    (4, 256, 1024),   # wider vocab
]
DTYPE = torch.float32
N_ITERS = 1000
N_TRIALS = 5
WARMUP_ITERS = 200  # softmax is lighter than matmul; more warmup for stability
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "softmax.json")

# Tighter tolerance than matmul (1e-5) per spec.
ATOL = 1e-6
RTOL = 1e-6


def _make_input(B, T, V, seed=0xBEEF):
    g = torch.Generator()
    g.manual_seed(seed ^ (B * 7919 + T * 31 + V))
    # Uniform [-3, 3) stresses the max-subtract + exp numerical stability
    # path; same range as the correctness test for consistency.
    x = (torch.rand((B, T, V), generator=g, dtype=DTYPE) * 6.0 - 3.0)
    return x


def _time_call(fn, x, n_iters):
    """Returns ms per call (median of n_iters calls)."""
    # Warm up
    for _ in range(WARMUP_ITERS):
        fn(x)
    # Time
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(x)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e3  # ms per call


def bench_shape(B, T, V):
    """Returns a dict with per-trial timings, medians, and speedup."""
    x = _make_input(B, T, V)

    # Correctness check before the benchmark.
    y_torch = F.softmax(x, dim=-1)
    y_rk = rk.softmax(x)
    max_diff = (y_rk - y_torch).abs().max().item()
    ok = torch.allclose(y_rk, y_torch, atol=ATOL, rtol=RTOL)
    print(f"Correctness check on ({B},{T},{V}): max_abs_diff={max_diff:.2e}, "
          f"allclose(atol={ATOL})={ok}")
    if not ok:
        print("  ERROR: correctness check FAILED; skipping this shape.",
              file=sys.stderr)
        return None

    print(f"  Iters/trial: {N_ITERS}, trials: {N_TRIALS}, warmup: {WARMUP_ITERS}")
    print(f"  OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")

    rk_fn = lambda x: rk.softmax(x)
    torch_fn = lambda x: F.softmax(x, dim=-1)

    rk_trials = []
    pt_trials = []

    # Interleave: alternate trials so thermal state is shared.
    for i in range(N_TRIALS):
        ms_rk = _time_call(rk_fn, x, N_ITERS)
        ms_pt = _time_call(torch_fn, x, N_ITERS)
        rk_trials.append(ms_rk)
        pt_trials.append(ms_pt)
        print(f"  trial {i+1}/{N_TRIALS}: rk={ms_rk*1000:8.2f} us  "
              f"pt={ms_pt*1000:8.2f} us")

    rk_median = statistics.median(rk_trials)
    pt_median = statistics.median(pt_trials)
    speedup = pt_median / rk_median if rk_median > 0 else float("inf")

    print(f"  PyTorch   F.softmax median = {pt_median*1000:8.2f} us/call "
          f"({pt_median:.4f} ms)")
    print(f"  Reikernel rk.softmax  median = {rk_median*1000:8.2f} us/call "
          f"({rk_median:.4f} ms)")
    print(f"  Speedup: {speedup:.2f}x  (median of {N_TRIALS} trials x "
          f"{N_ITERS} iters)")
    print()

    return {
        "shape": [B, T, V],
        "dtype": str(DTYPE),
        "n_iters": N_ITERS,
        "n_trials": N_TRIALS,
        "warmup_iters": WARMUP_ITERS,
        "rk_trials_ms": rk_trials,
        "pt_trials_ms": pt_trials,
        "rk_median_ms": rk_median,
        "pt_median_ms": pt_median,
        "speedup": speedup,
        "correctness": {
            "max_abs_diff": max_diff,
            "allclose_atol_1e-6": ok,
        },
    }


def main() -> int:
    # Load library early.
    rk.load_library()

    print(f"reikernel {rk.__version__} -- rk.softmax vs F.softmax microbench")
    print(f"Shapes: {SHAPES}, dtype={DTYPE}")
    print()

    results = []
    for (B, T, V) in SHAPES:
        print(f"--- Shape ({B},{T},{V}) ---")
        r = bench_shape(B, T, V)
        if r is not None:
            results.append(r)

    # Save JSON.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "shapes": [list(s) for s in SHAPES],
        "dtype": str(DTYPE),
        "n_iters": N_ITERS,
        "n_trials": N_TRIALS,
        "warmup_iters": WARMUP_ITERS,
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", "default"),
        "results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON to {RESULTS_PATH}")

    # Summary
    print()
    print("=" * 64)
    print(f"{'Shape':<20} {'F.softmax':>12} {'rk.softmax':>12} {'Speedup':>10}")
    print("-" * 64)
    for r in results:
        s = r["shape"]
        print(f"({s[0]:>2},{s[1]:>3},{s[2]:>4})        "
              f"{r['pt_median_ms']*1000:>9.2f} us "
              f"{r['rk_median_ms']*1000:>9.2f} us "
              f"{r['speedup']:>8.2f}x")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
