"""Microbenchmark: reikernel.mm vs torch.mm.

Shapes: (128,128,128), (256,256,256), (512,512,512) FP32.
Method:
  - 1000 iterations per trial
  - 5 trials per impl, interleaved (so thermal/cache state hits both equally)
  - median of the 5 trials reported

Saves JSON to bench/results/mm.json with per-shape, per-trial timings and
the median + speedup. Prints PyTorch ms, Reikernel ms, speedup ratio for
each shape.
"""

import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

import reikernel as rk  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
]
DTYPE = torch.float32
N_ITERS = 1000
N_TRIALS = 5
WARMUP_ITERS = 50  # matmul is heavier than rms_norm; fewer warmup iters
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "mm.json")


def _make_inputs(M, K, N, seed=0xBEEF):
    g = torch.Generator()
    g.manual_seed(seed ^ (M * 7919 + K * 31 + N))
    A = (torch.rand((M, K), generator=g, dtype=DTYPE) * 2.0 - 1.0)
    B = (torch.rand((K, N), generator=g, dtype=DTYPE) * 2.0 - 1.0)
    return A, B


def _time_call(fn, A, B, n_iters):
    """Returns ms per call (median of n_iters calls)."""
    # Warm up
    for _ in range(WARMUP_ITERS):
        fn(A, B)
    # Time
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(A, B)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e3  # ms per call


def bench_shape(M, K, N):
    """Returns a dict with per-trial timings, medians, and speedup."""
    A, B = _make_inputs(M, K, N)

    # Correctness check before the benchmark.
    y_torch = torch.mm(A, B)
    y_rk = rk.mm(A, B)
    max_diff = (y_rk - y_torch).abs().max().item()
    ok = torch.allclose(y_rk, y_torch, atol=1e-5, rtol=1e-5)
    print(f"Correctness check on ({M},{K},{N}): max_abs_diff={max_diff:.2e}, "
          f"allclose(atol=1e-5)={ok}")
    if not ok:
        print("  ERROR: correctness check FAILED; skipping this shape.",
              file=sys.stderr)
        return None

    print(f"  Iters/trial: {N_ITERS}, trials: {N_TRIALS}, warmup: {WARMUP_ITERS}")
    print(f"  OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")

    rk_fn = lambda A, B: rk.mm(A, B)
    torch_fn = lambda A, B: torch.mm(A, B)

    rk_trials = []
    pt_trials = []

    # Interleave: alternate trials so thermal state is shared.
    for i in range(N_TRIALS):
        ms_rk = _time_call(rk_fn, A, B, N_ITERS)
        ms_pt = _time_call(torch_fn, A, B, N_ITERS)
        rk_trials.append(ms_rk)
        pt_trials.append(ms_pt)
        print(f"  trial {i+1}/{N_TRIALS}: rk={ms_rk*1000:8.2f} us  "
              f"pt={ms_pt*1000:8.2f} us")

    rk_median = statistics.median(rk_trials)
    pt_median = statistics.median(pt_trials)
    speedup = pt_median / rk_median if rk_median > 0 else float("inf")

    print(f"  PyTorch   torch.mm median = {pt_median*1000:8.2f} us/call "
          f"({pt_median:.4f} ms)")
    print(f"  Reikernel rk.mm     median = {rk_median*1000:8.2f} us/call "
          f"({rk_median:.4f} ms)")
    print(f"  Speedup: {speedup:.2f}x  (median of {N_TRIALS} trials x "
          f"{N_ITERS} iters)")
    print()

    return {
        "shape": [M, K, N],
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
            "allclose_atol_1e-5": ok,
        },
    }


def main() -> int:
    # Load library early.
    rk.load_library()

    print(f"reikernel {rk.__version__} -- rk.mm vs torch.mm microbench")
    print(f"Shapes: {SHAPES}, dtype={DTYPE}")
    print()

    results = []
    for (M, K, N) in SHAPES:
        print(f"--- Shape ({M},{K},{N}) ---")
        r = bench_shape(M, K, N)
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
    print(f"{'Shape':<20} {'torch.mm':>12} {'rk.mm':>12} {'Speedup':>10}")
    print("-" * 64)
    for r in results:
        s = r["shape"]
        print(f"({s[0]:>3},{s[1]:>3},{s[2]:>3})        "
              f"{r['pt_median_ms']*1000:>9.2f} us "
              f"{r['rk_median_ms']*1000:>9.2f} us "
              f"{r['speedup']:>8.2f}x")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
