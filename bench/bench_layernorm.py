"""Microbenchmark: reikernel.layer_norm vs torch.nn.functional.layer_norm.

Shapes:
  - (4, 256, 128)   -- same as the rms_norm bench (apples-to-apples)
  - (4, 256, 320)   -- wider C (closer to the LLaMA hidden size 320 = 1/8 of 2560)

Method:
  - 1000 iterations per trial
  - 5 trials per impl, interleaved (so thermal/cache state hits both equally)
  - median of the 5 trials reported

Saves JSON to bench/results/layernorm.json with per-shape, per-trial timings
and the median + speedup. Prints PyTorch us, Reikernel us, speedup ratio
for each shape.

Correctness check before the benchmark: torch.allclose(rk.layer_norm(x, w, b),
F.layer_norm(x, (C,), w, b, eps), atol=1e-5, rtol=1e-5). Matches the v0.1
rms_norm tolerance (layernorm is numerically benign, no exp chain).
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
    (4, 256, 128),    # same as the rms_norm bench (apples-to-apples)
    (4, 256, 320),    # wider C
]
DTYPE = torch.float32
N_ITERS = 1000
N_TRIALS = 5
WARMUP_ITERS = 200  # layernorm is light; more warmup for stability
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "layernorm.json")

# Matches v0.1 rms_norm tolerance.
ATOL = 1e-5
RTOL = 1e-5
EPS = 1e-5  # default eps


def _make_inputs(B, T, C, seed=0xBEEF):
    """Deterministic input so timings are reproducible across runs.

    Uniform [-1, 1) keeps variance well-conditioned (~1/3); weight is
    biased positive (1.5 +/- 1) so the scale-by-weight step doesn't
    accidentally zero-out the output; bias is small (|b| < 0.5).
    """
    g = torch.Generator()
    g.manual_seed(seed ^ (B * 7919 + T * 31 + C))
    x = (torch.rand((B, T, C), generator=g, dtype=DTYPE) * 2.0 - 1.0)
    w = (torch.rand((C,), generator=g, dtype=DTYPE) * 2.0 - 1.0) + 1.5
    b = (torch.rand((C,), generator=g, dtype=DTYPE) * 2.0 - 1.0) * 0.5
    return x, w, b


def _time_call(fn, x, w, b, n_iters):
    """Returns ms per call (median of n_iters calls)."""
    # Warm up
    for _ in range(WARMUP_ITERS):
        fn(x, w, b)
    # Time
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(x, w, b)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e3  # ms per call


def bench_shape(B, T, C):
    """Returns a dict with per-trial timings, medians, and speedup."""
    x, w, b = _make_inputs(B, T, C)

    # Correctness check before the benchmark.
    y_torch = F.layer_norm(x, normalized_shape=(C,), weight=w, bias=b,
                          eps=EPS)
    y_rk = rk.layer_norm(x, w, b, eps=EPS)
    max_diff = (y_rk - y_torch).abs().max().item()
    ok = torch.allclose(y_rk, y_torch, atol=ATOL, rtol=RTOL)
    print(f"Correctness check on ({B},{T},{C}): max_abs_diff={max_diff:.2e}, "
          f"allclose(atol={ATOL})={ok}")
    if not ok:
        print("  ERROR: correctness check FAILED; skipping this shape.",
              file=sys.stderr)
        return None

    print(f"  Iters/trial: {N_ITERS}, trials: {N_TRIALS}, warmup: {WARMUP_ITERS}")
    print(f"  OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")

    rk_fn = lambda x, w, b: rk.layer_norm(x, w, b, eps=EPS)
    torch_fn = lambda x, w, b: F.layer_norm(
        x, normalized_shape=(C,), weight=w, bias=b, eps=EPS
    )

    rk_trials = []
    pt_trials = []

    # Interleave: alternate trials so thermal state is shared.
    for i in range(N_TRIALS):
        ms_rk = _time_call(rk_fn, x, w, b, N_ITERS)
        ms_pt = _time_call(torch_fn, x, w, b, N_ITERS)
        rk_trials.append(ms_rk)
        pt_trials.append(ms_pt)
        print(f"  trial {i+1}/{N_TRIALS}: rk={ms_rk*1000:8.2f} us  "
              f"pt={ms_pt*1000:8.2f} us")

    rk_median = statistics.median(rk_trials)
    pt_median = statistics.median(pt_trials)
    speedup = pt_median / rk_median if rk_median > 0 else float("inf")

    print(f"  PyTorch   F.layer_norm median = {pt_median*1000:8.2f} us/call "
          f"({pt_median:.4f} ms)")
    print(f"  Reikernel rk.layer_norm median = {rk_median*1000:8.2f} us/call "
          f"({rk_median:.4f} ms)")
    print(f"  Speedup: {speedup:.2f}x  (median of {N_TRIALS} trials x "
          f"{N_ITERS} iters)")
    print()

    return {
        "shape": [B, T, C],
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

    print(f"reikernel {rk.__version__} -- rk.layer_norm vs F.layer_norm microbench")
    print(f"Shapes: {SHAPES}, dtype={DTYPE}")
    print()

    results = []
    for (B, T, C) in SHAPES:
        print(f"--- Shape ({B},{T},{C}) ---")
        r = bench_shape(B, T, C)
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
    print(f"{'Shape':<20} {'F.layer_norm':>14} {'rk.layer_norm':>14} "
          f"{'Speedup':>10}")
    print("-" * 64)
    for r in results:
        s = r["shape"]
        print(f"({s[0]:>2},{s[1]:>3},{s[2]:>4})        "
              f"{r['pt_median_ms']*1000:>11.2f} us "
              f"{r['rk_median_ms']*1000:>11.2f} us "
              f"{r['speedup']:>8.2f}x")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
