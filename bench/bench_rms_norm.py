"""Microbenchmark: reikernel.rms_norm vs torch.nn.functional.rms_norm.

Shape: (4, 256, 128) FP32.
Method:
  - 1000 iterations per trial
  - 5 trials per impl, interleaved (so thermal/cache state hits both equally)
  - median of the 5 trials reported

Saves JSON to bench/results/rms_norm.json with per-trial timings and the
median + speedup. Prints PyTorch ms, Reikernel ms, speedup ratio.
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

SHAPE = (4, 256, 128)
DTYPE = torch.float32
N_ITERS = 1000
N_TRIALS = 5
WARMUP_ITERS = 200
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "rms_norm.json")


def _make_inputs():
    g = torch.Generator()
    g.manual_seed(0xBEEF)
    B, T, C = SHAPE
    x = (torch.rand((B, T, C), generator=g, dtype=DTYPE) * 2.0 - 1.0)
    w = (torch.rand((C,), generator=g, dtype=DTYPE) * 2.0 - 1.0) + 1.5
    return x, w


def _time_call(fn, x, w, n_iters):
    """Returns ms per call (median of n_iters calls)."""
    # Warm up
    for _ in range(WARMUP_ITERS):
        fn(x, w)
    # Time
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn(x, w)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e3  # ms per call


def main() -> int:
    # Load library early.
    rk.load_library()

    x, w = _make_inputs()

    # Correctness check before the benchmark.
    y_torch = F.rms_norm(x, normalized_shape=(SHAPE[-1],), weight=w, eps=1e-5)
    y_rk = rk.rms_norm(x, w, eps=1e-5)
    max_diff = (y_rk - y_torch).abs().max().item()
    ok = torch.allclose(y_rk, y_torch, atol=1e-5, rtol=1e-5)
    print(f"Correctness check on {SHAPE}: max_abs_diff={max_diff:.2e}, "
          f"allclose(atol=1e-5)={ok}")
    if not ok:
        print("ERROR: correctness check FAILED; aborting benchmark.",
              file=sys.stderr)
        return 2

    print()
    print(f"Shape: {SHAPE}, dtype={DTYPE}")
    print(f"Iters/trial: {N_ITERS}, trials: {N_TRIALS}, warmup: {WARMUP_ITERS}")
    print(f"OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")
    print()

    rk_fn = lambda x, w: rk.rms_norm(x, w, eps=1e-5)
    torch_fn = lambda x, w: F.rms_norm(
        x, normalized_shape=(SHAPE[-1],), weight=w, eps=1e-5
    )

    rk_trials = []
    pt_trials = []

    # Interleave: alternate trials so thermal state is shared.
    for i in range(N_TRIALS):
        ms_rk = _time_call(rk_fn, x, w, N_ITERS)
        ms_pt = _time_call(torch_fn, x, w, N_ITERS)
        rk_trials.append(ms_rk)
        pt_trials.append(ms_pt)
        print(f"  trial {i+1}/{N_TRIALS}: rk={ms_rk*1000:7.2f} us  "
              f"pt={ms_pt*1000:7.2f} us")

    rk_median = statistics.median(rk_trials)
    pt_median = statistics.median(pt_trials)
    speedup = pt_median / rk_median if rk_median > 0 else float("inf")

    print()
    print(f"PyTorch   F.rms_norm median = {pt_median*1000:8.2f} us/call "
          f"({pt_median:.4f} ms)")
    print(f"Reikernel rk.rms_norm median = {rk_median*1000:8.2f} us/call "
          f"({rk_median:.4f} ms)")
    print(f"Speedup: {speedup:.2f}x  (median of {N_TRIALS} trials x "
          f"{N_ITERS} iters)")

    # Save JSON.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "shape": list(SHAPE),
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
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON to {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
