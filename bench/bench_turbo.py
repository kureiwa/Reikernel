"""Smoke benchmark: rk.rms_norm inside vs outside `with rk.turbo()`.

This is a *smoke* benchmark: the goal is to verify the turbo() context
manager doesn't break anything and to give a rough idea of the speedup,
not to produce a precise training-loop measurement (that comes next).

The spec: "run a small computation (e.g. 1000 rms_norm calls on
(4, 256, 128)) inside vs outside `with rk.turbo()`. Print speedup."

Why this is a SMOKE bench, not a real bench:

  - On the 2-vCPU sandbox with threads_per_core == 1, "physical cores" ==
    "all logical CPUs" == {0, 1}. The turbo() affinity mask is identical
    to the default mask, so pinning is effectively a no-op. The "skip
    hyperthreads" win only materialises on a system with SMT.

  - 1000 rms_norm calls at ~22 us/call = ~22 ms total. Way under the
    ~500 ms Python GC cycle, so gc.disable() likely doesn't avoid any
    GC passes in this short a run. The GC-disable win only shows up in
    a multi-second training loop where GC has time to fire multiple
    times.

  - The torch.set_num_threads(2) is the default on the 2-vCPU sandbox
    anyway, so there's no thread-count change to measure.

So the expected speedup here is ~1.0x: turbo() is correctly activated
and doesn't break anything, but the bench is too short to see the wins
that would matter in a real training loop. The honest assessment is
in the printed result + saved JSON.

Method:
  - 1000 iterations per trial, 5 trials, 200-iter warmup, interleaved
    (turbo trial, no-turbo trial, repeat) so thermal state hits both
    equally.
  - Before each no-turbo trial, snapshot and clear the OMP_* env vars
    (so a previous turbo call's setenv doesn't contaminate the no-turbo
    measurement). Restore them afterwards so the next turbo call works
    as expected.

Saves JSON to bench/results/turbo.json.
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "turbo.json")

# Env vars the C side sets on rk_turbo_enter; we snapshot+clear them
# before each no-turbo trial so they don't leak across the comparison.
_OMP_ENV_VARS = ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")


def _make_inputs():
    g = torch.Generator()
    g.manual_seed(0xBEEF)
    B, T, C = SHAPE
    x = (torch.rand((B, T, C), generator=g, dtype=DTYPE) * 2.0 - 1.0)
    w = (torch.rand((C,), generator=g, dtype=DTYPE) * 2.0 - 1.0) + 1.5
    return x, w


def _snapshot_omp_env() -> dict:
    return {k: os.environ.pop(k, None) for k in _OMP_ENV_VARS}


def _restore_omp_env(saved: dict) -> None:
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _time_no_turbo(x, w, n_iters) -> float:
    """Returns ms per call with NO turbo context active.

    Snapshots and clears OMP_* env vars so a previous turbo call's
    setenv doesn't leak into the no-turbo measurement; restores them
    afterwards.
    """
    saved = _snapshot_omp_env()
    try:
        # Warmup (outside the timing loop).
        for _ in range(WARMUP_ITERS):
            rk.rms_norm(x, w, eps=1e-5)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            rk.rms_norm(x, w, eps=1e-5)
        t1 = time.perf_counter()
    finally:
        _restore_omp_env(saved)
    return (t1 - t0) / n_iters * 1e3  # ms per call


def _time_with_turbo(x, w, n_iters) -> float:
    """Returns ms per call WITH the turbo context active.

    Enters the turbo context, runs warmup + timing inside, exits.
    The setup overhead (topo_probe + setenv + sched_setaffinity) is
    amortised across the n_iters rms_norm calls, so the per-call cost
    reported is steady-state, not amortised setup.
    """
    with rk.turbo() as n:
        # Warmup inside the turbo context.
        for _ in range(WARMUP_ITERS):
            rk.rms_norm(x, w, eps=1e-5)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            rk.rms_norm(x, w, eps=1e-5)
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

    topo = rk.topo_detect()
    print()
    print(f"Shape: {SHAPE}, dtype={DTYPE}")
    print(f"Iters/trial: {N_ITERS}, trials: {N_TRIALS}, warmup: {WARMUP_ITERS}")
    print(f"Topology: {topo['total_cpus']} cpus "
          f"({topo['cores_per_package']}c x {topo['num_packages']}p, "
          f"{topo['threads_per_core']} threads/core, "
          f"{topo['num_numa_nodes']} numa node(s))")
    print(f"Physical cores (turbo pins to): {topo['physical_cores']}")
    print(f"Default torch threads: {torch.get_num_threads()}")
    print()

    # Interleave turbo / no-turbo trials so thermal state hits both.
    turbo_trials = []
    plain_trials = []
    for i in range(N_TRIALS):
        ms_turbo = _time_with_turbo(x, w, N_ITERS)
        ms_plain = _time_no_turbo(x, w, N_ITERS)
        turbo_trials.append(ms_turbo)
        plain_trials.append(ms_plain)
        print(f"  trial {i+1}/{N_TRIALS}: turbo={ms_turbo*1000:7.2f} us  "
              f"plain={ms_plain*1000:7.2f} us  "
              f"ratio={ms_plain/ms_turbo:.2f}x")

    turbo_median = statistics.median(turbo_trials)
    plain_median = statistics.median(plain_trials)
    speedup = plain_median / turbo_median if turbo_median > 0 else float("inf")

    print()
    print(f"Without rk.turbo() median = {plain_median*1000:8.2f} us/call "
          f"({plain_median:.4f} ms)")
    print(f"With    rk.turbo() median = {turbo_median*1000:8.2f} us/call "
          f"({turbo_median:.4f} ms)")
    print(f"Speedup: {speedup:.2f}x  (median of {N_TRIALS} trials x "
          f"{N_ITERS} iters)")

    print()
    print("Note: this is a SMOKE benchmark. On the 2-vCPU sandbox with")
    print("threads_per_core == 1, 'pinning to physical cores' is a no-op")
    print("(physical == logical). The 22 ms run is too short to hit a GC")
    print("cycle, so gc.disable() also doesn't show a measurable win.")
    print("Expected speedup ~1.0x; the real benefit shows up in a")
    print("multi-second training loop where GC fires + SMT contention")
    print("matters. See worklog REIKERNEL-V0.5.")

    # Save JSON.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "shape": list(SHAPE),
        "dtype": str(DTYPE),
        "n_iters": N_ITERS,
        "n_trials": N_TRIALS,
        "warmup_iters": WARMUP_ITERS,
        "topology": topo,
        "turbo_trials_ms": turbo_trials,
        "plain_trials_ms": plain_trials,
        "turbo_median_ms": turbo_median,
        "plain_median_ms": plain_median,
        "speedup": speedup,
        "correctness": {
            "max_abs_diff": max_diff,
            "allclose_atol_1e-5": ok,
        },
        "note": (
            "Smoke benchmark. On the 2-vCPU sandbox with threads_per_core==1, "
            "pinning is a no-op (physical==logical) and the 22 ms run is too "
            "short to hit a GC cycle, so the expected speedup is ~1.0x. The "
            "real benefit of turbo() shows up in a multi-second training loop "
            "where GC fires + SMT contention matters."
        ),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON to {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
