"""Correctness tests for reikernel.rms_norm vs torch.nn.functional.rms_norm.

Run:
    python3 tests/test_rms_norm.py

Exit code 0 means all numerical + validation tests passed.
"""

import os
import sys
import traceback

import torch
import torch.nn.functional as F

# Make sure the repo's python/ is importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

import reikernel as rk  # noqa: E402


# ----------------------------------------------------------------------------
# Numerical correctness: 6 shapes, 2 eps values, vs F.rms_norm at atol=1e-5.
# ----------------------------------------------------------------------------

SHAPES = [
    (1, 1, 128),       # tiny, B*T=1
    (4, 256, 128),     # the bench shape
    (8, 256, 320),     # larger C, AVX-512 friendly
    (2, 16, 1),        # C=1 edge (divisor C=1; rms = sqrt(x^2+eps))
    (2, 4, 7),         # non-aligned C (tail not 16-wide; simd remainder)
    (1, 3, 64),        # smallest non-trivial, even C
]

EPS_VALUES = [1e-5, 1e-6]


def _make_inputs(shape, dtype=torch.float32, seed=0xC0FFEE):
    """Deterministic input so failures are reproducible."""
    g = torch.Generator()
    g.manual_seed(seed ^ (sum(c << (8 * i) for i, c in enumerate(shape))))
    B, T, C = shape
    # Uniform [-1, 1) so x^2 stays bounded and the rms is well-defined.
    x = (torch.rand((B, T, C), generator=g, dtype=dtype) * 2.0 - 1.0)
    w = (torch.rand((C,), generator=g, dtype=dtype) * 2.0 - 1.0) + 1.5
    return x, w


def _torch_rms_norm(x, w, eps):
    """Reference: torch.nn.functional.rms_norm with normalized_shape=[C]."""
    C = x.shape[-1]
    return F.rms_norm(x, normalized_shape=(C,), weight=w, eps=eps)


def test_numerical():
    print("=== Numerical correctness: rk.rms_norm vs F.rms_norm ===")
    n_pass = 0
    n_fail = 0
    max_diff_seen = 0.0
    for shape in SHAPES:
        for eps in EPS_VALUES:
            x, w = _make_inputs(shape)
            y_torch = _torch_rms_norm(x, w, eps)
            try:
                y_rk = rk.rms_norm(x, w, eps=eps)
            except Exception as e:
                print(f"  FAIL shape={shape} eps={eps}: rk raised {e}")
                n_fail += 1
                continue
            ok = torch.allclose(y_rk, y_torch, atol=1e-5, rtol=1e-5)
            max_diff = (y_rk - y_torch).abs().max().item()
            max_diff_seen = max(max_diff_seen, max_diff)
            if ok:
                print(f"  PASS shape={str(shape):<16} eps={eps}  "
                      f"max_abs_diff={max_diff:.2e}")
                n_pass += 1
            else:
                print(f"  FAIL shape={str(shape):<16} eps={eps}  "
                      f"max_abs_diff={max_diff:.2e} (>{1e-5:.0e})")
                n_fail += 1
    print(f"  Numerical: {n_pass} pass, {n_fail} fail, "
          f"max_abs_diff_seen={max_diff_seen:.2e}")
    return n_fail == 0


# ----------------------------------------------------------------------------
# Input validation: each bad input must raise (ValueError or TypeError).
# ----------------------------------------------------------------------------

def _expect_raises(fn, label, exc_types=(ValueError, TypeError)):
    try:
        fn()
    except exc_types as e:
        print(f"  PASS {label}: raised {type(e).__name__}: {e}")
        return True
    except Exception as e:
        print(f"  FAIL {label}: raised wrong type {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    print(f"  FAIL {label}: did NOT raise")
    return False


def test_validation():
    print("\n=== Input validation ===")
    n_pass = 0
    n_fail = 0

    x_ok, w_ok = _make_inputs((4, 256, 128))

    # 1. non-float32 x
    def t1():
        x = x_ok.to(torch.float64)
        rk.rms_norm(x, w_ok)
    if _expect_raises(t1, "non-float32 x"): n_pass += 1
    else: n_fail += 1

    # 2. non-contiguous x
    def t2():
        x = x_ok.transpose(0, 1)  # not contiguous
        assert not x.is_contiguous()
        rk.rms_norm(x, w_ok)
    if _expect_raises(t2, "non-contiguous x"): n_pass += 1
    else: n_fail += 1

    # 3. wrong rank x (2D)
    def t3():
        x = x_ok.reshape(4 * 256, 128)  # 2D
        rk.rms_norm(x, w_ok)
    if _expect_raises(t3, "wrong rank x (2D)"): n_pass += 1
    else: n_fail += 1

    # 4. weight shape mismatch
    def t4():
        w = torch.randn(C_BAD := 127, dtype=torch.float32)  # off by one
        rk.rms_norm(x_ok, w)
    if _expect_raises(t4, "weight shape mismatch"): n_pass += 1
    else: n_fail += 1

    # 5. negative eps
    def t5():
        rk.rms_norm(x_ok, w_ok, eps=-1e-5)
    if _expect_raises(t5, "negative eps"): n_pass += 1
    else: n_fail += 1

    # 6. non-tensor input
    def t6():
        rk.rms_norm([1, 2, 3], w_ok)  # type: ignore
    if _expect_raises(t6, "non-tensor x"): n_pass += 1
    else: n_fail += 1

    # 7. non-float32 weight
    def t7():
        rk.rms_norm(x_ok, w_ok.to(torch.float64))
    if _expect_raises(t7, "non-float32 weight"): n_pass += 1
    else: n_fail += 1

    # 8. wrong weight rank (2D)
    def t8():
        w = w_ok.unsqueeze(0)  # (1, C)
        rk.rms_norm(x_ok, w)
    if _expect_raises(t8, "wrong weight rank (2D)"): n_pass += 1
    else: n_fail += 1

    print(f"  Validation: {n_pass} pass, {n_fail} fail")
    return n_fail == 0


# ----------------------------------------------------------------------------
# Output aliasing check: rk.rms_norm must NOT write into x.
# ----------------------------------------------------------------------------

def test_no_aliasing():
    print("\n=== Output aliasing check ===")
    x, w = _make_inputs((4, 256, 128))
    x_before = x.clone()
    _ = rk.rms_norm(x, w)
    if torch.equal(x, x_before):
        print("  PASS: input x unchanged after call")
        return True
    print("  FAIL: input x was mutated by rk.rms_norm")
    return False


def main() -> int:
    # Touch the library first to get an early, clear error if it isn't built.
    try:
        rk.load_library()
    except ImportError as e:
        print(f"ERROR: could not load libreikernel.so: {e}", file=sys.stderr)
        return 2

    ok_num = test_numerical()
    ok_val = test_validation()
    ok_alias = test_no_aliasing()

    all_ok = ok_num and ok_val and ok_alias
    print()
    print("=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
