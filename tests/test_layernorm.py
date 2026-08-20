"""Correctness tests for reikernel.layer_norm vs torch.nn.functional.layer_norm.

Run:
    python3 tests/test_layernorm.py

Exit code 0 means all numerical + validation tests passed.

Math tolerance: atol=1e-5, rtol=1e-5, matches the v0.1 rms_norm tolerance
per spec. Layernorm is numerically benign (mean + var, no exp chain) so
1e-5 catches reordering bugs while leaving room for the ~1 ULP FP32
differences between our two-pass reduction and PyTorch's Welford-based
RowwiseMoments.
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
# Numerical correctness: 6 shapes, 2 eps values, vs F.layer_norm at atol=1e-5.
#
# Shapes cover (per spec):
#   - (1, 1, 128)      -- tiny, B*T=1
#   - (4, 256, 128)    -- the bench shape (same as rms_norm bench)
#   - (8, 256, 320)    -- larger C, AVX-512 friendly
#   - (2, 16, 1)       -- C=1 edge (var=0, rstd=1/sqrt(eps))
#   - (2, 4, 7)        -- non-aligned C (tail not 16-wide; simd remainder)
#   - (1, 3, 64)       -- smallest non-trivial, even C
# ----------------------------------------------------------------------------

SHAPES = [
    (1, 1, 128),
    (4, 256, 128),
    (8, 256, 320),
    (2, 16, 1),
    (2, 4, 7),
    (1, 3, 64),
]

EPS_VALUES = [1e-5, 1e-6]

ATOL = 1e-5
RTOL = 1e-5


def _make_inputs(shape, dtype=torch.float32, seed=0xC0FFEE):
    """Deterministic input so failures are reproducible.

    Uniform [-1, 1) keeps x^2 bounded so the variance stays well-defined
    and well-conditioned (var ~ 1/3 for uniform [-1,1)); max-subtract-like
    catastrophic cancellation doesn't apply for layernorm.
    """
    g = torch.Generator()
    g.manual_seed(seed ^ (sum(c << (8 * i) for i, c in enumerate(shape))))
    B, T, C = shape
    x = (torch.rand((B, T, C), generator=g, dtype=dtype) * 2.0 - 1.0)
    w = (torch.rand((C,), generator=g, dtype=dtype) * 2.0 - 1.0) + 1.5
    b = (torch.rand((C,), generator=g, dtype=dtype) * 2.0 - 1.0) * 0.5
    return x, w, b


def _torch_layer_norm(x, w, b, eps):
    """Reference: torch.nn.functional.layer_norm with normalized_shape=[C]."""
    C = x.shape[-1]
    return F.layer_norm(x, normalized_shape=(C,), weight=w, bias=b, eps=eps)


def test_numerical():
    print("=== Numerical correctness: rk.layer_norm vs F.layer_norm ===")
    n_pass = 0
    n_fail = 0
    max_diff_seen = 0.0
    for shape in SHAPES:
        for eps in EPS_VALUES:
            x, w, b = _make_inputs(shape)
            y_torch = _torch_layer_norm(x, w, b, eps)
            try:
                y_rk = rk.layer_norm(x, w, b, eps=eps)
            except Exception as e:
                print(f"  FAIL shape={shape} eps={eps}: rk raised {e}")
                traceback.print_exc()
                n_fail += 1
                continue
            ok = torch.allclose(y_rk, y_torch, atol=ATOL, rtol=RTOL)
            max_diff = (y_rk - y_torch).abs().max().item()
            max_diff_seen = max(max_diff_seen, max_diff)
            # Shape and dtype checks
            shape_ok = (y_rk.shape == y_torch.shape == shape)
            dtype_ok = (y_rk.dtype == torch.float32)
            if ok and shape_ok and dtype_ok:
                print(f"  PASS shape={str(shape):<16} eps={eps}  "
                      f"max_abs_diff={max_diff:.2e}")
                n_pass += 1
            else:
                print(f"  FAIL shape={str(shape):<16} eps={eps}  "
                      f"max_abs_diff={max_diff:.2e} (>{ATOL:.0e})  "
                      f"allclose={ok} shape_ok={shape_ok} dtype_ok={dtype_ok}")
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
        print(f"  PASS {label}: raised {type(e).__name__}")
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

    x_ok, w_ok, b_ok = _make_inputs((4, 256, 128))

    # 1. non-float32 x
    def t1():
        x = x_ok.to(torch.float64)
        rk.layer_norm(x, w_ok, b_ok)
    if _expect_raises(t1, "non-float32 x"): n_pass += 1
    else: n_fail += 1

    # 2. non-contiguous x
    def t2():
        x = x_ok.transpose(0, 1)  # not contiguous
        assert not x.is_contiguous()
        rk.layer_norm(x, w_ok, b_ok)
    if _expect_raises(t2, "non-contiguous x"): n_pass += 1
    else: n_fail += 1

    # 3. wrong rank x (2D)
    def t3():
        x = x_ok.reshape(4 * 256, 128)  # 2D
        rk.layer_norm(x, w_ok, b_ok)
    if _expect_raises(t3, "wrong rank x (2D)"): n_pass += 1
    else: n_fail += 1

    # 4. weight shape mismatch
    def t4():
        w = torch.randn(127, dtype=torch.float32)  # off by one
        rk.layer_norm(x_ok, w, b_ok)
    if _expect_raises(t4, "weight shape mismatch"): n_pass += 1
    else: n_fail += 1

    # 5. bias shape mismatch
    def t5():
        b = torch.randn(127, dtype=torch.float32)  # off by one
        rk.layer_norm(x_ok, w_ok, b)
    if _expect_raises(t5, "bias shape mismatch"): n_pass += 1
    else: n_fail += 1

    # 6. negative eps
    def t6():
        rk.layer_norm(x_ok, w_ok, b_ok, eps=-1e-5)
    if _expect_raises(t6, "negative eps"): n_pass += 1
    else: n_fail += 1

    # 7. non-tensor x
    def t7():
        rk.layer_norm([1, 2, 3], w_ok, b_ok)  # type: ignore
    if _expect_raises(t7, "non-tensor x"): n_pass += 1
    else: n_fail += 1

    # 8. non-float32 weight
    def t8():
        rk.layer_norm(x_ok, w_ok.to(torch.float64), b_ok)
    if _expect_raises(t8, "non-float32 weight"): n_pass += 1
    else: n_fail += 1

    # 9. non-float32 bias
    def t9():
        rk.layer_norm(x_ok, w_ok, b_ok.to(torch.float64))
    if _expect_raises(t9, "non-float32 bias"): n_pass += 1
    else: n_fail += 1

    # 10. wrong weight rank (2D)
    def t10():
        w = w_ok.unsqueeze(0)  # (1, C)
        rk.layer_norm(x_ok, w, b_ok)
    if _expect_raises(t10, "wrong weight rank (2D)"): n_pass += 1
    else: n_fail += 1

    # 11. wrong bias rank (2D)
    def t11():
        b = b_ok.unsqueeze(0)  # (1, C)
        rk.layer_norm(x_ok, w_ok, b)
    if _expect_raises(t11, "wrong bias rank (2D)"): n_pass += 1
    else: n_fail += 1

    # 12. non-contiguous bias
    def t12():
        # Build a (2*C,) buffer and slice every other element -> stride 2,
        # which is not contiguous.
        C_dim = b_ok.shape[0]
        big = torch.empty(2 * C_dim, dtype=torch.float32)
        b = big[::2]  # stride 2 along the only dim -> not contiguous
        assert not b.is_contiguous()
        rk.layer_norm(x_ok, w_ok, b)
    if _expect_raises(t12, "non-contiguous bias"): n_pass += 1
    else: n_fail += 1

    print(f"  Validation: {n_pass} pass, {n_fail} fail")
    return n_fail == 0


# ----------------------------------------------------------------------------
# bias=None support: drop-in for F.layer_norm(input, normalized_shape,
# weight, bias=None, eps). Should NOT raise, and should match
# F.layer_norm(x, (C,), weight=w, bias=None, eps) numerically.
# ----------------------------------------------------------------------------

def test_bias_none():
    print("\n=== bias=None support ===")
    x, w, _ = _make_inputs((4, 256, 128))
    y_torch = F.layer_norm(x, normalized_shape=(128,), weight=w, bias=None,
                          eps=1e-5)
    try:
        y_rk = rk.layer_norm(x, w, bias=None, eps=1e-5)
    except Exception as e:
        print(f"  FAIL bias=None: raised {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    ok = torch.allclose(y_rk, y_torch, atol=ATOL, rtol=RTOL)
    max_diff = (y_rk - y_torch).abs().max().item()
    if ok:
        print(f"  PASS bias=None path matches F.layer_norm(bias=None); "
              f"max_abs_diff={max_diff:.2e}")
        return True
    print(f"  FAIL bias=None path mismatch: max_abs_diff={max_diff:.2e}")
    return False


# ----------------------------------------------------------------------------
# Output aliasing check: rk.layer_norm must NOT write into x / w / b.
# ----------------------------------------------------------------------------

def test_no_aliasing():
    print("\n=== Output aliasing check ===")
    x, w, b = _make_inputs((4, 256, 128))
    x_before = x.clone()
    w_before = w.clone()
    b_before = b.clone()
    _ = rk.layer_norm(x, w, b)
    ok = torch.equal(x, x_before) and torch.equal(w, w_before) \
        and torch.equal(b, b_before)
    if ok:
        print("  PASS: x, w, b unchanged after call")
        return True
    print("  FAIL: input x/w/b was mutated by rk.layer_norm")
    return False


# ----------------------------------------------------------------------------
# Output independence: a fresh output tensor is allocated each call.
# ----------------------------------------------------------------------------

def test_output_independence():
    print("\n=== Output independence check ===")
    x, w, b = _make_inputs((4, 256, 128))
    y1 = rk.layer_norm(x, w, b)
    y2 = rk.layer_norm(x, w, b)
    if y1.data_ptr() == y2.data_ptr():
        print("  FAIL: rk.layer_norm returned the same tensor object across "
              "calls (data_ptr matches)")
        return False
    if not torch.allclose(y1, y2, atol=ATOL, rtol=RTOL):
        print("  FAIL: rk.layer_norm(x) != rk.layer_norm(x) numerically")
        return False
    print("  PASS: two calls return distinct tensors with matching values")
    return True


def main() -> int:
    # Touch the library first to get an early, clear error if it isn't built.
    try:
        rk.load_library()
    except ImportError as e:
        print(f"ERROR: could not load libreikernel.so: {e}", file=sys.stderr)
        return 2

    ok_num = test_numerical()
    ok_val = test_validation()
    ok_bias_none = test_bias_none()
    ok_alias = test_no_aliasing()
    ok_indep = test_output_independence()

    all_ok = ok_num and ok_val and ok_bias_none and ok_alias and ok_indep
    print()
    print("=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
