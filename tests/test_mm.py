"""Correctness tests for reikernel.mm vs torch.mm.

Run:
    python3 tests/test_mm.py

Exit code 0 means all numerical + validation tests passed.
"""

import os
import sys
import traceback

import torch

# Make sure the repo's python/ is importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

import reikernel as rk  # noqa: E402


# ----------------------------------------------------------------------------
# Numerical correctness: 10 shapes, vs torch.mm at atol=1e-5, rtol=1e-5.
#
# Shapes cover:
#   - Trivial: (1,1,1)
#   - Small symmetric: (4,4,4), (8,8,8)
#   - Powers of two: (128,128,128), (256,256,256), (512,512,512)
#   - Non-square: (4,8,16) -- M<K<N
#   - Non-square: (8,16,4) -- M<K, K>N
#   - Tall/skinny: (1,128,128) -- M=1 (single-row output)
#   - Short/wide: (128,1,128) -- K=1 (outer product of two vectors)
# ----------------------------------------------------------------------------

SHAPES = [
    (1, 1, 1),         # trivial
    (4, 4, 4),         # small symmetric
    (8, 8, 8),         # MR-aligned (M=8=MR)
    (128, 128, 128),   # bench size 1
    (256, 256, 256),   # bench size 2
    (512, 512, 512),   # bench size 3
    (4, 8, 16),        # M<K<N; one micro-tile (4 rows < MR=8, 16 cols = NR)
    (8, 16, 4),        # M=MR, K=16, N=4 (NR tail = 4, masked store)
    (1, 128, 128),     # M=1 (GEMV-like); MR tail = 1
    (128, 1, 128),     # K=1 (outer product); kc tail = 1
]


def _make_inputs(M, K, N, seed=0xC0FFEE):
    """Deterministic input so failures are reproducible.

    Uniform [-1, 1) keeps output magnitudes bounded (~sqrt(K/3) std),
    which keeps torch.allclose(atol=1e-5, rtol=1e-5) honest.
    """
    g = torch.Generator()
    g.manual_seed(seed ^ (M * 7919 + K * 31 + N))
    A = (torch.rand((M, K), generator=g, dtype=torch.float32) * 2.0 - 1.0)
    B = (torch.rand((K, N), generator=g, dtype=torch.float32) * 2.0 - 1.0)
    return A, B


def test_numerical():
    print("=== Numerical correctness: rk.mm vs torch.mm ===")
    n_pass = 0
    n_fail = 0
    max_diff_seen = 0.0
    for (M, K, N) in SHAPES:
        A, B = _make_inputs(M, K, N)
        y_torch = torch.mm(A, B)
        try:
            y_rk = rk.mm(A, B)
        except Exception as e:
            print(f"  FAIL shape=({M},{K},{N}): rk raised {e}")
            traceback.print_exc()
            n_fail += 1
            continue
        ok = torch.allclose(y_rk, y_torch, atol=1e-5, rtol=1e-5)
        max_diff = (y_rk - y_torch).abs().max().item()
        max_diff_seen = max(max_diff_seen, max_diff)
        # Shape check
        shape_ok = (y_rk.shape == y_torch.shape == (M, N))
        # dtype check
        dtype_ok = (y_rk.dtype == torch.float32)
        if ok and shape_ok and dtype_ok:
            print(f"  PASS shape=({M:>3},{K:>3},{N:>3})  "
                  f"max_abs_diff={max_diff:.2e}  "
                  f"out={tuple(y_rk.shape)}")
            n_pass += 1
        else:
            print(f"  FAIL shape=({M:>3},{K:>3},{N:>3})  "
                  f"max_abs_diff={max_diff:.2e} (>{1e-5:.0e})  "
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

    A_ok, B_ok = _make_inputs(8, 16, 32)

    # 1. non-float32 A
    def t1():
        A = A_ok.to(torch.float64)
        rk.mm(A, B_ok)
    if _expect_raises(t1, "non-float32 A"): n_pass += 1
    else: n_fail += 1

    # 2. non-float32 B
    def t2():
        B = B_ok.to(torch.float64)
        rk.mm(A_ok, B)
    if _expect_raises(t2, "non-float32 B"): n_pass += 1
    else: n_fail += 1

    # 3. non-contiguous A
    def t3():
        A = A_ok.t()  # transpose -> (16, 8) but not contiguous
        assert not A.is_contiguous()
        rk.mm(A, B_ok)
    if _expect_raises(t3, "non-contiguous A"): n_pass += 1
    else: n_fail += 1

    # 4. non-contiguous B
    def t4():
        B = B_ok.t()  # (32, 16) non-contiguous
        assert not B.is_contiguous()
        rk.mm(A_ok, B)
    if _expect_raises(t4, "non-contiguous B"): n_pass += 1
    else: n_fail += 1

    # 5. wrong rank A (3D)
    def t5():
        A = A_ok.unsqueeze(0)  # (1, 8, 16)
        rk.mm(A, B_ok)
    if _expect_raises(t5, "wrong rank A (3D)"): n_pass += 1
    else: n_fail += 1

    # 6. wrong rank B (1D)
    def t6():
        B = B_ok[:, 0]  # (16,) 1D
        rk.mm(A_ok, B)
    if _expect_raises(t6, "wrong rank B (1D)"): n_pass += 1
    else: n_fail += 1

    # 7. shape mismatch (K_A != K_B)
    def t7():
        B = torch.randn(17, 32, dtype=torch.float32)  # K=17 != A's K=16
        rk.mm(A_ok, B)
    if _expect_raises(t7, "shape mismatch K_A != K_B"): n_pass += 1
    else: n_fail += 1

    # 8. non-tensor A
    def t8():
        rk.mm([[1.0, 2.0]], B_ok)  # type: ignore
    if _expect_raises(t8, "non-tensor A"): n_pass += 1
    else: n_fail += 1

    # 9. non-tensor B
    def t9():
        rk.mm(A_ok, "not a tensor")  # type: ignore
    if _expect_raises(t9, "non-tensor B"): n_pass += 1
    else: n_fail += 1

    # 10. 0-sized dim (M=0): torch.mm supports this (returns empty tensor)
    def t10():
        A = torch.empty((0, 16), dtype=torch.float32)
        B = torch.empty((16, 32), dtype=torch.float32)
        C = rk.mm(A, B)
        # Should succeed and return an empty (0, 32) tensor matching torch.mm
        y_torch = torch.mm(A, B)
        assert C.shape == (0, 32), f"expected (0, 32), got {tuple(C.shape)}"
        assert C.shape == y_torch.shape
    try:
        t10()
        print(f"  PASS zero-sized dim M=0: returned empty (0, 32) tensor")
        n_pass += 1
    except Exception as e:
        print(f"  FAIL zero-sized dim M=0: raised {type(e).__name__}: {e}")
        traceback.print_exc()
        n_fail += 1

    print(f"  Validation: {n_pass} pass, {n_fail} fail")
    return n_fail == 0


# ----------------------------------------------------------------------------
# Output aliasing check: rk.mm must NOT write into A or B.
# ----------------------------------------------------------------------------

def test_no_aliasing():
    print("\n=== Output aliasing check ===")
    A, B = _make_inputs(16, 16, 16)
    A_before = A.clone()
    B_before = B.clone()
    _ = rk.mm(A, B)
    ok_a = torch.equal(A, A_before)
    ok_b = torch.equal(B, B_before)
    if ok_a and ok_b:
        print("  PASS: inputs A and B unchanged after call")
        return True
    print(f"  FAIL: A unchanged={ok_a}, B unchanged={ok_b}")
    return False


# ----------------------------------------------------------------------------
# Output independence: a fresh output tensor is allocated each call.
# ----------------------------------------------------------------------------

def test_output_independence():
    print("\n=== Output independence check ===")
    A, B = _make_inputs(16, 16, 16)
    C1 = rk.mm(A, B)
    C2 = rk.mm(A, B)
    if C1.data_ptr() == C2.data_ptr():
        print("  FAIL: rk.mm returned the same tensor object across calls "
              "(data_ptr matches)")
        return False
    if not torch.allclose(C1, C2, atol=1e-5, rtol=1e-5):
        print("  FAIL: rk.mm(A,B) != rk.mm(A,B) numerically")
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
    ok_alias = test_no_aliasing()
    ok_indep = test_output_independence()

    all_ok = ok_num and ok_val and ok_alias and ok_indep
    print()
    print("=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
