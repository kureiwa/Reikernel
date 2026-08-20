"""Correctness tests for reikernel.softmax vs torch.nn.functional.softmax.

Run:
    python3 tests/test_softmax.py

Exit code 0 means all numerical + validation tests passed.

Math tolerance: atol=1e-6, rtol=1e-6. Tighter than matmul (1e-5)
because softmax is more numerically sensitive (the per-row max-subtract
+ exp + sum + divide chain compounds rounding; the spec mandates 1e-6 to
catch reordering bugs that would slip past a looser tolerance).
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
# Numerical correctness: 7 shapes, vs F.softmax(x, dim=-1) at atol=1e-6.
#
# Shapes cover (per spec):
#   - (1, 1, 257)      -- vocab=257 edge: single row
#   - (4, 256, 257)    -- the model's final logits shape (vocab=257)
#   - (8, 256, 1024)   -- wider vocab
#   - (2, 16, 1)       -- V=1 edge (softmax of a length-1 vector = [1.0])
#   - (1, 3, 64)       -- small
#   - (4, 1, 257)      -- T=1 (single-token batch)
#   - (1, 1, 1)        -- trivial
# ----------------------------------------------------------------------------

SHAPES = [
    (1, 1, 257),
    (4, 256, 257),
    (8, 256, 1024),
    (2, 16, 1),
    (1, 3, 64),
    (4, 1, 257),
    (1, 1, 1),
]

# Tighter tolerance than matmul (1e-5) because softmax is more numerically
# sensitive; the spec mandates 1e-6.
ATOL = 1e-6
RTOL = 1e-6


def _make_input(B, T, V, seed=0xC0FFEE):
    """Deterministic input so failures are reproducible.

    Uniform [-3, 3) keeps output magnitudes bounded; the wider range vs
    matmul's [-1, 1) stresses the max-subtract + exp numerical stability
    path (max - min can be up to 6, so exp(6) ~ 403 vs exp(-6) ~ 0.0025).
    """
    g = torch.Generator()
    g.manual_seed(seed ^ (B * 7919 + T * 31 + V))
    x = (torch.rand((B, T, V), generator=g, dtype=torch.float32) * 6.0 - 3.0)
    return x


def test_numerical():
    print("=== Numerical correctness: rk.softmax vs F.softmax(dim=-1) ===")
    n_pass = 0
    n_fail = 0
    max_diff_seen = 0.0
    for (B, T, V) in SHAPES:
        x = _make_input(B, T, V)
        y_torch = F.softmax(x, dim=-1)
        try:
            y_rk = rk.softmax(x)
        except Exception as e:
            print(f"  FAIL shape=({B},{T},{V}): rk raised {e}")
            traceback.print_exc()
            n_fail += 1
            continue
        ok = torch.allclose(y_rk, y_torch, atol=ATOL, rtol=RTOL)
        max_diff = (y_rk - y_torch).abs().max().item()
        max_diff_seen = max(max_diff_seen, max_diff)
        # Shape check
        shape_ok = (y_rk.shape == y_torch.shape == (B, T, V))
        # dtype check
        dtype_ok = (y_rk.dtype == torch.float32)
        # Sanity check: softmax output sums to 1 along last dim
        sum_ok = (V == 1) or torch.allclose(
            y_rk.sum(dim=-1),
            torch.ones((B, T), dtype=torch.float32),
            atol=1e-5,
        )
        if ok and shape_ok and dtype_ok and sum_ok:
            print(f"  PASS shape=({B:>2},{T:>3},{V:>4})  "
                  f"max_abs_diff={max_diff:.2e}  "
                  f"out={tuple(y_rk.shape)}  sum=1.0")
            n_pass += 1
        else:
            print(f"  FAIL shape=({B:>2},{T:>3},{V:>4})  "
                  f"max_abs_diff={max_diff:.2e} (>{ATOL:.0e})  "
                  f"allclose={ok} shape_ok={shape_ok} dtype_ok={dtype_ok} "
                  f"sum_ok={sum_ok}")
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

    x_ok = _make_input(4, 16, 32)

    # 1. non-float32 x
    def t1():
        x = x_ok.to(torch.float64)
        rk.softmax(x)
    if _expect_raises(t1, "non-float32 x"): n_pass += 1
    else: n_fail += 1

    # 2. non-float32 x (fp16)
    def t2():
        x = x_ok.to(torch.float16)
        rk.softmax(x)
    if _expect_raises(t2, "non-float32 x (fp16)"): n_pass += 1
    else: n_fail += 1

    # 3. non-contiguous x (transpose the last two dims)
    def t3():
        # x_ok shape (4, 16, 32); transpose -> (4, 32, 16), non-contiguous
        x = x_ok.transpose(1, 2)
        assert not x.is_contiguous()
        rk.softmax(x)
    if _expect_raises(t3, "non-contiguous x"): n_pass += 1
    else: n_fail += 1

    # 4. non-contiguous x (stride trick on inner dim)
    def t4():
        # Build a (4, 16, 32) tensor with a non-trivial stride along V.
        big = torch.empty((4, 16, 64), dtype=torch.float32)
        x = big[:, :, ::2]  # stride 2 along last dim -> not contiguous
        assert not x.is_contiguous()
        rk.softmax(x)
    if _expect_raises(t4, "non-contiguous x (strided)"): n_pass += 1
    else: n_fail += 1

    # 5. wrong rank (2D)
    def t5():
        x = torch.randn((16, 32), dtype=torch.float32)
        rk.softmax(x)
    if _expect_raises(t5, "wrong rank (2D)"): n_pass += 1
    else: n_fail += 1

    # 6. wrong rank (4D)
    def t6():
        x = torch.randn((1, 4, 16, 32), dtype=torch.float32)
        rk.softmax(x)
    if _expect_raises(t6, "wrong rank (4D)"): n_pass += 1
    else: n_fail += 1

    # 7. dim != -1 (dim=0)
    def t7():
        rk.softmax(x_ok, dim=0)
    if _expect_raises(t7, "dim=0 (not supported)"): n_pass += 1
    else: n_fail += 1

    # 8. dim != -1 (dim=1)
    def t8():
        rk.softmax(x_ok, dim=1)
    if _expect_raises(t8, "dim=1 (not supported)"): n_pass += 1
    else: n_fail += 1

    # 9. non-tensor x
    def t9():
        rk.softmax([[1.0, 2.0, 3.0]])  # type: ignore
    if _expect_raises(t9, "non-tensor x"): n_pass += 1
    else: n_fail += 1

    # 10. explicit last-dim (dim=2 for 3D x) should SUCCEED (== dim=-1)
    def t10():
        x = _make_input(2, 4, 16)
        y_rk = rk.softmax(x, dim=2)
        y_pt = F.softmax(x, dim=2)
        assert torch.allclose(y_rk, y_pt, atol=ATOL, rtol=RTOL), \
            "dim=2 path should match dim=-1 numerically"
    try:
        t10()
        print(f"  PASS explicit dim=2 (== dim=-1) accepts and matches")
        n_pass += 1
    except Exception as e:
        print(f"  FAIL explicit dim=2: raised {type(e).__name__}: {e}")
        traceback.print_exc()
        n_fail += 1

    print(f"  Validation: {n_pass} pass, {n_fail} fail")
    return n_fail == 0


# ----------------------------------------------------------------------------
# Output aliasing check: rk.softmax must NOT write into x.
# ----------------------------------------------------------------------------

def test_no_aliasing():
    print("\n=== Output aliasing check ===")
    x = _make_input(4, 16, 32)
    x_before = x.clone()
    _ = rk.softmax(x)
    ok = torch.equal(x, x_before)
    if ok:
        print("  PASS: input x unchanged after call")
        return True
    print("  FAIL: input x was modified during softmax")
    return False


# ----------------------------------------------------------------------------
# Output independence: a fresh output tensor is allocated each call.
# ----------------------------------------------------------------------------

def test_output_independence():
    print("\n=== Output independence check ===")
    x = _make_input(4, 16, 32)
    y1 = rk.softmax(x)
    y2 = rk.softmax(x)
    if y1.data_ptr() == y2.data_ptr():
        print("  FAIL: rk.softmax returned the same tensor object across "
              "calls (data_ptr matches)")
        return False
    if not torch.allclose(y1, y2, atol=ATOL, rtol=RTOL):
        print("  FAIL: rk.softmax(x) != rk.softmax(x) numerically")
        return False
    print("  PASS: two calls return distinct tensors with matching values")
    return True


# ----------------------------------------------------------------------------
# Numerical edge cases (additional cases beyond the spec shapes).
# ----------------------------------------------------------------------------

def test_edge_cases():
    print("\n=== Numerical edge cases ===")
    n_pass = 0
    n_fail = 0

    # Edge 1: all-equal row -> uniform output 1/V
    x = torch.zeros((1, 1, 8), dtype=torch.float32)
    y_rk = rk.softmax(x)
    y_pt = F.softmax(x, dim=-1)
    ok = torch.allclose(y_rk, y_pt, atol=ATOL, rtol=RTOL) and \
         torch.allclose(y_rk, torch.full((1, 1, 8), 0.125, dtype=torch.float32),
                        atol=ATOL, rtol=RTOL)
    print(f"  {'PASS' if ok else 'FAIL'} all-equal row -> uniform 1/V")
    if ok: n_pass += 1
    else: n_fail += 1

    # Edge 2: large positive values (numerical stability via max-subtract)
    x = torch.tensor([[[100.0, 200.0, 300.0, 400.0]]], dtype=torch.float32)
    y_rk = rk.softmax(x)
    y_pt = F.softmax(x, dim=-1)
    ok = torch.allclose(y_rk, y_pt, atol=ATOL, rtol=RTOL)
    # Should be ~[0, 0, 0, 1] (only the max survives)
    expected = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
    ok2 = torch.allclose(y_rk, expected, atol=1e-5, rtol=1e-5)
    print(f"  {'PASS' if (ok and ok2) else 'FAIL'} large positive values "
          f"(max-subtract stability), y={y_rk.squeeze().tolist()}")
    if ok and ok2: n_pass += 1
    else: n_fail += 1

    # Edge 3: large negative values
    x = torch.tensor([[[-100.0, -200.0, -300.0, -400.0]]], dtype=torch.float32)
    y_rk = rk.softmax(x)
    y_pt = F.softmax(x, dim=-1)
    ok = torch.allclose(y_rk, y_pt, atol=ATOL, rtol=RTOL)
    # Should be ~[1, 0, 0, 0]
    expected = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    ok2 = torch.allclose(y_rk, expected, atol=1e-5, rtol=1e-5)
    print(f"  {'PASS' if (ok and ok2) else 'FAIL'} large negative values")
    if ok and ok2: n_pass += 1
    else: n_fail += 1

    # Edge 4: mixed sign, mixed magnitude
    x = torch.tensor([[[5.0, -5.0, 0.0, 3.0, -2.0]]], dtype=torch.float32)
    y_rk = rk.softmax(x)
    y_pt = F.softmax(x, dim=-1)
    ok = torch.allclose(y_rk, y_pt, atol=ATOL, rtol=RTOL)
    sum_ok = torch.allclose(y_rk.sum(), torch.tensor(1.0), atol=1e-5)
    print(f"  {'PASS' if (ok and sum_ok) else 'FAIL'} mixed sign/magnitude, "
          f"sum={y_rk.sum().item():.6f}")
    if ok and sum_ok: n_pass += 1
    else: n_fail += 1

    # Edge 5: V=1 (degenerate softmax -> always 1.0)
    x = torch.tensor([[[3.14]], [[-2.71]], [[0.0]]], dtype=torch.float32)
    y_rk = rk.softmax(x)
    y_pt = F.softmax(x, dim=-1)
    ok = torch.allclose(y_rk, y_pt, atol=ATOL, rtol=RTOL) and \
         torch.allclose(y_rk, torch.ones_like(x), atol=ATOL)
    print(f"  {'PASS' if ok else 'FAIL'} V=1 edge (softmax of length-1 = 1.0)")
    if ok: n_pass += 1
    else: n_fail += 1

    print(f"  Edge cases: {n_pass} pass, {n_fail} fail")
    return n_fail == 0


def main() -> int:
    # Touch the library first to get an early, clear error if it isn't built.
    try:
        rk.load_library()
    except ImportError as e:
        print(f"ERROR: could not load libreikernel.so: {e}", file=sys.stderr)
        return 2

    ok_num = test_numerical()
    ok_val = test_validation()
    ok_edge = test_edge_cases()
    ok_alias = test_no_aliasing()
    ok_indep = test_output_independence()

    all_ok = ok_num and ok_val and ok_edge and ok_alias and ok_indep
    print()
    print("=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
