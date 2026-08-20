"""Tests for reikernel.turbo() context manager + topo_detect() (v0.5).

Run:
    python3 tests/test_turbo.py

Exit code 0 means all 6 spec tests + the topo_detect sanity check pass:
  1. `with rk.turbo() as n:` yields the physical core count (== 2 on the
     2-vCPU sandbox).
  2. Inside the context, `torch.get_num_threads() == n`.
  3. Inside the context, `gc.isenabled() is False`.
  4. Exiting restores the previous torch thread count AND the previous gc
     state (enabled or disabled).
  5. Calling turbo() while already inside a turbo context raises
     RuntimeError (no silent nesting).
  6. An exception raised inside the with-body still triggers cleanup:
     torch thread count + gc state are restored, and the exception
     propagates to the caller.
"""

import gc
import os
import sys

import torch

# Make sure the repo's python/ is importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

import reikernel as rk  # noqa: E402


def _expect(cond: bool, label: str, detail: str = "") -> bool:
    if cond:
        print(f"  PASS  {label}")
        return True
    print(f"  FAIL  {label}  {detail}")
    return False


def test_turbo_yields_physical_count() -> bool:
    """Test 1: `with rk.turbo() as n:` yields the physical core count."""
    print("=== Test 1: turbo() yields physical core count ===")
    topo = rk.topo_detect()
    expected = topo["physical_cores"]
    with rk.turbo() as n:
        ok = _expect(
            n == expected,
            f"with rk.turbo() as n -> n == {expected}",
            detail=f"got n={n}, expected={expected}",
        )
    # Also confirm it matches topo_detect's total on the 2-vCPU sandbox
    # (no SMT, so physical_cores == total_cpus).
    sandbox_ok = _expect(
        expected == topo["total_cpus"] or topo["threads_per_core"] == 1,
        "physical_cores == total_cpus when threads_per_core == 1",
        detail=f"physical_cores={expected}, total_cpus={topo['total_cpus']}, "
               f"threads_per_core={topo['threads_per_core']}",
    )
    return ok and sandbox_ok


def test_torch_threads_inside() -> bool:
    """Test 2: inside the context, torch.get_num_threads() == n."""
    print("\n=== Test 2: torch.get_num_threads() == n inside turbo() ===")
    with rk.turbo() as n:
        cur = torch.get_num_threads()
        ok = _expect(
            cur == n,
            f"inside: torch.get_num_threads() == n ({n})",
            detail=f"got {cur}, expected {n}",
        )
    return ok


def test_gc_disabled_inside() -> bool:
    """Test 3: inside the context, gc.isenabled() is False."""
    print("\n=== Test 3: gc.isenabled() is False inside turbo() ===")
    with rk.turbo() as n:
        gc_on = gc.isenabled()
        ok = _expect(
            gc_on is False,
            f"inside: gc.isenabled() == False",
            detail=f"got gc.isenabled()={gc_on}",
        )
    return ok


def test_state_restored_on_exit() -> bool:
    """Test 4: exiting restores the previous torch thread count + gc state."""
    print("\n=== Test 4: torch threads + gc state restored on exit ===")
    # Save the pre-turbo state.
    torch_threads_before = torch.get_num_threads()
    gc_enabled_before = gc.isenabled()

    with rk.turbo() as n:
        # Sanity: state is changed inside.
        inside_threads = torch.get_num_threads()
        inside_gc = gc.isenabled()
        _expect(
            inside_threads == n and inside_gc is False,
            f"inside sanity: threads={n} (got {inside_threads}), "
            f"gc=False (got {inside_gc})",
        )

    torch_threads_after = torch.get_num_threads()
    gc_enabled_after = gc.isenabled()

    ok1 = _expect(
        torch_threads_after == torch_threads_before,
        f"torch.get_num_threads() restored to {torch_threads_before}",
        detail=f"got {torch_threads_after}",
    )
    ok2 = _expect(
        gc_enabled_after == gc_enabled_before,
        f"gc.isenabled() restored to {gc_enabled_before}",
        detail=f"got {gc_enabled_after}",
    )
    return ok1 and ok2


def test_nested_raises() -> bool:
    """Test 5: nested `with rk.turbo(): with rk.turbo():` raises."""
    print("\n=== Test 5: nested turbo() raises RuntimeError ===")
    raised_expected = False
    raised_unexpected = None
    try:
        with rk.turbo() as n_outer:
            # Outer context is now active; entering again must raise.
            try:
                with rk.turbo() as n_inner:
                    raised_unexpected = (
                        f"inner turbo() did NOT raise; n_inner={n_inner}"
                    )
            except RuntimeError as e:
                raised_expected = True
                print(f"        inner turbo() raised as expected: {e}")
    except Exception as e:
        # The outer cleanup should still run normally; if anything else
        # propagates here that's a bug.
        raised_unexpected = f"outer turbo() raised {type(e).__name__}: {e}"

    ok = _expect(
        raised_expected and raised_unexpected is None,
        "nested turbo() raises RuntimeError",
        detail=raised_unexpected or "no RuntimeError raised on nested call",
    )
    # And the outer turbo state must be fully restored afterwards.
    restored = _expect(
        rk._turbo_active is False,
        "after nested attempt: _turbo_active is False",
        detail=f"_turbo_active={rk._turbo_active!r}",
    )
    return ok and restored


def test_exception_safety() -> bool:
    """Test 6: an exception inside the with-body triggers cleanup AND
    propagates."""
    print("\n=== Test 6: exception inside with-body -> cleanup runs ===")
    torch_threads_before = torch.get_num_threads()
    gc_enabled_before = gc.isenabled()

    caught = False
    caught_msg = ""
    try:
        with rk.turbo() as n:
            # Inside: state is changed.
            inside_threads = torch.get_num_threads()
            inside_gc = gc.isenabled()
            if inside_threads != n:
                raise AssertionError(
                    f"inside: threads={inside_threads}, expected {n}"
                )
            if inside_gc is not False:
                raise AssertionError(
                    f"inside: gc.isenabled()={inside_gc}, expected False"
                )
            # Now raise a user-side exception.
            raise ValueError("simulated training-loop failure")
    except ValueError as e:
        caught = True
        caught_msg = str(e)
    except Exception as e:
        # Wrong exception type: that's a bug.
        return _expect(
            False,
            "exception safety: caught ValueError",
            detail=f"caught {type(e).__name__}: {e}",
        )

    propagated = _expect(
        caught and caught_msg == "simulated training-loop failure",
        f"ValueError propagated to caller ({caught_msg!r})",
        detail=f"caught={caught}, msg={caught_msg!r}",
    )

    # Cleanup must have run despite the exception.
    torch_threads_after = torch.get_num_threads()
    gc_enabled_after = gc.isenabled()
    threads_restored = _expect(
        torch_threads_after == torch_threads_before,
        f"after exception: torch threads restored to {torch_threads_before}",
        detail=f"got {torch_threads_after}",
    )
    gc_restored = _expect(
        gc_enabled_after == gc_enabled_before,
        f"after exception: gc.isenabled() restored to {gc_enabled_before}",
        detail=f"got {gc_enabled_after}",
    )
    # And the C-side turbo state must also be cleared (so a subsequent
    # turbo() call works).
    c_cleared = _expect(
        rk._turbo_active is False,
        "after exception: _turbo_active is False",
        detail=f"_turbo_active={rk._turbo_active!r}",
    )
    return propagated and threads_restored and gc_restored and c_cleared


def test_topo_detect_sanity() -> bool:
    """Sanity: topo_detect() returns a dict with the expected keys + values
    matching libtopo's topology probe on this host."""
    print("\n=== topo_detect() sanity ===")
    topo = rk.topo_detect()
    expected_keys = {
        "threads_per_core", "cores_per_package", "num_packages",
        "num_numa_nodes", "total_cpus", "physical_cores",
    }
    ok_keys = _expect(
        set(topo.keys()) == expected_keys,
        f"topo_detect() keys == {expected_keys}",
        detail=f"got {set(topo.keys())}",
    )
    # Defensive defaults in libtopo: every field >= 1.
    ok_vals = True
    for k in ("threads_per_core", "cores_per_package", "num_packages",
              "num_numa_nodes", "total_cpus"):
        if not isinstance(topo.get(k), int) or topo[k] < 1:
            ok_vals = _expect(False, f"topo[{k!r}] >= 1",
                              detail=f"got {topo.get(k)!r}")
    # physical_cores == cores_per_package * num_packages (derived).
    derived = topo["cores_per_package"] * topo["num_packages"]
    ok_derived = _expect(
        topo["physical_cores"] == derived,
        f"physical_cores ({topo['physical_cores']}) == "
        f"cores_per_package * num_packages ({derived})",
    )
    # On the 2-vCPU sandbox specifically: 2 cpus, 1 thread/core, 1 package.
    sandbox_ok = _expect(
        topo["total_cpus"] == 2 and topo["physical_cores"] == 2,
        "sandbox: total_cpus == 2 and physical_cores == 2",
        detail=f"total_cpus={topo['total_cpus']}, "
               f"physical_cores={topo['physical_cores']}",
    )
    return ok_keys and ok_vals and ok_derived and sandbox_ok


def main() -> int:
    # Touch the library first to get an early, clear error if it isn't built.
    try:
        rk.load_library()
    except ImportError as e:
        print(f"ERROR: could not load libreikernel.so: {e}", file=sys.stderr)
        return 2

    results = []
    results.append(test_topo_detect_sanity())
    results.append(test_turbo_yields_physical_count())
    results.append(test_torch_threads_inside())
    results.append(test_gc_disabled_inside())
    results.append(test_state_restored_on_exit())
    results.append(test_nested_raises())
    results.append(test_exception_safety())

    print()
    print("=" * 60)
    if all(results):
        print(f"ALL TESTS PASSED ({len(results)}/{len(results)})")
        return 0
    n_pass = sum(1 for r in results if r)
    print(f"SOME TESTS FAILED ({n_pass}/{len(results)} passed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
