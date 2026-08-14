"""reikernel -- PyTorch CPU backend extension backed by EoSD.

Drop-in ops (rms_norm, mm, softmax, layer_norm) plus a turbo() context
manager that pins threads to physical cores and disables Python GC.

    import reikernel as rk
    with rk.turbo() as n:
        out = model(x); loss.backward(); optimizer.step()
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from contextlib import contextmanager

import torch

__all__ = [
    "rms_norm",
    "mm",
    "softmax",
    "layer_norm",
    "turbo",
    "topo_detect",
    "load_library",
    "__version__",
]
__version__ = "0.5.0"

# ----------------------------------------------------------------------------
# Shared library loading
# ----------------------------------------------------------------------------

def _candidate_lib_paths() -> list[str]:
    """Locations to look for libreikernel.so, in order of preference."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidates = [
        # 1) build/ next to the python/ dir (the canonical Makefile output)
        os.path.join(repo_root, "build", "libreikernel.so"),
        # 2) alongside the python module (post-`make install`)
        os.path.join(here, "libreikernel.so"),
        # 3) system install
        "libreikernel.so",
    ]
    # 4) environment override
    env = os.environ.get("REIKERNEL_LIB")
    if env:
        candidates.insert(0, env)
    return candidates


_lib = None


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load (or return the cached) libreikernel.so handle.

    If ``path`` is given, that exact file is loaded. Otherwise the candidate
    list from ``_candidate_lib_paths()`` is tried in order.
    """
    global _lib
    if _lib is not None and path is None:
        return _lib

    tried: list[str] = []
    candidates = [path] if path else _candidate_lib_paths()
    last_err: Exception | None = None
    for c in candidates:
        tried.append(c)
        if not os.path.exists(c):
            continue
        try:
            lib = ctypes.CDLL(c)
        except OSError as e:
            last_err = e
            continue
        _configure_lib(lib)
        if path is None:
            _lib = lib
        return lib

    raise ImportError(
        "Could not load libreikernel.so. Tried:\n  "
        + "\n  ".join(tried)
        + ("\n\nLast error: %s" % last_err if last_err else "")
        + "\nBuild it first with `make` from the repo root, or set the "
        "REIKERNEL_LIB env var to the .so path."
    )


def _configure_lib(lib: ctypes.CDLL) -> None:
    """Declare ctypes argtypes/restype for the rk_* functions we expose."""
    f = lib.rk_rms_norm
    f.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # y
        ctypes.c_int,     # B
        ctypes.c_int,     # T
        ctypes.c_int,     # C
        ctypes.c_float,   # eps
    ]
    f.restype = ctypes.c_int

    g = lib.rk_mm
    g.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # C
        ctypes.c_int,     # M
        ctypes.c_int,     # K
        ctypes.c_int,     # N
    ]
    g.restype = ctypes.c_int

    h = lib.rk_softmax
    h.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # y
        ctypes.c_int,     # B
        ctypes.c_int,     # T
        ctypes.c_int,     # V
    ]
    h.restype = ctypes.c_int

    i = lib.rk_layer_norm
    i.argtypes = [
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # bias (may be NULL)
        ctypes.c_void_p,  # y
        ctypes.c_int,     # B
        ctypes.c_int,     # T
        ctypes.c_int,     # C
        ctypes.c_float,   # eps
    ]
    i.restype = ctypes.c_int

    # rk_turbo_enter / rk_turbo_exit / rk_topo_detect (v0.5)
    t = lib.rk_turbo_enter
    t.argtypes = []
    t.restype = ctypes.c_int

    u = lib.rk_turbo_exit
    u.argtypes = []
    u.restype = ctypes.c_int

    v = lib.rk_topo_detect
    v.argtypes = [
        ctypes.POINTER(ctypes.c_uint),  # threads_per_core
        ctypes.POINTER(ctypes.c_uint),  # cores_per_package
        ctypes.POINTER(ctypes.c_uint),  # num_packages
        ctypes.POINTER(ctypes.c_uint),  # num_numa_nodes
        ctypes.POINTER(ctypes.c_uint),  # total_cpus
    ]
    v.restype = ctypes.c_int


# ----------------------------------------------------------------------------
# rms_norm
# ----------------------------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Drop-in replacement for ``torch.nn.functional.rms_norm``.

    Computes ``y = x / sqrt(mean(x^2) + eps) * weight`` on FP32 contiguous
    tensors of shape ``(B, T, C)`` with weight shape ``(C,)``.

    Args:
        x:      FP32 contiguous tensor, 3D ``(B, T, C)``.
        weight: FP32 contiguous tensor, 1D ``(C,)``.
        eps:    Non-negative smoothing constant (default 1e-5, matches torch).

    Returns:
        FP32 contiguous tensor, same shape as ``x``.

    Raises:
        TypeError:  if x or weight is not a torch.Tensor.
        ValueError: if dtype != float32, not contiguous, wrong rank,
                    weight shape mismatch, or eps < 0.
    """
    # --- type checks ---
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x).__name__}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"weight must be torch.Tensor, got {type(weight).__name__}"
        )

    # --- dtype checks ---
    if x.dtype != torch.float32:
        raise ValueError(
            f"x.dtype must be torch.float32, got {x.dtype}"
        )
    if weight.dtype != torch.float32:
        raise ValueError(
            f"weight.dtype must be torch.float32, got {weight.dtype}"
        )

    # --- contiguity checks ---
    if not x.is_contiguous():
        raise ValueError(
            "x must be contiguous. Call x = x.contiguous() before rms_norm."
        )
    if not weight.is_contiguous():
        raise ValueError(
            "weight must be contiguous. Call weight = weight.contiguous() "
            "before rms_norm."
        )

    # --- rank checks ---
    if x.dim() != 3:
        raise ValueError(
            f"x must be 3D (B, T, C); got {x.dim()}D with shape {tuple(x.shape)}"
        )
    if weight.dim() != 1:
        raise ValueError(
            f"weight must be 1D (C,); got {weight.dim()}D with shape "
            f"{tuple(weight.shape)}"
        )

    # --- shape match ---
    B, T, C = x.shape[0], x.shape[1], x.shape[2]
    if weight.shape[0] != C:
        raise ValueError(
            f"weight.shape[0] ({weight.shape[0]}) must match x.shape[-1] "
            f"({C})"
        )

    # --- eps check ---
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    # --- allocate output ---
    y = torch.empty_like(x)

    # --- call C ---
    lib = load_library()
    rc = lib.rk_rms_norm(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(T),
        ctypes.c_int(C),
        ctypes.c_float(eps),
    )
    if rc != 0:
        raise RuntimeError(
            f"rk_rms_norm returned error code {rc} "
            "(NULL pointer, bad shape, or integer overflow)."
        )
    return y


# ----------------------------------------------------------------------------
# mm (matmul)
# ----------------------------------------------------------------------------

def mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for ``torch.mm``.

    Computes ``C = A @ B`` on FP32 contiguous 2D tensors.
      A: shape ``(M, K)``
      B: shape ``(K, N)``
      C: shape ``(M, N)``

    Backend (v0.2): tiled FP32 matmul with MC=64 x KC=256 x NC=128 blocking,
    MR=8 x NR=16 micro-tile, AVX-512 FMA inner kernel, OpenMP ``parallel
    for collapse(2)`` over (MC, NC) blocks. Per-thread libbarrage arena
    (32 MiB) for the packed-panel scratch.

    Args:
        A: FP32 contiguous 2D tensor ``(M, K)``.
        B: FP32 contiguous 2D tensor ``(K, N)``.

    Returns:
        FP32 contiguous 2D tensor ``(M, N)``.

    Raises:
        TypeError:  if A or B is not a torch.Tensor.
        ValueError: if dtype != float32, not contiguous, not 2D,
                    or A.shape[1] != B.shape[0] (incompatible matmul dims).
    """
    # --- type checks ---
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"A must be torch.Tensor, got {type(A).__name__}")
    if not isinstance(B, torch.Tensor):
        raise TypeError(f"B must be torch.Tensor, got {type(B).__name__}")

    # --- dtype checks ---
    if A.dtype != torch.float32:
        raise ValueError(f"A.dtype must be torch.float32, got {A.dtype}")
    if B.dtype != torch.float32:
        raise ValueError(f"B.dtype must be torch.float32, got {B.dtype}")

    # --- contiguity checks ---
    if not A.is_contiguous():
        raise ValueError(
            "A must be contiguous. Call A = A.contiguous() before mm."
        )
    if not B.is_contiguous():
        raise ValueError(
            "B must be contiguous. Call B = B.contiguous() before mm."
        )

    # --- rank checks ---
    if A.dim() != 2:
        raise ValueError(
            f"A must be 2D (M, K); got {A.dim()}D with shape {tuple(A.shape)}"
        )
    if B.dim() != 2:
        raise ValueError(
            f"B must be 2D (K, N); got {B.dim()}D with shape {tuple(B.shape)}"
        )

    # --- shape match ---
    M, K_A = A.shape[0], A.shape[1]
    K_B, N = B.shape[0], B.shape[1]
    if K_A != K_B:
        raise ValueError(
            f"A.shape[1] ({K_A}) must match B.shape[0] ({K_B}) for matmul; "
            f"got A={tuple(A.shape)} B={tuple(B.shape)}"
        )
    K = K_A

    # --- allocate output (uninitialised; the C side zeroes it) ---
    C = torch.empty((M, N), dtype=torch.float32)

    # --- degenerate shapes: M=0 or N=0 means empty output, no work to do.
    # PyTorch's torch.empty may return a tensor with a NULL data_ptr for
    # 0-element tensors, which the C side's NULL check would reject. Handle
    # the empty case here so the C kernel only sees non-empty inputs. ---
    if M == 0 or N == 0:
        return C

    # --- call C ---
    lib = load_library()
    rc = lib.rk_mm(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(K),
        ctypes.c_int(N),
    )
    if rc != 0:
        raise RuntimeError(
            f"rk_mm returned error code {rc} "
            "(NULL pointer, bad shape, or integer overflow)."
        )
    return C


# ----------------------------------------------------------------------------
# softmax
# ----------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in replacement for ``torch.nn.functional.softmax`` along the
    last dim.

    Computes numerically-stable softmax along the last dim of a 3D FP32
    contiguous tensor of shape ``(B, T, V)``:

        m = max(x[b, t, :])
        s = sum(exp(x[b, t, :] - m))
        y[b, t, :] = exp(x[b, t, :] - m) / s

    Backend (v0.3): three-pass per-row kernel with OpenMP ``parallel for
    collapse(2)`` over ``(B, T)``, per-thread libbarrage arena (4 MiB) for
    the per-row max+sum scratch (two floats, bump-allocated). Max and sum
    reductions use ``#pragma omp simd reduction(...)`` so ``-march=native``
    selects AVX-512 zmm. The exp pass uses libm ``expf`` (scalar, cannot
    auto-vectorise without ``-ffast-math``; acceptable for v0.3 since the
    bottleneck cut is ATen dispatch removal, not exp throughput).

    Args:
        x:   FP32 contiguous 3D tensor ``(B, T, V)``.
        dim: Must be ``-1`` (the last dim) or ``x.dim() - 1`` (equivalent
             explicit form). Other dims are not supported in v0.3.

    Returns:
        FP32 contiguous 3D tensor ``(B, T, V)`` of softmax probabilities.

    Raises:
        TypeError:  if x is not a torch.Tensor.
        ValueError: if dtype != float32, not contiguous, not 3D,
                    or dim is not -1 / (ndim - 1).
    """
    # --- type check ---
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x).__name__}")

    # --- dtype check ---
    if x.dtype != torch.float32:
        raise ValueError(
            f"x.dtype must be torch.float32, got {x.dtype}"
        )

    # --- contiguity check ---
    if not x.is_contiguous():
        raise ValueError(
            "x must be contiguous. Call x = x.contiguous() before softmax."
        )

    # --- rank check ---
    if x.dim() != 3:
        raise ValueError(
            f"x must be 3D (B, T, V); got {x.dim()}D with shape "
            f"{tuple(x.shape)}"
        )

    # --- dim check: only -1 (or ndim-1) is supported in v0.3 ---
    last_dim = x.dim() - 1
    if dim != -1 and dim != last_dim:
        raise ValueError(
            f"only dim=-1 (last dim) is supported in v0.3; got dim={dim} "
            f"(x has {x.dim()} dims, so only -1 or {last_dim} are accepted)."
        )

    B, T, V = x.shape[0], x.shape[1], x.shape[2]

    # --- allocate output ---
    y = torch.empty_like(x)

    # --- call C ---
    lib = load_library()
    rc = lib.rk_softmax(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(T),
        ctypes.c_int(V),
    )
    if rc != 0:
        raise RuntimeError(
            f"rk_softmax returned error code {rc} "
            "(NULL pointer, bad shape, or integer overflow)."
        )
    return y


# ----------------------------------------------------------------------------
# layer_norm
# ----------------------------------------------------------------------------

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Drop-in replacement for ``torch.nn.functional.layer_norm``.

    Computes ``y = (x - mean) / sqrt(var + eps) * weight + bias`` on FP32
    contiguous tensors of shape ``(B, T, C)`` with weight/bias shape ``(C,)``.

    The variance is biased (divide by N, matching PyTorch's
    ``F.layer_norm``). Math is computed in FP32 op ordering identical to
    PyTorch's CPU LayerNorm kernel: mean via per-row sum, var via per-row
    sum of ``(x - mean)^2``, ``rstd = 1.0 / sqrt(var + eps)`` (computed once
    per row and multiplied into each element, NOT per-element divided),
    then ``y[i] = (x[i] - mean) * rstd * weight[i] + bias[i]``.

    Args:
        x:      FP32 contiguous tensor, 3D ``(B, T, C)``.
        weight: FP32 contiguous tensor, 1D ``(C,)``.
        bias:   FP32 contiguous tensor, 1D ``(C,)``, or ``None`` (matches
                ``F.layer_norm(..., bias=None)``; C side treats NULL bias
                as zero).
        eps:    Non-negative smoothing constant (default 1e-5, matches torch).

    Returns:
        FP32 contiguous tensor, same shape as ``x``.

    Raises:
        TypeError:  if x or weight is not a torch.Tensor.
        ValueError: if dtype != float32, not contiguous, wrong rank,
                    weight/bias shape mismatch, or eps < 0.
    """
    # --- type checks ---
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x).__name__}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"weight must be torch.Tensor, got {type(weight).__name__}"
        )
    # bias may be None (matches F.layer_norm with bias=None).
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"bias must be torch.Tensor or None, got {type(bias).__name__}"
        )

    # --- dtype checks ---
    if x.dtype != torch.float32:
        raise ValueError(f"x.dtype must be torch.float32, got {x.dtype}")
    if weight.dtype != torch.float32:
        raise ValueError(
            f"weight.dtype must be torch.float32, got {weight.dtype}"
        )
    if bias is not None and bias.dtype != torch.float32:
        raise ValueError(
            f"bias.dtype must be torch.float32, got {bias.dtype}"
        )

    # --- contiguity checks ---
    if not x.is_contiguous():
        raise ValueError(
            "x must be contiguous. Call x = x.contiguous() before layer_norm."
        )
    if not weight.is_contiguous():
        raise ValueError(
            "weight must be contiguous. Call weight = weight.contiguous() "
            "before layer_norm."
        )
    if bias is not None and not bias.is_contiguous():
        raise ValueError(
            "bias must be contiguous. Call bias = bias.contiguous() "
            "before layer_norm."
        )

    # --- rank checks ---
    if x.dim() != 3:
        raise ValueError(
            f"x must be 3D (B, T, C); got {x.dim()}D with shape {tuple(x.shape)}"
        )
    if weight.dim() != 1:
        raise ValueError(
            f"weight must be 1D (C,); got {weight.dim()}D with shape "
            f"{tuple(weight.shape)}"
        )
    if bias is not None and bias.dim() != 1:
        raise ValueError(
            f"bias must be 1D (C,) or None; got {bias.dim()}D with shape "
            f"{tuple(bias.shape)}"
        )

    # --- shape match ---
    B, T, C = x.shape[0], x.shape[1], x.shape[2]
    if weight.shape[0] != C:
        raise ValueError(
            f"weight.shape[0] ({weight.shape[0]}) must match x.shape[-1] "
            f"({C})"
        )
    if bias is not None and bias.shape[0] != C:
        raise ValueError(
            f"bias.shape[0] ({bias.shape[0]}) must match x.shape[-1] ({C})"
        )

    # --- eps check ---
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    # --- allocate output ---
    y = torch.empty_like(x)

    # --- call C ---
    # bias may be None -> pass NULL through to the C side, which treats NULL
    # bias as zero (matches F.layer_norm(input, normalized_shape, weight,
    # bias=None, eps)).
    lib = load_library()
    rc = lib.rk_layer_norm(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(0 if bias is None else bias.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(T),
        ctypes.c_int(C),
        ctypes.c_float(eps),
    )
    if rc != 0:
        raise RuntimeError(
            f"rk_layer_norm returned error code {rc} "
            "(NULL pointer, bad shape, or integer overflow)."
        )
    return y


# ----------------------------------------------------------------------------
# turbo() context manager
# ----------------------------------------------------------------------------

# Module-level flag so nested `with rk.turbo(): with rk.turbo(): ...` fails
# fast with a clear RuntimeError on the Python side, before the C side
# even gets the second rk_turbo_enter call. (rk_turbo_enter also returns -1
# on nested call, but the Python-side check gives a clearer message.)
_turbo_active = False


@contextmanager
def turbo():
    """Pin threads + sync OMP + disable GC for the hot loop.

    Wraps rk_turbo_enter (libtopo pin + OMP env + omp_set_num_threads) and
    rk_turbo_exit (restore affinity). Yields the physical core count.

        with rk.turbo() as n:
            for x, y in loader:
                out = model(x); loss.backward(); optimizer.step()

    Raises RuntimeError on nested call or libtopo init failure.
    """
    global _turbo_active
    if _turbo_active:
        raise RuntimeError(
            "rk.turbo() cannot be nested; exit the outer context first. "
            "turbo() is process-global state (saved CPU affinity + OMP env "
            "vars); nesting would clobber the saved affinity mask."
        )

    lib = load_library()
    n_physical = lib.rk_turbo_enter()
    if n_physical < 0:
        raise RuntimeError(
            f"rk_turbo_enter failed (returned {n_physical}). "
            "Causes: libtopo init failure, topo_set_affinity failure "
            "(cgroup cpuset excludes the physical core?), or nested call."
        )
    _turbo_active = True

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(n_physical)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield n_physical
    finally:
        # Cleanup runs even if the body raises. The order is:
        #   1. Re-enable GC + collect (so any finalizer-triggered cleanup
        #      runs while we're still pinned, on the off chance it matters).
        #   2. Restore torch's thread count.
        #   3. Call rk_turbo_exit to restore the CPU affinity mask.
        #   4. Clear _turbo_active last, so a re-entry between steps 1-3
        #      still sees the active flag (defensive; the GIL makes this
        #      single-threaded anyway).
        try:
            if gc_was_enabled:
                gc.enable()
                gc.collect()
            torch.set_num_threads(prev_threads)
        finally:
            lib.rk_turbo_exit()
            _turbo_active = False


# ----------------------------------------------------------------------------
# topo_detect(): Python view of libtopo's topology probe
# ----------------------------------------------------------------------------

def topo_detect() -> dict:
    """Probe the CPU/NUMA layout via EoSD libtopo.

    Returns a dict with the fields of ``topo_info_t`` (see
    ``vendor/EoSD/libtopo/include/topo.h``):
      - ``threads_per_core``  : logical CPUs per physical core (1 = no SMT)
      - ``cores_per_package`` : physical cores per socket
      - ``num_packages``      : sockets
      - ``num_numa_nodes``   : NUMA nodes visible to the kernel (>=1;
        non-NUMA systems expose a synthetic node0)
      - ``total_cpus``        : online logical CPUs (== sysconf(_SC_NPROCESSORS_ONLN))
      - ``physical_cores``    : ``cores_per_package * num_packages``
        (the count ``turbo()`` would pin to)

    On the 2-vCPU sandbox this returns:
        {'threads_per_core': 1, 'cores_per_package': 2, 'num_packages': 1,
         'num_numa_nodes': 1, 'total_cpus': 2, 'physical_cores': 2}

    Raises:
      RuntimeError: if rk_topo_detect returns non-zero (only happens if
        libtopo's topo_probe fails, which only happens for a NULL pointer,
        which we don't pass).
    """
    lib = load_library()
    threads_per_core  = ctypes.c_uint(0)
    cores_per_package = ctypes.c_uint(0)
    num_packages      = ctypes.c_uint(0)
    num_numa_nodes    = ctypes.c_uint(0)
    total_cpus        = ctypes.c_uint(0)
    rc = lib.rk_topo_detect(
        ctypes.byref(threads_per_core),
        ctypes.byref(cores_per_package),
        ctypes.byref(num_packages),
        ctypes.byref(num_numa_nodes),
        ctypes.byref(total_cpus),
    )
    if rc != 0:
        raise RuntimeError(f"rk_topo_detect failed (returned {rc})")
    n_physical = cores_per_package.value * num_packages.value
    return {
        "threads_per_core":  threads_per_core.value,
        "cores_per_package": cores_per_package.value,
        "num_packages":      num_packages.value,
        "num_numa_nodes":    num_numa_nodes.value,
        "total_cpus":        total_cpus.value,
        "physical_cores":    n_physical,
    }


# ----------------------------------------------------------------------------
# Smoke-test entrypoint: `python3 -m reikernel` prints the lib path + version
# and a topo_detect() summary.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        lib = load_library()
        print(f"reikernel {__version__} loaded OK")
        print(f"  lib: {_candidate_lib_paths()[0]}")
        print(f"  torch: {torch.__version__}")
        try:
            topo = topo_detect()
            print(f"  topo: {topo}")
        except Exception as e:
            print(f"  topo_detect failed: {e}", file=sys.stderr)
    except ImportError as e:
        print(f"reikernel {__version__} NOT loaded: {e}", file=sys.stderr)
        sys.exit(1)
