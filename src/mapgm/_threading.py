"""Size the BLAS and numba thread pools, which want opposite settings.

MAPGM's linear algebra is a stack of ``(d+1) x (d+1)`` systems with ``d`` in
the single or low double digits.  Handing work that small to a threaded BLAS
costs more in dispatch and spin-wait than the arithmetic itself: the same fit
measured 3.45 s of wall clock with OpenBLAS at 64 threads and 2.23 s at one.
So the BLAS is pinned to a single thread.

Numba is the opposite case.  The natural parameter update is about three
quarters of a fit and is embarrassingly parallel over samples, so it wants
many threads -- but not all of them.  Measured on the dominant kernel:

    threads      1      2      4      8     16     32     64     96
    speedup   1.07x  2.03x  3.48x  5.62x 11.16x 17.52x 19.26x  1.97x

Using every core collapses, so the default is capped well below it.

``OMP_NUM_THREADS`` is deliberately **not** set: numba's threading layer is
OpenMP here, so pinning that variable would pin ``prange`` to one thread and
make the parallel kernel about half the speed of the serial one.  OpenBLAS
reads its own ``OPENBLAS_NUM_THREADS`` in preference, so the two pools stay
independent.

Import this module -- or call :func:`limit_numeric_threads` -- *before* numpy
and numba, because both size their pools at load time.  ``MAPGM_BLAS_THREADS``
and ``NUMBA_NUM_THREADS`` override the defaults.
"""

from __future__ import annotations

import os

#: BLAS pools only.  OMP_NUM_THREADS is excluded on purpose; see above.
_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Past this the per-call dispatch stops paying for itself on this model's
#: kernel sizes, and at the full core count it collapses outright.
DEFAULT_NUMBA_THREADS = 32


def parallel_enabled(family: str) -> bool:
    """Whether one family of parallel kernels is switched on.

    ``MAPGM_PARALLEL`` turns every family off at once;
    ``MAPGM_PARALLEL_<FAMILY>`` overrides it for one.  The two exist
    separately because a timing run needs them to differ: it has to measure
    a fit that runs on one thread, while the data that fit is measured on
    can be drawn as fast as the machine allows.

    Families: ``fit`` (the ADMM natural parameter update) and ``gibbs``
    (the sampler that draws the count tensors).
    """
    specific = os.environ.get(f"MAPGM_PARALLEL_{family.upper()}")
    value = os.environ.get("MAPGM_PARALLEL", "1") if specific is None else specific
    return value not in ("0", "false", "False")


def final_fit_parallel_enabled() -> bool:
    """Whether the fit whose result and runtime are reported runs parallel.

    ``MAPGM_PARALLEL_FINAL_FIT`` decides it when set; otherwise it follows
    the ``fit`` family.  It exists so that one run can search
    hyperparameters as fast as the machine allows and still time the final
    fit on one thread: the search only has to rank candidates, while the
    final fit's runtime is what is reported.
    """
    value = os.environ.get("MAPGM_PARALLEL_FINAL_FIT")
    if value is None:
        return parallel_enabled("fit")
    return value not in ("0", "false", "False")


def limit_numeric_threads(threads: int | None = None) -> int:
    """Pin the BLAS thread pools, returning the value applied.

    A variable the caller set explicitly is left alone, so an environment
    that has already been tuned keeps its settings.
    """
    if threads is None:
        threads = int(os.environ.get("MAPGM_BLAS_THREADS", "1"))
    if threads < 1:
        raise ValueError("threads must be at least 1.")
    for name in _THREAD_VARIABLES:
        os.environ.setdefault(name, str(threads))
    return threads


def size_numba_threads(threads: int | None = None) -> int:
    """Cap numba's pool, returning the value applied.

    Leaving every core to ``prange`` is slower than using a third of them, so
    the count is capped rather than inherited from the machine.
    """
    if threads is None:
        threads = min(DEFAULT_NUMBA_THREADS, os.cpu_count() or 1)
    if threads < 1:
        raise ValueError("threads must be at least 1.")
    os.environ.setdefault("NUMBA_NUM_THREADS", str(threads))
    return int(os.environ["NUMBA_NUM_THREADS"])


limit_numeric_threads()
size_numba_threads()
