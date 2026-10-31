"""Numba kernels and helpers of the MAPGM ADMM solver.

Each mode's subproblem is solved by a two-block ADMM: a parameter-block
update of ``(A_t, b_t)`` from grouped normal equations, a consensus-block
update (the Poisson natural-parameter copies, the sparse copy of ``A_t`` and
the temporal-difference copies), a scaled-dual update, and the primal/dual
residuals that drive the stopping test.  The sample and matrix-index loops of
those steps live here; :class:`mapgm.methods.mapgm.model.MAPGM` keeps the
steps themselves.
"""

from __future__ import annotations

import contextlib
from typing import List, Union

# Sizes the BLAS and numba thread pools, which want opposite settings and are
# both fixed at load time; see mapgm._threading.
from mapgm import _threading

import numpy as np
from numba import njit, prange

PerModeFloat = Union[float, List[float]]
PerModePenalty = Union[str, List[str]]

PENALTY_L1 = 0
PENALTY_GROUP_L2 = 1
_PENALTY_CODES = {"l1": PENALTY_L1, "group_l2": PENALTY_GROUP_L2}
#: The model uses one penalty for every constraint block.  Splitting it
#: into eta/A/b weights was measured to never beat a single value, while the
#: value itself matters a great deal, so only the common one is adapted.
#: Changing it rescales every scaled dual so the unscaled duals are kept.
_RHO_BLOCKS = ("rho",)
_RHO_BLOCK_DUAL_NAMES = {
    "rho": ("u0", "U1", "U2", "U3", "u4", "u5"),
}


def _as_per_mode_floats(value: PerModeFloat, num_modes: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        values = [float(value)] * num_modes
    else:
        values = [float(item) for item in value]
    if len(values) != num_modes:
        raise ValueError(f"{name} must have one value per mode (M={num_modes}).")
    if any(item < 0.0 or not np.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite, non-negative values.")
    return values


def _as_per_mode_penalties(
    value: PerModePenalty,
    num_modes: int,
    name: str,
) -> list[str]:
    values = [value] * num_modes if isinstance(value, str) else list(value)
    if len(values) != num_modes:
        raise ValueError(f"{name} must have one value per mode (M={num_modes}).")
    allowed = {"l1", "group_l2"}
    if any(item not in allowed for item in values):
        raise ValueError(f"{name} must contain only {sorted(allowed)}.")
    return values


def _penalty_value(value: np.ndarray, kind: str) -> float:
    if kind == "l1":
        return float(np.sum(np.abs(value)))
    return float(np.linalg.norm(value))


@njit(cache=True)
def _poisson_prox_root_numba(
    count: float,
    center: float,
    log_C: float,
    rho: float,
    guess: float = np.nan,
    max_iter: int = 50,
) -> float:
    """Solve exp(log_C + z) - count + rho * (z - center) = 0.

    Written as ``z = apex - s`` with ``apex = center + count / rho``, the
    equation becomes ``exp(log_C + apex - s) = rho s``, that is

        phi(s) = s + log(s) - L = 0,    L = log_C + apex - log(rho),

    so ``s = W(exp(L))``.  ``phi`` is concave and increasing, so Newton
    started left of the root climbs to it without ever overshooting, and
    needs no exponential, so it cannot overflow.  The start
    ``L - log(L)`` (for ``L > 1``) or ``e^L / (1 + e^L)`` (otherwise) is
    always left of the root and close to it, which leaves two or three
    steps to go however extreme the input.

    Solving in ``z`` directly instead -- with Newton on the equation, or on
    its logarithm -- needs a dozen or more steps wherever the exponential
    dominates, as it does over most of the range a five-mode fit reaches,
    and that made those fits three times slower per iteration.

    ``apex - s`` loses the bits ``apex`` has above ``z`` when ``count /
    rho`` is large; a Newton step on the original equation afterwards
    restores them, since the answer is then close enough for its
    quadratic convergence.

    ``guess`` -- the previous ADMM iteration's root -- replaces the start
    when it is left of the root and nearer, or, after one Newton step that
    lands left of the root by concavity, when that is.
    """
    apex = center + count / rho
    level = log_C + apex - np.log(rho)
    if level < -700.0:
        # s = e^L (1 - e^L + ...), and e^L is below 1e-304.
        return apex - np.exp(level)

    if level > 1.0:
        s = level - np.log(level)
    else:
        e = np.exp(level)
        s = e / (1.0 + e)

    if np.isfinite(guess):
        s_guess = apex - guess
        if s_guess > s:
            phi = s_guess + np.log(s_guess) - level
            if phi <= 0.0:
                s = s_guess
            else:
                stepped = s_guess - phi * s_guess / (1.0 + s_guess)
                if stepped > s:
                    s = stepped

    for _ in range(max_iter):
        phi = s + np.log(s) - level
        candidate = s - phi * s / (1.0 + s)
        # Left of the root every step moves right; one that does not is
        # rounding at the root.
        if not candidate > s:
            break
        done = candidate - s <= 4e-16 * candidate
        s = candidate
        if done:
            break

    value = apex - s
    # One step squares the error apex - s leaves.  When apex dwarfs the
    # root -- tiny rho -- that error is large enough to need squaring again.
    steps = 1 if abs(apex) <= 1e6 * max(1.0, abs(value)) else 3
    for _ in range(steps):
        exponent = log_C + value
        if exponent > 700.0:
            break
        exponential = np.exp(exponent) if exponent >= -745.0 else 0.0
        residual = exponential - count + rho * (value - center)
        polished = value - residual / (exponential + rho)
        if not np.isfinite(polished):
            break
        done = abs(polished - value) <= 4e-16 * max(1.0, abs(value))
        value = polished
        if done:
            break
    return value


@njit(cache=True)
def _static_base_parameter_block_update_numba(
    x_m: np.ndarray,
    group_offsets: np.ndarray,
    z0: np.ndarray,
    u0: np.ndarray,
    Z1: np.ndarray,
    U1: np.ndarray,
    Z2: np.ndarray,
    U2: np.ndarray,
    Z3: np.ndarray,
    U3: np.ndarray,
    z4: np.ndarray,
    u4: np.ndarray,
    z5: np.ndarray,
    u5: np.ndarray,
    rho: float,
    scale_A: float,
    scale_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter update with one base vector shared by all times.

    The time-specific network rows are eliminated from the joint normal
    equations with a Schur complement.  This enforces a genuinely static base
    instead of approximating it with a very large temporal penalty.
    """
    T = Z1.shape[0]
    dimension = x_m.shape[1]
    gram_A = np.zeros((T, dimension, dimension), dtype=np.float64)
    cross = np.zeros((T, dimension), dtype=np.float64)
    rhs_A = np.zeros((T, dimension, dimension), dtype=np.float64)
    rhs_b = np.zeros(dimension, dtype=np.float64)
    gram_b = float(
        rho * scale_b * scale_b * x_m.shape[0]
        + 2.0 * rho * max(T - 1, 0)
    )

    for t in range(T):
        start = group_offsets[t]
        stop = group_offsets[t + 1]
        nu_t = int(t > 0) + int(t < T - 1)
        gamma_t = 1.0 + nu_t

        for j in range(dimension):
            gram_A[t, j, j] = rho * gamma_t

        for sample in range(start, stop):
            for j in range(dimension):
                x_j = x_m[sample, j]
                cross[t, j] += rho * scale_A * scale_b * x_j
                for k in range(dimension):
                    gram_A[t, j, k] += (
                        rho
                        * scale_A
                        * scale_A
                        * x_j
                        * x_m[sample, k]
                    )

            for i in range(dimension):
                c_i = z0[sample, i] + u0[sample, i]
                rhs_b[i] += rho * scale_b * c_i
                for j in range(dimension):
                    rhs_A[t, j, i] += (
                        rho * scale_A * c_i * x_m[sample, j]
                    )

        for i in range(dimension):
            for j in range(dimension):
                consensus = Z1[t, i, j] + U1[t, i, j]
                if t < T - 1:
                    consensus += Z2[t, i, j] + U2[t, i, j]
                if t > 0:
                    consensus += Z3[t - 1, i, j] + U3[t - 1, i, j]
                rhs_A[t, j, i] += rho * consensus

    for edge in range(T - 1):
        for i in range(dimension):
            rhs_b[i] += rho * (
                z4[edge, i]
                + u4[edge, i]
                + z5[edge, i]
                + u5[edge, i]
            )

    solved_cross = np.empty((T, dimension), dtype=np.float64)
    solved_rhs = np.empty((T, dimension, dimension), dtype=np.float64)
    schur = gram_b
    for t in range(T):
        solved_cross[t] = np.linalg.solve(gram_A[t], cross[t])
        solved_rhs[t] = np.linalg.solve(gram_A[t], rhs_A[t])
        schur -= np.dot(cross[t], solved_cross[t])
        for i in range(dimension):
            for j in range(dimension):
                rhs_b[i] -= cross[t, j] * solved_rhs[t, j, i]

    schur = max(schur, np.finfo(np.float64).eps)
    common_b_scaled = rhs_b / schur
    new_A = np.empty((T, dimension, dimension), dtype=np.float64)
    new_b = np.empty((T, dimension), dtype=np.float64)
    for t in range(T):
        for i in range(dimension):
            new_b[t, i] = scale_b * common_b_scaled[i]
            for j in range(dimension):
                new_A[t, i, j] = scale_A * (
                    solved_rhs[t, j, i]
                    - solved_cross[t, j] * common_b_scaled[i]
                )
    return new_A, new_b


@njit(cache=True)
def _parameter_gram_cholesky_numba(
    x_m: np.ndarray,
    group_offsets: np.ndarray,
    T: int,
    scale_A: float,
    scale_b: float,
) -> np.ndarray:
    """Cholesky factors of the normal equations in :func:`_parameter_block_update_numba`.

    The Gram matrix depends only on the data and the variable scaling, both
    fixed for the life of a solver, so it is built and factored once instead
    of being reaccumulated -- an ``n_t * d^2`` cost -- on every ADMM
    iteration.  ``rho`` is absent because it cancels; see the update itself.
    """
    dimension = x_m.shape[1]
    factors = np.empty((T, dimension + 1, dimension + 1), dtype=np.float64)
    for t in range(T):
        nu_t = int(t > 0) + int(t < T - 1)
        gram = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
        for j in range(dimension):
            gram[j, j] = 1.0 + nu_t
        gram[dimension, dimension] = float(nu_t)
        for sample in range(group_offsets[t], group_offsets[t + 1]):
            for j in range(dimension):
                x_j = x_m[sample, j]
                gram[j, dimension] += scale_A * scale_b * x_j
                gram[dimension, j] += scale_A * scale_b * x_j
                for k in range(dimension):
                    gram[j, k] += scale_A * scale_A * x_j * x_m[sample, k]
            gram[dimension, dimension] += scale_b * scale_b
        factors[t] = np.linalg.cholesky(gram)
    return factors


@njit(cache=True)
def _parameter_block_update_numba(
    x_m: np.ndarray,
    group_offsets: np.ndarray,
    z0: np.ndarray,
    u0: np.ndarray,
    Z1: np.ndarray,
    U1: np.ndarray,
    Z2: np.ndarray,
    U2: np.ndarray,
    Z3: np.ndarray,
    U3: np.ndarray,
    z4: np.ndarray,
    u4: np.ndarray,
    z5: np.ndarray,
    u5: np.ndarray,
    scale_A: float,
    scale_b: float,
    gram_cholesky: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped parameter update for samples sharing ``A[t], b[t]``.

    ``rho`` multiplies every term of both the Gram matrix and the right-hand
    side, so it cancels out of the solution exactly and is not a parameter.
    What remains is a fixed system per parameter time, solved here against
    the factors :func:`_parameter_gram_cholesky_numba` produced once.
    """
    T = Z1.shape[0]
    dimension = x_m.shape[1]
    size = dimension + 1
    new_A = np.empty((T, dimension, dimension), dtype=np.float64)
    new_b = np.empty((T, dimension), dtype=np.float64)
    rhs = np.empty((size, dimension), dtype=np.float64)
    forward = np.empty((size, dimension), dtype=np.float64)

    for t in range(T):
        rhs[:, :] = 0.0
        for sample in range(group_offsets[t], group_offsets[t + 1]):
            for i in range(dimension):
                c_i = z0[sample, i] + u0[sample, i]
                rhs[dimension, i] += scale_b * c_i
                for j in range(dimension):
                    rhs[j, i] += scale_A * c_i * x_m[sample, j]

        for i in range(dimension):
            q_i = 0.0
            if t < T - 1:
                q_i += z4[t, i] + u4[t, i]
            if t > 0:
                q_i += z5[t - 1, i] + u5[t - 1, i]
            rhs[dimension, i] += q_i

            for j in range(dimension):
                consensus = Z1[t, i, j] + U1[t, i, j]
                if t < T - 1:
                    consensus += Z2[t, i, j] + U2[t, i, j]
                if t > 0:
                    consensus += Z3[t - 1, i, j] + U3[t - 1, i, j]
                rhs[j, i] += consensus

        factor = gram_cholesky[t]
        for row in range(size):
            for column in range(dimension):
                accumulated = rhs[row, column]
                for k in range(row):
                    accumulated -= factor[row, k] * forward[k, column]
                forward[row, column] = accumulated / factor[row, row]
        for row in range(size - 1, -1, -1):
            for column in range(dimension):
                accumulated = forward[row, column]
                for k in range(row + 1, size):
                    accumulated -= factor[k, row] * rhs[k, column]
                rhs[row, column] = accumulated / factor[row, row]

        for i in range(dimension):
            new_b[t, i] = scale_b * rhs[dimension, i]
            for j in range(dimension):
                new_A[t, i, j] = scale_A * rhs[j, i]

    return new_A, new_b


@njit(cache=True)
def _natural_parameter_update_numba(
    x_m: np.ndarray,
    sample_time_index: np.ndarray,
    A_m: np.ndarray,
    b_m: np.ndarray,
    u0: np.ndarray,
    log_C_not_m: np.ndarray,
    rho: float,
    rho_scale: np.ndarray,
    z0: np.ndarray,
    eta_buffer: np.ndarray,
) -> None:
    """Natural-parameter (Poisson prox) update, solved without cancellation.

    A sample of parameter time ``t`` is solved with the penalty
    ``rho * rho_scale[t]``: the loss of that time enters the objective with
    weight ``1 / rho_scale[t]``, and dividing the objective through by it
    puts the whole factor on the quadratic term.  All ones is the plain sum.

    ``z0`` arrives holding the previous iteration's solution and is reused as
    the starting point for each root; see :func:`_poisson_prox_root_numba`.

    ``eta_buffer`` receives ``b + A x``.  ``A`` and ``b`` do not change again
    until the next parameter block update, so the dual update and the
    residuals read this buffer instead of recomputing the same product.
    """
    num_samples, dimension = x_m.shape
    for sample in range(num_samples):
        t = sample_time_index[sample]
        sample_rho = rho * rho_scale[t]
        for i in range(dimension):
            eta = b_m[t, i]
            for j in range(dimension):
                eta += A_m[t, i, j] * x_m[sample, j]
            eta_buffer[sample, i] = eta
            z0[sample, i] = _poisson_prox_root_numba(
                x_m[sample, i],
                eta - u0[sample, i],
                log_C_not_m[sample, 0],
                sample_rho,
                z0[sample, i],
            )


@njit(cache=True, parallel=True)
def _natural_parameter_update_parallel_numba(
    x_m: np.ndarray,
    sample_time_index: np.ndarray,
    A_m: np.ndarray,
    b_m: np.ndarray,
    u0: np.ndarray,
    log_C_not_m: np.ndarray,
    rho: float,
    rho_scale: np.ndarray,
    z0: np.ndarray,
    eta_buffer: np.ndarray,
) -> None:
    """Thread-parallel twin of :func:`_natural_parameter_update_numba`.

    The sample loop is embarrassingly parallel: each sample writes only its
    own row of ``z0`` and ``eta_buffer``, and the results are bit-identical
    to the serial kernel.  This is about three quarters of a fit and scales
    nearly linearly, so it is on by default.
    """
    num_samples, dimension = x_m.shape
    for sample in prange(num_samples):
        t = sample_time_index[sample]
        sample_rho = rho * rho_scale[t]
        for i in range(dimension):
            eta = b_m[t, i]
            for j in range(dimension):
                eta += A_m[t, i, j] * x_m[sample, j]
            eta_buffer[sample, i] = eta
            z0[sample, i] = _poisson_prox_root_numba(
                x_m[sample, i],
                eta - u0[sample, i],
                log_C_not_m[sample, 0],
                sample_rho,
                z0[sample, i],
            )


#: Whether to use the parallel natural parameter update.  On by default;
#: set ``MAPGM_PARALLEL_FIT=0`` (or ``MAPGM_PARALLEL=0`` for every kernel)
#: to force the serial one, which is what a timing run wants and what to
#: use when several fits already run side by side.  ``NUMBA_NUM_THREADS``
#: sets the thread count.  This is the only parallel kernel in the fit, so
#: turning it off makes the fit single-threaded whatever that is set to.
PARALLEL_NATURAL_PARAMETER_UPDATE = _threading.parallel_enabled("fit")


@contextlib.contextmanager
def parallel_fit(enabled: bool):
    """Switch the parallel natural parameter update on or off for a block.

    Fits read the switch on every update, so this changes the fits started
    inside the block and none outside it, and restores the previous setting
    on the way out.
    """
    global PARALLEL_NATURAL_PARAMETER_UPDATE
    previous = PARALLEL_NATURAL_PARAMETER_UPDATE
    PARALLEL_NATURAL_PARAMETER_UPDATE = bool(enabled)
    try:
        yield
    finally:
        PARALLEL_NATURAL_PARAMETER_UPDATE = previous

#: Below this many ``num_samples * dimension`` roots, the per-call thread
#: dispatch costs more than the loop it splits up: measured crossover is
#: around 500 samples, where the parallel kernel first breaks even and below
#: which it runs several times slower.  Streaming windows and small fits stay
#: on the serial path.
PARALLEL_MINIMUM_WORK = 4096


@njit(cache=True)
def _consensus_block_update_numba(
    A_m: np.ndarray,
    b_m: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    U3: np.ndarray,
    u4: np.ndarray,
    u5: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    z4: np.ndarray,
    z5: np.ndarray,
    sparse_weight: float,
    temporal_A_weight: float,
    temporal_b_weight: float,
    penalty_A_kind: int,
    penalty_b_kind: int,
    scale_A: float,
    scale_b: float,
) -> None:
    """Sparse and temporal-difference consensus updates, in scaled variables.

    ``Z1`` is the soft-thresholded projection of ``A_t`` onto symmetric,
    zero-diagonal matrices; ``Z2, Z3`` and ``z4, z5`` are the pairs of
    copies whose difference carries the temporal penalty on ``A`` and ``b``.
    """
    T, dimension = b_m.shape

    for t in range(T):
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    projected = 0.0
                else:
                    projected = 0.5 * (
                        A_m[t, i, j] / scale_A
                        - U1[t, i, j]
                        + A_m[t, j, i] / scale_A
                        - U1[t, j, i]
                    )
                magnitude = abs(projected) - sparse_weight
                Z1[t, i, j] = (
                    np.sign(projected) * magnitude if magnitude > 0.0 else 0.0
                )

    difference_A = np.empty((dimension, dimension), dtype=np.float64)
    left_A = np.empty((dimension, dimension), dtype=np.float64)
    right_A = np.empty((dimension, dimension), dtype=np.float64)
    difference_b = np.empty(dimension, dtype=np.float64)

    for edge in range(T - 1):
        squared_norm_A = 0.0
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    left = 0.0
                    right = 0.0
                else:
                    left = 0.5 * (
                        A_m[edge, i, j] / scale_A
                        - U2[edge, i, j]
                        + A_m[edge, j, i] / scale_A
                        - U2[edge, j, i]
                    )
                    right = 0.5 * (
                        A_m[edge + 1, i, j] / scale_A
                        - U3[edge, i, j]
                        + A_m[edge + 1, j, i] / scale_A
                        - U3[edge, j, i]
                    )
                left_A[i, j] = left
                right_A[i, j] = right
                difference = right - left
                difference_A[i, j] = difference
                squared_norm_A += difference * difference

        if penalty_A_kind == PENALTY_GROUP_L2:
            norm_A = np.sqrt(squared_norm_A)
            shrink_A = (
                max(1.0 - temporal_A_weight / norm_A, 0.0)
                if norm_A > 0.0
                else 0.0
            )
        else:
            shrink_A = 0.0

        for i in range(dimension):
            for j in range(dimension):
                if penalty_A_kind == PENALTY_L1:
                    difference = difference_A[i, j]
                    magnitude = abs(difference) - temporal_A_weight
                    prox_difference = (
                        np.sign(difference) * magnitude if magnitude > 0.0 else 0.0
                    )
                else:
                    prox_difference = shrink_A * difference_A[i, j]
                Z2[edge, i, j] = 0.5 * (
                    left_A[i, j] + right_A[i, j] - prox_difference
                )
                Z3[edge, i, j] = 0.5 * (
                    left_A[i, j] + right_A[i, j] + prox_difference
                )

        squared_norm_b = 0.0
        for i in range(dimension):
            left = b_m[edge, i] / scale_b - u4[edge, i]
            right = b_m[edge + 1, i] / scale_b - u5[edge, i]
            difference_b[i] = right - left
            squared_norm_b += difference_b[i] * difference_b[i]

        if penalty_b_kind == PENALTY_GROUP_L2:
            norm_b = np.sqrt(squared_norm_b)
            shrink_b = (
                max(1.0 - temporal_b_weight / norm_b, 0.0)
                if norm_b > 0.0
                else 0.0
            )
        else:
            shrink_b = 0.0

        for i in range(dimension):
            left = b_m[edge, i] / scale_b - u4[edge, i]
            right = b_m[edge + 1, i] / scale_b - u5[edge, i]
            if penalty_b_kind == PENALTY_L1:
                difference = difference_b[i]
                magnitude = abs(difference) - temporal_b_weight
                prox_difference = (
                    np.sign(difference) * magnitude if magnitude > 0.0 else 0.0
                )
            else:
                prox_difference = shrink_b * difference_b[i]
            z4[edge, i] = 0.5 * (left + right - prox_difference)
            z5[edge, i] = 0.5 * (left + right + prox_difference)


@njit(cache=True)
def _dual_update_numba(
    x_m: np.ndarray,
    eta_buffer: np.ndarray,
    A_m: np.ndarray,
    b_m: np.ndarray,
    z0: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    z4: np.ndarray,
    z5: np.ndarray,
    u0: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    U3: np.ndarray,
    u4: np.ndarray,
    u5: np.ndarray,
    scale_A: float,
    scale_b: float,
) -> None:
    """Scaled-dual update, in place.

    ``eta_buffer`` holds ``b + A x`` as left by the natural parameter update;
    neither ``A`` nor ``b`` has changed since, so it is read rather than
    recomputed.
    """
    num_samples, dimension = x_m.shape
    T = b_m.shape[0]
    for sample in range(num_samples):
        for i in range(dimension):
            u0[sample, i] += z0[sample, i] - eta_buffer[sample, i]

    for t in range(T):
        for i in range(dimension):
            for j in range(dimension):
                U1[t, i, j] += Z1[t, i, j] - A_m[t, i, j] / scale_A

    for edge in range(T - 1):
        for i in range(dimension):
            u4[edge, i] += z4[edge, i] - b_m[edge, i] / scale_b
            u5[edge, i] += z5[edge, i] - b_m[edge + 1, i] / scale_b
            for j in range(dimension):
                U2[edge, i, j] += (
                    Z2[edge, i, j] - A_m[edge, i, j] / scale_A
                )
                U3[edge, i, j] += (
                    Z3[edge, i, j] - A_m[edge + 1, i, j] / scale_A
                )


@njit(cache=True)
def _residuals_and_tolerances_numba(
    x_m: np.ndarray,
    sample_time_index: np.ndarray,
    eta_buffer: np.ndarray,
    A_m: np.ndarray,
    b_m: np.ndarray,
    z0: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    Z3: np.ndarray,
    z4: np.ndarray,
    z5: np.ndarray,
    u0: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    U3: np.ndarray,
    u4: np.ndarray,
    u5: np.ndarray,
    previous_z0: np.ndarray,
    previous_Z1: np.ndarray,
    previous_Z2: np.ndarray,
    previous_Z3: np.ndarray,
    previous_z4: np.ndarray,
    previous_z5: np.ndarray,
    rho: float,
    scale_A: float,
    scale_b: float,
    abs_tol: float,
    rel_tol: float,
) -> tuple:
    """Primal and dual residuals and their stopping tolerances."""
    num_samples, dimension = x_m.shape
    T = b_m.shape[0]
    block_primal_sq = np.zeros(3, dtype=np.float64)
    block_consensus_sq = np.zeros(3, dtype=np.float64)
    block_parameter_copy_sq = np.zeros(3, dtype=np.float64)
    dual_eta_b = np.zeros((T, dimension), dtype=np.float64)
    dual_scaled_eta_b = np.zeros((T, dimension), dtype=np.float64)
    dual_eta_A = np.zeros((T, dimension, dimension), dtype=np.float64)
    dual_scaled_eta_A = np.zeros((T, dimension, dimension), dtype=np.float64)
    dual_A = np.zeros((T, dimension, dimension), dtype=np.float64)
    dual_scaled_A = np.zeros((T, dimension, dimension), dtype=np.float64)
    dual_b = np.zeros((T, dimension), dtype=np.float64)
    dual_scaled_b = np.zeros((T, dimension), dtype=np.float64)

    for sample in range(num_samples):
        t = sample_time_index[sample]
        for i in range(dimension):
            eta = eta_buffer[sample, i]

            residual = z0[sample, i] - eta
            block_primal_sq[0] += residual * residual
            block_consensus_sq[0] += z0[sample, i] * z0[sample, i]
            block_parameter_copy_sq[0] += eta * eta

            delta_z0 = z0[sample, i] - previous_z0[sample, i]
            dual_eta_b[t, i] += scale_b * delta_z0
            dual_scaled_eta_b[t, i] += scale_b * u0[sample, i]

            for j in range(dimension):
                dual_eta_A[t, i, j] += (
                    scale_A * delta_z0 * x_m[sample, j]
                )
                dual_scaled_eta_A[t, i, j] += (
                    scale_A * u0[sample, i] * x_m[sample, j]
                )

    for t in range(T):
        for i in range(dimension):
            if t < T - 1:
                dual_b[t, i] += z4[t, i] - previous_z4[t, i]
                dual_scaled_b[t, i] += u4[t, i]
            if t > 0:
                dual_b[t, i] += z5[t - 1, i] - previous_z5[t - 1, i]
                dual_scaled_b[t, i] += u5[t - 1, i]

            for j in range(dimension):
                scaled_A = A_m[t, i, j] / scale_A
                residual = Z1[t, i, j] - scaled_A
                block_primal_sq[1] += residual * residual
                block_consensus_sq[1] += Z1[t, i, j] * Z1[t, i, j]
                block_parameter_copy_sq[1] += scaled_A * scaled_A

                dual_A[t, i, j] += (
                    Z1[t, i, j] - previous_Z1[t, i, j]
                )
                dual_scaled_A[t, i, j] += U1[t, i, j]
                if t < T - 1:
                    dual_A[t, i, j] += Z2[t, i, j] - previous_Z2[t, i, j]
                    dual_scaled_A[t, i, j] += U2[t, i, j]
                if t > 0:
                    dual_A[t, i, j] += (
                        Z3[t - 1, i, j] - previous_Z3[t - 1, i, j]
                    )
                    dual_scaled_A[t, i, j] += U3[t - 1, i, j]

    for edge in range(T - 1):
        for i in range(dimension):
            scaled_left_b = b_m[edge, i] / scale_b
            scaled_right_b = b_m[edge + 1, i] / scale_b
            residual = z4[edge, i] - scaled_left_b
            block_primal_sq[2] += residual * residual
            residual = z5[edge, i] - scaled_right_b
            block_primal_sq[2] += residual * residual
            block_consensus_sq[2] += z4[edge, i] * z4[edge, i]
            block_consensus_sq[2] += z5[edge, i] * z5[edge, i]
            block_parameter_copy_sq[2] += scaled_left_b * scaled_left_b
            block_parameter_copy_sq[2] += scaled_right_b * scaled_right_b

            for j in range(dimension):
                scaled_left_A = A_m[edge, i, j] / scale_A
                scaled_right_A = A_m[edge + 1, i, j] / scale_A
                residual = Z2[edge, i, j] - scaled_left_A
                block_primal_sq[1] += residual * residual
                residual = Z3[edge, i, j] - scaled_right_A
                block_primal_sq[1] += residual * residual
                block_consensus_sq[1] += Z2[edge, i, j] * Z2[edge, i, j]
                block_consensus_sq[1] += Z3[edge, i, j] * Z3[edge, i, j]
                block_parameter_copy_sq[1] += scaled_left_A * scaled_left_A
                block_parameter_copy_sq[1] += scaled_right_A * scaled_right_A

    rho_values = np.full(3, rho, dtype=np.float64)
    block_dual_unscaled_sq = np.array(
        [
            np.sum(dual_eta_b * dual_eta_b)
            + np.sum(dual_eta_A * dual_eta_A),
            np.sum(dual_A * dual_A),
            np.sum(dual_b * dual_b),
        ],
        dtype=np.float64,
    )
    block_dual_scaled_sq = np.array(
        [
            np.sum(dual_scaled_eta_b * dual_scaled_eta_b)
            + np.sum(dual_scaled_eta_A * dual_scaled_eta_A),
            np.sum(dual_scaled_A * dual_scaled_A),
            np.sum(dual_scaled_b * dual_scaled_b),
        ],
        dtype=np.float64,
    )
    block_primal = np.sqrt(block_primal_sq)
    block_dual = rho_values * np.sqrt(block_dual_unscaled_sq)

    total_dual_A = rho * dual_eta_A + rho * dual_A
    total_dual_b = rho * dual_eta_b + rho * dual_b
    total_scaled_dual_A = (
        rho * dual_scaled_eta_A + rho * dual_scaled_A
    )
    total_scaled_dual_b = (
        rho * dual_scaled_eta_b + rho * dual_scaled_b
    )
    dual_sq = np.sum(total_dual_A * total_dual_A) + np.sum(
        total_dual_b * total_dual_b
    )
    dual_scaled_sq = np.sum(
        total_scaled_dual_A * total_scaled_dual_A
    ) + np.sum(total_scaled_dual_b * total_scaled_dual_b)
    number_constraints = z0.size + Z1.size + Z2.size + Z3.size + z4.size + z5.size
    number_parameters = A_m.size + b_m.size
    primal_norm = np.sqrt(np.sum(block_primal_sq))
    dual_norm = np.sqrt(dual_sq)
    eps_primal = np.sqrt(number_constraints) * abs_tol + rel_tol * max(
        np.sqrt(np.sum(block_consensus_sq)),
        np.sqrt(np.sum(block_parameter_copy_sq)),
    )
    eps_dual = (
        np.sqrt(number_parameters) * abs_tol
        + rel_tol * np.sqrt(dual_scaled_sq)
    )
    normalized_primal = primal_norm / max(eps_primal, np.finfo(np.float64).tiny)
    normalized_dual = dual_norm / max(eps_dual, np.finfo(np.float64).tiny)

    # A single penalty leaves a single block, so it carries the overall
    # residuals rather than an eta/A/b split.
    single_primal = np.full(1, primal_norm, dtype=np.float64)
    single_dual = np.full(1, dual_norm, dtype=np.float64)
    single_eps_primal = np.full(1, eps_primal, dtype=np.float64)
    single_eps_dual = np.full(1, eps_dual, dtype=np.float64)
    single_normalized_primal = np.full(1, normalized_primal, dtype=np.float64)
    single_normalized_dual = np.full(1, normalized_dual, dtype=np.float64)
    return (
        primal_norm,
        dual_norm,
        eps_primal,
        eps_dual,
        normalized_primal,
        normalized_dual,
        single_primal,
        single_dual,
        single_eps_primal,
        single_eps_dual,
        single_normalized_primal,
        single_normalized_dual,
    )
