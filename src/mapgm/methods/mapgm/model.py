"""Time-varying Multi-Aspect Poisson Graphical Model fitted by ADMM.

Each mode is optimized while the natural parameters of the other modes are
held fixed.  Consecutive samples can share one parameter time through
``tau``; ``tau=1`` is the single-sample case of the same grouped update.  For
ablation studies, ``time_varying_base=False`` fits one base vector shared by
every parameter time while leaving ``A[t]`` time-varying.

The loss of a parameter time is the *mean* over its ``N_t`` samples, so the
weights ``lambda`` keep one meaning whatever ``tau`` is::

    sum_t (1/N_t) sum_{n in t} {1' exp(eta_{t,n}) - x_{t,n}' eta_{t,n}}
        + lambda_1 sum_t |A_t|_1
        + lambda_2 sum_t phi(A_t - A_{t-1})
        + lambda_3 sum_t psi(b_t - b_{t-1})

``N_t`` is ``tau`` for every time but a final short group.  The solver
minimizes ``tau`` times this objective, which has the same minimizer: each
full group then carries unit weight, so ``rho`` and the stopping tolerances
mean the same whatever ``tau`` is, and only the final short group differs.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from mapgm.methods.mapgm import _kernels
from mapgm.methods.mapgm._kernels import (
    PARALLEL_MINIMUM_WORK,
    PerModeFloat,
    PerModePenalty,
    _PENALTY_CODES,
    _RHO_BLOCK_DUAL_NAMES,
    _RHO_BLOCKS,
    _as_per_mode_floats,
    _as_per_mode_penalties,
    _consensus_block_update_numba,
    _dual_update_numba,
    _natural_parameter_update_numba,
    _natural_parameter_update_parallel_numba,
    _parameter_block_update_numba,
    _parameter_gram_cholesky_numba,
    _penalty_value,
    _residuals_and_tolerances_numba,
    _static_base_parameter_block_update_numba,
)


class MAPGM:
    """Time-varying Multi-aspect Poisson Graphical Model fitted by ADMM.

    ``x_summaries[m]`` has one row per sample.  Every consecutive ``tau``
    samples share one parameter time, so ``b[m]`` has shape ``(T, d_m)`` and
    ``A[m]`` has shape ``(T, d_m, d_m)`` for
    ``T = ceil(num_samples / tau)``.  With ``time_varying_base=False``, one
    base vector is estimated jointly across all parameter times and repeated
    along the first axis of ``b[m]`` for API compatibility.  In that case the
    temporal base penalty is identically zero.  The Python methods retain the
    ADMM step structure while the sample and matrix-index loops run in Numba
    kernels.

    The loss of parameter time ``t`` is the mean over its ``N_t`` samples, so
    every ``lambda`` is a weight on a per-sample objective and means the same
    for any ``tau``.  See the module docstring.
    """

    def __init__(
        self,
        sum_log_factorials: float,
        x_summaries: list,
        dims: list,
        lambda_regs: PerModeFloat,
        lambda_temporal_A: PerModeFloat = 0.0,
        lambda_temporal_b: PerModeFloat = 0.0,
        temporal_penalty_A: PerModePenalty = "l1",
        temporal_penalty_b: PerModePenalty = "l1",
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
        rho: PerModeFloat = 1.0,
        residual_check_interval: int = 10,
        adaptive_rho: bool = False,
        rho_balance: float = 10.0,
        rho_scale: float = 10.0,
        rho_scale_decay: float = 0.5,
        rho_scale_min: float = 10.0,
        rho_stable_iterations: int | None = None,
        rho_min: PerModeFloat | None = None,
        rho_max: PerModeFloat | None = None,
        post_projection_tolerance_factor: float = 10.0,
        normalize_mode_offsets: bool = True,
        divergence_guard: bool = True,
        divergence_factor: float = 1e4,
        divergence_patience: int = 2,
        # Disabled by default: the guard restores the best objective seen,
        # and the best objective is often the pre-iteration state (A=0,
        # zero penalties).  Restoring it silently discards every iterate
        # and returns an empty network, which reads as a converged fit but
        # scores at chance.  Opt in when a run is known to diverge.
        outer_objective_guard: bool = False,
        outer_objective_rel_tol: float = 1e-6,
        outer_objective_patience: int = 3,
        variable_scaling: bool = True,
        scale_A: PerModeFloat | None = None,
        scale_b: PerModeFloat | None = None,
        time_varying_base: bool = True,
        tau: int = 1,
    ) -> None:
        self.dims = [int(dimension) for dimension in dims]
        if not self.dims or any(dimension <= 0 for dimension in self.dims):
            raise ValueError("dims must contain positive integers.")
        self.M = len(self.dims)
        if len(x_summaries) != self.M:
            raise ValueError("x_summaries and dims must contain the same number of modes.")

        self.x_summaries = [
            np.ascontiguousarray(value, dtype=np.float64) for value in x_summaries
        ]
        self.num_samples = int(self.x_summaries[0].shape[0])
        if self.num_samples <= 0:
            raise ValueError("At least one sample is required.")
        if (
            isinstance(tau, (bool, np.bool_))
            or not isinstance(tau, (int, np.integer))
            or int(tau) <= 0
        ):
            raise ValueError("tau must be a positive integer.")
        self.tau = int(tau)
        self.T = (self.num_samples + self.tau - 1) // self.tau
        self.num_parameter_times = self.T
        self.sample_time_index = np.ascontiguousarray(
            np.arange(self.num_samples, dtype=np.int64) // self.tau
        )
        self.group_offsets = np.ascontiguousarray(
            np.minimum(
                np.arange(self.T + 1, dtype=np.int64) * self.tau,
                self.num_samples,
            )
        )
        self.samples_per_time = np.diff(self.group_offsets)
        # The objective gives the loss of time t the weight 1/N_t.  The
        # solver minimizes ``tau`` times it, which has the same minimizer and
        # leaves a full group at weight one, so rho, the tolerances and the
        # residuals keep the meaning they have for one sample per time.
        #
        # In that scaled problem a sample of time t has loss weight tau/N_t.
        # Dividing it out of the quadratic term gives the natural-parameter
        # prox a penalty of rho * N_t/tau, and the penalty weights enter the
        # consensus prox as tau * lambda.
        self._admm_objective_scale = float(self.tau)
        self._prox_rho_scale = np.ascontiguousarray(
            self.samples_per_time / float(self.tau), dtype=np.float64
        )
        reported_weight = 1.0 / self.samples_per_time[self.sample_time_index]
        # Per-sample weight of the *reported* loss, as a column.
        self._loss_weight = np.ascontiguousarray(
            reported_weight.astype(np.float64).reshape(-1, 1)
        )
        if not isinstance(time_varying_base, (bool, np.bool_)):
            raise ValueError("time_varying_base must be a boolean.")
        self.time_varying_base = bool(time_varying_base)
        for m, (x_m, d_m) in enumerate(zip(self.x_summaries, self.dims)):
            if x_m.shape != (self.num_samples, d_m):
                raise ValueError(
                    f"x_summaries[{m}] has shape {x_m.shape}; "
                    f"expected {(self.num_samples, d_m)}."
                )
            if np.any(x_m < 0.0) or not np.all(np.isfinite(x_m)):
                raise ValueError("x_summaries must contain finite, non-negative values.")

        if not isinstance(variable_scaling, (bool, np.bool_)):
            raise ValueError("variable_scaling must be a boolean.")
        self.variable_scaling = bool(variable_scaling)
        if not self.variable_scaling and (
            scale_A is not None or scale_b is not None
        ):
            raise ValueError(
                "scale_A and scale_b require variable_scaling=True."
            )
        if self.variable_scaling:
            self.scale_b_by_mode = (
                [1.0] * self.M
                if scale_b is None
                else _as_per_mode_floats(scale_b, self.M, "scale_b")
            )
            if scale_A is None:
                self.scale_A_by_mode = []
                for x_m, b_scale in zip(
                    self.x_summaries,
                    self.scale_b_by_mode,
                ):
                    rms = float(np.sqrt(np.mean(x_m * x_m)))
                    self.scale_A_by_mode.append(
                        b_scale / max(rms, 1.0)
                    )
            else:
                self.scale_A_by_mode = _as_per_mode_floats(
                    scale_A, self.M, "scale_A"
                )
        else:
            self.scale_A_by_mode = [1.0] * self.M
            self.scale_b_by_mode = [1.0] * self.M
        if any(
            value <= 0.0
            for values in (self.scale_A_by_mode, self.scale_b_by_mode)
            for value in values
        ):
            raise ValueError("scale_A and scale_b must be finite and positive.")

        self.sum_log_factorials = float(sum_log_factorials)
        self.lambda_regs = _as_per_mode_floats(lambda_regs, self.M, "lambda_regs")
        self.lambda_temporal_A = _as_per_mode_floats(
            lambda_temporal_A, self.M, "lambda_temporal_A"
        )
        self.lambda_temporal_b = _as_per_mode_floats(
            lambda_temporal_b, self.M, "lambda_temporal_b"
        )
        self.temporal_penalty_A = _as_per_mode_penalties(
            temporal_penalty_A, self.M, "temporal_penalty_A"
        )
        self.temporal_penalty_b = _as_per_mode_penalties(
            temporal_penalty_b, self.M, "temporal_penalty_b"
        )

        self._rho_by_mode = _as_per_mode_floats(rho, self.M, "rho")
        self.rho_by_block = {"rho": self._rho_by_mode}
        if any(
            value <= 0.0
            for values in self.rho_by_block.values()
            for value in values
        ):
            raise ValueError("rho values must be finite and positive.")
        common_rho_min = (
            None
            if rho_min is None
            else _as_per_mode_floats(rho_min, self.M, "rho_min")
        )
        common_rho_max = (
            None
            if rho_max is None
            else _as_per_mode_floats(rho_max, self.M, "rho_max")
        )
        self.rho_min_by_block = {
            block: (
                [value / 16.0 for value in values]
                if common_rho_min is None
                else common_rho_min.copy()
            )
            for block, values in self.rho_by_block.items()
        }
        self.rho_max_by_block = {
            block: (
                [value * 16.0 for value in values]
                if common_rho_max is None
                else common_rho_max.copy()
            )
            for block, values in self.rho_by_block.items()
        }
        self.rho_min_by_mode = self.rho_min_by_block["rho"]
        self.rho_max_by_mode = self.rho_max_by_block["rho"]
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        if (
            isinstance(residual_check_interval, (bool, np.bool_))
            or int(residual_check_interval) != residual_check_interval
            or int(residual_check_interval) < 1
        ):
            raise ValueError(
                "residual_check_interval must be a positive integer."
            )
        self.residual_check_interval = int(residual_check_interval)
        if not isinstance(adaptive_rho, (bool, np.bool_)):
            raise ValueError("adaptive_rho must be a boolean.")
        self.adaptive_rho = bool(adaptive_rho)
        self.rho_balance = float(rho_balance)
        self.rho_scale = float(rho_scale)
        self.rho_scale_decay = float(rho_scale_decay)
        self.rho_scale_min = float(rho_scale_min)
        if (
            rho_stable_iterations is not None
            and (
                isinstance(rho_stable_iterations, (bool, np.bool_))
                or int(rho_stable_iterations) != rho_stable_iterations
                or int(rho_stable_iterations) < 0
            )
        ):
            raise ValueError("rho_stable_iterations must be a non-negative integer.")
        self.rho_stable_iterations = (
            self.residual_check_interval
            if rho_stable_iterations is None
            else int(rho_stable_iterations)
        )
        self.post_projection_tolerance_factor = float(
            post_projection_tolerance_factor
        )
        if not isinstance(normalize_mode_offsets, (bool, np.bool_)):
            raise ValueError("normalize_mode_offsets must be a boolean.")
        self.normalize_mode_offsets = bool(normalize_mode_offsets)
        if not isinstance(divergence_guard, (bool, np.bool_)):
            raise ValueError("divergence_guard must be a boolean.")
        self.divergence_guard = bool(divergence_guard)
        self.divergence_factor = float(divergence_factor)
        if (
            isinstance(divergence_patience, (bool, np.bool_))
            or int(divergence_patience) != divergence_patience
            or int(divergence_patience) <= 0
        ):
            raise ValueError("divergence_patience must be a positive integer.")
        self.divergence_patience = int(divergence_patience)
        if not isinstance(outer_objective_guard, (bool, np.bool_)):
            raise ValueError("outer_objective_guard must be a boolean.")
        self.outer_objective_guard = bool(outer_objective_guard)
        self.outer_objective_rel_tol = float(outer_objective_rel_tol)
        if (
            isinstance(outer_objective_patience, (bool, np.bool_))
            or int(outer_objective_patience) != outer_objective_patience
            or int(outer_objective_patience) <= 0
        ):
            raise ValueError("outer_objective_patience must be a positive integer.")
        self.outer_objective_patience = int(outer_objective_patience)
        if self.abs_tol <= 0.0 or self.rel_tol <= 0.0:
            raise ValueError("abs_tol and rel_tol must be positive.")
        if self.rho_balance <= 1.0 or not np.isfinite(self.rho_balance):
            raise ValueError("rho_balance must be finite and greater than 1.")
        if self.rho_scale <= 1.0 or not np.isfinite(self.rho_scale):
            raise ValueError("rho_scale must be finite and greater than 1.")
        if self.rho_scale_decay < 0.0 or not np.isfinite(self.rho_scale_decay):
            raise ValueError("rho_scale_decay must be finite and non-negative.")
        if (
            self.rho_scale_min <= 1.0
            or self.rho_scale_min > self.rho_scale
            or not np.isfinite(self.rho_scale_min)
        ):
            raise ValueError(
                "rho_scale_min must be finite and satisfy 1 < min <= rho_scale."
            )
        if (
            self.post_projection_tolerance_factor < 1.0
            or not np.isfinite(self.post_projection_tolerance_factor)
        ):
            raise ValueError(
                "post_projection_tolerance_factor must be finite and at least 1."
            )
        if self.divergence_factor <= 1.0 or not np.isfinite(self.divergence_factor):
            raise ValueError("divergence_factor must be finite and greater than 1.")
        if self.outer_objective_rel_tol < 0.0 or not np.isfinite(
            self.outer_objective_rel_tol
        ):
            raise ValueError("outer_objective_rel_tol must be finite and non-negative.")
        if any(
            value <= 0.0
            for values in self.rho_min_by_block.values()
            for value in values
        ):
            raise ValueError("rho bounds must be finite and satisfy 0 < min <= max.")
        for block in _RHO_BLOCKS:
            for initial, lower, upper in zip(
                self.rho_by_block[block],
                self.rho_min_by_block[block],
                self.rho_max_by_block[block],
            ):
                if upper < lower or not lower <= initial <= upper:
                    raise ValueError(
                        f"rho_{block} must lie between rho_min and rho_max."
                    )
        if self.normalize_mode_offsets and self.M > 1:
            reference_totals = np.sum(self.x_summaries[0], axis=1)
            if any(
                not np.allclose(
                    np.sum(x_m, axis=1),
                    reference_totals,
                    rtol=1e-10,
                    atol=1e-8,
                )
                for x_m in self.x_summaries[1:]
            ):
                raise ValueError(
                    "normalize_mode_offsets requires equal per-time totals "
                    "across x_summaries modes."
                )

        # Start from no edges.  A random start was measured to converge no
        # faster, and zeros satisfy the model's symmetric, hollow constraint
        # from the first iteration instead of leaving the consensus
        # projection to repair an asymmetric draw.  It also consumes no
        # random numbers, so a fit depends on the seed only through the
        # data it is given.
        self.A = [
            np.zeros((self.T, d_m, d_m), dtype=float) for d_m in self.dims
        ]

        # Initialize the shared intercept from all samples assigned to time t.
        self.b = [np.zeros((self.T, d_m), dtype=float) for d_m in self.dims]
        eta_all = [self._compute_eta_(m) for m in range(self.M)]
        for m in range(self.M):
            log_C_not_m = self._compute_log_C_not_m(eta_all, m)
            self.b[m] = self._initialize_intercept(
                self.x_summaries[m], log_C_not_m
            )
            eta_all[m] = self._compute_eta_(m)
 
        self._states = [self._new_state(m) for m in range(self.M)]
        # Filled on first use by _parameter_gram_cholesky.
        self._gram_cholesky: list[np.ndarray | None] = [None] * self.M
        self.initial_mode_offset_normalization = self._normalize_mode_offsets()
        self.history: list[dict] = []
        self.mode_diagnostics: list[dict] = [{} for _ in range(self.M)]
        self.rho_history_by_block: dict[str, list[list[float]]] = {
            block: [[value] for value in values]
            for block, values in self.rho_by_block.items()
        }
        self.rho_history = self.rho_history_by_block["rho"]
        self.admm_history: list[list[dict]] = [[] for _ in range(self.M)]
        self.fit_diagnostics: dict = {}
        self.rho_update_counts = [0] * self.M
        self.rho_update_counts_by_block = {
            block: [0] * self.M for block in _RHO_BLOCKS
        }
        self._last_rho_scale_by_mode: list[float | None] = [None] * self.M
        self._last_rho_scale_by_block: dict[str, list[float | None]] = {
            block: [None] * self.M for block in _RHO_BLOCKS
        }

    @property
    def rho(self) -> float:
        """The common penalty, as a scalar (mode 0)."""
        return self._rho_by_mode[0]

    @property
    def rho_by_mode(self) -> list[float]:
        """The common penalty, one entry per mode."""
        return self._rho_by_mode

    @staticmethod
    def _stable_row_logsumexp(value: np.ndarray) -> np.ndarray:
        row_max = np.max(value, axis=1, keepdims=True)
        return row_max + np.log(np.sum(np.exp(value - row_max), axis=1, keepdims=True))

    def _initialize_intercept(
        self,
        x_m: np.ndarray,
        log_C_not_m: np.ndarray,
    ) -> np.ndarray:
        """Initialize grouped or globally shared intercepts from the samples."""
        initialized = np.empty((self.T, x_m.shape[1]), dtype=float)
        tiny = np.finfo(float).tiny
        if not self.time_varying_base:
            log_C_values = log_C_not_m[:, 0]
            maximum = float(np.max(log_C_values))
            log_denominator = maximum + np.log(
                np.sum(np.exp(log_C_values - maximum))
            )
            common_base = (
                np.log(np.clip(np.sum(x_m, axis=0), tiny, None))
                - log_denominator
            )
            initialized[:] = common_base
            return initialized

        for t in range(self.T):
            start = self.group_offsets[t]
            stop = self.group_offsets[t + 1]
            log_C_group = log_C_not_m[start:stop, 0]
            maximum = float(np.max(log_C_group))
            log_denominator = maximum + np.log(
                np.sum(np.exp(log_C_group - maximum))
            )
            initialized[t] = (
                np.log(np.clip(np.sum(x_m[start:stop], axis=0), tiny, None))
                - log_denominator
            )
        return initialized

    def _compute_eta_(
        self,
        m: int,
        b: np.ndarray | None = None,
        A: np.ndarray | None = None,
    ) -> np.ndarray:
        b_m = self.b[m] if b is None else np.asarray(b, dtype=float)
        A_m = self.A[m] if A is None else np.asarray(A, dtype=float)
        return b_m[self.sample_time_index] + np.einsum(
            "sij,sj->si",
            A_m[self.sample_time_index],
            self.x_summaries[m],
        )

    def _compute_log_C_not_m(self, eta_all: list[np.ndarray], m: int) -> np.ndarray:
        value = np.zeros((self.num_samples, 1), dtype=float)
        for other_mode in range(self.M):
            if other_mode != m:
                value += self._stable_row_logsumexp(eta_all[other_mode])
        return value

    def _new_state(self, m: int) -> dict[str, np.ndarray]:
        d_m = self.dims[m]
        eta = self._compute_eta_(m)
        scaled_A = self.A[m] / self.scale_A_by_mode[m]
        scaled_b = self.b[m] / self.scale_b_by_mode[m]
        state = {
            "z0": eta.copy(),
            # Scratch for b + A x, filled by the natural parameter update and
            # read back by the dual update and the residuals.
            "eta": eta.copy(),
            "Z1": scaled_A.copy(),
            "u0": np.zeros((self.num_samples, d_m), dtype=float),
            "U1": np.zeros((self.T, d_m, d_m), dtype=float),
            "Z2": scaled_A[:-1].copy(),
            "Z3": scaled_A[1:].copy(),
            "U2": np.zeros((self.T - 1, d_m, d_m), dtype=float),
            "U3": np.zeros((self.T - 1, d_m, d_m), dtype=float),
            "z4": scaled_b[:-1].copy(),
            "z5": scaled_b[1:].copy(),
            "u4": np.zeros((self.T - 1, d_m), dtype=float),
            "u5": np.zeros((self.T - 1, d_m), dtype=float),
        }
        return state

    def _normalize_mode_offsets(
        self,
        eta_all: list[np.ndarray] | None = None,
    ) -> dict:
        """Fix the constant cross-mode offset without changing the objective.

        A scalar shift is applied to every time point and component of one
        mode.  The shifts sum to zero, so the likelihood is unchanged when
        ``x_summaries`` are marginals with equal per-time totals.  Constant
        shifts also leave every temporal difference unchanged.

        The matching natural-parameter and temporal consensus copies are
        translated with ``b``.  Constraint residuals therefore stay fixed,
        so dual variables and ``rho`` must not be reset or rescaled.
        """
        means_before = [float(np.mean(b_m)) for b_m in self.b]
        common_mean = float(np.mean(means_before))
        shifts = [
            common_mean - mode_mean if self.normalize_mode_offsets else 0.0
            for mode_mean in means_before
        ]

        for m, shift in enumerate(shifts):
            if shift == 0.0:
                continue
            self.b[m] += shift
            state = self._states[m]
            state["z0"] += shift
            scaled_shift = shift / self.scale_b_by_mode[m]
            state["z4"] += scaled_shift
            state["z5"] += scaled_shift
            if eta_all is not None:
                eta_all[m] += shift

        means_after = [float(np.mean(b_m)) for b_m in self.b]
        return {
            "enabled": self.normalize_mode_offsets,
            "common_mean": common_mean,
            "means_before": means_before,
            "means_after": means_after,
            "shifts": shifts,
            "max_mean_gap": float(np.ptp(means_after)),
        }

    def _parameter_block_update(self, m: int, state: dict[str, np.ndarray]) -> None:
        """Parameter-block update of ``(A_t, b_t)`` over each time's samples."""
        common = (
            self.x_summaries[m],
            self.group_offsets,
            state["z0"],
            state["u0"],
            state["Z1"],
            state["U1"],
            state["Z2"],
            state["U2"],
            state["Z3"],
            state["U3"],
            state["z4"],
            state["u4"],
            state["z5"],
            state["u5"],
        )
        if self.time_varying_base:
            self.A[m], self.b[m] = _parameter_block_update_numba(
                *common,
                self.scale_A_by_mode[m],
                self.scale_b_by_mode[m],
                self._parameter_gram_cholesky(m),
            )
        else:
            self.A[m], self.b[m] = _static_base_parameter_block_update_numba(
                *common,
                self._rho_by_mode[m],
                self.scale_A_by_mode[m],
                self.scale_b_by_mode[m],
            )

    def _parameter_gram_cholesky(self, m: int) -> np.ndarray:
        """Cholesky factors for mode ``m``, built once and kept.

        The data, the grouping and the variable scaling are all fixed at
        construction, so the factors never go stale within a solver.
        """
        factors = self._gram_cholesky[m]
        if factors is None:
            factors = _parameter_gram_cholesky_numba(
                self.x_summaries[m],
                self.group_offsets,
                self.T,
                self.scale_A_by_mode[m],
                self.scale_b_by_mode[m],
            )
            self._gram_cholesky[m] = factors
        return factors

    def _natural_parameter_update(
        self,
        m: int,
        state: dict[str, np.ndarray],
        log_C_not_m: np.ndarray,
    ) -> None:
        """Natural-parameter (Poisson prox) update, evaluated without overflow."""
        x_m = self.x_summaries[m]
        use_parallel = (
            _kernels.PARALLEL_NATURAL_PARAMETER_UPDATE
            and x_m.size >= PARALLEL_MINIMUM_WORK
        )
        update = (
            _natural_parameter_update_parallel_numba
            if use_parallel
            else _natural_parameter_update_numba
        )
        update(
            x_m,
            self.sample_time_index,
            self.A[m],
            self.b[m],
            state["u0"],
            log_C_not_m,
            self._rho_by_mode[m],
            self._prox_rho_scale,
            state["z0"],
            state["eta"],
        )

    def _consensus_block_update(
        self,
        m: int,
        state: dict[str, np.ndarray],
        log_C_not_m: np.ndarray,
    ) -> None:
        """Consensus-block update: natural-parameter, sparse and temporal copies."""
        self._natural_parameter_update(m, state, log_C_not_m)
        rho = self._rho_by_mode[m]
        # The solver's objective is ``_admm_objective_scale`` times the
        # reported one, so its penalty weights are scaled the same way.
        weight = self._admm_objective_scale
        _consensus_block_update_numba(
            self.A[m],
            self.b[m],
            state["U1"],
            state["U2"],
            state["U3"],
            state["u4"],
            state["u5"],
            state["Z1"],
            state["Z2"],
            state["Z3"],
            state["z4"],
            state["z5"],
            weight * self.lambda_regs[m] * self.scale_A_by_mode[m] / rho,
            2.0
            * weight
            * self.lambda_temporal_A[m]
            * self.scale_A_by_mode[m]
            / rho,
            (
                2.0
                * weight
                * self.lambda_temporal_b[m]
                * self.scale_b_by_mode[m]
                / rho
                if self.time_varying_base
                else 0.0
            ),
            _PENALTY_CODES[self.temporal_penalty_A[m]],
            _PENALTY_CODES[self.temporal_penalty_b[m]],
            self.scale_A_by_mode[m],
            self.scale_b_by_mode[m],
        )

    def _dual_update(self, m: int, state: dict[str, np.ndarray]) -> None:
        """Scaled-dual update."""
        _dual_update_numba(
            self.x_summaries[m],
            state["eta"],
            self.A[m],
            self.b[m],
            state["z0"],
            state["Z1"],
            state["Z2"],
            state["Z3"],
            state["z4"],
            state["z5"],
            state["u0"],
            state["U1"],
            state["U2"],
            state["U3"],
            state["u4"],
            state["u5"],
            self.scale_A_by_mode[m],
            self.scale_b_by_mode[m],
        )

    def _residuals_and_tolerances(
        self,
        m: int,
        state: dict[str, np.ndarray],
        previous: dict[str, np.ndarray],
        eta: np.ndarray | None = None,
    ) -> tuple:
        """Primal and dual residuals with their stopping tolerances.

        ``eta`` defaults to the buffer the natural parameter update filled,
        which is valid only while ``A`` and ``b`` are the ones it was built
        from.  A caller that has since replaced them -- the post-projection
        feasibility check does -- must pass a freshly computed ``eta``.
        """
        return _residuals_and_tolerances_numba(
            self.x_summaries[m],
            self.sample_time_index,
            state["eta"] if eta is None else eta,
            self.A[m],
            self.b[m],
            state["z0"],
            state["Z1"],
            state["Z2"],
            state["Z3"],
            state["z4"],
            state["z5"],
            state["u0"],
            state["U1"],
            state["U2"],
            state["U3"],
            state["u4"],
            state["u5"],
            previous["z0"],
            previous["Z1"],
            previous["Z2"],
            previous["Z3"],
            previous["z4"],
            previous["z5"],
            self._rho_by_mode[m],
            self.scale_A_by_mode[m],
            self.scale_b_by_mode[m],
            self.abs_tol,
            self.rel_tol,
        )

    def _mode_rhos(self, m: int) -> dict[str, float]:
        return {block: self.rho_by_block[block][m] for block in _RHO_BLOCKS}

    def _rho_snapshot(self) -> dict[str, list[float]]:
        return {
            block: self.rho_by_block[block].copy() for block in _RHO_BLOCKS
        }

    def _set_block_rho(self, m: int, block: str, new_rho: float) -> bool:
        """Set one constraint-block penalty and preserve its unscaled dual."""
        if block not in _RHO_BLOCKS:
            raise ValueError(f"Unknown rho block: {block}")
        old_rho = self.rho_by_block[block][m]
        if new_rho == old_rho:
            return False

        scaled_dual_factor = old_rho / new_rho
        state = self._states[m]
        for name in _RHO_BLOCK_DUAL_NAMES[block]:
            state[name] *= scaled_dual_factor
        self.rho_by_block[block][m] = new_rho
        self.rho_history_by_block[block][m].append(new_rho)
        return True

    def _effective_rho_scale(self, m: int, block: str) -> float:
        """Return a multiplicative rho step that approaches one over time."""
        update_count = self.rho_update_counts_by_block[block][m]
        exponent = 1.0 / (update_count + 1) ** self.rho_scale_decay
        return max(self.rho_scale_min, self.rho_scale**exponent)

    def _adapt_rho(
        self,
        m: int,
        block: str,
        normalized_primal: float,
        normalized_dual: float,
    ) -> bool:
        """Balance one block's normalized residuals with a finer rho step."""
        old_rho = self.rho_by_block[block][m]
        scale = self._effective_rho_scale(m, block)
        if normalized_primal > self.rho_balance * normalized_dual:
            new_rho = min(old_rho * scale, self.rho_max_by_block[block][m])
        elif normalized_dual > self.rho_balance * normalized_primal:
            new_rho = max(old_rho / scale, self.rho_min_by_block[block][m])
        else:
            return False
        updated = self._set_block_rho(m, block, new_rho)
        if updated:
            self.rho_update_counts[m] += 1
            self.rho_update_counts_by_block[block][m] += 1
            self._last_rho_scale_by_mode[m] = scale
            self._last_rho_scale_by_block[block][m] = scale
        return updated

    def _capture_mode_checkpoint(
        self,
        m: int,
        state: dict[str, np.ndarray],
    ) -> dict:
        """Copy one mode's complete ADMM state for guarded recovery."""
        return {
            "A": self.A[m].copy(),
            "b": self.b[m].copy(),
            "state": {name: value.copy() for name, value in state.items()},
            "rho": self._mode_rhos(m),
            "mode_diagnostics": self.mode_diagnostics[m].copy(),
        }

    def _restore_mode_checkpoint(
        self,
        m: int,
        state: dict[str, np.ndarray],
        checkpoint: dict,
    ) -> tuple[str, ...]:
        """Restore the best finite iterate and increase adaptive block rhos."""
        rho_before_recovery = self._mode_rhos(m)
        checkpoint_rho = {
            block: float(checkpoint["rho"][block]) for block in _RHO_BLOCKS
        }
        self.A[m] = checkpoint["A"].copy()
        self.b[m] = checkpoint["b"].copy()
        for name, value in checkpoint["state"].items():
            state[name][...] = value
        for block in _RHO_BLOCKS:
            self.rho_by_block[block][m] = checkpoint_rho[block]

        recovery_blocks = list(_RHO_BLOCKS)
        updated_blocks = []
        for block in recovery_blocks:
            scale = self._effective_rho_scale(m, block)
            recovery_rho = min(
                max(checkpoint_rho[block], rho_before_recovery[block]) * scale,
                self.rho_max_by_block[block][m],
            )
            if self._set_block_rho(m, block, recovery_rho):
                self.rho_update_counts[m] += 1
                self.rho_update_counts_by_block[block][m] += 1
                self._last_rho_scale_by_mode[m] = scale
                self._last_rho_scale_by_block[block][m] = scale
                updated_blocks.append(block)
        return tuple(updated_blocks)

    def _mode_parameter_magnitude(self, m: int) -> float:
        return max(
            float(np.max(np.abs(self.A[m] / self.scale_A_by_mode[m]))),
            float(np.max(np.abs(self.b[m] / self.scale_b_by_mode[m]))),
        )

    def _parameter_contribution_norms(self, m: int) -> dict[str, float]:
        """Compare the natural-parameter contributions of ``Ax`` and ``b``."""
        A_by_sample = self.A[m][self.sample_time_index]
        b_by_sample = self.b[m][self.sample_time_index]
        Ax = np.einsum("sij,sj->si", A_by_sample, self.x_summaries[m])
        norm_Ax = float(np.linalg.norm(Ax))
        norm_b = float(np.linalg.norm(b_by_sample))
        return {
            "norm_Ax": norm_Ax,
            "norm_b": norm_b,
            "norm_Ax_over_b": norm_Ax / max(norm_b, np.finfo(float).tiny),
            "max_abs_A": float(np.max(np.abs(self.A[m]))),
            "max_abs_b": float(np.max(np.abs(self.b[m]))),
            "max_abs_scaled_A": float(
                np.max(np.abs(self.A[m] / self.scale_A_by_mode[m]))
            ),
            "max_abs_scaled_b": float(
                np.max(np.abs(self.b[m] / self.scale_b_by_mode[m]))
            ),
            "scale_A": self.scale_A_by_mode[m],
            "scale_b": self.scale_b_by_mode[m],
        }

    def _post_projection_feasibility(
        self,
        m: int,
        state: dict[str, np.ndarray],
    ) -> tuple[float, float, float]:
        """Measure primal feasibility after exporting the sparse copy ``Z1``."""
        consensus_names = ("z0", "Z1", "Z2", "Z3", "z4", "z5")
        unchanged = {name: state[name] for name in consensus_names}
        # A and b now hold the exported projection, so eta must be rebuilt
        # from them rather than read from the pre-projection buffer.
        metrics = self._residuals_and_tolerances(
            m, state, unchanged, eta=self._compute_eta_(m)
        )
        primal, _, eps_primal, _, normalized_primal, _ = metrics[:6]
        return primal, eps_primal, normalized_primal

    def _primal_residuals_by_block(
        self,
        m: int,
        state: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Expose consensus residual norms without changing the ADMM update."""
        eta = self._compute_eta_(m)
        scaled_A = self.A[m] / self.scale_A_by_mode[m]
        scaled_b = self.b[m] / self.scale_b_by_mode[m]
        temporal_A_sq = float(
            np.sum((state["Z2"] - scaled_A[:-1]) ** 2)
            + np.sum((state["Z3"] - scaled_A[1:]) ** 2)
        )
        temporal_b_sq = float(
            np.sum((state["z4"] - scaled_b[:-1]) ** 2)
            + np.sum((state["z5"] - scaled_b[1:]) ** 2)
        )
        return {
            "natural": float(np.linalg.norm(state["z0"] - eta)),
            "sparsity_A": float(np.linalg.norm(state["Z1"] - scaled_A)),
            "temporal_A": float(np.sqrt(temporal_A_sq)),
            "temporal_b": float(np.sqrt(temporal_b_sq)),
        }

    def _capture_model_checkpoint(self, eta_all: list[np.ndarray]) -> dict:
        """Copy the complete multi-mode iterate at a finite outer objective."""
        return {
            "A": [value.copy() for value in self.A],
            "b": [value.copy() for value in self.b],
            "states": [
                {name: value.copy() for name, value in state.items()}
                for state in self._states
            ],
            "rho": self._rho_snapshot(),
            "rho_update_counts": self.rho_update_counts.copy(),
            "rho_update_counts_by_block": {
                block: values.copy()
                for block, values in self.rho_update_counts_by_block.items()
            },
            "last_rho_scale_by_mode": self._last_rho_scale_by_mode.copy(),
            "last_rho_scale_by_block": {
                block: values.copy()
                for block, values in self._last_rho_scale_by_block.items()
            },
            "eta": [value.copy() for value in eta_all],
            "mode_diagnostics": [value.copy() for value in self.mode_diagnostics],
        }

    def _restore_model_checkpoint(
        self,
        eta_all: list[np.ndarray],
        checkpoint: dict,
    ) -> None:
        """Restore a complete outer iterate without changing its scaled duals."""
        for m in range(self.M):
            self.A[m] = checkpoint["A"][m].copy()
            self.b[m] = checkpoint["b"][m].copy()
            for name, value in checkpoint["states"][m].items():
                self._states[m][name][...] = value
            for block in _RHO_BLOCKS:
                restored_rho = float(checkpoint["rho"][block][m])
                if restored_rho != self.rho_by_block[block][m]:
                    self.rho_history_by_block[block][m].append(restored_rho)
                self.rho_by_block[block][m] = restored_rho
            eta_all[m] = checkpoint["eta"][m].copy()
        self.rho_update_counts = checkpoint["rho_update_counts"].copy()
        self.rho_update_counts_by_block = {
            block: values.copy()
            for block, values in checkpoint["rho_update_counts_by_block"].items()
        }
        self._last_rho_scale_by_mode = checkpoint[
            "last_rho_scale_by_mode"
        ].copy()
        self._last_rho_scale_by_block = {
            block: values.copy()
            for block, values in checkpoint["last_rho_scale_by_block"].items()
        }
        self.mode_diagnostics = [
            value.copy() for value in checkpoint["mode_diagnostics"]
        ]

    def _fit_mode(
        self,
        m: int,
        eta_all: list[np.ndarray],
        admm_iter: int,
        outer_iteration: int,
        *,
        max_admm_iter: int | None = None,
    ) -> dict:
        """Fit one mode with residual-balanced, progressively finer rho steps."""
        if max_admm_iter is None:
            max_admm_iter = admm_iter
        log_C_not_m = self._compute_log_C_not_m(eta_all, m)
        state = self._states[m]
        diagnostics = {}
        consensus_names = ("z0", "Z1", "Z2", "Z3", "z4", "z5")
        rho_updates = 0
        guard_restarts = 0
        divergence_checks = 0
        best_merit = np.inf
        best_parameter_magnitude = self._mode_parameter_magnitude(m)
        checkpoint = self._capture_mode_checkpoint(m, state)
        last_rho_update_iteration: int | None = None
        iteration_limit = int(admm_iter)
        iteration = 0

        while iteration < iteration_limit:
            iteration_number = iteration + 1
            # The residuals drive the stopping test, the rho balance and the
            # divergence guard, and nothing else, so one interval sets the
            # cadence of all three.  Splitting the rho balance onto its own
            # interval only allowed the two to fall out of step: rho could
            # then be adapted solely on iterations divisible by both, and a
            # pair like 7 and 10 left it practically frozen.  The snapshot
            # the residuals need is taken only on these iterations, which
            # keeps the dual residual a one-step difference rather than a
            # k-step one.
            check_residuals = (
                self.residual_check_interval <= 1
                or iteration_number % self.residual_check_interval == 0
                or iteration_number >= iteration_limit
            )
            previous = (
                {name: state[name].copy() for name in consensus_names}
                if check_residuals
                else None
            )
            self._parameter_block_update(m, state)
            self._consensus_block_update(m, state, log_C_not_m)
            self._dual_update(m, state)
            if not check_residuals:
                iteration += 1
                continue
            metrics = self._residuals_and_tolerances(m, state, previous)
            (
                primal,
                dual,
                eps_primal,
                eps_dual,
                normalized_primal,
                normalized_dual,
            ) = metrics[:6]
            (
                block_primal_values,
                block_dual_values,
                block_eps_primal_values,
                block_eps_dual_values,
                block_normalized_primal_values,
                block_normalized_dual_values,
            ) = metrics[6:]
            block_metrics = {
                block: {
                    "primal": float(block_primal_values[index]),
                    "dual": float(block_dual_values[index]),
                    "eps_primal": float(block_eps_primal_values[index]),
                    "eps_dual": float(block_eps_dual_values[index]),
                    "normalized_primal": float(
                        block_normalized_primal_values[index]
                    ),
                    "normalized_dual": float(
                        block_normalized_dual_values[index]
                    ),
                }
                for index, block in enumerate(_RHO_BLOCKS)
            }
            block_residual_converged = all(
                values["primal"] <= values["eps_primal"]
                and values["dual"] <= values["eps_dual"]
                for values in block_metrics.values()
            )
            residual_converged = (
                primal <= eps_primal
                and dual <= eps_dual
                and block_residual_converged
            )
            rho_used = self._mode_rhos(m)
            rho_updated = False
            updated_rho_blocks: tuple[str, ...] = ()
            guard_triggered = False
            metrics_are_finite = bool(
                np.all(np.isfinite(np.asarray(metrics[:6], dtype=float)))
                and all(
                    np.all(np.isfinite(np.asarray(value, dtype=float)))
                    for value in metrics[6:]
                )
            )
            normalized_merit = max(
                normalized_primal,
                normalized_dual,
                *(values["normalized_primal"] for values in block_metrics.values()),
                *(values["normalized_dual"] for values in block_metrics.values()),
            )
            parameter_magnitude = self._mode_parameter_magnitude(m)
            # Every iteration that reaches here is a residual-check one, so
            # the guard and the rho balance run on that same cadence.
            should_check_guard = True

            if not metrics_are_finite:
                divergence_checks = self.divergence_patience
            elif should_check_guard:
                merit_diverged = (
                    np.isfinite(best_merit)
                    and normalized_merit > self.divergence_factor * best_merit
                )
                parameters_diverged = parameter_magnitude > self.divergence_factor * max(
                    best_parameter_magnitude, 1.0
                )
                if merit_diverged or parameters_diverged:
                    divergence_checks += 1
                else:
                    divergence_checks = 0
                    if normalized_merit < best_merit:
                        best_merit = normalized_merit
                        best_parameter_magnitude = parameter_magnitude
                        checkpoint = self._capture_mode_checkpoint(m, state)

            if self.divergence_guard and divergence_checks >= self.divergence_patience:
                guard_restarts += 1
                if guard_restarts > 2:
                    raise FloatingPointError(
                        f"ADMM mode {m} repeatedly diverged despite guarded recovery."
                    )
                updated_rho_blocks = self._restore_mode_checkpoint(
                    m, state, checkpoint
                )
                rho_updated = bool(updated_rho_blocks)
                guard_triggered = True
                divergence_checks = 0
                residual_converged = False
            elif self.adaptive_rho and not residual_converged:
                adaptive_blocks = list(_RHO_BLOCKS)
                updated_rho_blocks = tuple(
                    block
                    for block in adaptive_blocks
                    if self._adapt_rho(
                        m,
                        block,
                        block_metrics[block]["normalized_primal"],
                        block_metrics[block]["normalized_dual"],
                    )
                )
                rho_updated = bool(updated_rho_blocks)
            if rho_updated:
                rho_updates += len(updated_rho_blocks)
                last_rho_update_iteration = iteration_number
                iteration_limit = min(
                    int(max_admm_iter),
                    max(
                        iteration_limit,
                        iteration_number + self.rho_stable_iterations,
                    ),
                )

            iterations_since_rho_update = (
                None
                if last_rho_update_iteration is None
                else iteration_number - last_rho_update_iteration
            )
            rho_is_stable = (
                iterations_since_rho_update is None
                or iterations_since_rho_update >= self.rho_stable_iterations
            )
            converged = residual_converged and rho_is_stable and not guard_triggered
            block_residuals = (
                self._primal_residuals_by_block(m, state)
                if should_check_guard
                or converged
                or iteration_number == iteration_limit
                else None
            )
            diagnostics = {
                "outer_iteration": outer_iteration,
                "iterations": iteration_number,
                "primal_residual": primal,
                "dual_residual": dual,
                "eps_primal": eps_primal,
                "eps_dual": eps_dual,
                "normalized_primal": normalized_primal,
                "normalized_dual": normalized_dual,
                "block_residuals": block_metrics,
                "block_residual_converged": block_residual_converged,
                "converged": converged,
                "rho_used": rho_used,
                "rho": self._mode_rhos(m),
                "rho_updated": rho_updated,
                "rho_updated_blocks": updated_rho_blocks,
                "rho_updates": rho_updates,
                "rho_update_count_total": self.rho_update_counts[m],
                "rho_update_counts_by_block": {
                    block: self.rho_update_counts_by_block[block][m]
                    for block in _RHO_BLOCKS
                },
                "rho_scale_used": {
                    block: self._last_rho_scale_by_block[block][m]
                    for block in updated_rho_blocks
                },
                "iterations_since_rho_update": iterations_since_rho_update,
                "rho_stable": rho_is_stable,
                "guard_triggered": guard_triggered,
                "guard_restarts": guard_restarts,
                "max_abs_parameter": parameter_magnitude,
                "initial_admm_iter": int(admm_iter),
                "iteration_limit": iteration_limit,
                "max_admm_iter": int(max_admm_iter),
                "extended": iteration_limit > int(admm_iter),
                "block_primal_residuals": block_residuals,
            }
            self.admm_history[m].append(diagnostics.copy())
            iteration += 1
            if guard_triggered:
                continue
            if converged:
                break

        # The sparse consensus copy is the relation network reported by the
        # model.  At convergence it agrees with the parameter-block copy.
        pre_projection_converged = bool(diagnostics["converged"])
        self.A[m] = self.scale_A_by_mode[m] * state["Z1"].copy()
        eta_all[m] = self._compute_eta_(m)
        post_primal, post_eps_primal, post_normalized_primal = (
            self._post_projection_feasibility(m, state)
        )
        diagnostics.update(
            {
                "pre_projection_converged": pre_projection_converged,
                "post_projection_primal_residual": post_primal,
                "post_projection_eps_primal": post_eps_primal,
                "post_projection_normalized_primal": post_normalized_primal,
                "block_primal_residuals_after_projection": (
                    self._primal_residuals_by_block(m, state)
                ),
                "parameter_contributions": self._parameter_contribution_norms(m),
                "converged": pre_projection_converged
                and post_primal
                <= self.post_projection_tolerance_factor * post_eps_primal,
            }
        )
        self.admm_history[m][-1].update(
            {
                "pre_projection_converged": pre_projection_converged,
                "post_projection_primal_residual": post_primal,
                "post_projection_eps_primal": post_eps_primal,
                "post_projection_normalized_primal": post_normalized_primal,
                "block_primal_residuals_after_projection": diagnostics[
                    "block_primal_residuals_after_projection"
                ],
                "parameter_contributions": diagnostics[
                    "parameter_contributions"
                ],
                "converged": diagnostics["converged"],
            }
        )
        self.mode_diagnostics[m] = diagnostics
        return diagnostics

    def fit(
        self,
        max_iter: int,
        admm_iter: int = 100,
        is_print: bool = False,
        *,
        max_admm_iter: int | None = None,
        progress_callback: Callable[["MAPGM", dict], None] | None = None,
        progress_every: int = 1,
    ) -> list[dict]:
        """Run cyclic mode updates, each solved by an inner two-block ADMM.

        ``max_iter`` is the number of outer mode sweeps.  ``admm_iter`` is the
        initial inner-iteration budget.  A late rho update extends that budget
        long enough to observe ``rho_stable_iterations``, up to
        ``max_admm_iter``.  The hard limit defaults to four times
        ``admm_iter``.

        ``residual_check_interval`` sets how often the inner loop evaluates
        its residuals, and with them the stopping test, the rho balance and
        the divergence guard.  Results are insensitive to it -- measured
        across 1..30 the AUC-PR moved by 0.4% and the rho trajectory not at
        all -- while the cost is not, so it is worth more than 1.

        ``progress_callback`` is invoked as ``callback(self, record)`` every
        ``progress_every`` outer iterations, and always on the final one.
        The record carries the objective and its weighted components, so a
        caller holding the ground truth can score the current ``A`` and ``b``
        without reaching into the solver.
        """
        if progress_callback is not None and progress_every <= 0:
            raise ValueError("progress_every must be positive.")
        if max_iter <= 0 or admm_iter is None or admm_iter <= 0:
            raise ValueError("max_iter and admm_iter must be positive integers.")
        admm_iter = int(admm_iter)
        if max_admm_iter is None:
            max_admm_iter = 4 * admm_iter
        if (
            isinstance(max_admm_iter, (bool, np.bool_))
            or int(max_admm_iter) != max_admm_iter
            or int(max_admm_iter) < admm_iter
        ):
            raise ValueError("max_admm_iter must be an integer at least admm_iter.")
        max_admm_iter = int(max_admm_iter)

        eta_all = [self._compute_eta_(m) for m in range(self.M)]
        self.history = []
        self.rho_history_by_block = {
            block: [[value] for value in values]
            for block, values in self.rho_by_block.items()
        }
        self.rho_history = self.rho_history_by_block["rho"]
        self.admm_history = [[] for _ in range(self.M)]
        self.fit_diagnostics = {}
        initial_objective = self._objective(eta_all)
        best_objective = (
            initial_objective if np.isfinite(initial_objective) else np.inf
        )
        best_outer_iteration = 0
        # The checkpoint exists only so the guard can roll back to it.  It
        # copies every state array, z0 and eta included, so it is not taken
        # when nothing can consume it.
        best_checkpoint = (
            self._capture_model_checkpoint(eta_all)
            if self.outer_objective_guard
            else None
        )
        objective_increase_checks = 0
        stopped_reason = "max_iter"
        restored_best = False
        outer_seconds: list[float] = []
        for outer in range(int(max_iter)):
            outer_started = time.time()
            mode_results = []
            for m in range(self.M):
                mode_result = self._fit_mode(
                    m,
                    eta_all,
                    admm_iter,
                    outer + 1,
                    max_admm_iter=max_admm_iter,
                )
                mode_results.append(mode_result)

            offset_normalization = self._normalize_mode_offsets(eta_all)
            objective = self._objective(eta_all)
            objective_is_finite = bool(np.isfinite(objective))
            objective_tolerance = (
                self.outer_objective_rel_tol * max(abs(best_objective), 1.0)
                if np.isfinite(best_objective)
                else 0.0
            )
            if objective_is_finite and objective < best_objective:
                best_objective = objective
                best_outer_iteration = outer + 1
                if self.outer_objective_guard:
                    best_checkpoint = self._capture_model_checkpoint(eta_all)
                objective_increase_checks = 0
            elif (
                objective_is_finite
                and objective <= best_objective + objective_tolerance
            ):
                objective_increase_checks = 0
            elif objective_is_finite:
                objective_increase_checks += 1
            else:
                objective_increase_checks = self.outer_objective_patience

            record = {
                "outer_iteration": outer + 1,
                "outer_seconds": time.time() - outer_started,
                "objective": objective,
                "rho": self._rho_by_mode.copy(),
                "rho_blocks": self._rho_snapshot(),
                "modes": mode_results,
                "all_modes_converged": all(
                    bool(item["converged"]) for item in mode_results
                )
                and len(mode_results) == self.M,
                "mode_offset_normalization": offset_normalization,
                "best_objective": best_objective,
                "best_outer_iteration": best_outer_iteration,
                "objective_increase_checks": objective_increase_checks,
                "restored_best": False,
                "stopped_reason": None,
            }
            outer_seconds.append(record["outer_seconds"])
            self.history.append(record)
            if progress_callback is not None and (
                (outer + 1) % progress_every == 0
                or outer + 1 == int(max_iter)
            ):
                penalties = self._compute_penalty_breakdown()
                progress_callback(
                    self,
                    {
                        "outer_iteration": outer + 1,
                        "log_marginal_likelihood": float(
                            self._compute_log_marginal_likelihood(eta_all)
                        ),
                        "lasso": float(penalties["lasso"]),
                        "temporal_A": float(penalties["temporal_A"]),
                        "temporal_b": float(penalties["temporal_b"]),
                        "objective": float(objective),
                        "best_objective": float(best_objective),
                        "converged_modes": int(
                            sum(
                                bool(item["converged"])
                                for item in mode_results
                            )
                        ),
                        "rho": self._rho_snapshot(),
                    },
                )
            if is_print:
                converged = sum(bool(item["converged"]) for item in mode_results)
                log_ml = self._compute_log_marginal_likelihood(eta_all)
                penalties = self._compute_penalty_breakdown()
                print(
                    f"Outer {outer + 1}/{max_iter}: "
                    f"log_ml={log_ml:.6g}, "
                    f"lasso(λ1)={penalties['lasso']:.6g}, "
                    f"temporal_A(λ2)={penalties['temporal_A']:.6g}, "
                    f"temporal_b(λ3)={penalties['temporal_b']:.6g}, "
                    f"objective={objective:.6g}, "
                    f"rho={self._rho_snapshot()}, "
                    f"ADMM converged modes={converged}/{self.M}"
                )

            if (
                self.outer_objective_guard
                and objective_increase_checks >= self.outer_objective_patience
            ):
                self._restore_model_checkpoint(eta_all, best_checkpoint)
                stopped_reason = "objective_divergence_guard"
                restored_best = True
                record["restored_best"] = True
                record["stopped_reason"] = stopped_reason
                record["rho_after_restore"] = self._rho_snapshot()
                if is_print:
                    print(
                        "Outer objective increased repeatedly; restored "
                        f"iteration {best_outer_iteration} "
                        f"(objective={best_objective:.6g})."
                    )
                break

        timings = np.asarray(outer_seconds, dtype=float)
        self.outer_seconds = outer_seconds
        self.fit_diagnostics = {
            "stopped_reason": stopped_reason,
            "iterations": len(self.history),
            # Elapsed time per outer sweep, so a slow run can be attributed
            # to either many cheap sweeps or a few expensive ones.
            "outer_seconds_total": float(timings.sum()),
            "outer_seconds_mean": (
                float(timings.mean()) if timings.size else float("nan")
            ),
            "outer_seconds_variance": (
                float(timings.var()) if timings.size else float("nan")
            ),
            "outer_seconds_std": (
                float(timings.std()) if timings.size else float("nan")
            ),
            "outer_seconds_min": (
                float(timings.min()) if timings.size else float("nan")
            ),
            "outer_seconds_max": (
                float(timings.max()) if timings.size else float("nan")
            ),
            "best_outer_iteration": best_outer_iteration,
            "best_objective": best_objective,
            "final_objective": self._objective(eta_all),
            "restored_best": restored_best,
            "all_modes_converged": all(
                bool(item) and bool(item.get("converged"))
                for item in self.mode_diagnostics
            ),
            "rho_update_counts": self.rho_update_counts.copy(),
            "rho_update_counts_by_block": {
                block: values.copy()
                for block, values in self.rho_update_counts_by_block.items()
            },
            "rho": self._rho_snapshot(),
            "initial_admm_iter": admm_iter,
            "max_admm_iter": max_admm_iter,
            "final_mode_means": [float(np.mean(b_m)) for b_m in self.b],
            "final_mode_mean_gap": float(
                np.ptp([float(np.mean(b_m)) for b_m in self.b])
            ),
            "time_varying_base": self.time_varying_base,
            "variable_scaling": self.variable_scaling,
            "scale_A": self.scale_A_by_mode.copy(),
            "scale_b": self.scale_b_by_mode.copy(),
        }
        return self.history

    def _compute_temporal_penalty_breakdown(self) -> tuple[float, float]:
        temporal_A = 0.0
        temporal_b = 0.0
        for m in range(self.M):
            for t in range(1, self.T):
                temporal_A += self.lambda_temporal_A[m] * _penalty_value(
                    self.A[m][t] - self.A[m][t - 1], self.temporal_penalty_A[m]
                )
                temporal_b += self.lambda_temporal_b[m] * _penalty_value(
                    self.b[m][t] - self.b[m][t - 1], self.temporal_penalty_b[m]
                )
        return float(temporal_A), float(temporal_b)

    def _compute_penalty_breakdown(self) -> dict[str, float]:
        """Return weighted lasso and temporal penalty components."""
        temporal_A, temporal_b = self._compute_temporal_penalty_breakdown()
        return {
            "lasso": self._compute_lasso_loss(),
            "temporal_A": temporal_A,
            "temporal_b": temporal_b,
        }

    def _temporal_regularization(self) -> float:
        temporal_A, temporal_b = self._compute_temporal_penalty_breakdown()
        return temporal_A + temporal_b

    def _compute_lasso_loss(self) -> float:
        return float(
            sum(
                self.lambda_regs[m] * np.sum(np.abs(self.A[m]))
                for m in range(self.M)
            )
        )

    def _compute_log_marginal_likelihood(self, eta_all: list[np.ndarray] | None = None) -> float:
        """Log-likelihood, each sample weighted by ``1/N_t`` of its time.

        The constant ``sum_log_factorials`` is the whole-data total, so it is
        divided by ``tau`` here: that is exactly its weighted value when every
        group is full, and the constant only shifts the reported objective in
        any case.
        """
        if eta_all is None:
            eta_all = [self._compute_eta_(m) for m in range(self.M)]
        weight = self._loss_weight
        log_likelihood = -self.sum_log_factorials / self._admm_objective_scale
        for m in range(self.M):
            log_likelihood += float(
                np.sum(weight * eta_all[m] * self.x_summaries[m])
            )
        log_partition = np.zeros((self.num_samples, 1), dtype=float)
        for m in range(self.M):
            log_partition += self._stable_row_logsumexp(eta_all[m])
        log_likelihood -= float(
            np.sum(weight * np.exp(np.clip(log_partition, -745.0, 700.0)))
        )
        return float(log_likelihood)

    def _objective(self, eta_all: list[np.ndarray] | None = None) -> float:
        return (
            -self._compute_log_marginal_likelihood(eta_all)
            + self._compute_lasso_loss()
            + self._temporal_regularization()
        )

    def objective_breakdown(
        self,
        A: Sequence[np.ndarray] | None = None,
        b: Sequence[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Objective and its weighted components at the given parameters.

        With no arguments this scores the fitted parameters.  Passing the
        ground truth scores that instead, which is the cheapest check that
        the optimizer is working: a solver that reached a better point than
        the truth returns a smaller ``objective`` here, and one that did not
        returns a larger one.
        """
        original_A, original_b = self.A, self.b
        if A is not None:
            self.A = [np.asarray(value, dtype=float) for value in A]
        if b is not None:
            self.b = [np.asarray(value, dtype=float) for value in b]
        try:
            for m in range(self.M):
                if self.A[m].shape != original_A[m].shape:
                    raise ValueError(
                        f"A[{m}] has shape {self.A[m].shape}, "
                        f"expected {original_A[m].shape}."
                    )
                if self.b[m].shape != original_b[m].shape:
                    raise ValueError(
                        f"b[{m}] has shape {self.b[m].shape}, "
                        f"expected {original_b[m].shape}."
                    )
            eta_all = [self._compute_eta_(m) for m in range(self.M)]
            log_likelihood = self._compute_log_marginal_likelihood(eta_all)
            penalties = self._compute_penalty_breakdown()
            return {
                "objective": float(
                    -log_likelihood
                    + penalties["lasso"]
                    + penalties["temporal_A"]
                    + penalties["temporal_b"]
                ),
                "log_marginal_likelihood": float(log_likelihood),
                "lasso": float(penalties["lasso"]),
                "temporal_A": float(penalties["temporal_A"]),
                "temporal_b": float(penalties["temporal_b"]),
            }
        finally:
            self.A, self.b = original_A, original_b
