"""Streaming MAPGM: an initial batch window followed by sliding refits."""

from __future__ import annotations

import numpy as np

from mapgm.methods.mapgm._kernels import _RHO_BLOCKS
from mapgm.methods.mapgm.model import MAPGM


class _FixedAnchorMAPGM(MAPGM):
    """Window-local MAPGM whose first parameter time is kept fixed.

    The first time point supplies the left-hand endpoint of the temporal
    penalty.  Its likelihood terms are constant with respect to the active
    window, while the ordinary ADMM consensus copies still give the first
    active time the required temporal boundary condition.
    """

    def __init__(
        self,
        *args,
        fixed_A: list[np.ndarray],
        fixed_b: list[np.ndarray],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._fixed_A = [np.asarray(value, dtype=float).copy() for value in fixed_A]
        self._fixed_b = [np.asarray(value, dtype=float).copy() for value in fixed_b]
        if len(self._fixed_A) != self.M or len(self._fixed_b) != self.M:
            raise ValueError("fixed_A and fixed_b must contain one value per mode.")
        for m, dimension in enumerate(self.dims):
            if self._fixed_A[m].shape != (dimension, dimension):
                raise ValueError(
                    f"fixed_A[{m}] must have shape {(dimension, dimension)}."
                )
            if self._fixed_b[m].shape != (dimension,):
                raise ValueError(f"fixed_b[{m}] must have shape {(dimension,)}.")
            self.A[m][0] = self._fixed_A[m]
            self.b[m][0] = self._fixed_b[m]
        self._states = [self._new_state(m) for m in range(self.M)]

    def _parameter_block_update(self, m: int, state: dict[str, np.ndarray]) -> None:
        """Update only the active window, leaving its temporal anchor fixed."""
        super()._parameter_block_update(m, state)
        self.A[m][0] = self._fixed_A[m]
        self.b[m][0] = self._fixed_b[m]

    def _fit_mode(self, m: int, *args, **kwargs) -> dict:
        diagnostics = super()._fit_mode(m, *args, **kwargs)
        self.A[m][0] = self._fixed_A[m]
        self.b[m][0] = self._fixed_b[m]
        eta_all = args[0]
        eta_all[m] = self._compute_eta_(m)
        return diagnostics


class StreamMAPGM(MAPGM):
    """ADMM MAPGM with an initial batch fit followed by sliding updates.

    ``init_len`` and ``slide_len`` are expressed in parameter-time units after
    the optional ``tau`` grouping.  ``initial_update`` fits times 1 through
    ``init_len`` with the ordinary batch ADMM solver.  Each
    ``streaming_update`` then advances one time point and refits the inclusive
    interval ``t - slide_len`` through ``t``.  When that interval has a prior
    time point, it is included as a fixed first local time so that the
    temporal penalty is applied without modifying its parameters.

    ``current_time`` is one-based: it is zero before ``initial_update`` and is
    ``init_len`` immediately afterwards.  A streaming call returns ``False``
    after it has updated the final time, enabling
    ``while model.streaming_update(...): ...``.
    """

    def __init__(
        self,
        *args,
        init_len: int,
        slide_len: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.time_varying_base:
            raise ValueError(
                "StreamMAPGM currently requires time_varying_base=True."
            )
        self.init_len = self._validate_parameter_length(
            init_len, "init_len", minimum=1
        )
        if self.init_len > self.T:
            raise ValueError(f"init_len={self.init_len} must be at most T={self.T}.")
        self.slide_len = self._validate_parameter_length(
            slide_len, "slide_len", minimum=0
        )
        self._current_time = 0
        self._initial_window_fitted = False
        self.initial_history: list[dict] = []
        self.initial_fit_diagnostics: dict = {}
        self.stream_history: list[dict] = []
        self.last_stream_diagnostics: dict | None = None

    @staticmethod
    def _validate_parameter_length(value: int, name: str, *, minimum: int) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            comparator = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {comparator} integer.")
        return int(value)

    @property
    def current_time(self) -> int:
        """Last fitted parameter time, using the public one-based convention."""
        return self._current_time

    @property
    def current_time_index(self) -> int | None:
        """Zero-based counterpart of ``current_time`` for array indexing."""
        return None if self._current_time == 0 else self._current_time - 1

    @property
    def is_initialized(self) -> bool:
        """Whether the initial batch window has already been fitted."""
        return self._initial_window_fitted

    def _local_solver_options(self, *, normalize_mode_offsets: bool) -> dict:
        """Return the current ADMM configuration for a local window solver."""
        return {
            "lambda_temporal_A": self.lambda_temporal_A.copy(),
            "lambda_temporal_b": self.lambda_temporal_b.copy(),
            "temporal_penalty_A": self.temporal_penalty_A.copy(),
            "temporal_penalty_b": self.temporal_penalty_b.copy(),
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "rho": self._rho_by_mode.copy(),

            "adaptive_rho": self.adaptive_rho,
            "rho_balance": self.rho_balance,
            "rho_scale": self.rho_scale,
            "rho_scale_decay": self.rho_scale_decay,
            "rho_scale_min": self.rho_scale_min,
            "rho_stable_iterations": self.rho_stable_iterations,
            "rho_min": [
                min(self.rho_min_by_block[block][m] for block in _RHO_BLOCKS)
                for m in range(self.M)
            ],
            "rho_max": [
                max(self.rho_max_by_block[block][m] for block in _RHO_BLOCKS)
                for m in range(self.M)
            ],
            "post_projection_tolerance_factor": self.post_projection_tolerance_factor,
            "normalize_mode_offsets": normalize_mode_offsets,
            "divergence_guard": self.divergence_guard,
            "divergence_factor": self.divergence_factor,
            "divergence_patience": self.divergence_patience,
            "outer_objective_guard": self.outer_objective_guard,
            "outer_objective_rel_tol": self.outer_objective_rel_tol,
            "outer_objective_patience": self.outer_objective_patience,
            "variable_scaling": self.variable_scaling,
            "scale_A": (
                self.scale_A_by_mode.copy() if self.variable_scaling else None
            ),
            "scale_b": (
                self.scale_b_by_mode.copy() if self.variable_scaling else None
            ),
            "time_varying_base": True,
            "tau": self.tau,
        }

    def _make_window_solver(
        self,
        start_time_index: int,
        end_time_index: int,
        *,
        fixed_anchor_index: int | None,
    ) -> tuple[MAPGM, int]:
        """Create a warm-started local solver and return its active offset."""
        if fixed_anchor_index is None:
            local_start = start_time_index
            solver_class = MAPGM
            solver_kwargs = self._local_solver_options(
                normalize_mode_offsets=self.normalize_mode_offsets
            )
        else:
            local_start = fixed_anchor_index
            solver_class = _FixedAnchorMAPGM
            solver_kwargs = self._local_solver_options(normalize_mode_offsets=False)
            solver_kwargs["fixed_A"] = [
                self.A[m][fixed_anchor_index].copy() for m in range(self.M)
            ]
            solver_kwargs["fixed_b"] = [
                self.b[m][fixed_anchor_index].copy() for m in range(self.M)
            ]

        sample_start = int(self.group_offsets[local_start])
        sample_stop = int(self.group_offsets[end_time_index + 1])
        local_x = [x_m[sample_start:sample_stop].copy() for x_m in self.x_summaries]
        solver = solver_class(
            self.sum_log_factorials,
            local_x,
            self.dims,
            self.lambda_regs.copy(),
            **solver_kwargs,
        )
        solver.rho_update_counts = self.rho_update_counts.copy()
        solver.rho_update_counts_by_block = {
            block: values.copy()
            for block, values in self.rho_update_counts_by_block.items()
        }
        solver._last_rho_scale_by_mode = self._last_rho_scale_by_mode.copy()
        solver._last_rho_scale_by_block = {
            block: values.copy()
            for block, values in self._last_rho_scale_by_block.items()
        }
        expected_times = end_time_index - local_start + 1
        if solver.T != expected_times:
            raise RuntimeError("Local parameter-time groups do not align with tau.")

        for m in range(self.M):
            solver.A[m][...] = self.A[m][local_start : end_time_index + 1]
            solver.b[m][...] = self.b[m][local_start : end_time_index + 1]
        if fixed_anchor_index is not None:
            for m in range(self.M):
                solver.A[m][0] = self.A[m][fixed_anchor_index]
                solver.b[m][0] = self.b[m][fixed_anchor_index]
        solver._states = [solver._new_state(m) for m in range(self.M)]
        active_offset = 1 if fixed_anchor_index is not None else 0
        return solver, active_offset

    def _synchronize_local_solver(self, solver: MAPGM, *, reset_history: bool) -> None:
        """Preserve the rho and diagnostic history produced by a local fit."""
        for block in _RHO_BLOCKS:
            self.rho_by_block[block][:] = solver.rho_by_block[block]
        if reset_history:
            self.rho_history_by_block = {
                block: [history.copy() for history in histories]
                for block, histories in solver.rho_history_by_block.items()
            }
            self.rho_history = self.rho_history_by_block["rho"]
            self.admm_history = [history.copy() for history in solver.admm_history]
        else:
            for block in _RHO_BLOCKS:
                for m in range(self.M):
                    self.rho_history_by_block[block][m].extend(
                        solver.rho_history_by_block[block][m][1:]
                    )
            for m in range(self.M):
                self.admm_history[m].extend(solver.admm_history[m])
        self.rho_update_counts = solver.rho_update_counts.copy()
        self.rho_update_counts_by_block = {
            block: values.copy()
            for block, values in solver.rho_update_counts_by_block.items()
        }
        self._last_rho_scale_by_mode = solver._last_rho_scale_by_mode.copy()
        self._last_rho_scale_by_block = {
            block: values.copy()
            for block, values in solver._last_rho_scale_by_block.items()
        }
        self.mode_diagnostics = [value.copy() for value in solver.mode_diagnostics]

    def _replace_unseen_parameters(self) -> None:
        """Remove future-data initialization after fitting the initial window."""
        for m in range(self.M):
            self.A[m][self.init_len :] = 0.0
            if self.init_len < self.T:
                self.b[m][self.init_len :] = self.b[m][self.init_len - 1]
        self._states = [self._new_state(m) for m in range(self.M)]

    def initial_update(
        self,
        max_iter: int,
        admm_iter: int = 100,
        is_print: bool = False,
        *,
        max_admm_iter: int | None = None,
    ) -> list[dict]:
        """Fit the first ``init_len`` times with the ordinary batch ADMM.

        This is intentionally separate from ``streaming_update`` so that no
        future samples are used until the corresponding time is reached.
        """
        sample_stop = int(self.group_offsets[self.init_len])
        initial_x = [x_m[:sample_stop].copy() for x_m in self.x_summaries]
        solver = MAPGM(
            self.sum_log_factorials,
            initial_x,
            self.dims,
            self.lambda_regs.copy(),
            **self._local_solver_options(
                normalize_mode_offsets=self.normalize_mode_offsets
            ),
        )
        if solver.T != self.init_len:
            raise RuntimeError("Initial parameter-time groups do not align with tau.")
        history = solver.fit(
            max_iter=max_iter,
            admm_iter=admm_iter,
            is_print=is_print,
            max_admm_iter=max_admm_iter,
        )
        for m in range(self.M):
            self.A[m][: self.init_len] = solver.A[m]
            self.b[m][: self.init_len] = solver.b[m]
        self._replace_unseen_parameters()
        self._synchronize_local_solver(solver, reset_history=True)
        self.initial_history = history
        self.history = history.copy()
        self.initial_fit_diagnostics = solver.fit_diagnostics.copy()
        self.fit_diagnostics = solver.fit_diagnostics.copy()
        self.stream_history = []
        self.last_stream_diagnostics = None
        self._current_time = self.init_len
        self._initial_window_fitted = True
        return history

    def streaming_update(
        self,
        max_iter: int,
        admm_iter: int = 100,
        is_print: bool = False,
        *,
        max_admm_iter: int | None = None,
    ) -> bool:
        """Advance one time and refit its sliding window.

        The return value indicates whether a subsequent time remains.  Thus
        the call that fits the final time returns ``False`` after committing
        that update.
        """
        if not self._initial_window_fitted:
            raise RuntimeError("Call initial_update before streaming_update.")
        if self._current_time >= self.T:
            return False

        end_time_index = self._current_time
        start_time_index = max(0, end_time_index - self.slide_len)
        fixed_anchor_index = (
            start_time_index - 1 if start_time_index > 0 else None
        )
        solver, active_offset = self._make_window_solver(
            start_time_index,
            end_time_index,
            fixed_anchor_index=fixed_anchor_index,
        )
        history = solver.fit(
            max_iter=max_iter,
            admm_iter=admm_iter,
            is_print=is_print,
            max_admm_iter=max_admm_iter,
        )
        for m in range(self.M):
            self.A[m][start_time_index : end_time_index + 1] = solver.A[m][
                active_offset:
            ]
            self.b[m][start_time_index : end_time_index + 1] = solver.b[m][
                active_offset:
            ]
        self._states = [self._new_state(m) for m in range(self.M)]
        self._synchronize_local_solver(solver, reset_history=False)

        self._current_time += 1
        record = {
            "time": self._current_time,
            "time_index": end_time_index,
            "window_start": start_time_index + 1,
            "window_end": end_time_index + 1,
            "window_start_index": start_time_index,
            "window_end_index": end_time_index,
            "fixed_anchor_time": (
                None if fixed_anchor_index is None else fixed_anchor_index + 1
            ),
            "fixed_anchor_index": fixed_anchor_index,
            "history": history,
            "fit_diagnostics": solver.fit_diagnostics.copy(),
        }
        self.stream_history.append(record)
        self.last_stream_diagnostics = record
        self.fit_diagnostics = solver.fit_diagnostics.copy()
        self.fit_diagnostics.update(
            {
                "streaming": True,
                "current_time": self.current_time,
                "remaining_times": self.T - self.current_time,
            }
        )
        return self._current_time < self.T
