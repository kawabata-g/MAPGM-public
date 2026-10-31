"""Synthetic count tensors with time-varying bases and mode networks.

A scenario fixes the sizes, the temporal shape of the base, the change points
of each mode's network and the seed.  :func:`generate_scenario_data` draws
the base trajectories and the piecewise-constant signed networks, then
samples the count tensor from the sublinear Poisson graphical model (SPGM)
by Gibbs sampling, and reduces it to the mode marginals MAPGM reads.

Every random stream is derived from the scenario seed alone, and each Gibbs
chain carries its own seed, so a scenario gives the same tensor bit for bit
on any machine and thread count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Sequence

# Sizes the BLAS and numba thread pools; see mapgm._threading.
from mapgm import _threading

import numpy as np
from numba import njit, prange
from scipy.special import gammaln


@dataclass(frozen=True)
class SyntheticScenario:
    """Complete, reproducible setup for one synthetic condition."""

    name: str
    dims: tuple[int, ...]
    parameter_times: int
    samples_per_time: int
    base_style: str
    change_points: tuple[tuple[int, ...], ...]
    seed: int

    # Base design.  Every coefficient is drawn per (mode, node) from a
    # uniform range, so the scenarios differ only in the temporal shape that
    # these draws are placed into.
    #: Log rate of a *cell*, not of one mode.  A cell's log rate is the sum
    #: of the M mode trajectories, so each mode receives this trajectory
    #: divided by M and the count level stays put as modes are added.
    base_low: float = 6.0
    base_high: float = 8.0
    amplitude_low: float = 0.5
    amplitude_high: float = 1.0
    frequency_low: float = 1.5
    frequency_high: float = 3.0
    rectangle_low: float = 0.5
    rectangle_high: float = 1.0
    rectangle_width_low: float = 0.2
    rectangle_width_high: float = 0.5

    # Network design: one Erdos-Renyi draw per segment.
    edge_probability: float = 0.25
    #: Edge-weight range per tensor cell; see :func:`edge_weight_range`.
    edge_weight_low: float = -1e-3
    edge_weight_high: float = 1e-3
    #: Probability that a change point resamples a given node pair.  1.0
    #: redraws the whole network.
    edge_turnover: float = 1.0

    #: Knots of the sublinear sufficient statistic of the SPGM sampler.
    r0: float = 5.0e3
    r: float = 1.0e4

    gibbs_sweeps: int = 40

    # MAPGM solver settings shared by every variant fitted in the study.
    mapgm_rho: float = 1e10
    mapgm_adaptive_rho: bool = True
    mapgm_outer_objective_patience: int = 200
    mapgm_divergence_patience: int = 200
    mapgm_outer_iter: int = 500
    mapgm_admm_iter: int = 120
    #: Evaluate the ADMM residuals every this many inner iterations.
    mapgm_residual_check_interval: int = 10
    #: Outer iterations of a hyperparameter-search fit.  ``None`` derives one
    #: fifth of ``mapgm_outer_iter``: the search only ranks candidates.
    mapgm_search_outer_iter: int | None = None
    # Streaming windows, in parameter times.
    mapgm_init_len: int = 100
    mapgm_slide_len: int = 100
    #: Outer iterations of each sliding refit, which starts from the
    #: previous window's estimate.
    mapgm_stream_max_iter: int = 10
    #: Outer iterations of the initial batch window, which starts from
    #: scratch.
    mapgm_stream_init_max_iter: int = 100

    @property
    def num_modes(self) -> int:
        return len(self.dims)

    def outer_iterations(self, *, search: bool) -> int:
        """Outer-iteration budget for a search fit or for the final fit."""
        if not search:
            return self.mapgm_outer_iter
        if self.mapgm_search_outer_iter is not None:
            return max(1, int(self.mapgm_search_outer_iter))
        return max(1, self.mapgm_outer_iter // 5)


SYNTHETIC_SCENARIO_NAMES = ("synt2", "synt3", "synt4")

#: Scenario labels as the paper names them.
SCENARIO_LABELS = {
    "synt2": "Linear",
    "synt3": "Trigonometric",
    "synt4": "Rectangular",
}


def _registered_scenarios() -> dict[str, SyntheticScenario]:
    return {
        "synt2": SyntheticScenario(
            name="synt2",
            dims=(15, 15, 15),
            parameter_times=300,
            samples_per_time=50,
            base_style="linear",
            change_points=((100, 200), (150, 250), (50, 200)),
            seed=202,
        ),
        "synt3": SyntheticScenario(
            name="synt3",
            dims=(15, 15, 15),
            parameter_times=300,
            samples_per_time=50,
            base_style="trigonometric",
            # Ten sets so the mode-count sweep can reach M=10; a run takes
            # the leading ``num_modes`` of them.
            change_points=(
                (100, 200), (150, 250), (50, 200),
                (75, 225), (125, 275), (60, 180),
                (90, 240), (110, 210), (140, 260),
                (80, 190),
            ),
            seed=300,
        ),
        "synt4": SyntheticScenario(
            name="synt4",
            dims=(5, 5),
            parameter_times=300,
            samples_per_time=50,
            base_style="rectangular",
            change_points=((100, 200), (150, 250), (50, 200)),
            seed=404,
        ),
    }


def resolve_change_points(
    change_points: Sequence[Sequence[int]],
    num_modes: int,
    registry_times: int,
    parameter_times: int,
) -> tuple[tuple[int, ...], ...]:
    """Take the first ``num_modes`` change-point sets, rescaled to ``T``.

    A scenario lists one set per mode, so a smaller mode count uses the
    leading sets and a larger one is an error.  The listed times are
    absolute, so a different horizon rescales them by
    ``parameter_times / registry_times``, which keeps their relative
    positions.  Rounding can collide, so each mode's times are deduplicated.
    """
    if num_modes > len(change_points):
        raise ValueError(
            f"num_modes={num_modes} exceeds the {len(change_points)} "
            "change-point set(s) this scenario defines."
        )
    scale = parameter_times / registry_times
    resolved = []
    for mode_points in change_points[:num_modes]:
        rescaled = sorted(
            {
                int(np.clip(round(point * scale), 1, parameter_times - 1))
                for point in mode_points
            }
        )
        resolved.append(tuple(rescaled))
    return tuple(resolved)


def get_scenario(
    name: str,
    *,
    quick: bool = False,
    dimension: int | None = None,
    num_modes: int | None = None,
    parameter_times: int | None = None,
    samples_per_time: int | None = None,
    seed: int | None = None,
    overrides: dict[str, object] | None = None,
) -> SyntheticScenario:
    """Return a registered scenario, optionally resized.

    ``dimension`` applies the same size ``d`` to every mode.  ``overrides``
    sets any other scenario field.  ``quick`` cuts the Gibbs and solver
    iterations for a smoke test; its results are not meaningful.
    """
    scenarios = _registered_scenarios()
    if name not in scenarios:
        raise ValueError(
            f"Unknown scenario {name!r}; choose from {sorted(scenarios)}."
        )
    scenario = scenarios[name]
    if overrides:
        unknown = set(overrides) - {
            field.name for field in fields(SyntheticScenario)
        }
        if unknown:
            raise ValueError(f"Unknown scenario overrides {sorted(unknown)}.")
        scenario = replace(scenario, **overrides)
    resolved_times = (
        scenario.parameter_times
        if parameter_times is None
        else int(parameter_times)
    )
    resolved_modes = scenario.num_modes if num_modes is None else int(num_modes)
    if resolved_modes < 2:
        raise ValueError("num_modes must be at least 2.")
    if resolved_times < 8:
        raise ValueError("parameter_times must be at least 8.")

    if dimension is None and resolved_modes == scenario.num_modes:
        resolved_dims = scenario.dims
    else:
        resolved_dimension = (
            scenario.dims[0] if dimension is None else int(dimension)
        )
        if resolved_dimension < 3:
            raise ValueError("dimension must be at least 3.")
        resolved_dims = (resolved_dimension,) * resolved_modes

    resolved_change_points = resolve_change_points(
        scenario.change_points,
        resolved_modes,
        scenario.parameter_times,
        resolved_times,
    )
    scenario = replace(
        scenario,
        dims=resolved_dims,
        parameter_times=resolved_times,
        samples_per_time=(
            scenario.samples_per_time
            if samples_per_time is None
            else int(samples_per_time)
        ),
        seed=scenario.seed if seed is None else int(seed),
        change_points=resolved_change_points,
    )
    if scenario.samples_per_time <= 0:
        raise ValueError("samples_per_time must be positive.")
    if not 0.0 < scenario.edge_probability <= 1.0:
        raise ValueError("edge_probability must be in (0, 1].")
    if not 0.0 < scenario.edge_turnover <= 1.0:
        raise ValueError("edge_turnover must be in (0, 1].")
    if not 0.0 <= scenario.r0 < scenario.r:
        raise ValueError("r0 and r must satisfy 0 <= r0 < r.")
    if scenario.edge_weight_low > scenario.edge_weight_high:
        raise ValueError("edge_weight_low must not exceed edge_weight_high.")
    if scenario.base_low > scenario.base_high:
        raise ValueError("base_low must not exceed base_high.")
    if quick:
        return replace(
            scenario,
            gibbs_sweeps=min(scenario.gibbs_sweeps, 12),
            mapgm_outer_iter=min(scenario.mapgm_outer_iter, 3),
            mapgm_admm_iter=min(scenario.mapgm_admm_iter, 20),
        )
    return scenario


# ---------------------------------------------------------------------------
# true parameters
# ---------------------------------------------------------------------------


def _network_sequence(
    dimension: int,
    parameter_times: int,
    change_points: Sequence[int],
    seed: int,
    edge_probability: float,
    weight_low: float,
    weight_high: float,
    edge_turnover: float,
) -> np.ndarray:
    """Draw a piecewise-constant network with partial change-point turnover.

    A ``uniform < p_edge`` draw is kept on the strict upper triangle and then
    mirrored, so every unordered pair is an edge with probability
    ``edge_probability``.  Each edge takes one uniform signed weight.

    At a change point only an ``edge_turnover`` share of node pairs is
    resampled from the same prior; the rest keeps its presence and weight.
    ``edge_turnover=1.0`` redraws the whole network.
    """
    boundaries = (0, *change_points, parameter_times)
    sequence = np.zeros((parameter_times, dimension, dimension), dtype=float)
    rng = np.random.default_rng(seed)
    previous = np.zeros((dimension, dimension), dtype=float)

    def _mirror(values: np.ndarray) -> np.ndarray:
        upper = np.triu(values, 1)
        return upper + upper.T

    for segment, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        mask = _mirror(
            rng.uniform(size=(dimension, dimension)) < edge_probability
        ).astype(bool)
        weights = _mirror(
            rng.uniform(weight_low, weight_high, size=(dimension, dimension))
        )
        fresh = np.where(mask, weights, 0.0)
        if segment == 0:
            matrix = fresh
        else:
            resampled = _mirror(
                rng.uniform(size=(dimension, dimension)) < edge_turnover
            ).astype(bool)
            matrix = np.where(resampled, fresh, previous)
        np.fill_diagonal(matrix, 0.0)
        sequence[start:stop] = matrix
        previous = matrix
    return sequence


def _base_trajectory(scenario: SyntheticScenario, mode: int) -> np.ndarray:
    """Draw one mode's ``(T, d_m)`` base trajectory.

    Every coefficient is uniform over a configurable range and is drawn once
    per node, so the scenarios share one intercept distribution and differ
    only in the temporal shape added on top of it.  The ranges describe a
    cell, which sums one trajectory per mode, so the finished trajectory is
    divided by the mode count.
    """
    dimension = scenario.dims[mode]
    time = np.linspace(0.0, 1.0, scenario.parameter_times)
    rng = np.random.default_rng(scenario.seed + 7919 * (mode + 1))

    if scenario.base_style == "linear":
        start = rng.uniform(scenario.base_low, scenario.base_high, dimension)
        stop = rng.uniform(scenario.base_low, scenario.base_high, dimension)
        profile = start[None, :] + time[:, None] * (stop - start)[None, :]
    elif scenario.base_style == "trigonometric":
        intercept = rng.uniform(scenario.base_low, scenario.base_high, dimension)
        amplitude = rng.uniform(
            scenario.amplitude_low, scenario.amplitude_high, dimension
        )
        frequency = rng.uniform(
            scenario.frequency_low, scenario.frequency_high, dimension
        )
        phase = rng.uniform(0.0, 2.0 * np.pi, dimension)
        profile = intercept[None, :] + amplitude[None, :] * np.sin(
            2.0 * np.pi * time[:, None] * frequency[None, :]
            + phase[None, :]
        )
    elif scenario.base_style == "rectangular":
        intercept = rng.uniform(scenario.base_low, scenario.base_high, dimension)
        width = rng.uniform(
            scenario.rectangle_width_low, scenario.rectangle_width_high, dimension
        )
        starts = rng.uniform(0.0, 1.0 - width, dimension)
        stops = starts + width
        multiplier = rng.uniform(
            scenario.rectangle_low, scenario.rectangle_high, dimension
        )
        active = (time[:, None] >= starts[None, :]) & (
            time[:, None] <= stops[None, :]
        )
        profile = intercept[None, :] + active * multiplier[None, :]
    else:
        raise ValueError(
            "base_style must be one of 'linear', 'trigonometric', or "
            "'rectangular'."
        )
    return profile / float(scenario.num_modes)


def edge_weight_range(scenario: SyntheticScenario) -> tuple[float, float]:
    """The edge-weight range, per cell of the tensor.

    ``A_m`` multiplies a mode marginal, a sum over ``prod(dims) / d_m``
    cells, and the product sums ``d_m`` of them, so the interaction inside
    ``exp`` grows with ``prod(dims)``.  Dividing by the cell count makes the
    configured range mean one thing at every size.
    """
    cells = float(np.prod(scenario.dims))
    return (
        scenario.edge_weight_low / cells,
        scenario.edge_weight_high / cells,
    )


def build_true_parameters(
    scenario: SyntheticScenario,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build the base trajectories and the segment-wise networks."""
    weight_low, weight_high = edge_weight_range(scenario)
    b_true = []
    A_true = []
    for mode, dimension in enumerate(scenario.dims):
        b_true.append(_base_trajectory(scenario, mode))
        A_true.append(
            _network_sequence(
                dimension,
                scenario.parameter_times,
                scenario.change_points[mode],
                scenario.seed + 1009 * mode,
                scenario.edge_probability,
                weight_low,
                weight_high,
                scenario.edge_turnover,
            )
        )
    return b_true, A_true


def count_scale(b_true: Sequence[np.ndarray]) -> float:
    """Typical cell Poisson rate implied by the drawn base."""
    total_log_rate = sum(float(np.mean(base)) for base in b_true)
    return float(np.exp(np.clip(total_log_rate, -20.0, 30.0)))


# ---------------------------------------------------------------------------
# SPGM Gibbs sampler
# ---------------------------------------------------------------------------

#: Whether to draw the Gibbs chains in parallel.  Each ``(parameter time,
#: sample)`` chain carries its own seed, so the tensor is bit-for-bit the
#: same either way; set ``MAPGM_PARALLEL_GIBBS=0`` (or ``MAPGM_PARALLEL=0``)
#: to force the serial sampler.
PARALLEL_GIBBS = _threading.parallel_enabled("gibbs")

#: A handful of chains would leave most threads idle.
PARALLEL_GIBBS_MINIMUM_CHAINS = 64


@njit(cache=True)
def _sublinear_statistic(value: float, r0: float, r: float) -> float:
    if value <= r0:
        return value
    if value <= r:
        return (2.0 * r * value - value * value - r0 * r0) / (
            2.0 * (r - r0)
        )
    return 0.5 * (r + r0)


@njit(cache=True)
def _conditional_log_mass(
    value: int,
    linear: float,
    interaction: float,
    r0: float,
    r: float,
) -> float:
    return (
        (linear + interaction) * _sublinear_statistic(value, r0, r)
        - math.lgamma(value + 1.0)
    )


@njit(cache=True)
def _discrete_slice_sample(
    current: int,
    linear: float,
    interaction: float,
    upper: int,
    width: int,
    r0: float,
    r: float,
) -> int:
    """One integer slice update with stepping out and interval shrinkage."""
    log_level = _conditional_log_mass(
        current, linear, interaction, r0, r
    ) + math.log(max(np.random.random(), 1e-300))
    offset = int(np.random.random() * width)
    left = max(0, current - offset)
    right = min(upper, left + width)

    while left > 0:
        candidate = max(0, left - width)
        if _conditional_log_mass(candidate, linear, interaction, r0, r) <= log_level:
            break
        left = candidate
    while right < upper:
        candidate = min(upper, right + width)
        if _conditional_log_mass(candidate, linear, interaction, r0, r) <= log_level:
            break
        right = candidate

    for _ in range(1000):
        proposal = left + int(np.random.random() * (right - left + 1))
        if _conditional_log_mass(proposal, linear, interaction, r0, r) >= log_level:
            return proposal
        if proposal < current:
            left = proposal + 1
        else:
            right = proposal - 1
        if right < left:
            break
    return current


@njit(cache=True)
def _gibbs_chain_numba(
    bases: np.ndarray,
    networks: np.ndarray,
    node_offsets: np.ndarray,
    edge_offsets: np.ndarray,
    coordinates: np.ndarray,
    dims: np.ndarray,
    time_index: int,
    sweeps: int,
    upper: int,
    slice_width: int,
    r0: float,
    r: float,
    values: np.ndarray,
    marginals: np.ndarray,
) -> None:
    """Run one independent chain: the draw at one parameter time and sample.

    ``values`` is that chain's flat cell vector and ``marginals`` its
    concatenated per-mode sufficient statistics.  Nothing outside them is
    written, which is what lets the chains run concurrently.
    """
    num_modes = dims.shape[0]
    flat_dimension = coordinates.shape[0]

    for flat in range(flat_dimension):
        base_log_rate = 0.0
        for mode in range(num_modes):
            base_log_rate += bases[
                time_index, node_offsets[mode] + coordinates[flat, mode]
            ]
        rate = math.exp(min(max(base_log_rate, -10.0), math.log(upper)))
        values[flat] = min(np.random.poisson(rate), upper)

    for _ in range(sweeps):
        marginals[:] = 0.0
        for flat in range(flat_dimension):
            statistic = _sublinear_statistic(values[flat], r0, r)
            for mode in range(num_modes):
                marginals[node_offsets[mode] + coordinates[flat, mode]] += statistic

        for flat in range(flat_dimension):
            base_log_rate = 0.0
            interaction = 0.0
            for mode in range(num_modes):
                node = coordinates[flat, mode]
                dimension = dims[mode]
                node_start = node_offsets[mode]
                base_log_rate += bases[time_index, node_start + node]
                row = edge_offsets[mode] + node * dimension
                for neighbor in range(dimension):
                    interaction += (
                        networks[time_index, row + neighbor]
                        * marginals[node_start + neighbor]
                    )

            old = int(values[flat])
            new = _discrete_slice_sample(
                old,
                base_log_rate,
                interaction,
                upper,
                slice_width,
                r0,
                r,
            )
            values[flat] = new
            delta = _sublinear_statistic(new, r0, r) - _sublinear_statistic(
                old, r0, r
            )
            if delta != 0.0:
                for mode in range(num_modes):
                    marginals[
                        node_offsets[mode] + coordinates[flat, mode]
                    ] += delta


@njit(cache=True)
def _chain_seed(seed: int, chain: int) -> int:
    """A per-chain seed that does not depend on how the work is scheduled."""
    mixed = (seed + 1) * 6364136223846793005 + (chain + 1) * 1442695040888963407
    mixed ^= mixed >> 33
    mixed = mixed * 1099511628211
    mixed ^= mixed >> 29
    return int(mixed & 0x7FFFFFFF)


@njit(cache=True)
def _gibbs_tensor_numba(
    bases: np.ndarray,
    networks: np.ndarray,
    node_offsets: np.ndarray,
    edge_offsets: np.ndarray,
    coordinates: np.ndarray,
    dims: np.ndarray,
    samples_per_time: int,
    sweeps: int,
    seed: int,
    upper: int,
    slice_width: int,
    r0: float,
    r: float,
) -> np.ndarray:
    parameter_times = bases.shape[0]
    flat_dimension = coordinates.shape[0]
    total_nodes = node_offsets[dims.shape[0]]
    values = np.empty(
        (parameter_times, samples_per_time, flat_dimension),
        dtype=np.float64,
    )
    scratch = np.zeros(
        (parameter_times * samples_per_time, total_nodes), dtype=np.float64
    )
    for chain in range(parameter_times * samples_per_time):
        time_index = chain % parameter_times
        sample = chain // parameter_times
        np.random.seed(_chain_seed(seed, chain))
        _gibbs_chain_numba(
            bases,
            networks,
            node_offsets,
            edge_offsets,
            coordinates,
            dims,
            time_index,
            sweeps,
            upper,
            slice_width,
            r0,
            r,
            values[time_index, sample],
            scratch[chain],
        )
    return values


@njit(cache=True, parallel=True)
def _gibbs_tensor_parallel_numba(
    bases: np.ndarray,
    networks: np.ndarray,
    node_offsets: np.ndarray,
    edge_offsets: np.ndarray,
    coordinates: np.ndarray,
    dims: np.ndarray,
    samples_per_time: int,
    sweeps: int,
    seed: int,
    upper: int,
    slice_width: int,
    r0: float,
    r: float,
) -> np.ndarray:
    """Thread-parallel twin of :func:`_gibbs_tensor_numba`.

    Each ``(parameter time, sample)`` pair is its own chain with its own
    seed, so the tensor is identical to the serial one whatever the thread
    count.  Time varies fastest over the chains, so a contiguous block of
    them spans the whole horizon and the threads get even work.
    """
    parameter_times = bases.shape[0]
    flat_dimension = coordinates.shape[0]
    total_nodes = node_offsets[dims.shape[0]]
    values = np.empty(
        (parameter_times, samples_per_time, flat_dimension),
        dtype=np.float64,
    )
    scratch = np.zeros(
        (parameter_times * samples_per_time, total_nodes), dtype=np.float64
    )
    for chain in prange(parameter_times * samples_per_time):
        time_index = chain % parameter_times
        sample = chain // parameter_times
        np.random.seed(_chain_seed(seed, chain))
        _gibbs_chain_numba(
            bases,
            networks,
            node_offsets,
            edge_offsets,
            coordinates,
            dims,
            time_index,
            sweeps,
            upper,
            slice_width,
            r0,
            r,
            values[time_index, sample],
            scratch[chain],
        )
    return values


def _pack_mode_parameters(
    bases: Sequence[np.ndarray],
    networks: Sequence[np.ndarray],
    dims: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate the per-mode bases and networks into flat arrays."""
    parameter_times = bases[0].shape[0]
    node_offsets = np.zeros(len(dims) + 1, dtype=np.int64)
    edge_offsets = np.zeros(len(dims) + 1, dtype=np.int64)
    for mode, dimension in enumerate(dims):
        node_offsets[mode + 1] = node_offsets[mode] + dimension
        edge_offsets[mode + 1] = edge_offsets[mode] + dimension * dimension
    packed_bases = np.zeros((parameter_times, node_offsets[-1]), dtype=np.float64)
    packed_networks = np.zeros(
        (parameter_times, edge_offsets[-1]), dtype=np.float64
    )
    for mode, dimension in enumerate(dims):
        packed_bases[:, node_offsets[mode] : node_offsets[mode + 1]] = bases[mode]
        packed_networks[:, edge_offsets[mode] : edge_offsets[mode + 1]] = (
            networks[mode].reshape(parameter_times, dimension * dimension)
        )
    return packed_bases, packed_networks, node_offsets, edge_offsets


def generate_spgm_tensor(
    b_list: Sequence[np.ndarray],
    A_list: Sequence[np.ndarray],
    *,
    samples_per_time: int,
    count_scale: float,
    n_gibbs: int,
    seed: int,
    r0: float,
    r: float,
) -> np.ndarray:
    """Sample a ``(T, samples_per_time, d_1, ..., d_M)`` SPGM count tensor.

    ``b_list[m]`` has shape ``(T, d_m)`` and ``A_list[m]`` shape
    ``(T, d_m, d_m)``, symmetric with a zero diagonal.  ``count_scale`` is the
    typical Poisson rate implied by ``b_list``; it only sets the sampler's
    support bound and slice width.
    """
    if samples_per_time <= 0 or n_gibbs <= 0:
        raise ValueError("samples_per_time and n_gibbs must be positive.")
    if count_scale <= 0.0 or not np.isfinite(count_scale):
        raise ValueError("count_scale must be positive and finite.")
    if not 0.0 <= r0 < r:
        raise ValueError("r0 and r must satisfy 0 <= r0 < r.")
    if not b_list or len(b_list) != len(A_list):
        raise ValueError("b_list and A_list must contain the same modes.")
    bases = [np.ascontiguousarray(value, dtype=np.float64) for value in b_list]
    networks = [np.ascontiguousarray(value, dtype=np.float64) for value in A_list]
    parameter_times = bases[0].shape[0]
    dims = tuple(value.shape[1] for value in bases)
    for mode, (base, network, dimension) in enumerate(zip(bases, networks, dims)):
        if base.shape != (parameter_times, dimension):
            raise ValueError(f"b_list[{mode}] has an inconsistent shape.")
        if network.shape != (parameter_times, dimension, dimension):
            raise ValueError(f"A_list[{mode}] has an inconsistent shape.")
        if not np.allclose(network, np.swapaxes(network, 1, 2)):
            raise ValueError("Every network must be symmetric.")
        if not np.allclose(np.diagonal(network, axis1=1, axis2=2), 0.0):
            raise ValueError("Every network must have a zero diagonal.")

    coordinates = np.stack(
        np.unravel_index(np.arange(int(np.prod(dims))), dims),
        axis=1,
    ).astype(np.int64)
    packed_bases, packed_networks, node_offsets, edge_offsets = (
        _pack_mode_parameters(bases, networks, dims)
    )
    # The support must reach past the saturation knot of the statistic.
    support_scale = max(count_scale, r)
    upper = max(
        32,
        int(math.ceil(6.0 * support_scale + 12.0 * math.sqrt(support_scale))),
    )
    slice_width = max(4, int(math.ceil(4.0 * math.sqrt(support_scale))))
    chains = parameter_times * int(samples_per_time)
    sampler = (
        _gibbs_tensor_parallel_numba
        if PARALLEL_GIBBS and chains >= PARALLEL_GIBBS_MINIMUM_CHAINS
        else _gibbs_tensor_numba
    )
    flat = sampler(
        packed_bases,
        packed_networks,
        node_offsets,
        edge_offsets,
        coordinates,
        np.asarray(dims, dtype=np.int64),
        int(samples_per_time),
        int(n_gibbs),
        int(seed),
        upper,
        slice_width,
        float(r0),
        float(r),
    )
    return flat.reshape((parameter_times, int(samples_per_time), *dims))


# ---------------------------------------------------------------------------
# one scenario's data
# ---------------------------------------------------------------------------


def mode_summaries(X: np.ndarray, scenario: SyntheticScenario) -> list[np.ndarray]:
    """Reduce the count tensor to the ``(T * n, d_m)`` mode marginals."""
    sample_tensor = np.asarray(
        X.reshape(
            scenario.parameter_times * scenario.samples_per_time,
            *scenario.dims,
        ),
        dtype=float,
    )
    summaries = []
    for mode in range(scenario.num_modes):
        reduce_axes = tuple(
            axis
            for axis in range(1, scenario.num_modes + 1)
            if axis != mode + 1
        )
        summaries.append(np.sum(sample_tensor, axis=reduce_axes))
    return summaries


def generate_scenario_data(scenario: SyntheticScenario) -> dict:
    """Draw the true parameters, then sample one count tensor.

    The returned dict holds the tensor ``X`` with shape
    ``(T, samples_per_time, d_1, ..., d_M)``, the truth ``b_true`` and
    ``A_true`` (one array per mode), the mode marginals ``x_summaries`` and
    the flattened cells ``flat_summary`` (one row per sample), and
    ``sum_log_factorials``, the constant term of the Poisson log-likelihood.
    """
    b_true, A_true = build_true_parameters(scenario)
    reference = count_scale(b_true)
    X = generate_spgm_tensor(
        b_true,
        A_true,
        samples_per_time=scenario.samples_per_time,
        count_scale=reference,
        n_gibbs=scenario.gibbs_sweeps,
        seed=scenario.seed,
        r0=scenario.r0,
        r=scenario.r,
    )
    cells = np.asarray(
        X.reshape(
            scenario.parameter_times * scenario.samples_per_time,
            *scenario.dims,
        ),
        dtype=float,
    )
    return {
        "scenario": scenario,
        "X": X,
        "b_true": b_true,
        "A_true": A_true,
        "x_summaries": mode_summaries(X, scenario),
        "flat_summary": cells.reshape(cells.shape[0], -1),
        "sum_log_factorials": float(np.sum(gammaln(cells + 1.0))),
        "count_scale": reference,
        "observed_mean": float(np.mean(X)),
        # Share of cells past the knots of the sampler's sublinear statistic.
        "saturated_fraction": float(np.mean(X > scenario.r0)),
        "clipped_fraction": float(np.mean(X > scenario.r)),
    }
