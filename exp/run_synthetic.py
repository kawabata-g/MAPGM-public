"""Run the synthetic experiments of the paper.

For every scenario, mode count M, mode size d and seed, the evaluation tensor
is drawn with that seed and each requested MAPGM variant is fitted to it:

    tensor       MAPGM
    stream       MAPGM-Stream (an initial window, then sliding refits)
    static-base  MAPGM with one base shared by every time (ablation)
    flattened    MAPGM on the vectorized tensor, M=1 (ablation)

The regularization weights are those selected for the paper, read from
configs/synthetic_lambdas.csv; the stream uses the tensor MAPGM's.  With
--search they are selected again instead: every setting of the grid is fitted
to a separate draw of the same scenario (seed = the scenario's own seed +
500000) and the one with the best AUC-PR wins.  The search belongs to the
condition, not to the seed, so every seed uses the same weights.

Results go to <output-dir>/<run-name>/seed=<s>/<scenario>/M=<M>/dims=<d>/.
<run-name>/summary_metrics.csv gathers every seed's metrics and
<run-name>/summary_metrics_mean_std.csv their mean and standard deviation.

Examples
--------
The paper's grid for the tensor MAPGM (5 seeds)::

    uv run python exp/run_synthetic.py --methods tensor --run-name paper

A quick check of the whole code path (values are not meaningful)::

    uv run python exp/run_synthetic.py --scenarios synt3 --num-modes 2 \\
        --dims 5 --seeds 1 --quick --parameter-times 24 --samples-per-time 3
"""

from __future__ import annotations

# Sizes the BLAS and numba thread pools before numpy and numba load.
from mapgm import _threading  # isort: skip

import argparse
import itertools
import json
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mapgm import MAPGM, StreamMAPGM  # noqa: E402
from mapgm.common import viz  # noqa: E402
from mapgm.common.metric import (  # noqa: E402
    change_point_metrics,
    edge_recovery_metrics,
    project_flat_network_to_modes,
)
from mapgm.common.synthetic import (  # noqa: E402
    SYNTHETIC_SCENARIO_NAMES,
    SyntheticScenario,
    generate_scenario_data,
    get_scenario,
)
from mapgm.methods.mapgm._kernels import parallel_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMBDA_TABLE = ROOT / "configs" / "synthetic_lambdas.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "synthetic"

METHODS = ("tensor", "stream", "static-base", "flattened")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)

#: The grid searched with --search, as the full product (18 settings).
SEARCH_LAMBDA_REGS = (1e1, 1e2, 1e3)
SEARCH_LAMBDA_TEMPORAL_A = (1e4, 1e5, 1e6)
SEARCH_LAMBDA_TEMPORAL_B = (1e0, 1e1)

#: Seed distance between the search draw and the scenario's own seed.
SEARCH_SEED_OFFSET = 500_000

METRIC_COLUMNS = ("AUC-ROC", "AUC-PR", "Best-F1", "TDR", "CP-recall@K")


@dataclass(frozen=True)
class Lambdas:
    """The three penalty weights of one fit."""

    lambda_regs: float
    lambda_temporal_A: float
    lambda_temporal_b: float

    @property
    def name(self) -> str:
        return (
            f"l1={self.lambda_regs:g}_l2={self.lambda_temporal_A:g}"
            f"_l3={self.lambda_temporal_b:g}"
        )


def lambda_grid(
    lambda_regs: Sequence[float] = SEARCH_LAMBDA_REGS,
    lambda_temporal_A: Sequence[float] = SEARCH_LAMBDA_TEMPORAL_A,
    lambda_temporal_b: Sequence[float] = SEARCH_LAMBDA_TEMPORAL_B,
) -> tuple[Lambdas, ...]:
    """Every combination of the three weights, in a fixed order."""
    return tuple(
        Lambdas(float(l1), float(l2), float(l3))
        for l1, l2, l3 in itertools.product(
            lambda_regs, lambda_temporal_A, lambda_temporal_b
        )
    )


def load_lambda_table(path: Path) -> dict[tuple, Lambdas]:
    """Read the weights per ``(method, scenario, num_modes, dimension)``."""
    table = pd.read_csv(path)
    return {
        (row.method, row.scenario, int(row.num_modes), int(row.dimension)): Lambdas(
            float(row.lambda_regs),
            float(row.lambda_temporal_A),
            float(row.lambda_temporal_b),
        )
        for row in table.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit_mapgm(
    data: dict,
    lambdas: Lambdas,
    *,
    method: str,
    search: bool = False,
    is_print: bool = False,
) -> MAPGM:
    """Fit one MAPGM variant to a scenario's data with the study's settings.

    ``search`` uses the reduced outer-iteration budget of a search fit.
    """
    scenario: SyntheticScenario = data["scenario"]
    flattened = method == "flattened"
    options = dict(
        sum_log_factorials=data["sum_log_factorials"],
        x_summaries=[data["flat_summary"]] if flattened else data["x_summaries"],
        dims=[int(np.prod(scenario.dims))] if flattened else list(scenario.dims),
        lambda_regs=lambdas.lambda_regs,
        lambda_temporal_A=lambdas.lambda_temporal_A,
        lambda_temporal_b=lambdas.lambda_temporal_b,
        temporal_penalty_A="l1",
        temporal_penalty_b="group_l2",
        rho=scenario.mapgm_rho,
        rho_min=1e2,
        rho_max=1e20,
        adaptive_rho=scenario.mapgm_adaptive_rho,
        residual_check_interval=scenario.mapgm_residual_check_interval,
        outer_objective_patience=scenario.mapgm_outer_objective_patience,
        divergence_patience=scenario.mapgm_divergence_patience,
        tau=scenario.samples_per_time,
        normalize_mode_offsets=not flattened,
    )
    if method != "stream":
        model = MAPGM(**options, time_varying_base=method != "static-base")
        model.fit(
            max_iter=scenario.outer_iterations(search=search),
            admm_iter=scenario.mapgm_admm_iter,
            is_print=is_print,
        )
        return model

    times = scenario.parameter_times
    model = StreamMAPGM(
        **options,
        init_len=max(1, min(scenario.mapgm_init_len, times)),
        slide_len=max(0, min(scenario.mapgm_slide_len, times - 1)),
    )
    started = time.time()
    model.initial_update(
        max_iter=scenario.mapgm_stream_init_max_iter,
        admm_iter=scenario.mapgm_admm_iter,
        is_print=is_print,
    )
    model.initial_seconds_ = time.time() - started
    model.window_seconds_ = []
    while True:
        started = time.time()
        advanced = model.streaming_update(
            max_iter=scenario.mapgm_stream_max_iter,
            admm_iter=scenario.mapgm_admm_iter,
            is_print=is_print,
        )
        model.window_seconds_.append(time.time() - started)
        if not advanced:
            break
    return model


def network_estimate(model: MAPGM, scenario: SyntheticScenario, method: str) -> list[np.ndarray]:
    """The estimated ``A`` per mode; the flattened fit is projected onto them."""
    if method == "flattened":
        return project_flat_network_to_modes(np.asarray(model.A[0]), scenario.dims)
    return [np.asarray(value) for value in model.A]


def evaluate(estimate: Sequence[np.ndarray], data: dict) -> dict[str, float]:
    """Signed edge recovery and change-point detection of one estimate."""
    return {
        **edge_recovery_metrics(estimate, data["A_true"]),
        **change_point_metrics(estimate, data["scenario"].change_points),
    }


def search_lambdas(
    search_data: dict,
    method: str,
    grid: Sequence[Lambdas],
    *,
    is_print: bool = False,
) -> tuple[Lambdas, pd.DataFrame]:
    """Fit every setting to the search draw and keep the best AUC-PR.

    Ties go to the earlier setting of the grid.
    """
    rows = []
    for lambdas in grid:
        model = fit_mapgm(search_data, lambdas, method=method, search=True, is_print=is_print)
        metrics = evaluate(network_estimate(model, search_data["scenario"], method), search_data)
        rows.append({"candidate": lambdas.name, **asdict(lambdas), **metrics})
    scores = [
        row["AUC-PR"] if np.isfinite(row["AUC-PR"]) else -np.inf for row in rows
    ]
    best = max(range(len(rows)), key=lambda index: (scores[index], -index))
    table = pd.DataFrame(rows)
    table["selected"] = False
    table.loc[best, "selected"] = True
    return grid[best], table


# ---------------------------------------------------------------------------
# one condition
# ---------------------------------------------------------------------------


def condition_dir(run_dir: Path, scenario: SyntheticScenario) -> Path:
    """``<run>/seed=<s>/<scenario>/M=<M>/dims=<d>``."""
    sizes = list(scenario.dims)
    label = str(sizes[0]) if len(set(sizes)) == 1 else "x".join(map(str, sizes))
    return (
        run_dir / f"seed={scenario.seed}" / scenario.name
        / f"M={scenario.num_modes}" / f"dims={label}"
    )


def _merge_rows(path: Path, frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Add rows to a table on disk, replacing earlier rows with the same keys."""
    if path.is_file():
        previous = pd.read_csv(path)
        mine = set(map(tuple, frame[list(keys)].astype(str).to_numpy().tolist()))
        stale = previous[list(keys)].astype(str).apply(tuple, axis=1).isin(mine)
        frame = pd.concat([previous.loc[~stale], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def run_condition(
    scenario: SyntheticScenario,
    methods: Sequence[str],
    lambdas_by_method: dict[str, tuple[Lambdas, str]],
    output_dir: Path,
    *,
    search_tables: dict[str, pd.DataFrame] | None = None,
    plots: bool = True,
    is_print: bool = False,
) -> pd.DataFrame:
    """Fit every method to one evaluation draw and save everything."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = generate_scenario_data(scenario)
    generation_seconds = time.time() - started
    np.savez_compressed(
        output_dir / "truth.npz",
        **{f"A_true__mode{m + 1}": a for m, a in enumerate(data["A_true"])},
        **{f"b_true__mode{m + 1}": b for m, b in enumerate(data["b_true"])},
    )

    metric_rows = []
    summary_rows = []
    for method in methods:
        lambdas, source = lambdas_by_method[method]
        print(f"  {method}: {lambdas.name} ({source})", flush=True)
        # The search above stays parallel; the reported fit runs on one
        # thread when MAPGM_PARALLEL_FINAL_FIT=0, so its runtime can be timed.
        parallel = _threading.final_fit_parallel_enabled()
        started = time.time()
        with parallel_fit(parallel):
            model = fit_mapgm(data, lambdas, method=method, is_print=is_print)
        runtime = time.time() - started
        estimate = network_estimate(model, scenario, method)
        metrics = evaluate(estimate, data)
        metric_rows.append({"method": method, **metrics})

        diagnostics = model.fit_diagnostics
        row = {
            "method": method,
            "lambda_source": source,
            **asdict(lambdas),
            "runtime_seconds": runtime,
            "final_fit_parallel": parallel,
            "fit_converged": bool(diagnostics.get("all_modes_converged", False)),
            "fit_iterations": diagnostics.get("iterations"),
            "stopped_reason": diagnostics.get("stopped_reason"),
        }
        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)
        if method == "stream":
            seconds = np.asarray(model.window_seconds_, dtype=float)
            row.update(
                initial_seconds=model.initial_seconds_,
                window_count=len(seconds),
                window_seconds_mean=float(seconds.mean()),
                window_seconds_max=float(seconds.max()),
            )
            pd.DataFrame(
                {"window": np.arange(len(seconds)), "seconds": seconds}
            ).to_csv(method_dir / "window_runtimes.csv", index=False)
        summary_rows.append(row)

        saved = {f"A__mode{m + 1}": a for m, a in enumerate(estimate)}
        if method == "flattened":
            saved["b_flat"] = np.asarray(model.b[0])
        else:
            saved.update({f"b__mode{m + 1}": np.asarray(b) for m, b in enumerate(model.b)})
        np.savez_compressed(method_dir / "networks.npz", **saved)
        if search_tables and method in search_tables:
            search_tables[method].to_csv(method_dir / "hyperparameter_search.csv", index=False)
        if plots:
            plots_dir = method_dir / "plots"
            viz.viz_temporal_deviation(estimate, scenario.change_points, save_dir=plots_dir)
            viz.viz_error(estimate, data["A_true"], save_dir=plots_dir)
            if method != "flattened":
                viz.viz_base(model.b, save_dir=plots_dir)
            times = sorted({0, *(t for points in scenario.change_points for t in points)})
            viz.viz_edges(estimate, times, A_true=data["A_true"], save_dir=plots_dir)

    metrics = pd.DataFrame(metric_rows)
    _merge_rows(output_dir / "metrics.csv", metrics, ["method"])
    _merge_rows(output_dir / "fit_summary.csv", pd.DataFrame(summary_rows), ["method"])
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    config.update(
        scenario=asdict(scenario),
        generation_seconds=generation_seconds,
        observed_mean=data["observed_mean"],
        count_scale=data["count_scale"],
        saturated_fraction=data["saturated_fraction"],
        clipped_fraction=data["clipped_fraction"],
    )
    config.setdefault("lambdas", {}).update(
        {
            method: {**asdict(lambdas_by_method[method][0]), "source": lambdas_by_method[method][1]}
            for method in methods
        }
    )
    config_path.write_text(json.dumps(config, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# summaries over seeds
# ---------------------------------------------------------------------------

SUMMARY_KEYS = ("scenario", "num_modes", "dims", "parameter_times", "samples_per_time", "method")


def summarize_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the run's summary tables from its seed directories."""
    frames = [
        pd.read_csv(path)
        for path in sorted(run_dir.glob("seed=*/summary_metrics.csv"))
    ]
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(list(SUMMARY_KEYS), sort=False)
    aggregate = grouped["seed"].nunique().rename("n_seeds").to_frame()
    for column in METRIC_COLUMNS:
        aggregate[f"{column}_mean"] = grouped[column].mean()
        aggregate[f"{column}_std"] = grouped[column].std()
    aggregate = aggregate.reset_index()
    combined.to_csv(run_dir / "summary_metrics.csv", index=False)
    aggregate.to_csv(run_dir / "summary_metrics_mean_std.csv", index=False)
    return combined, aggregate


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenarios", nargs="+", choices=SYNTHETIC_SCENARIO_NAMES,
                        default=list(SYNTHETIC_SCENARIO_NAMES))
    parser.add_argument("--num-modes", nargs="+", type=int, default=[2, 3],
                        help="Mode counts M (default: 2 3).")
    parser.add_argument("--dims", nargs="+", type=int, default=[5, 10, 15],
                        help="Mode sizes d (default: 5 10 15).")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                        help="Seeds of the evaluation draws (default: 1 2 3 4 5).")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["tensor"])
    parser.add_argument("--lambda-table", type=Path, default=DEFAULT_LAMBDA_TABLE,
                        help="Weights per condition (default: the paper's).")
    parser.add_argument("--search", action="store_true",
                        help="Select the weights by grid search instead.")
    parser.add_argument("--lambda-regs", nargs="+", type=float, default=None,
                        help="L1 weight on A_t: one value to fix it, or the grid to search.")
    parser.add_argument("--lambda-temporal-a", nargs="+", type=float, default=None,
                        help="Weight on A_t - A_(t-1): one value, or the grid to search.")
    parser.add_argument("--lambda-temporal-b", nargs="+", type=float, default=None,
                        help="Weight on b_t - b_(t-1): one value, or the grid to search.")
    parser.add_argument("--max-flat-dimension", type=int, default=125,
                        help="Skip the flattened fit above this many cells (0: never).")
    parser.add_argument("--parameter-times", type=int, default=None,
                        help="Override T (change points are rescaled).")
    parser.add_argument("--samples-per-time", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Cut the Gibbs and solver iterations; for a smoke test only.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print every outer iteration.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None, help="Default: a UTC timestamp.")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    explicit = [args.lambda_regs, args.lambda_temporal_a, args.lambda_temporal_b]
    if args.search:
        grid = lambda_grid(
            args.lambda_regs or SEARCH_LAMBDA_REGS,
            args.lambda_temporal_a or SEARCH_LAMBDA_TEMPORAL_A,
            args.lambda_temporal_b or SEARCH_LAMBDA_TEMPORAL_B,
        )
    elif any(value is not None for value in explicit):
        if not all(value is not None and len(value) == 1 for value in explicit):
            parser.error(
                "Without --search, give exactly one value each of --lambda-regs, "
                "--lambda-temporal-a and --lambda-temporal-b."
            )
        fixed = Lambdas(*(value[0] for value in explicit))
    table = None if args.search or any(explicit) else load_lambda_table(args.lambda_table)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_name
    searched: dict[tuple, tuple[Lambdas, pd.DataFrame]] = {}
    failures = []
    for seed in args.seeds:
        for name, num_modes, dimension in itertools.product(
            args.scenarios, args.num_modes, args.dims
        ):
            condition = (name, num_modes, dimension)
            tag = f"[{name} M={num_modes} d={dimension} seed={seed}]"
            try:
                scenario = get_scenario(
                    name,
                    quick=args.quick,
                    dimension=dimension,
                    num_modes=num_modes,
                    parameter_times=args.parameter_times,
                    samples_per_time=args.samples_per_time,
                )
                lambdas_by_method: dict[str, tuple[Lambdas, str]] = {}
                search_tables: dict[str, pd.DataFrame] = {}
                methods = []
                for method in args.methods:
                    if method == "flattened" and 0 < args.max_flat_dimension < dimension**num_modes:
                        print(f"{tag} skip flattened: {dimension**num_modes} cells "
                              f"> --max-flat-dimension {args.max_flat_dimension}")
                        continue
                    # The stream fits the batch objective window by window,
                    # so it takes the tensor MAPGM's weights.
                    source_method = "tensor" if method == "stream" else method
                    if args.search:
                        key = (*condition, source_method)
                        if key not in searched:
                            search_scenario = replace(
                                scenario, seed=scenario.seed + SEARCH_SEED_OFFSET
                            )
                            print(f"{tag} searching {len(grid)} settings for "
                                  f"{source_method} on seed {search_scenario.seed}", flush=True)
                            searched[key] = search_lambdas(
                                generate_scenario_data(search_scenario),
                                source_method,
                                grid,
                                is_print=args.verbose,
                            )
                        lambdas, search_tables[method] = searched[key]
                        source = f"search ({source_method})"
                    elif table is None:
                        lambdas, source = fixed, "command line"
                    else:
                        key = (source_method, name, num_modes, dimension)
                        if key not in table:
                            print(f"{tag} skip {method}: no weights for it in "
                                  f"{args.lambda_table}; use --search")
                            continue
                        lambdas, source = table[key], f"table ({source_method})"
                    lambdas_by_method[method] = (lambdas, source)
                    methods.append(method)
                if not methods:
                    continue
                scenario = replace(scenario, seed=seed)
                print(f"{tag} fitting {', '.join(methods)}", flush=True)
                metrics = run_condition(
                    scenario,
                    methods,
                    lambdas_by_method,
                    condition_dir(run_dir, scenario),
                    search_tables=search_tables,
                    plots=not args.no_plots,
                    is_print=args.verbose,
                )
            except Exception as error:  # noqa: BLE001
                failures.append(tag)
                print(f"{tag} failed: {error}")
                traceback.print_exc()
                continue
            print(metrics.to_string(index=False, float_format="{:.3f}".format), flush=True)
            metrics.insert(0, "scenario", name)
            metrics.insert(1, "num_modes", num_modes)
            metrics.insert(2, "dims", dimension)
            metrics.insert(3, "parameter_times", scenario.parameter_times)
            metrics.insert(4, "samples_per_time", scenario.samples_per_time)
            metrics.insert(5, "seed", seed)
            _merge_rows(
                run_dir / f"seed={seed}" / "summary_metrics.csv",
                metrics,
                [*SUMMARY_KEYS, "seed"],
            )
    if any(run_dir.glob("seed=*/summary_metrics.csv")):
        summarize_run(run_dir)
        print(f"summaries written to {run_dir}")
    if failures:
        print(f"{len(failures)} condition(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
