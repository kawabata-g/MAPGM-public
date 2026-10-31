"""Fit MAPGM to one of the paper's real datasets.

    car          Google Trends, 8 car brands x 10 countries, weekly
    gadget       Google Trends, 7 gadgets x 10 countries, weekly
    electronics  purchase events of an electronics store, 5 component
                 categories x 10 brands, daily

Each case is a long-format CSV under data/ with one row per (time, mode-1
label, mode-2 label) and a count.  The study period, the labels kept, and the
solver settings of every case are registered in ``CASES`` below.

Results go to <output-dir>/<run-name>/<case>/: the estimated networks and
bases (networks.npz, and as labeled long tables edges.csv and baseline.csv),
the fit summary, and diagnostic plots.

Example::

    uv run python exp/run_case_study.py --case gadget --snapshot-times 399 400 509
"""

from __future__ import annotations

# Sizes the BLAS and numba thread pools before numpy and numba load.
from mapgm import _threading  # noqa: F401  isort: skip

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mapgm import MAPGM, TensorDataLoader  # noqa: E402
from mapgm.common import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "case_study"


@dataclass(frozen=True)
class CaseConfig:
    """Everything that distinguishes one dataset from another."""

    name: str
    long_csv: Path
    time_col: str
    mode_cols: tuple[str, ...]
    value_col: str
    start_datetime: str | None = None
    end_datetime: str | None = None
    #: Keep only the ``k`` labels of a mode with the largest total count, as
    #: ``(mode column, k)`` pairs, applied in order.
    mode_top_k: tuple[tuple[str, int], ...] = ()

    # MAPGM solver settings.
    tau: int = 1
    lambda_regs: float = 1e4
    lambda_temporal_A: float = 1e7
    lambda_temporal_b: float = 1e5
    temporal_penalty_A: str = "l1"
    temporal_penalty_b: str = "group_l2"
    rho: float = 1e3
    adaptive_rho: bool = False
    max_iter: int = 2000
    admm_iter: int = 100
    max_admm_iter: int = 1000


CASES: dict[str, CaseConfig] = {
    "car": CaseConfig(
        name="car",
        long_csv=ROOT / "data" / "google_car.csv.gz",
        time_col="date",
        mode_cols=("query", "geo"),
        value_col="volume",
        start_datetime="2007-01-01",
        end_datetime="2023-12-31",
        lambda_regs=2e4,
        lambda_temporal_A=1e7,
        lambda_temporal_b=1e6,
        rho=1e3,
    ),
    "gadget": CaseConfig(
        name="gadget",
        long_csv=ROOT / "data" / "google_gadget.csv.gz",
        time_col="date",
        mode_cols=("query", "geo"),
        value_col="volume",
        start_datetime="2007-01-01",
        end_datetime="2023-12-31",
        lambda_regs=5e6,
        lambda_temporal_A=5e10,
        lambda_temporal_b=1e4,
        rho=1e7,
        adaptive_rho=True,
    ),
    "electronics": CaseConfig(
        name="electronics",
        long_csv=ROOT / "data" / "electronics.csv.gz",
        time_col="event_time",
        mode_cols=("category_code", "brand"),
        value_col="count",
        # The top 10 of 143 brands hold 87% of the counts; of the categories
        # left, the top 5 hold all but 0.2% of theirs.
        mode_top_k=(("brand", 10), ("category_code", 5)),
        lambda_regs=1e0,
        lambda_temporal_A=1e1,
        lambda_temporal_b=1e0,
        rho=1e3,
        max_iter=10000,
    ),
}


def keep_top_labels(frame: pd.DataFrame, case: CaseConfig) -> pd.DataFrame:
    """Apply ``case.mode_top_k``: keep each mode's most counted labels."""
    for column, k in case.mode_top_k:
        totals = frame.groupby(column)[case.value_col].sum()
        # Ties broken by label so the selection does not depend on row order.
        ranked = sorted(totals.items(), key=lambda item: (-item[1], str(item[0])))
        keep = [label for label, _ in ranked[: int(k)]]
        frame = frame[frame[column].isin(keep)]
    return frame.reset_index(drop=True)


def load_case(case: CaseConfig):
    """Read a case and reduce it to the mode marginals MAPGM fits."""
    loader = TensorDataLoader(
        file_path=str(case.long_csv),
        index_cols=list(case.mode_cols),
        value_col=case.value_col,
        time_col=case.time_col,
    )
    frame = loader.load_df(
        start_datetime=case.start_datetime,
        end_datetime=case.end_datetime,
    )
    frame = keep_top_labels(frame, case)
    return loader.mode_summaries(frame), loader.sum_log_factorials(frame)


def fit_case(case: CaseConfig, summaries, sum_log_factorials: float, *, is_print: bool = False) -> MAPGM:
    """Fit MAPGM with the case's registered settings."""
    model = MAPGM(
        sum_log_factorials=sum_log_factorials,
        x_summaries=summaries.summaries,
        dims=list(summaries.dims),
        lambda_regs=case.lambda_regs,
        lambda_temporal_A=case.lambda_temporal_A,
        lambda_temporal_b=case.lambda_temporal_b,
        temporal_penalty_A=case.temporal_penalty_A,
        temporal_penalty_b=case.temporal_penalty_b,
        rho=case.rho,
        adaptive_rho=case.adaptive_rho,
        outer_objective_patience=10,
        tau=case.tau,
    )
    model.fit(
        max_iter=case.max_iter,
        admm_iter=case.admm_iter,
        max_admm_iter=case.max_admm_iter,
        is_print=is_print,
    )
    return model


def labeled_tables(case: CaseConfig, summaries, model: MAPGM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The nonzero edges and the bases as long tables with labels and dates."""
    times = summaries.times
    edges = []
    bases = []
    for mode, labels in enumerate(summaries.labels):
        matrices = np.asarray(model.A[mode])
        upper = np.triu_indices(matrices.shape[1], k=1)
        for position in range(matrices.shape[0]):
            weights = matrices[position][upper]
            for source, target, weight in zip(upper[0], upper[1], weights):
                if abs(weight) > 1e-14:
                    edges.append((mode + 1, case.mode_cols[mode], position,
                                  times[position], labels[source], labels[target], float(weight)))
        base = np.asarray(model.b[mode])
        for position in range(base.shape[0]):
            for index, label in enumerate(labels):
                bases.append((mode + 1, case.mode_cols[mode], position,
                              times[position], label, float(base[position, index])))
    return (
        pd.DataFrame(edges, columns=["mode", "mode_column", "time_index", "date",
                                     "source", "target", "weight"]),
        pd.DataFrame(bases, columns=["mode", "mode_column", "time_index", "date",
                                     "label", "base"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--start", default=None, help="First date (default: the case's).")
    parser.add_argument("--end", default=None, help="Last date (default: the case's).")
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--lambda-regs", type=float, default=None)
    parser.add_argument("--lambda-temporal-a", type=float, default=None)
    parser.add_argument("--lambda-temporal-b", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--admm-iter", type=int, default=None)
    parser.add_argument("--snapshot-times", nargs="+", type=int, default=[-1],
                        help="Parameter times of the edge heatmaps; negative "
                             "counts from the end (default: -1).")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print every outer iteration.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None, help="Default: a UTC timestamp.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    overrides = {
        field: value
        for field, value in (
            ("start_datetime", args.start),
            ("end_datetime", args.end),
            ("tau", args.tau),
            ("lambda_regs", args.lambda_regs),
            ("lambda_temporal_A", args.lambda_temporal_a),
            ("lambda_temporal_b", args.lambda_temporal_b),
            ("rho", args.rho),
            ("max_iter", args.max_iter),
            ("admm_iter", args.admm_iter),
        )
        if value is not None
    }
    case = replace(CASES[args.case], **overrides)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_name / case.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, sum_log_factorials = load_case(case)
    print(f"[{case.name}] T={summaries.parameter_times} dims={summaries.dims} "
          f"{summaries.times[0]:%Y-%m-%d}..{summaries.times[-1]:%Y-%m-%d}", flush=True)
    started = time.time()
    model = fit_case(case, summaries, sum_log_factorials, is_print=args.verbose)
    runtime = time.time() - started
    diagnostics = model.fit_diagnostics
    summary = {
        "runtime_seconds": runtime,
        "fit_converged": bool(diagnostics.get("all_modes_converged", False)),
        "fit_iterations": diagnostics.get("iterations"),
        "stopped_reason": diagnostics.get("stopped_reason"),
        "nonzero_fraction": float(np.mean([np.mean(np.abs(a) > 1e-14) for a in model.A])),
    }
    print(f"[{case.name}] " + ", ".join(f"{k}={v}" for k, v in summary.items()), flush=True)

    pd.DataFrame([summary]).to_csv(output_dir / "fit_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "networks.npz",
        **{f"A__mode{m + 1}": np.asarray(a) for m, a in enumerate(model.A)},
        **{f"b__mode{m + 1}": np.asarray(b) for m, b in enumerate(model.b)},
    )
    edges, bases = labeled_tables(case, summaries, model)
    edges.to_csv(output_dir / "edges.csv", index=False)
    bases.to_csv(output_dir / "baseline.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(
        {
            "case": asdict(case),
            "parameter_times": summaries.parameter_times,
            "dims": list(summaries.dims),
            "labels": summaries.labels,
            "period": [str(summaries.times[0]), str(summaries.times[-1])],
        },
        indent=2,
        default=str,
    ))
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        viz.viz_temporal_deviation(model.A, save_dir=plots_dir)
        viz.viz_base(model.b, save_dir=plots_dir)
        times = sorted({t % summaries.parameter_times for t in args.snapshot_times})
        viz.viz_edges(model.A, times, labels=summaries.labels, save_dir=plots_dir)
    print(f"[{case.name}] saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
