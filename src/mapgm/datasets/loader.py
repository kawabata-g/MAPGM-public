"""Load long-format count CSVs and reduce them to mode marginals.

A long-format file has one row per (time, mode-1 label, ..., mode-M label)
with a non-negative count.  ``TensorDataLoader`` reads such a file and sums
over every mode but one to give the ``(T, d_m)`` design matrices that
:class:`mapgm.MAPGM` consumes.  The counts are passed through as they
are, so every method reads the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln


@dataclass(frozen=True)
class ModeSummaries:
    """Mode-marginal design matrices and the labels that index them."""

    #: One ``(T, d_m)`` matrix per mode.
    summaries: list[np.ndarray]
    #: Size of each mode, ``d_m``.
    dims: list[int]
    #: Column labels of each mode, in the order its matrix uses.
    labels: list[list[str]]
    #: The ``T`` distinct timestamps, ascending.
    times: pd.DatetimeIndex

    @property
    def num_modes(self) -> int:
        return len(self.dims)

    @property
    def parameter_times(self) -> int:
        return len(self.times)


class TensorDataLoader:
    """Read one long-format CSV and reduce it to mode marginals."""

    def __init__(
        self,
        file_path: str,
        index_cols: Sequence[str],
        value_col: str,
        time_col: str = "time",
    ) -> None:
        if not index_cols:
            raise ValueError("index_cols needs at least one mode column.")
        self.file_path = str(file_path)
        self.index_cols = list(index_cols)
        self.value_col = str(value_col)
        self.time_col = str(time_col)

    @property
    def required_cols(self) -> list[str]:
        return [self.time_col, *self.index_cols, self.value_col]

    def load_df(
        self,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
    ) -> pd.DataFrame:
        """Read the file and return the normalized rows this loader needs.

        Normalizing here rather than in each caller means every downstream
        step sees parsed timestamps, ascending order, no missing times, and
        counts that have already been checked for negatives.
        """
        frame = pd.read_csv(self.file_path)
        missing = [
            column
            for column in self.required_cols
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = frame[self.required_cols].copy()
        frame[self.time_col] = pd.to_datetime(
            frame[self.time_col],
            errors="coerce",
        )
        frame = frame.dropna(subset=[self.time_col])
        frame[self.value_col] = frame[self.value_col].astype(float)
        if (frame[self.value_col] < 0).any():
            raise ValueError("Counts must be non-negative.")

        if start_datetime is not None:
            frame = frame[frame[self.time_col] >= pd.to_datetime(start_datetime)]
        if end_datetime is not None:
            frame = frame[frame[self.time_col] <= pd.to_datetime(end_datetime)]
        if frame.empty:
            raise ValueError(
                "No rows remain after filtering; check the requested period."
            )
        return frame.sort_values(self.time_col).reset_index(drop=True)

    def mode_summaries(self, df: pd.DataFrame) -> ModeSummaries:
        """Reduce the long frame to one design matrix per mode.

        The counts are summed over the other modes and nothing else is done
        to them, so every method reads the same numbers.  Times where a
        label is absent contribute zero rather than dropping the row, which
        keeps every mode on the same time axis.
        """
        work = df.copy()
        work["_statistic"] = work[self.value_col].to_numpy(dtype=float)
        times = pd.DatetimeIndex(
            work[self.time_col].drop_duplicates().sort_values()
        )

        summaries: list[np.ndarray] = []
        dims: list[int] = []
        labels: list[list[str]] = []
        for column in self.index_cols:
            pivot = (
                work.pivot_table(
                    index=self.time_col,
                    columns=column,
                    values="_statistic",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reindex(index=times, fill_value=0.0)
                .sort_index(axis=1)
            )
            summaries.append(pivot.to_numpy(dtype=float))
            dims.append(pivot.shape[1])
            labels.append([str(label) for label in pivot.columns])
        return ModeSummaries(
            summaries=summaries,
            dims=dims,
            labels=labels,
            times=times,
        )

    def sum_log_factorials(self, df: pd.DataFrame) -> float:
        """Return the constant ``sum(log(x!))`` term of the log-likelihood."""
        return float(np.sum(gammaln(df[self.value_col].to_numpy(float) + 1.0)))
