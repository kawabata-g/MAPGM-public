#!/usr/bin/env bash
# Reproduce the synthetic experiments of the paper.
#
#   bash bin/run_synthetic.sh                # everything
#   bash bin/run_synthetic.sh --no-plots     # extra arguments go to every run
#
# Accuracy goes to results/synthetic/paper/summary_metrics_mean_std.csv, and
# the per-window runtime of MAPGM-Stream to results/synthetic/stream_runtime.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# MAPGM, MAPGM-Stream and the static-base ablation:
# synt2-4 x M in {2, 3} x d in {5, 10, 15} x seeds 1-5.
uv run python exp/run_synthetic.py --methods tensor stream static-base \
    --run-name paper "$@"

# The flattened ablation, where its single mode stays small: M=2, d in {5, 10}.
uv run python exp/run_synthetic.py --methods flattened --num-modes 2 --dims 5 10 \
    --run-name paper "$@"

# The runtime of every MAPGM-Stream window, with the reported fit on one thread.
MAPGM_PARALLEL_FINAL_FIT=0 uv run python exp/run_synthetic.py --methods stream \
    --scenarios synt2 --num-modes 2 3 --dims 5 --seeds 202 \
    --run-name stream_runtime "$@"
