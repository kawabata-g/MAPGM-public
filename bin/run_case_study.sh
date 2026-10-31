#!/usr/bin/env bash
# Fit MAPGM to the three real datasets of the paper.
#
#   bash bin/run_case_study.sh
#
# Results go to results/case_study/paper/<case>/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python exp/run_case_study.py --case car --run-name paper "$@"
# The week before and of the Apple Watch announcement, and after Pixel's.
uv run python exp/run_case_study.py --case gadget --run-name paper \
    --snapshot-times 399 400 509 "$@"
uv run python exp/run_case_study.py --case electronics --run-name paper "$@"
