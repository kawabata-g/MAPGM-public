# MAPGM: Multi-Aspect Poisson Graphical Model

Implementation of the Multi-Aspect Poisson Graphical Model (MAPGM), with the
code that reproduces the experiments of the paper.

MAPGM takes a sequence of count tensors (for example, search volume by query ×
country × week) and estimates, for every mode and every time,

- a **dependency network** `A_t^(m)`: a sparse, symmetric, signed matrix over
  the labels of mode `m`, and
- a **base** `b_t^(m)`: the activity level of each label,

so that changes in activity are separated from changes in dependency
structure. Each mode is fitted in turn by ADMM while the others are held
fixed. A streaming variant, `StreamMAPGM`, updates the estimate one time
point at a time.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.
`uv.lock` pins the package versions used for the paper.

```bash
git clone https://github.com/kawabata-g/MAPGM-public.git
cd MAPGM-public
uv sync
```

`pip install -e .` also works, without the pinned versions.

## Quick start

```python
from mapgm import MAPGM
from mapgm.common.metric import edge_recovery_metrics
from mapgm.common.synthetic import generate_scenario_data, get_scenario

# A small synthetic tensor sequence: 2 modes of size 5, 60 times, 10 samples each.
scenario = get_scenario("synt3", num_modes=2, dimension=5,
                        parameter_times=60, samples_per_time=10, seed=1)
data = generate_scenario_data(scenario)

model = MAPGM(
    sum_log_factorials=data["sum_log_factorials"],
    x_summaries=data["x_summaries"],    # one (N, d_m) array per mode
    dims=list(scenario.dims),
    lambda_regs=10.0,                   # L1 on A_t
    lambda_temporal_A=1e4,              # l1 on A_t - A_(t-1)
    lambda_temporal_b=1.0,              # l2 on b_t - b_(t-1)
    temporal_penalty_b="group_l2",
    rho=1e10, rho_min=1e2, rho_max=1e20, adaptive_rho=True,
    tau=scenario.samples_per_time,      # samples sharing one parameter time
)
model.fit(max_iter=100, admm_iter=120)

print(model.A[0].shape, model.b[0].shape)   # (60, 5, 5) (60, 5)
print(edge_recovery_metrics(model.A, data["A_true"]))
```

## Input format

`MAPGM` reads the **mode marginals** of the tensors rather than the tensors
themselves:

- `x_summaries[m]` has shape `(N, d_m)`: for each of the `N` samples, the
  counts summed over every mode but `m`. Every mode must have the same total
  per sample.
- `sum_log_factorials` is `sum(log(x!))` over every cell of every sample, the
  constant term of the Poisson log-likelihood. It only shifts the reported
  objective.
- `tau` consecutive samples share one parameter time, so the model has
  `T = ceil(N / tau)` times: `A[m]` has shape `(T, d_m, d_m)` and `b[m]` shape
  `(T, d_m)`. The loss of a time is the mean over its samples, so a weight
  `lambda` means the same for any `tau`.

`TensorDataLoader` builds these from a long-format CSV with one row per
(time, mode-1 label, ..., mode-M label) and a count:

```python
from mapgm import TensorDataLoader

loader = TensorDataLoader("data/google_gadget.csv.gz",
                          index_cols=["query", "geo"],
                          value_col="volume", time_col="date")
frame = loader.load_df(start_datetime="2007-01-01", end_datetime="2023-12-31")
summaries = loader.mode_summaries(frame)          # .summaries, .dims, .labels, .times
constant = loader.sum_log_factorials(frame)
```

## Model options

The objective of each fit is

```text
sum_t (1/N_t) sum_{n in t} { 1' exp(eta_{t,n}) - x_{t,n}' eta_{t,n} }
    + lambda_regs       * sum_t ||A_t||_1
    + lambda_temporal_A * sum_t phi(A_t - A_(t-1))
    + lambda_temporal_b * sum_t psi(b_t - b_(t-1))
```

| Option | Default | Meaning |
|---|---|---|
| `lambda_regs`, `lambda_temporal_A`, `lambda_temporal_b` | — , 0, 0 | Penalty weights; a scalar or one value per mode |
| `temporal_penalty_A`, `temporal_penalty_b` | `"l1"` | `phi` and `psi`: `"l1"` (element-wise, individual changes) or `"group_l2"` (the unsquared Euclidean / Frobenius norm of the whole difference, coordinated changes) |
| `tau` | 1 | Consecutive samples sharing one parameter time |
| `rho` | 1.0 | ADMM penalty, common to every constraint block |
| `adaptive_rho` | `False` | Balance the primal and dual residuals by adjusting `rho` within `[rho_min, rho_max]` |
| `variable_scaling` | `True` | Solve internally for `A / scale_A`, with `scale_A = 1 / max(RMS(x_m), 1)` per mode, so that `A` and `b` have comparable magnitudes. The objective is unchanged; `scale_A` and `scale_b` can be given explicitly |
| `time_varying_base` | `True` | `False` fits one base shared by every time |
| `normalize_mode_offsets` | `True` | Fix the constant offset between the bases of different modes, which the likelihood cannot identify |

`model.fit(max_iter, admm_iter)` runs `max_iter` sweeps over the modes, each
with at most `admm_iter` inner ADMM iterations per mode. The diagnostics are in
`model.history` and `model.fit_diagnostics`.

`StreamMAPGM` takes the same options plus `init_len` and `slide_len` (in
parameter times). `initial_update` fits the first `init_len` times; each
`streaming_update` then advances one time and refits the last `slide_len + 1`
times, with the time before them held fixed:

```python
from mapgm import StreamMAPGM

model = StreamMAPGM(..., init_len=100, slide_len=100)
model.initial_update(max_iter=100, admm_iter=120)
while model.streaming_update(max_iter=10, admm_iter=120):
    pass
```

## Reproducing the experiments

| Experiment | Command | Output |
|---|---|---|
| Synthetic accuracy (MAPGM, MAPGM-Stream, ablations) | `bash bin/run_synthetic.sh` | `results/synthetic/paper/summary_metrics_mean_std.csv` |
| Runtime per MAPGM-Stream window | (included above) | `results/synthetic/stream_runtime/` |
| Case studies | `bash bin/run_case_study.sh` | `results/case_study/paper/<case>/` |

**Synthetic data.** The scenarios `synt2`, `synt3` and `synt4` differ in the
temporal shape of the base (linear, trigonometric, rectangular). The counts are
drawn from the sublinear Poisson graphical model by Gibbs sampling, with
signed Erdős–Rényi networks that change at fixed change points. Every
condition (scenario × M ∈ {2, 3} × d ∈ {5, 10, 15}) is run on seeds 1–5, and
the tables report the mean and standard deviation over them. A draw depends
on its seed alone, not on the thread count.

**Regularization weights.** `exp/run_synthetic.py` takes the weights selected
for the paper from `configs/synthetic_lambdas.csv` (MAPGM-Stream uses the
MAPGM ones). `--search` selects them again: every setting of the grid
`lambda_regs ∈ {10, 100, 1000}`, `lambda_temporal_A ∈ {1e4, 1e5, 1e6}`,
`lambda_temporal_b ∈ {1, 10}` is fitted to a separate draw of the same
scenario (seed + 500000), and the one with the best AUC-PR is used for every
seed.

**Metrics.** Edge recovery (AUC-ROC, AUC-PR, Best-F1) scores positive and
negative edges as separate channels, so an edge estimated with the wrong sign
does not count. Change-point detection uses the temporal deviation
`||A_t - A_(t-1)||_F / ||A_t||_F`: TDR is its mean at the true change points
divided by its mean over all times, and Recall@K the share of the true change
points among the K largest deviations (within one time).

**Runtime.** On a 96-core machine the final fits of the synthetic grid took
about 7 hours for MAPGM, 15 for MAPGM-Stream, 5 for the static-base ablation
and 39 for the flattened ablation. Every experiment script also takes `--quick` for a smoke
test, and `exp/run_synthetic.py --help` lists the options to run part of the
grid.

**Threads.** The natural-parameter update and the Gibbs sampler run in
parallel through numba. `NUMBA_NUM_THREADS` sets the thread count (at most 32
by default), `MAPGM_PARALLEL_FIT=0` makes the fit serial,
`MAPGM_PARALLEL_FINAL_FIT=0` only the reported fit (not the search), and
`MAPGM_PARALLEL_GIBBS=0` the sampler. The BLAS is pinned to one thread
(`MAPGM_BLAS_THREADS`). Serial and parallel runs give identical results.

**Plots.** Every run saves diagnostic plots next to its results (disable with
`--no-plots`): the temporal deviation of `A_t`, the error against the truth
(synthetic data only), the bases `b_t`, and heatmaps of `A_t` at chosen times.

## Data

| File | Content | Source |
|---|---|---|
| `data/google_car.csv.gz` | Weekly search interest in 8 car brands in 10 countries | Google Trends |
| `data/google_gadget.csv.gz` | Weekly search interest in 7 gadgets in 10 countries | Google Trends |
| `data/electronics.csv.gz` | Daily event counts of computer components by category and brand, 2020-09-25 to 2021-02-28 | [eCommerce events history in electronics store](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store), collected by the Open CDP project of [REES46](https://rees46.com) |

## Repository layout

```text
src/mapgm/
  methods/mapgm/   MAPGM (model.py), StreamMAPGM (stream.py), numba kernels
  datasets/        TensorDataLoader for long-format CSVs
  common/          synthetic data, metrics, diagnostic plots
exp/               experiment scripts (synthetic, case studies)
bin/               shell scripts that run the paper's experiments
configs/           regularization weights selected for the paper
data/              case-study datasets
```

## Citation

To be added.

## License

To be added.
