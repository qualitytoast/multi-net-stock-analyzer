# Det-MASO reproduction (MVP, CSI300)

A faithful reproduction of **Det-MASO** (Wu and Yang, arXiv 2610.34936, Oct 2026): five MLP agents
tied to K-means market regimes, an attention scheduler that mixes their per-stock scores, a
log-determinant diversity penalty, a top-decile long-only backtest and a Shapley interaction
analysis. The MVP's scope was the paper's method only. Follow-up experiments ([TODO.md](TODO.md))
add baselines, ablations and improvements, each behind a config flag, without changing the paper
configuration's results (a test pins its fingerprint to the one recorded when the test period was opened).

- The full design is in [docs/PLAN.md](docs/PLAN.md): a spec table tying every component to the paper,
  about 45 documented defaults for details the paper leaves out (IDs A1-A45, each a value in
  `src/detmaso/config.py` / `configs/csi300.yaml`), the test list, success criteria, and 27 verified
  internal inconsistencies in the paper.
- Changes made to those planned defaults during the build are logged with evidence in
  [DEVIATIONS.md](DEVIATIONS.md).
- Every training run is logged in [EXPERIMENTS.md](EXPERIMENTS.md); finished milestones are in
  [ROADMAP.md](ROADMAP.md) and open work in [TODO.md](TODO.md).

## Status

CSI300 MVP complete. The test period (2024-03-01 to 2026-03-01) was evaluated once, on 2026-11-06,
with the frozen configuration and all 5 seeds; the opening is recorded in
[locks/csi300_test.json](locks/csi300_test.json). **Verdict: not reproduced**: the paper's
Det-MASO numbers, and even its own single-MLP floor, were not reached (details below).

Main findings: the paper's objective makes the scheduler collapse onto the two agents tied to the
most common regimes, and the agents tied to rare regimes get almost no learning signal. Making
training run at all required four documented changes to planned defaults (GELU, decoupled weight
decay, float64, a defined score for constant agent output); a fifth change concerns only the
excess-return benchmark. All five were approved by the project owner (see DEVIATIONS.md).
S&P 500 is deferred until a point-in-time data source is chosen.

**Step 1 (2026-11-06): does the 5-agent idea help?** No, not on our data. On the validation period, a
single plain network the size of one Det-MASO agent, trained with only the IC and Sharpe loss terms,
scored better than Det-MASO on every metric (IC 0.103 vs 0.062, Sharpe 0.66 vs 0.02, max drawdown
16.7% vs 28.0%). By the decision rule fixed before the runs the gap is not statistically resolved on
278 validation days ("no detectable difference"), so Det-MASO does not beat a single network. A
Det-MASO-sized single network does the same; switching off the diversity penalty changes nothing
detectable. Details: [write-up](results/csi300/single_vs_ensemble/README.md).

The market data is not included in this repository: it is downloaded by `scripts/get_cn_data.sh`
from the community Qlib dataset (chenditc/investment_data), whose data has no explicit license, so
the dataset is not redistributed here.

## Results

- [Test report, 5 seeds](results/csi300/test/report.md) (the reproduction result):

  | | Ours (mean +- std) | Paper (Table 5) |
  |---|---|---|
  | IC | 0.038 +- 0.008 | 0.081 |
  | ICIR | 0.18 | 0.62 |
  | Sharpe | 0.81 +- 0.29 | 1.85 |
  | Annual return | 14.7% +- 6.0% | 31.5% |
  | Max drawdown | 19.1% +- 3.0% | 8.5% |

  Against the equal-weighted average of the stocks the portfolio could trade, the portfolio returned
  about -4% a year (seeds range from -10% to -0.1%). On Mar 2024 to Apr 2025, the span the paper's
  Figure 4 actually covers, mean IC was 0.058. Scheduler entropy was 0.03 of a possible 1.61 (collapsed);
  I(Low-Vol Bull, Tail-Risk) > 0 in 4 of 5 seeds, the largest pair in 2.
- [Validation report, 5 seeds](results/csi300/validation/report.md) (development): IC 0.062 +- 0.010,
  ICIR 0.42; absolute portfolio returns flat in the falling 2023 market.
- [Data QA report](results/csi300/data_report.md): splits, universe sizes, spot-price checks, missing
  values, silhouette curve (K = 2 is optimal; the paper's 5 is kept per decision D2), regime coverage.
- [Step 1: Det-MASO vs a single network](results/csi300/single_vs_ensemble/README.md) (validation only,
  5 seeds per model): the answer paragraph, results with 95% bootstrap ranges, figures, assumptions.

## Setup

```bash
# Python 3.12 is required (pyqlib 0.9.7 has no 3.13 wheels); uv installs it for you.
uv sync
./scripts/get_cn_data.sh        # pinned community Qlib CN dataset, verified by sha256 (about 570 MB)
uv run pytest                   # 93 tests; data-dependent ones skip until `prepare` has run
```

## Run

```bash
uv run detmaso prepare  --config configs/csi300.yaml                  # features, regimes, panel, QA report
uv run detmaso train    --config configs/csi300.yaml --seeds 0 1 2 3 4 --tag final
uv run detmaso evaluate --config configs/csi300.yaml --seeds 0 1 2 3 4 --tag final --split valid
uv run detmaso evaluate --config configs/csi300.yaml --seeds 0 1 2 3 4 --tag final --split test --unlock-test
```

Variants of the paper config live in `configs/` and inherit it with `base: csi300.yaml`, overriding only
what differs (for example `configs/csi300_nodiv.yaml` sets `loss.c_div: 0.0`; `configs/csi300_single_mlp.yaml`
switches to a single network). A run group (`--tag`) belongs to one config: training refuses to add to a
group trained with a different config, and evaluation refuses a mismatched config. Variants are compared on
the validation period only:

```bash
uv run detmaso train    --config configs/csi300_single_mlp.yaml --tag single_mlp
uv run detmaso evaluate --config configs/csi300_single_mlp.yaml --tag single_mlp --split valid
uv run detmaso compare  --runs configs/csi300.yaml:final configs/csi300_single_mlp.yaml:single_mlp \
                        --reference final --out results/csi300/single_vs_ensemble
```

`compare` reports each model's metrics (mean +- std over seeds) and the reference's advantage over each
variant with a 95% range from a paired block bootstrap (validation days redrawn in 20-day blocks shared by
all models, seeds redrawn per model), plus an early-stopping check. It refuses the test split.

The test period (2024-03-01 to 2026-03-01) is locked. Evaluating it needs `--unlock-test` and an
explicit `--tag`, always uses all configured seeds, and is recorded in the git-tracked
`locks/csi300_test.json` with the config and data hashes, a fingerprint of the source code and the
sha256 of every model. Re-running an identical evaluation is allowed; anything different is refused
unless `--extra-test-run "reason"` records it. Development uses the training and validation periods only.

## Outputs

| Path | Content |
|---|---|
| `runs/csi300/prepared/data_report.md` | data QA: splits, universe sizes, spot prices, missing values, silhouette curve, regime coverage vs the paper, training-period single-feature ICs |
| `runs/csi300/<tag>/seed<k>/` | `model.pt` (best epoch), `history.jsonl` (per-epoch diagnostics), `summary.json` |
| `runs/csi300/<tag>/report_<split>.md` | metrics vs the paper (paper config only), per-seed table, diversity, Shapley matrix, qualitative checks (multi-agent models), verdict (paper config, test only), figures |
| `runs/csi300/<tag>/series_<split>.parquet` | per seed and decision date: daily IC, portfolio and benchmark returns, scheduler weights and entropy (input to `compare`) |
| `<compare --out>/` | `comparison.md`, `comparison.json`, `figures/` (metrics by seed, advantage ranges, cumulative return) |

## Layout

```
configs/               csi300.yaml (the paper config) and variants that inherit it with `base:`
src/detmaso/
  config.py            typed config; every assumption is a named value; post-paper flags, `base:` inheritance
  data/                qlib_io, returns, labels, features (34), market_state (12), regimes, splits, panel, qa
  model/               agent MLP, scheduler (window encoders + attention + head), DetMASO,
                       single network baseline (single.py), factory (model.kind)
  losses/              soft rank (Blondel et al. L2, exact PAV), masked statistics, the full objective
  train/trainer.py     Algorithm 1: one Adam(W) step per date, cosine LR, clipping, early stopping
  eval/                metrics, top-decile backtest, Shapley interaction, report, test lock,
                       compare (+ compare_plots): variants side by side on validation
  cli.py               prepare | train | evaluate | compare
tests/                 the plan's 30 checks plus the post-MVP ones (frozen paper config, golden training
                       run, single network, config inheritance, comparison statistics)
```

## Copyright

Copyright (c) 2026 Nathan Hung. All rights reserved. This code is shared for viewing only; any other use requires permission.
