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
- Every configuration trained with validation switched on is counted as a try in
  [results/csi300/trials.csv](results/csi300/trials.csv), and each change is judged by rules written in
  [docs/RULES.md](docs/RULES.md) before its first run (see "Guards" in CLAUDE.md).

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

**Scheduler collapse fix (2026-11-07): dropped.** A load-balancing penalty (each agent's average scheduler
weight is pulled toward its market mood's share of the training days) fixes the collapse in the usage sense: from
strength 0.1 up, every agent gets at least half its target share in every seed. The agents mostly become mood
specialists, but not reliably in every seed, mainly for the rarest mood (Tail-Risk: 59 training days, all in
2015), which the rule fixed in advance does not allow; the owner kept the rule as written, so the change was
dropped (the penalty stays in the code, off by default). Validation IC did not change detectably either
(0.036-0.072 vs 0.062; the validation period contains only the two common moods). A placebo run on shuffled answers
found nothing, as it should. Details: [write-up](results/csi300/scheduler_balance/README.md).
Every result now carries its try number, deflated Sharpe and deflated IC (see "Guards" in CLAUDE.md and
[docs/RULES.md](docs/RULES.md)).

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
- [Scheduler collapse fix](results/csi300/scheduler_balance/README.md) (validation and training days, 5 seeds per
  model): the penalty, the placebo, the strength pilot, the rules and their outcomes, scheduler usage and routing.

## What didn't work

Every idea tried so far that did not help, with its result and why it was dropped (guard "report everything";
try numbers from [results/csi300/trials.csv](results/csi300/trials.csv), validation numbers as mean +- std over
5 seeds with 95% ranges from the paired 20-day block resampling).

| Idea | Try # | Result | Why it was dropped | Details |
|---|---|---|---|---|
| ReLU agents (the planned default) | 1-2 | Training aborted in epoch 2-3: a rare-mood agent's last hidden layer switched off for every stock | Unstable; replaced by GELU | DEVIATIONS 1, E1 |
| Adam with coupled L2 weight decay (planned default) | 1, 3 | Rare-mood agents' output scale shrank about 35% per epoch | Unstable; replaced by AdamW | DEVIATIONS 2, E1 |
| float32 networks (planned default) | 1-4 | Starved agents' output spread fell below float32 precision | Replaced by float64 | DEVIATIONS 3, E1 |
| Aborting training when an agent's output is constant (planned safeguard) | 5 | Training aborted in epoch 11 | Replaced by "score 0 for that day" | DEVIATIONS 4, E1 |
| Det-MASO without the diversity penalty | 7 | IC 0.088 +- 0.058 vs 0.062 +- 0.010; Det-MASO's lead -0.026 [-0.086, +0.031] | No detectable gain; seeds very unstable (IC 0.035 to 0.181) | E5 |
| A Det-MASO-sized single network | 9 | IC 0.101 +- 0.007 vs 0.103 +- 0.002 for the one-agent size | More weights changed nothing | E7 |
| Det-MASO's 5-agent machinery itself, as the paper trains it | 6 vs 8 | IC 0.062 +- 0.010 vs 0.103 +- 0.002 for one plain network; Det-MASO's lead -0.041 [-0.088, +0.011] | Does not beat a single network on our data (still the reproduction's subject, so kept as the reference) | [Step 1](results/csi300/single_vs_ensemble/README.md) |
| Load-balancing penalty on the scheduler (strengths 0.03-10) | 11-16 | Usage fixed from 0.1 up; specialists not reliable (mostly Tail-Risk); IC 0.036-0.072, no detectable difference from 0.062; deflated IC at most 72% | Its pre-written rule found no reliable specialists ("all agents used, but no specialists"); the owner kept the rule as written | [write-up](results/csi300/scheduler_balance/README.md), E12-E17 |

## Setup

```bash
# Python 3.12 is required (pyqlib 0.9.7 has no 3.13 wheels); uv installs it for you.
uv sync
./scripts/get_cn_data.sh        # pinned community Qlib CN dataset, verified by sha256 (about 570 MB)
uv run pytest                   # 160 tests; data-dependent ones skip until `prepare` has run
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
all models, seeds redrawn per model from the model's own random stream, so a model's range does not depend on
which other models are compared), plus an early-stopping check. It refuses the test split.

Guards against fooling ourselves (CLAUDE.md "Guards", rules in [docs/RULES.md](docs/RULES.md)):

- **Try ledger.** `train` registers every new configuration in `results/<market>/trials.csv` before training
  starts (aborted runs count too) and `evaluate --split valid` adds its validation metrics. `compare` then reports
  each model's try number, a 95% range for its IC and Sharpe, a deflated Sharpe ratio (on the daily excess return)
  and a deflated IC: the probability that the value beats the best of that many tries of pure luck.
- **Rules as code.** `compare --rule scheduler_balance` applies a change's pre-written rules mechanically and writes
  their outcome into the comparison.
- **Placebo.** A config with `placebo: {shuffle_seed: 0}` trains and evaluates on shuffled answers (each 20-day
  block's stocks swap their future returns), so there is nothing real to find; placebo runs are never compared with
  real ones and can never open the test period.
- **Training-only runs.** `train.validate: false` trains a fixed number of epochs without ever looking at the
  validation period (used for the strength pilot); `evaluate` refuses such run groups.

```bash
# training-day scheduler usage, routing by mood and the Rule M checks (never touches validation)
uv run detmaso usage   --runs configs/scheduler_balance/bal_c1e0.yaml:bal_c1e0 [--grid] [--out DIR]
# a sweep judged by its rules (the try ledger is read from results/csi300/trials.csv)
uv run detmaso compare --rule scheduler_balance --reference final --out results/csi300/scheduler_balance \
                       --runs configs/csi300.yaml:final configs/scheduler_balance/bal_c3e-2.yaml:bal_c3e-2 ...
```

The load-balancing configs live in `configs/scheduler_balance/`: the strength pilot in `pilot/`, the six grid
strengths as `bal_c*.yaml`, and the placebo versions in `placebo/`. The full command sequence is in the
[write-up](results/csi300/scheduler_balance/README.md).

The test period (2024-03-01 to 2026-03-01) is locked. Evaluating it needs `--unlock-test` and an
explicit `--tag`, always uses all configured seeds, and is recorded in the git-tracked
`locks/csi300_test.json` with the config and data hashes, a fingerprint of the source code and the
sha256 of every model. Re-running an identical evaluation is allowed; anything different is refused
unless `--extra-test-run "reason"` records it. Development uses the training and validation periods only.

## Outputs

| Path | Content |
|---|---|
| `runs/csi300/prepared/data_report.md` | data QA: splits, universe sizes, spot prices, missing values, silhouette curve, regime coverage vs the paper, training-period single-feature ICs |
| `runs/csi300/<tag>/seed<k>/` | `model.pt` (restored best epoch), `model_last.pt` (last epoch trained), `history.jsonl` (per-epoch diagnostics; with the balance penalty also its value, training-day usage and gradient-clipping statistics), `summary.json` |
| `runs/csi300/<tag>/report_<split>.md` | metrics vs the paper (paper config only), per-seed table, diversity, Shapley matrix, qualitative checks (multi-agent models), verdict (paper config, test only), figures |
| `runs/csi300/<tag>/series_<split>.parquet` | per seed and decision date: daily IC, portfolio and benchmark returns, scheduler weights and entropy (input to `compare`) |
| `runs/csi300/<tag>/usage_train.parquet` | from `usage`: per seed, checkpoint (restored or last) and training day: the day's mood, every agent's scheduler weight and dead flag |
| `<compare --out>/` | `comparison.md`, `comparison.json`, `figures/` (metrics by seed, advantage ranges, cumulative return; with `--rule` also the strength sweep, usage relative to target and routing by mood) |
| `results/csi300/trials.csv` | the try ledger (git-tracked): every trained configuration, its kind (counted try, placebo or training-only), run groups, seed statuses and validation metrics |
| `docs/RULES.md` | the decision rules, written before each change's first run, and the grids set by pilots |

## Layout

```
configs/               csi300.yaml (the paper config) and variants that inherit it with `base:`;
                       scheduler_balance/ (pilot, grid strengths, placebo versions)
docs/                  PLAN.md (the MVP design), RULES.md (decision rules fixed before each change's first run)
src/detmaso/
  config.py            typed config; every assumption is a named value; post-paper flags, `base:` inheritance,
                       placebo section
  data/                qlib_io, returns, labels, features (34), market_state (12), regimes, splits, panel, qa,
                       placebo (answers shuffled in 20-day blocks)
  model/               agent MLP, scheduler (window encoders + attention + head), DetMASO,
                       single network baseline (single.py), factory (model.kind)
  losses/              soft rank (Blondel et al. L2, exact PAV), masked statistics, the full objective,
                       balance (scheduler load-balancing penalty)
  train/trainer.py     Algorithm 1: one Adam(W) step per date, cosine LR, clipping, early stopping;
                       optional balance penalty and training-only runs
  eval/                metrics, top-decile backtest, Shapley interaction, report, test lock,
                       compare (+ compare_plots): variants side by side on validation,
                       ledger (try ledger), guard (deflated Sharpe and IC, the rules as code),
                       scheduler_diag (training-day usage and routing)
  cli.py               prepare | train | evaluate | usage | compare
tests/                 the plan's 30 checks plus the post-MVP ones (frozen paper config, golden training
                       run, single network, config inheritance, comparison statistics, balance penalty,
                       placebo, deflated metrics, try ledger, rules)
```

## Copyright

Copyright (c) 2026 Nathan Hung. All rights reserved. This code is shared for viewing only; any other use requires permission.
