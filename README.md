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
S&P 500 is deferred: the free data route failed its coverage check (see "More data" below).

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

**Walk-forward evaluation (2026-11-08): built, not yet used on real answers.** Every later change is judged on 11
yearly folds (test years 2013-2023; each fold trains from scratch on 2008 up to two years before its test year and
early-stops on the year before), with trading costs (realistic China costs: buys 8 bps, sells 18 bps, 13 bps since
the 2023 stamp-duty cut), daily turnover, blocked trades at price limits, the long-short spread, and the keep rule
applied by code. Its placebo on shuffled answers found nothing (average IC +0.001 and +0.003, both ranges containing
0). It did show that trading costs matter a lot for this daily-rebalanced portfolio: even without any signal it
replaces 26-37% of its holdings a day, and realistic costs turn an annual return of about +4% into -12% and -18%.
(Re-evaluated 2026-11-11 after the placebo valuation fix in docs/RULES.md, "Rule changes": only a few 2015
numbers moved, in the second decimal.)
Details: [placebo
comparison](results/csi300/walkforward/placebo/comparison.md), rules in [docs/RULES.md](docs/RULES.md).

**More data (2026-11-12): built, no model run yet.** Beyond CSI300 there are now four more datasets, each with its
own config, locked test period (2024-03-01 on), try ledger and data QA report: CSI500, CSI1000 and the whole
A-share market from the same pinned archive, and a South Korea top 200 built from a free daily snapshot of every
Korean stock (its go/no-go checks, written beforehand, passed: delisted stocks are present, split adjustments come
out right, suspensions are identified). They are extra checks next to CSI300; the keep rule is unchanged. The free
S&P 500 route failed its check: only 91.06% of member-days (one stock on one day it was in the index) can be matched
to a listed ticker, against a bar of 95.0%, and only 70.37% for stocks that later left the index. A membership audit
found no look-ahead in the China archive's index lists, which show index reviews 9 to 14 trading days late. Known
gaps: the whole-market walk-forward data does not fit in memory yet and has no ST filter; the Korea data has no cash
dividends and misses re-opening moves after capital reductions and long halts. Details:
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md), EXPERIMENTS E20-E23.

The market data is not included in this repository. The China data is downloaded by `scripts/get_cn_data.sh`
from the community Qlib dataset (chenditc/investment_data), and the Korea data by `scripts/get_kr_data.sh` from
FinanceData/marcap; neither has an explicit license, so no data is redistributed here.

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
- [Walk-forward placebo](results/csi300/walkforward/placebo/comparison.md) (11 test years, 5 seeds per model, shuffled
  answers): Rule WP passed; per-model reports with costs, turnover and moods per fold in `placebo/paper/` and
  `placebo/single_mlp/`; the [walk-forward data report](results/csi300/walkforward/data_report.md) with its per-year
  history check.

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
| Free S&P 500 data (fja05680 member list + Tiingo prices) | - (data check) | 91.06% of member-days matched to a listed ticker (1,926,082 of 2,115,048); 70.37% for stocks that left the index; bar 95.0% | Failed the coverage check written beforehand: the gap is survivorship (renamed and delisted companies are missing) | E22, [report](results/sp500/coverage_check.md) |

## Setup

```bash
# Python 3.12 is required (pyqlib 0.9.7 has no 3.13 wheels); uv installs it for you.
uv sync
./scripts/get_cn_data.sh        # pinned community Qlib CN dataset, verified by sha256 (about 570 MB)
uv run pytest                   # 401 tests; data-dependent ones skip until `prepare` has run

# Optional, South Korea: pinned source (about 1 GB), then the go/no-go checks and the conversion it prints
./scripts/get_kr_data.sh
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
the validation period (or on walk-forward folds, see "Walk-forward evaluation" below), never on the test period:

```bash
uv run detmaso train    --config configs/csi300_single_mlp.yaml --tag single_mlp
uv run detmaso evaluate --config configs/csi300_single_mlp.yaml --tag single_mlp --split valid
uv run detmaso compare  --runs configs/csi300.yaml:final configs/csi300_single_mlp.yaml:single_mlp \
                        --reference final --out results/csi300/single_vs_ensemble
```

`compare` reports each model's metrics (mean +- std over seeds) and the reference's advantage over each
variant with a 95% range from a paired block bootstrap (validation days redrawn in 20-day blocks shared by
all models, seeds redrawn per model from the model's own random stream, so a model's range does not depend on
which other models are compared), plus an early-stopping check. It refuses the test split and any run group
not evaluated on all its configured seeds.

Guards against fooling ourselves (CLAUDE.md "Guards", rules in [docs/RULES.md](docs/RULES.md)):

- **Try ledger.** `train` registers every new configuration in `results/<market>/trials.csv` before training
  starts (aborted runs count too) and `evaluate --split valid` over all configured seeds adds its validation
  metrics (a subset of seeds is not recorded). `compare` then reports each model's try number, a 95% range for
  its IC and Sharpe, a deflated Sharpe ratio (on the daily excess return) and a deflated IC: the probability that
  the value beats the best of that many tries of pure luck.
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

### More markets

Each market has a validation-period config and walk-forward variants; only `prepare` has been run on them so far:

```bash
uv run detmaso prepare --config configs/csi500.yaml                    # also csi1000, ashare, kr200
uv run detmaso prepare --config configs/walkforward/csi500/paper.yaml  # also csi1000, kr200 (ashare does not fit in 24 GB)
```

Data checks behind them: `scripts/audit_membership.py` (index membership timing against BaoStock) and
`python -m detmaso.data.ingest.us_coverage` (the free S&P 500 coverage check; inputs from
`scripts/get_us_coverage_data.sh`). Sources and results: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

### Walk-forward evaluation

The yardstick for every later change (docs/RULES.md, "Walk-forward evaluation"). A walk-forward config
(`configs/walkforward/`) is a validation-period config plus the walk-forward layer `_layer.yaml`, e.g.
`base: [../csi300_single_mlp.yaml, _layer.yaml]`. It has 11 folds: test years 2013-2023 (the last fold runs to
2024-02-28). Fold Y trains from scratch on 2008 .. Y-2, early-stops on Y-1 and is tested on Y, with 20-day purges
at every boundary; the market moods and the scheduler's input scaling are refit on each fold's pre-test days.
Code refuses any fold that would touch the locked test period.

```bash
uv run detmaso prepare  --config configs/walkforward/paper.yaml          # data from 2007 into runs/csi300_wf
uv run detmaso train    --config configs/walkforward/paper.yaml --tag wf --jobs 8   # 11 folds x 5 seeds, resumable
uv run detmaso evaluate --config configs/walkforward/paper.yaml --tag wf --split walkforward
uv run detmaso train    --config configs/walkforward/single_mlp.yaml --tag wf_single --jobs 8   # each config has its own tag
uv run detmaso evaluate --config configs/walkforward/single_mlp.yaml --tag wf_single --split walkforward
uv run detmaso compare  --split walkforward --reference wf --out results/csi300/walkforward/<name> \
                        --runs configs/walkforward/paper.yaml:wf configs/walkforward/single_mlp.yaml:wf_single \
                        --rule keep --candidate wf_single --no-tunable-setting [--target-years 2015 2016]
```

- **Report** (`report_wf.md`): IC by test year and over all years with 95% ranges, the paper backtest's Sharpe,
  the after-cost Sharpe, daily turnover, the long-short spread (top 10% minus bottom 10%), results at 0, 10 and 20
  bps per trade and at realistic China costs (buys 8 bps, sells 18 bps, 13 bps from 2023-08-28 when the stamp duty
  was halved), with and without blocked trades (price limits, suspensions), the moods of each fold, early-stopping
  facts, the try number, deflated Sharpe and deflated IC.
- **Keep rule as code.** `compare --rule keep` applies the CLAUDE.md keep rule (Rule WK): the average IC gain's 95%
  range above 0, at least 8 of 11 years won (or the targeted form), at least 4 of 5 seeds gaining, the next setting
  over also gaining (`--neighbor <tag>` names its run group; `--no-tunable-setting` says the change has none), and
  the after-cost Sharpe not detectably worse. `--rule placebo` applies Rule WP to two placebo run groups. Both
  refuse to give a verdict when a number they need could not be computed (not finite).
- A walk-forward configuration is one try in the ledger; each fold and seed is a job (`<tag>:Y<year>:s<seed>`,
  marked `aborted: <reason>` when it fails or the run is interrupted, and retrained by the next `train`), and
  `evaluate --split walkforward` adds its walk-forward metrics before it computes the report's deflated numbers.

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
| `runs/csi300_wf/prepared/` | walk-forward data (from 2008), `data_report.md` with a per-year history check, and `folds/fold<Y>/` each fold's moods |
| `runs/csi300_wf/<tag>/fold<Y>/seed<k>/` | a walk-forward job: the same files as a validation-period seed folder |
| `runs/csi300_wf/<tag>/` | `series_wf.parquet` (per seed and test day: fold, IC, paper backtest, benchmark, long-short, buys and sells with and without blocked trades), `results_wf.json`, `report_wf.md`, `figures_wf/`, `folds.json` |
| `results/csi300/trials.csv` | the try ledger (git-tracked): every trained configuration, its kind (counted try, placebo or training-only), run groups, seed (or walk-forward job) statuses, validation and walk-forward metrics |
| `docs/RULES.md` | the decision rules, written before each change's first run, and the grids set by pilots |
| `results/<market>/data_report.md`, `results/<market>/walkforward/data_report.md` | data QA of the newer datasets (csi500, csi1000, ashare, kr200): per-year members, priced share of member-days, stocks whose data ends early, longest unpriced spans |
| `results/kr200/go_no_go.md`, `results/kr200/conversion_report.md` | the Korea go/no-go checks and the conversion report (universe per year, factor steps, known gaps) |
| `results/sp500/coverage_check.md` | the free US coverage check (failed) |
| `results/csi300/membership_audit.md` | index membership timing of the China archive against BaoStock |

## Layout

```
configs/               csi300.yaml (the paper config) and variants that inherit it with `base:`;
                       scheduler_balance/ (pilot, grid strengths, placebo versions);
                       walkforward/ (the walk-forward layer, the paper config and the single network, placebo versions,
                       and one folder per newer market);
                       csi500.yaml, csi1000.yaml, ashare.yaml, kr200.yaml (the paper method on other universes)
docs/                  PLAN.md (the MVP design), RULES.md (decision rules fixed before each change's first run),
                       DATA_SOURCES.md (free data sources: what is used, what was measured, what to avoid)
scripts/               pinned downloads (China, Korea, US coverage inputs) and the membership audit
src/detmaso/
  config.py            typed config; every assumption is a named value; post-paper flags, `base:` inheritance
                       (one file or a list), placebo, walkforward and trading (evaluation-only) sections
  data/                qlib_io, returns, labels, features (34), market_state (12), regimes, splits, panel, qa,
                       placebo (answers shuffled in 20-day blocks),
                       ingest/ (kr: the Korea converter and its go/no-go checks; us_coverage: the free US check)
  model/               agent MLP, scheduler (window encoders + attention + head), DetMASO,
                       single network baseline (single.py), factory (model.kind)
  losses/              soft rank (Blondel et al. L2, exact PAV), masked statistics, the full objective,
                       balance (scheduler load-balancing penalty)
  train/trainer.py     Algorithm 1: one Adam(W) step per date, cosine LR, clipping, early stopping;
                       optional balance penalty and training-only runs
  eval/                metrics, top-decile backtest, Shapley interaction, report, test lock,
                       compare (+ compare_plots): variants side by side on validation,
                       ledger (try ledger), guard (deflated Sharpe and IC, the rules as code),
                       scheduler_diag (training-day usage and routing),
                       trading (turnover, costs, blocked trades, long-short), walkforward (evaluation and report),
                       wf_stats (metrics and the stratified paired resampling), wf_compare (Rules WK and WP), wf_plots
  walkforward/         folds (layout, purges, test-period guard, per-fold moods), runner (fold x seed jobs, parallel)
  cli.py               prepare | train | evaluate | usage | compare
tests/                 the plan's 30 checks plus the post-MVP ones (frozen paper config, golden training
                       run, single network, config inheritance, comparison statistics, balance penalty,
                       placebo, deflated metrics, try ledger, rules, walk-forward folds and runner, trading,
                       the keep rule)
```

## Copyright

Copyright (c) 2026 Nathan Hung. All rights reserved. This code is shared for viewing only; any other use requires permission.
