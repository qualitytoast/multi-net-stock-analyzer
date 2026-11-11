# Roadmap

What has been done and what is next for the Det-MASO reproduction. Milestones (M0-M10) and
decisions (D1-D10) refer to [docs/PLAN.md](docs/PLAN.md); changes to planned defaults are in
[DEVIATIONS.md](DEVIATIONS.md); every run is logged in [EXPERIMENTS.md](EXPERIMENTS.md). Open work
is in [TODO.md](TODO.md).

Last updated: 2026-11-08

## Status

| Area | State |
|---|---|
| CSI300 MVP (M0-M9) | Done. Verdict: **not reproduced** (IC 0.038 vs paper 0.081; Sharpe 0.81 vs 1.85). See [test report](results/csi300/test/report.md). |
| Test period (2024-03-01 to 2026-03-01) | Opened once, 2026-11-06. **Locked** (see below). |
| Deviations 1-5 | Approved by the owner, 2026-11-06. |
| Step 1: 5 agents vs a single network | Done. Det-MASO does **not** beat a single network on validation (single network IC 0.103 vs 0.062; no detectable difference under the fixed rule, every estimate favoring the single network). See [write-up](results/csi300/single_vs_ensemble/README.md). |
| Guards against fooling ourselves | Tooling built 2026-11-07 (try ledger, rules as code, placebo, deflated Sharpe and IC); standing rules in CLAUDE.md. 16 tries counted so far ([trials.csv](results/csi300/trials.csv)). |
| Scheduler collapse fix | Done 2026-11-07: **dropped**. The load-balancing penalty fixes the collapse in the usage sense from strength 0.1 up, but the agents do not reliably become mood specialists (mostly the Tail-Risk mood fails), so the rule written in advance said "no specialists"; the owner kept the rule. Validation IC does not change detectably. See [write-up](results/csi300/scheduler_balance/README.md). |
| Walk-forward evaluation | Tooling built 2026-11-08: 11 yearly folds (test years 2013-2023, history from 2008, moods refit per fold), trading costs, turnover, blocked trades, long-short spread, the keep rule as code. Placebo passed (E19). No real-answer walk-forward run yet. Rules: [docs/RULES.md](docs/RULES.md), Rules WL, WT, WK, WP. |
| S&P 500 (M10) | Deferred. Waiting on the owner's WRDS access check. |
| Repo | Public on GitHub, "all rights reserved, viewing only". |

## Done

- [x] **Plan** approved, decisions D1-D10 answered (2026-11-05). [docs/PLAN.md](docs/PLAN.md)
- [x] **M0** uv project on Python 3.12, pinned deps, pytest, pinned CN data download with sha256 check
- [x] **M1** Raw panel, index membership, labels, splits with 20-day purge, masks, cache
- [x] **M2** Suspension-aware returns, 34 features, cross-sectional processing
- [x] **M3** Market state, K-means regimes, regime-to-agent mapping
- [x] **M4** Agents and scheduler forward pass
- [x] **M5** Soft rank, all loss terms, numerical safeguards, skip rules
- [x] **M6** Trainer (one step per date, cosine LR, early stopping)
- [x] **M7** Inference, metrics, top-decile backtest on validation
- [x] **M8** Shapley interaction analysis on validation
  - M0-M8 landed in `d89ebaf` (2026-11-05); 59 tests.
- [x] **Code review** fixes applied; final 5-seed models trained (tag `final`); validation results. `d84829c` (2026-11-05)
- [x] **M9** Config frozen; test period opened once with all 5 seeds; test report and verdict. `1568a39` (2026-11-06)
- [x] **Deviations 1-5** approved: GELU, AdamW, float64, score 0 for constant agent output, tradable-stock benchmark. `1568a39`
- [x] **Published** to github.com/qualitytoast/multi-net-stock-analyzer with copyright notice. `f637f6c`
- [x] **Does the 5-agent idea help?** (2026-11-06; EXPERIMENTS E4-E8; [write-up](results/csi300/single_vs_ensemble/README.md))
  - Built: a single-network model mode (`model.kind: single_mlp`, optional `model.market_state_input`), config inheritance (`base:`), four variant configs, `detmaso compare` (validation only, paired 20-day block bootstrap over days and seeds), and a frozen paper-config fingerprint plus a golden training test; 93 tests.
  - Paper config unchanged: retrained models identical to `final` in every weight; validation report reproduced byte for byte.
  - Result (validation): single network IC 0.103, Sharpe 0.66, MDD 16.7% vs Det-MASO 0.062, 0.02, 28.0%. Det-MASO's IC advantage -0.041 [95% range -0.088, +0.011]: no detectable difference under the rule fixed in advance; Det-MASO does not beat a single network. A Det-MASO-sized single network is the same (0.101); removing the diversity penalty changes nothing detectable but destabilizes seeds; a single network with the 12 market-mood inputs beats Det-MASO (IC 0.140) but not detectably the plain single network.
  - To re-check under walk-forward (TODO.md).
- [x] **Guard tooling** (2026-11-07; CLAUDE.md "Guards"; [docs/RULES.md](docs/RULES.md))
  - Try ledger `results/csi300/trials.csv`: `train` registers every configuration before training (aborted runs count), `evaluate --split valid` adds its metrics; backfilled with E1-E8.
  - `compare` reports each model's try number, deflated Sharpe (Bailey and Lopez de Prado 2014, on daily excess returns) and deflated IC, and a 95% range for every IC level; its resampling no longer depends on which other models are compared. This also covers the old TODO item "multiple-testing correction across everything in EXPERIMENTS.md".
  - Placebo config section (answers shuffled in 20-day blocks; never mixed with real runs, never opens the test period), training-only runs (`train.validate: false`), rules applied by code (`compare --rule`), `detmaso usage` (training-day scheduler diagnostics), `model_last.pt` saved by every run; 159 tests.
  - Paper config unchanged: retrained models identical to `final` in every weight (E9); validation report reproduced byte for byte.
- [x] **Fix the scheduler collapse** (2026-11-07; EXPERIMENTS E9-E17; [write-up](results/csi300/scheduler_balance/README.md)): **dropped**
  - Built: a load-balancing penalty behind `loss.c_bal` (KL between each agent's target share, its market mood's share of the training days, and its average scheduler weight over all training days), configs in `configs/scheduler_balance/`.
  - Process, in the order the rules require: placebo first (whole procedure on shuffled answers: average IC +0.005 [-0.006, +0.016], passed); training-only pilot set the grid 0.03, 0.1, 0.3, 1, 3, 10; 6 strengths x 5 seeds.
  - Result: every agent gets at least half its target share from 0.1 up (paper config: the rare-mood agents get almost nothing), and mood specialists mostly form (own agent 79-93% at strength 10), but not in every seed, mostly for Tail-Risk (59 training days, all in 2015). Rule S: "all agents used, but no specialists"; the owner kept the rule, so the change is dropped. Validation IC 0.036-0.072 vs 0.062 for the paper config: no detectable difference; all below the single network (0.103).
  - Walk-forward notes recorded in TODO.md (moods must be redefined per fold; the rare moods are only partly testable).

- [x] **Walk-forward evaluation tooling** (2026-11-08; EXPERIMENTS E18-E19; [placebo results](results/csi300/walkforward/placebo/comparison.md))
  - Built: a `walkforward` config section (`configs/walkforward/`: a layer merged on top of any validation-period
    config with a list `base:`); 11 folds, test years 2013-2023 (the last to 2024-02-28), each trained from scratch on
    2008 .. Y-2 and early-stopped on Y-1, 20-day purges, a code guard against the locked test period; moods and the
    scheduler's input scaling refit per fold; fold x seed jobs in parallel, resumable, one try per configuration.
  - Trading tools: daily turnover against drifted holdings, costs at 0, 10, 20 bps and realistic China costs (buys
    8 bps, sells 18 bps, 13 bps from 2023-08-28), blocked trades at price limits and suspensions, the long-short spread.
  - `evaluate --split walkforward` (per-year and joined metrics with 95% ranges, cost table, moods per fold, try number,
    deflated Sharpe and IC) and `compare --split walkforward` with `--rule keep` (Rule WK, including the new after-cost
    Sharpe condition K5 approved by the owner) and `--rule placebo` (Rule WP); 194 tests.
  - Data: 2007-2026 prepared separately (`runs/csi300_wf`); 2008-2011 pass the per-year check.
  - Paper config unchanged: retrained models identical to `final` in every weight (E18); validation report reproduced
    byte for byte.
  - Placebo passed (E19): average IC +0.0010 and +0.0028, both ranges containing 0; "no improvement" both ways. The
    after-cost Sharpe of the two placebo models differs detectably because Det-MASO turns over 37% of its portfolio a
    day against 26% for the single network.

Test-period checks: q1 (agents stay different) pass; q2 (scheduler does not collapse) **fail**,
entropy 0.03 of 1.61; q3 (Tail-Risk share 3-7%) **fail**, 2.2%; q4 (Shapley claim) pass, 4 of 5 seeds.