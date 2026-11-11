# Experiments

Every training or evaluation run is logged here: date, config, data used, results, and whether the
change was kept or dropped. Comparisons use the training and validation periods only; the test period
(2024-03-01 to 2026-03-01) is locked and was opened once, for E3 (see `locks/csi300_test.json`).

Every configuration trained with validation switched on is also a counted try in the git-tracked try ledger
[results/csi300/trials.csv](results/csi300/trials.csv) (CLAUDE.md, "Guards"): `detmaso train` registers it before
training starts, so aborted runs count too. "Try #" below is its number there; identical reruns, placebo runs
and training-only pilots are not new tries. Decision rules are fixed in [docs/RULES.md](docs/RULES.md) before a
change's first run.

Common to every entry unless stated otherwise:

- **Data:** CSI300, community Qlib dataset tag `2026-11-04` (sha256 `dc917293...`), data hash `8e81084fb70adad6`.
- **Training dates:** 2012-01-04 to 2022-12-02 (2,653 dates after the 20-day purge).
- **Validation:** 2023-01-03 to 2024-02-28 (278 trading days; IC on the 258 dates whose 20-day labels end inside it).
- **Seeds:** 0-4. Numbers are mean +- std over seeds. Metrics: IC and ICIR (rank correlation of scores with 20-day returns), and Sharpe, annual return (AR) and max drawdown (MDD) of the daily top-10% portfolio.

## Index

| ID | Date | Try # | What | Config (hash) | Run group | Validation IC | Kept / dropped |
|---|---|---|---|---|---|---|---|
| E1 | 2026-11-05 | 1-5 | Stability diagnostics behind DEVIATIONS 1-4 | paper config with ReLU/GELU x coupled/decoupled decay, float32/float64 | `runs/diag/*` (partly not saved) | not compared | GELU + decoupled decay + float64 + "constant output scores 0" kept |
| E2 | 2026-11-05 | 6 | First 5-seed training of the paper config | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `dev` | 0.062 +- 0.010 | superseded by E3 (identical models) |
| E3 | 2026-11-05 | 6 | Final paper-config models (MVP result) | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `final` | 0.062 +- 0.010 | kept: the reference Det-MASO |
| E4 | 2026-11-06 | 6 (rerun) | Step 1: retrain the paper config with the Step 1 code (regression check) | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `paper_rerun` | identical to E3 | check passed; not a candidate |
| E5 | 2026-11-06 | 7 | Step 1: Det-MASO without the diversity penalty (c_div = 0) | `configs/csi300_nodiv.yaml` (`95dc0657f33ab915`) | `nodiv` | 0.088 +- 0.058 | dropped: no detectable gain, unstable across seeds |
| E6 | 2026-11-06 | 8 | Step 1: single network, one-agent size (main baseline) | `configs/csi300_single_mlp.yaml` (`ef0560a807c0b155`) | `single_mlp` | 0.103 +- 0.002 | kept as the reference baseline |
| E7 | 2026-11-06 | 9 | Step 1: single network, Det-MASO size (capacity check) | `configs/csi300_single_mlp_wide.yaml` (`6964fbf8c1ad4259`) | `single_mlp_wide` | 0.101 +- 0.007 | dropped: same as E6 |
| E8 | 2026-11-06 | 10 | Step 1: single network + 12 market-mood inputs | `configs/csi300_single_mlp_state.yaml` (`d38afb377e8a652b`) | `single_mlp_state` | 0.140 +- 0.014 | kept as a baseline to re-check under walk-forward (TODO.md) |
| E9 | 2026-11-07 | 6 (rerun) | Scheduler balance: retrain the paper config with the new code (regression check) | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `paper_rerun2` | identical to E3 | check passed; not a candidate |
| E10 | 2026-11-07 | - (placebo) | Scheduler balance: the whole procedure on shuffled answers (Rule P) | `configs/scheduler_balance/placebo/` | `placebo_*` (pilot + 8 groups) | average +0.005 [-0.006, +0.016] | check passed |
| E11 | 2026-11-07 | - (training only) | Scheduler balance: strength pilot, validation off (Rule G) | `configs/scheduler_balance/pilot/` | `bal_pilot_*` | not evaluated | set the grid 0.03-10 |
| E12 | 2026-11-07 | 11 | Scheduler balance, strength 0.03 | `configs/scheduler_balance/bal_c3e-2.yaml` (`5e6205840602a67a`) | `bal_c3e-2` | 0.038 +- 0.020 | dropped (Rule S: no reliable specialists; rule kept by the owner); usage and routing fail |
| E13 | 2026-11-07 | 13 | Scheduler balance, strength 0.1 | `configs/scheduler_balance/bal_c1e-1.yaml` (`36f4924f33ff1153`) | `bal_c1e-1` | 0.072 +- 0.015 | dropped (Rule S: no reliable specialists; rule kept by the owner); usage ok, routing fails |
| E14 | 2026-11-07 | 16 | Scheduler balance, strength 0.3 | `configs/scheduler_balance/bal_c3e-1.yaml` (`7cec9a9df1982f8d`) | `bal_c3e-1` | 0.058 +- 0.029 | dropped (Rule S: no reliable specialists; rule kept by the owner); usage ok, routing fails |
| E15 | 2026-11-07 | 12 | Scheduler balance, strength 1 | `configs/scheduler_balance/bal_c1e0.yaml` (`6cf6709dc4c7b083`) | `bal_c1e0` | 0.055 +- 0.016 | dropped (Rule S: no reliable specialists; rule kept by the owner); usage ok, routing fails |
| E16 | 2026-11-07 | 14 | Scheduler balance, strength 3 | `configs/scheduler_balance/bal_c3e0.yaml` (`2be8c2d2b3b14e8d`) | `bal_c3e0` | 0.054 +- 0.021 | dropped (Rule S: no reliable specialists; rule kept by the owner); usage ok, routing fails |
| E17 | 2026-11-07 | 15 | Scheduler balance, strength 10 | `configs/scheduler_balance/bal_c1e1.yaml` (`521e530361af914f`) | `bal_c1e1` | 0.036 +- 0.033 | dropped (Rule S: no reliable specialists; rule kept by the owner); usage ok, routing fails |
| E18 | 2026-11-08 | 6 (rerun) | Walk-forward tooling: retrain the paper config with the new code (regression check) | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `paper_rerun3` | identical to E3 | check passed; not a candidate |
| E19 | 2026-11-08 | - (placebo) | Walk-forward placebo (Rule WP): paper config and single network, 11 folds x 5 seeds on shuffled answers | `configs/walkforward/placebo/` | `runs/csi300_wf`: `placebo`, `placebo_single`, `smoke` | walk-forward IC +0.0010 [-0.0031, +0.0051] and +0.0028 [-0.0025, +0.0081] (not validation) | check passed |

## E1: stability diagnostics (2026-11-05)

- **Config:** the paper config with the planned defaults (ReLU, Adam with coupled L2 weight decay, float32 networks), then one change at a time. Seed 0, up to 6 epochs per diagnostic.
- **Data:** training period; validation IC watched only as a sanity signal.
- **Results** (details and evidence in [DEVIATIONS.md](DEVIATIONS.md), entries 1-4):
  - ReLU died in epoch 2 (coupled decay) or epoch 3 (decoupled decay): the rare-regime agent's last hidden layer went permanently inactive.
  - GELU with coupled decay: rare-regime agents' output spread shrank about 35% per epoch (saved: `runs/diag/gelu_coupled`, config `4ee45b94fc9a810d`, best smoothed validation IC 0.112 after 6 epochs).
  - GELU with decoupled decay (AdamW): stable (saved: `runs/diag/gelu_decoupled`, config `38ca1b54da2a14d8`, 0.099 after 6 epochs).
  - float32: starved agents' output spread fell below float32 precision; float64 fixed it. A full float64 run still aborted at epoch 11 on a constant agent output, which led to DEVIATIONS 4 (score 0 for a constant output).
- **Kept:** GELU, decoupled weight decay, float64, score 0 for a constant output. Chosen for training stability, not for validation performance. All four approved by the owner on 2026-11-06.

## E2: `dev`, first 5-seed paper-config training (2026-11-05)

- **Config:** `configs/csi300.yaml`, hash `d2a65e9a6a237d2c` (includes the E1 changes).
- **Results (validation):** IC 0.062 +- 0.010, ICIR 0.420 +- 0.091, Sharpe 0.02 +- 0.41, AR -0.9% +- 7.0%, MDD 28.0% +- 2.8%.
- **Kept / dropped:** superseded. Retrained as `final` (E3) after the code-review fixes; the models came out identical (same weights), so the numbers are the same.

## E3: `final`, the MVP's paper-config models (2026-11-05)

- **Config:** `configs/csi300.yaml`, hash `d2a65e9a6a237d2c`; best epochs 13, 4, 8, 10, 9 of 21, 12, 16, 18, 17 trained.
- **Results (validation):** IC 0.062 +- 0.010, ICIR 0.420 +- 0.091, Sharpe 0.02 +- 0.41, AR -0.9% +- 7.0%, MDD 28.0% +- 2.8%. Scheduler entropy 0.014 of a possible 1.61 (collapsed onto two agents). Report: [results/csi300/validation/report.md](results/csi300/validation/report.md).
- **Test period (opened once, 2026-11-06, the only opening):** IC 0.038 +- 0.008, ICIR 0.18, Sharpe 0.81 +- 0.29, AR 14.7% +- 6.0%, MDD 19.1% +- 3.0%. Verdict: not reproduced. Report: [results/csi300/test/report.md](results/csi300/test/report.md).
- **Kept:** the reference Det-MASO for all later comparisons.

## Step 1: does the 5-agent idea help? (2026-11-06, E4-E8)

Write-up, figures and the answer: [results/csi300/single_vs_ensemble/README.md](results/csi300/single_vs_ensemble/README.md).
All five run groups: 5 seeds, the paper's training recipe (E3), validation only. Comparison statistics:
paired block bootstrap (20-day blocks shared across models, seeds redrawn per model, 10,000 draws),
with the decision rule fixed before the runs (Step 1 decision D3). "Advantage" below = Det-MASO (E3) minus
the model; the range is the 95% bootstrap range. These ranges come from the earlier resampling method, in which
every model's seeds were redrawn from one shared random stream; `detmaso compare` now redraws each model's seeds
from a stream of its own (docs/RULES.md, "Rule changes", 2026-11-07), which moves the ranges by about 0.01 and
changes no verdict.

### E4: `paper_rerun`, regression check

- **Config:** `configs/csi300.yaml` (`d2a65e9a6a237d2c`), unchanged.
- **Result:** all 5 models identical to `final` (E3) in every weight; identical validation-IC curves, epochs and best epochs; every model file's sha256 equals the one recorded in `locks/csi300_test.json`. Re-evaluating `final` with the Step 1 code also reproduced its validation report and results file byte for byte.
- **Kept / dropped:** check passed (the Step 1 code leaves the paper configuration's results unchanged).

### E5: `nodiv`, Det-MASO with c_div = 0

- **Config:** `configs/csi300_nodiv.yaml` (`95dc0657f33ab915`): the paper config with `loss.c_div: 0.0`.
- **Results (validation):** IC 0.088 +- 0.058 (seeds 0.035 to 0.181), ICIR 0.50 +- 0.28, Sharpe 0.33 +- 0.67, AR 3.2% +- 10.4%, MDD 25.8% +- 6.7%. Advantage on IC -0.026 [-0.086, +0.031], Sharpe -0.31 [-1.37, +0.65]: no detectable difference. Validation IC averaged over all epochs 0.043 (same as E3). Agents more alike (mean abs rank correlation 0.41 vs 0.10); scheduler still collapsed (1.07 effective agents per day).
- **Kept / dropped:** dropped as a change (the paper config keeps c_div = 0.1). Removing the penalty gave no detectable gain and much less stable seeds.

### E6: `single_mlp`, one network of one-agent size (main comparison)

- **Config:** `configs/csi300_single_mlp.yaml` (`ef0560a807c0b155`): `model.kind: single_mlp`, c_div = c_ent = 0; 4,865 weights.
- **Results (validation):** IC 0.103 +- 0.002, ICIR 0.67 +- 0.03, Sharpe 0.66 +- 0.29, AR 8.3% +- 4.0%, MDD 16.7% +- 3.8%. Advantage on IC -0.041 [-0.088, +0.011] (Det-MASO ahead in 5% of draws), Sharpe -0.64 [-1.91, +0.58]: no detectable difference under the fixed rule, every point estimate favoring the single network. Validation IC averaged over all epochs 0.093 (Det-MASO 0.043).
- **Kept / dropped:** kept as the reference baseline for later steps. Answer to Step 1: Det-MASO does not beat a single network on our data.

### E7: `single_mlp_wide`, one network of Det-MASO size

- **Config:** `configs/csi300_single_mlp_wide.yaml` (`6964fbf8c1ad4259`): E6 with hidden widths 180, 90, 45; 26,731 weights (Det-MASO 26,634).
- **Results (validation):** IC 0.101 +- 0.007, ICIR 0.67 +- 0.05, Sharpe 0.62 +- 0.19, AR 7.7% +- 2.8%, MDD 16.9% +- 2.6%. Advantage on IC -0.040 [-0.086, +0.012]: no detectable difference.
- **Kept / dropped:** dropped: more weights change nothing (same as E6).

### E8: `single_mlp_state`, one network + market-mood inputs

- **Config:** `configs/csi300_single_mlp_state.yaml` (`d38afb377e8a652b`): E6 with `model.market_state_input: true` (46 inputs); 5,633 weights.
- **Results (validation):** IC 0.140 +- 0.014, ICIR 0.73 +- 0.09, Sharpe 0.69 +- 0.26, AR 10.6% +- 4.7%, MDD 20.6% +- 3.3%. Advantage on IC -0.078 [-0.144, -0.017]: the single network is better under the fixed rule; Sharpe -0.67 [-2.22, +0.86]: no detectable difference. Against E6: IC +0.038 [-0.025, +0.101], Sharpe +0.02 [-1.12, +1.34], no detectable difference. Validation IC averaged over all epochs 0.099 (picking the epoch adds +0.042, the most of any single network).
- **Kept / dropped:** kept as a baseline to re-check under walk-forward (TODO.md); its extra edge over E6 is not established.

## Scheduler collapse fix (2026-11-07, E9-E17)

Plan `give-me-a-plan-abundant-beacon.md` (approved 2026-11-07); rules in [docs/RULES.md](docs/RULES.md), fixed before
the first run; write-up, tables and figures: [results/csi300/scheduler_balance/README.md](results/csi300/scheduler_balance/README.md).
The change: `loss.c_bal x KL(target || usage)`, usage = the scheduler's mean weight per agent over all 2,653 training
days, target = each mood's share of the training days (52.6, 7.9, 32.2, 5.1, 2.2%). Everything else is the paper
config (E3); 5 seeds. "Usage" (Rule M1) = every agent gets at least half its target share of the training days;
"routing" (M2) = each mood's own agent has the largest weight on that mood's days; both in every seed, for the
restored and the last-epoch model.

### E9: `paper_rerun2`, regression check

- **Config:** `configs/csi300.yaml` (`d2a65e9a6a237d2c`), unchanged; trained with the scheduler-balance code.
- **Result:** all 5 models identical to `final` (E3) in every weight, validation-IC curve, epoch count and best epoch;
  every model file's sha256 equals `locks/csi300_test.json`. Re-evaluating `final` reproduced its validation report,
  results file and daily series byte for byte.
- **Kept / dropped:** check passed.

### E10: placebo of the whole procedure (Rule P; not tries)

- **Data:** answers shuffled within 20-day blocks: the stocks labelled on every day of a block (93.0% of training
  stock-days, 99.2% of validation; at least 67 stocks per training day) get another such stock's 20-day labels and
  next-day returns (one random cycle per block, so nobody keeps its own); the other stocks are left out of training,
  IC and the portfolio for that block. Placebo seed 0.
- **Pilot** (6 strengths x 2 seeds x 8 epochs, validation off): every agent at least half its target at epoch 8 in
  both seeds at 1e-5, 1e-4, 1e-2, 0.1 and 1 (not 1e-3); placebo grid 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3.
- **Sweep:** `placebo_paper`, `placebo_single_mlp` and the 6 placebo-grid strengths, 5 seeds each (configs
  `configs/scheduler_balance/placebo/`). Validation IC per group: 0.0055, 0.0096, 0.0038, 0.0102, -0.0007, -0.0008,
  0.0050, 0.0056; average **+0.0048 [-0.0056, +0.0157]** (P1 passes). No placebo strength passed Rule M (without a
  signal the scheduler drifts, and these weak penalties do not hold every seed at the target), so none was chosen
  (P2 does not apply).
- **Excess Sharpe** was positive in all 8 groups (0.13-0.99), but their daily excess returns correlate 0.43 on
  average (shared tilt, same shuffled returns): one draw, t = 1.7 by the daily formula, which overstates certainty.
- **Kept / dropped:** check passed: the process finds nothing on shuffled answers. Results:
  [results/csi300/scheduler_balance/placebo/](results/csi300/scheduler_balance/placebo/).

### E11: strength pilot (Rule G; training only, not tries)

- **Config:** `configs/scheduler_balance/pilot/pilot_c*.yaml`: strengths 1e-5 to 1, seeds 0-1, 8 epochs,
  `train.validate: false` (the validation period is never evaluated).
- **Result:** every agent at least half its target at epoch 8 in both seeds only at 0.1 and 1; anchor 0.1, so the
  grid is 0.03, 0.1, 0.3, 1, 3, 10. Gradient clipping was active on 1-3% of steps at every strength.
- **Kept / dropped:** set the grid (logged in docs/RULES.md before the sweep).

### E12-E17: the strength sweep (tries 11-16)

| ID | Strength | Try # | IC | Sharpe | Det-MASO's IC lead [95%] | Usage (M1) | Routing (M2) | Routing failures (of 50) |
|---|---|---|---|---|---|---|---|---|
| E12 | 0.03 | 11 | 0.038 +- 0.020 | -0.32 +- 0.48 | +0.024 [-0.011, +0.058] | fail | fail | 11 (Tail-Risk 7) |
| E13 | 0.1 | 13 | 0.072 +- 0.015 | -0.21 +- 0.32 | -0.010 [-0.038, +0.015] | pass | fail | 11 (Tail-Risk 6) |
| E14 | 0.3 | 16 | 0.058 +- 0.029 | -0.31 +- 0.43 | +0.004 [-0.031, +0.041] | pass | fail | 6 (all Tail-Risk) |
| E15 | 1 | 12 | 0.055 +- 0.016 | -0.17 +- 0.27 | +0.007 [-0.024, +0.040] | pass | fail | 4 (Tail-Risk 2) |
| E16 | 3 | 14 | 0.054 +- 0.021 | -0.31 +- 0.31 | +0.008 [-0.024, +0.040] | pass | fail | 6 (Tail-Risk 4) |
| E17 | 10 | 15 | 0.036 +- 0.033 | -0.30 +- 0.33 | +0.026 [-0.012, +0.074] | pass | fail | 1 (Tail-Risk) |

- **Data:** as in E3. **Configs:** `configs/scheduler_balance/bal_c*.yaml` (hashes in the index).
- **Rule S:** usage is fixed from 0.1 up, but routing fails at every strength (mostly on Tail-Risk, 59 training
  days, all in 2015), so the outcome is **"all agents used, but no specialists"** and the process stopped for the
  owner's decision; the owner kept the rules as written. Rule K was not applied (no c*).
- **Results (validation):** no strength differs detectably from the paper Det-MASO (every IC range above spans 0;
  Sharpe likewise); all stay below the single network (E6, 0.103). Deflated Sharpe 5-8%, deflated IC 19-72% (16 tries).
  Picking the epoch on validation added +0.005 to +0.034 IC.
- **Scheduler:** effective agents over time on the training days 2.96-3.13 (paper 1.99, target 3.13); own agent's
  weight on its mood's days at strength 10: 93, 86, 87, 92, 79% (paper 77, 19, 83, 20, 0%). Strengths 3 and 10 each
  had one seed whose rare-mood agent was dead (constant output) for long stretches of training.
- **Kept / dropped:** **dropped**. At the Rule S stop the owner kept the rules as written (2026-11-07), so none of
  the six strengths is carried forward; the penalty stays in the code behind `loss.c_bal`, off by default.

## Walk-forward tooling (2026-11-08, E18-E19)

Plan `make-a-plan-for-melodic-ocean.md` (approved 2026-11-08, decisions D1-D8 all "a"); rules in
[docs/RULES.md](docs/RULES.md) ("Walk-forward evaluation", Rules WL, WT, WK, WP), fixed before any walk-forward run.
No walk-forward run on the real answers was made: the first one belongs to the testing phase.

### E18: `paper_rerun3`, regression check

- **Config:** `configs/csi300.yaml` (`d2a65e9a6a237d2c`), unchanged; trained with the walk-forward code, 5 seeds in parallel.
- **Result:** all 5 models identical to `final` (E3) in every weight, validation-IC curve, epoch count and best epoch;
  every model file's sha256 equals `locks/csi300_test.json`. Re-evaluating `final` reproduced its validation report,
  results file and daily series byte for byte.
- **Kept / dropped:** check passed.

### E19: walk-forward placebo (Rule WP; not tries)

- **Data:** the walk-forward data (`runs/csi300_wf`, data hash `7470fa641726f353`): loaded from 2007-01-01, model dates
  2008-01-02 onward; the per-year check in [results/csi300/walkforward/data_report.md](results/csi300/walkforward/data_report.md)
  shows 2008-2011 like the later years (medians of 288-293 stocks per day, under 1% missing features, one one-day move above
  21% in 2009). Answers shuffled within 20-day blocks (placebo seed 0), as in E10.
- **Configs:** `configs/walkforward/placebo/paper.yaml` (`9515c4dbd6f711d2`) and `configs/walkforward/placebo/single_mlp.yaml`
  (`d820628005c10957`); 11 folds (test years 2013-2023, the last to 2024-02-28) x seeds 0-4; moods refit per fold.
- **Smoke run** (`smoke`, fold 2013, seed 0): about 1 minute of setup (data, shuffled answers, moods of all 11 folds),
  then 7 s per epoch on the smallest fold with the CPU shared; the evaluation path ran on the real layout.
- **Training:** all 110 jobs trained, no aborts. Wall time with the CPU shared: 65 min (paper config, 5 jobs at a
  time), 22 min (single network, 4 at a time).
- **Results** (11 test years joined; mean +- std over seeds [95% range]):

  | Run group | Average IC | Sharpe, paper backtest | Sharpe after costs | Turnover/day | Long-short (bps/day) |
  |---|---|---|---|---|---|
  | `placebo` (paper config) | 0.0010 +- 0.0023 [-0.0031, +0.0051] | 0.29 [-0.28, +0.88] | -0.69 +- 0.15 [-1.23, -0.14] | 37.4% | -0.2 |
  | `placebo_single` (single network) | 0.0028 +- 0.0015 [-0.0025, +0.0081] | 0.30 [-0.25, +0.89] | -0.40 +- 0.07 [-0.92, +0.16] | 26.4% | 0.1 |

- **Rule WP:** P1 both average-IC ranges contain 0 (pass); P2 Rule WK in both directions says "no improvement"
  (single minus paper: average gain +0.0018 [-0.0031, +0.0068], 6 of 11 years, 4 of 5 seeds) (pass). **Placebo passed.**
  One year of one group (2019, paper config) has an IC range just above 0, as chance gives about once in 20 ranges.
- **Finding for the owner:** the after-cost Sharpe difference (single minus paper) is +0.29 [+0.11, +0.48], entirely
  above 0, although the gross difference is not (+0.01, [-0.15, +0.18]). On shuffled answers the paper config still
  replaces 36-40% of its portfolio a day against 24-28% for the single network (every seed), so it pays more costs.
  K5 therefore also responds to turnover, which is a real property of a model even when there is nothing to predict.
- **Costs:** on shuffled answers the daily top-10% portfolio's annual return falls from about +4% before costs to
  -12% (single network) and -18% (paper config) after realistic costs (max drawdown 84-93%); at a flat 10 bps it is
  -8% and -13%. Blocked trades are rare (0.3-0.4 a day, most in 2015).
- **Kept / dropped:** check passed. Results: [results/csi300/walkforward/placebo/](results/csi300/walkforward/placebo/).
