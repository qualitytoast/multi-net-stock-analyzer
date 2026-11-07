# Experiments

Every training or evaluation run is logged here: date, config, data used, results, and whether the
change was kept or dropped. Comparisons use the training and validation periods only; the test period
(2024-03-01 to 2026-03-01) is locked and was opened once, for E3 (see `locks/csi300_test.json`).

Common to every entry unless stated otherwise:

- **Data:** CSI300, community Qlib dataset tag `2026-11-04` (sha256 `dc917293...`), data hash `8e81084fb70adad6`.
- **Training dates:** 2012-01-04 to 2022-12-02 (2,653 dates after the 20-day purge).
- **Validation:** 2023-01-03 to 2024-02-28 (278 trading days; IC on the 258 dates whose 20-day labels end inside it).
- **Seeds:** 0-4. Numbers are mean +- std over seeds. Metrics: IC and ICIR (rank correlation of scores with 20-day returns), and Sharpe, annual return (AR) and max drawdown (MDD) of the daily top-10% portfolio.

## Index

| ID | Date | What | Config (hash) | Run group | Validation IC | Kept / dropped |
|---|---|---|---|---|---|---|
| E1 | 2026-11-05 | Stability diagnostics behind DEVIATIONS 1-4 | paper config with ReLU/GELU x coupled/decoupled decay, float32/float64 | `runs/diag/*` (partly not saved) | not compared | GELU + decoupled decay + float64 + "constant output scores 0" kept |
| E2 | 2026-11-05 | First 5-seed training of the paper config | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `dev` | 0.062 +- 0.010 | superseded by E3 (identical models) |
| E3 | 2026-11-05 | Final paper-config models (MVP result) | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `final` | 0.062 +- 0.010 | kept: the reference Det-MASO |
| E4 | 2026-11-06 | Step 1: retrain the paper config with the Step 1 code (regression check) | `configs/csi300.yaml` (`d2a65e9a6a237d2c`) | `paper_rerun` | identical to E3 | check passed; not a candidate |
| E5 | 2026-11-06 | Step 1: Det-MASO without the diversity penalty (c_div = 0) | `configs/csi300_nodiv.yaml` (`95dc0657f33ab915`) | `nodiv` | 0.088 +- 0.058 | dropped: no detectable gain, unstable across seeds |
| E6 | 2026-11-06 | Step 1: single network, one-agent size (main baseline) | `configs/csi300_single_mlp.yaml` (`ef0560a807c0b155`) | `single_mlp` | 0.103 +- 0.002 | kept as the reference baseline |
| E7 | 2026-11-06 | Step 1: single network, Det-MASO size (capacity check) | `configs/csi300_single_mlp_wide.yaml` (`6964fbf8c1ad4259`) | `single_mlp_wide` | 0.101 +- 0.007 | dropped: same as E6 |
| E8 | 2026-11-06 | Step 1: single network + 12 market-mood inputs | `configs/csi300_single_mlp_state.yaml` (`d38afb377e8a652b`) | `single_mlp_state` | 0.140 +- 0.014 | kept as a baseline to re-check under walk-forward (Step 3) |

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
with the decision rule fixed before the runs (plan decision D3). "Advantage" below = Det-MASO (E3) minus
the model; the range is the 95% bootstrap range.

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
- **Kept / dropped:** kept as a baseline to re-check under walk-forward (Step 3); its extra edge over E6 is not established.
