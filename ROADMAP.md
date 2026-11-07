# Roadmap

What has been done and what is next for the Det-MASO reproduction. Milestones (M0-M10) and
decisions (D1-D10) refer to [docs/PLAN.md](docs/PLAN.md); changes to planned defaults are in
[DEVIATIONS.md](DEVIATIONS.md); every run is logged in [EXPERIMENTS.md](EXPERIMENTS.md). Open work
is in [TODO.md](TODO.md).

Last updated: 2026-11-06

## Status

| Area | State |
|---|---|
| CSI300 MVP (M0-M9) | Done. Verdict: **not reproduced** (IC 0.038 vs paper 0.081; Sharpe 0.81 vs 1.85). See [test report](results/csi300/test/report.md). |
| Test period (2024-03-01 to 2026-03-01) | Opened once, 2026-11-06. **Locked** (see below). |
| Deviations 1-5 | Approved by the owner, 2026-11-06. |
| Step 1: 5 agents vs a single network | Done. Det-MASO does **not** beat a single network on validation (single network IC 0.103 vs 0.062; no detectable difference under the fixed rule, every estimate favoring the single network). See [write-up](results/csi300/single_vs_ensemble/README.md). |
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
- [x] **Step 1: does the 5-agent idea help?** (2026-11-06; EXPERIMENTS E4-E8; [write-up](results/csi300/single_vs_ensemble/README.md))
  - Built: a single-network model mode (`model.kind: single_mlp`, optional `model.market_state_input`), config inheritance (`base:`), four variant configs, `detmaso compare` (validation only, paired 20-day block bootstrap over days and seeds), and a frozen paper-config fingerprint plus a golden training test; 92 tests.
  - Paper config unchanged: retrained models identical to `final` in every weight; validation report reproduced byte for byte.
  - Result (validation): single network IC 0.103, Sharpe 0.66, MDD 16.7% vs Det-MASO 0.062, 0.02, 28.0%. Det-MASO's IC advantage -0.041 [95% range -0.088, +0.011]: no detectable difference under the rule fixed in advance; Det-MASO does not beat a single network. A Det-MASO-sized single network is the same (0.101); removing the diversity penalty changes nothing detectable but destabilizes seeds; a single network with the 12 market-mood inputs beats Det-MASO (IC 0.140) but not detectably the plain single network.
  - To re-check under walk-forward in Step 3.

Test-period checks: q1 (agents stay different) pass; q2 (scheduler does not collapse) **fail**,
entropy 0.03 of 1.61; q3 (Tail-Risk share 3-7%) **fail**, 2.2%; q4 (Shapley claim) pass, 4 of 5 seeds.