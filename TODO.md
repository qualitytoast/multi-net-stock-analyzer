# TODO

The guards against fooling myself (try ledger, rules first, placebo, report everything, keep rule) are standing
rules in CLAUDE.md; their tooling is built (see ROADMAP.md).

## Make the evaluation trustworthy

Why: 278 validation days are too few and too noisy to judge changes, and real trading has costs.

The tooling is built (2026-11-08; see ROADMAP.md): walk-forward evaluation (11 folds, test years 2013-2023, history
from 2008, moods refit per fold), trading costs (0, 10, 20 bps and realistic), turnover, blocked trades, the
long-short spread, and the keep rule as code (docs/RULES.md, Rules WL, WT, WK, WP).

- [ ] Re-check the single-network comparison (Step 1) under walk-forward: `configs/walkforward/paper.yaml` vs
  `configs/walkforward/single_mlp.yaml` (and the single network with mood inputs), judged by Rule WK. Write which
  run is the candidate before running.

Notes from the scheduler-collapse fix (2026-11-07; the load-balancing penalty was dropped, but stays in the code behind
`loss.c_bal` if a walk-forward version is ever tried, with its own pre-written rules):
- With the walk-forward history from 2008, the moods are refit in every fold, so the old counts (all 59 Tail-Risk
  days in 2015) no longer hold; each fold's mood shares of its training and test days are in its walk-forward report.
- With mood-share balance targets, a fold whose training years contain no days of a mood gives that agent a target
  of 0, so the penalty leaves it alone.
- Early stopping on a year that holds only common moods cannot pick a good epoch for the rare-mood agents.

Done when: every later step is judged on walk-forward IC and after-cost Sharpe.

## Number of agents

Why: the paper only tried 5, and our data suggests 2 regimes.

- [ ] Separate the number of agents from the number of regimes (agents per regime × number of regimes).
- [ ] Scale the diversity penalty by the number of agents, so the comparison stays fair.
- [ ] Sweep 1, 2, 3, 5 and 10 agents, then 20 and 50 if runs are fast enough.
- [ ] Shapley: compute it exactly up to about 12 agents, and estimate it by sampling above that.
- [ ] Plot walk-forward IC and the effective number of agents against the agent count.

Done when: the plot exists, and the best agent count has been chosen on walk-forward.

## Speed (only when runs get slow)

- [ ] Profile first.
- [ ] Run all agents as one batched network (torch.func.stack_module_state + vmap) instead of a loop.
- [ ] Train several seeds and configs in one batched run.
- [ ] Keep all data on the GPU.
- [ ] Revisit change 3 (float64): it is slow on most GPUs. Try float32 with a fix for the shrinking output scale.

## More data

The datasets are built (2026-11-12; see ROADMAP.md and docs/DATA_SOURCES.md): CSI500, CSI1000, the whole A-share
market (validation-period data only) and a South Korea top 200. No model has been run on them. They are extra
checks next to CSI300; the keep rule and the second-market rule (the US) are unchanged (docs/RULES.md, Rule ND).

- [ ] First model runs on CSI500, CSI1000 and Korea: the walk-forward placebo on each new market first, then the
  paper config and the single network (`configs/walkforward/<market>/`), reported next to CSI300.
- [ ] S&P 500. The free route failed its coverage check (E22: 91.06% against a 95.0% bar). The owner has no WRDS/CRSP
  access (2026-11-12). Remaining routes: a rename map (old ticker to new) and a new coverage check under a new rule
  written beforehand, or a paid source (docs/DATA_SOURCES.md).
- [ ] Second-market check (from the guards): changes kept on China must also help on the US (train on China, test on the US). Needs the S&P 500 data.
- [ ] Whole A-share market: make `prepare` fit in memory (its walk-forward data needs about 27 GiB; the machine has
  24) without changing the paper config's results. Then load more warm-up history, so the new-listing filter has
  settled before the first model date, and add an ST filter (BaoStock has daily ST flags).
- [ ] Korea: returns miss the move into a re-opening price after a capital reduction, spin-off, merger or long halt
  (82 cases on member-days since 2007), and cash dividends. Both need corporate-action data from another source.
- [ ] China membership dates: the archive shows index reviews 9 to 14 trading days late (membership audit, E21).
  Decide whether to correct the dates.
- [ ] Qlib's Alpha158 features (158 instead of 34).
- [ ] Other horizons: 5-day and 60-day labels.
- [ ] Documented only, if wanted later: Taiwan, Numerai, Binance (docs/DATA_SOURCES.md).

## Smarter models

- [ ] Let agents compare stocks with each other (attention across stocks).
- [ ] Give agents recent price history (an encoder over the last 60 days).
- [ ] Apply the diversity penalty to the stocks each agent would actually pick (soft top 10%), not to all scores.
- [ ] Add a turnover penalty to the training loss.
- [ ] Learn the regimes during training, instead of using frozen K-means.

## Later / optional

- [ ] Run the paper's other baselines on our data: AlphaGen (https://github.com/RL-MLDM/alphagen) and AlphaForge (https://github.com/DulyHao/AlphaForge).