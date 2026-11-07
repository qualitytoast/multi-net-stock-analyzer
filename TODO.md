# TODO

The guards against fooling myself (try ledger, rules first, placebo, report everything, keep rule) are standing
rules in CLAUDE.md; their tooling is built (see ROADMAP.md).

## Make the evaluation trustworthy

Why: 278 validation days are too few and too noisy to judge changes, and real trading has costs.

- [ ] Walk-forward evaluation:
  - retrain each year on all earlier data, then test on the next year;
  - folds run up to 2024-02-28 only;
  - inside each fold, use the last part of the training window for early stopping;
  - purge 20 days at every boundary.
- [ ] Trading costs: report results at 0, 10 and 20 basis points per trade. Claude: look up China's current stamp duty and typical commissions and include a realistic setting.
- [ ] Report daily turnover: the share of the portfolio swapped each day.
- [ ] Add a long-short spread (top 10% minus bottom 10%) as an extra signal measure.
- [ ] Re-check the single-network comparison (Step 1) under walk-forward.
- [ ] Decide how the market moods are defined inside each fold. The K-means moods are fit on 2012-2022, so using them in earlier folds is look-ahead; refitting per fold changes what each mood (and its agent) means from fold to fold.

Notes from the scheduler-collapse fix (2026-11-07; the load-balancing penalty was dropped, but stays in the code behind `loss.c_bal` if a walk-forward version is ever tried, with its own pre-written rules):
- The rare moods are only partly testable: all 59 Tail-Risk days are in 2015 and no earlier year has any, so no fold both trains and tests a Tail-Risk specialist; Low-Vol Bear is tested only in the 2016 fold; High-Vol Bull in 2015 (with 24 training days), 2019 and 2020.
- With mood-share balance targets, a fold whose training years contain no days of a mood gives that agent a target of 0, so the penalty leaves it alone.
- Early stopping on a window that holds only common moods cannot pick a good epoch for the rare-mood agents; the inner early-stopping window above helps.

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

- [ ] Bigger universes: CSI500 and CSI1000.
- [ ] Qlib's Alpha158 features (158 instead of 34).
- [ ] Other horizons: 5-day and 60-day labels.
- [ ] S&P 500. This needs a data source decision first: WRDS/CRSP if available, otherwise free data with its gaps documented.
- [ ] Second-market check (from the guards): changes kept on China must also help on the US (train on China, test on the US). Needs the S&P 500 data.

## Smarter models

- [ ] Let agents compare stocks with each other (attention across stocks).
- [ ] Give agents recent price history (an encoder over the last 60 days).
- [ ] Apply the diversity penalty to the stocks each agent would actually pick (soft top 10%), not to all scores.
- [ ] Add a turnover penalty to the training loss.
- [ ] Learn the regimes during training, instead of using frozen K-means.

## Later / optional

- [ ] Run the paper's other baselines on our data: AlphaGen (https://github.com/RL-MLDM/alphagen) and AlphaForge (https://github.com/DulyHao/AlphaForge).