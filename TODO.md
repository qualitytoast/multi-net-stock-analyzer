# TODO

## Step 2: Fix the scheduler collapse

Why: with 3 of 5 agents dead, the "5 specialists" idea isn't really being tested.

- [ ] Add a load-balancing penalty, the standard fix for collapse in mixture-of-experts models. It pushes the scheduler to use every agent over time. Put it behind a config flag and choose its strength on validation.
- [ ] Report:
  - average scheduler weight per agent;
  - scheduler entropy;
  - number of dead agents (how often change 4 fired);
  - validation IC and Sharpe compared with Step 1.

Done when: every agent gets meaningful weight, and we know whether IC went up.

## Step 3: Make the evaluation trustworthy

Why: 278 validation days are too few and too noisy to judge changes, and real trading has costs.

- [ ] Walk-forward evaluation:
  - retrain each year on all earlier data, then test on the next year;
  - folds run up to 2024-02-28 only;
  - inside each fold, use the last part of the training window for early stopping;
  - purge 20 days at every boundary.
- [ ] Trading costs: report results at 0, 10 and 20 basis points per trade. Claude: look up China's current stamp duty and typical commissions and include a realistic setting.
- [ ] Report daily turnover: the share of the portfolio swapped each day.
- [ ] Add a long-short spread (top 10% minus bottom 10%) as an extra signal measure.
- [ ] Re-check the Step 1 and Step 2 conclusions under walk-forward.

Done when: every later step is judged on walk-forward IC and after-cost Sharpe.

## Step 4: Number of agents

Why: the paper only tried 5, and our data suggests 2 regimes.

- [ ] Separate the number of agents from the number of regimes (agents per regime × number of regimes).
- [ ] Scale the diversity penalty by the number of agents, so the comparison stays fair.
- [ ] Sweep 1, 2, 3, 5 and 10 agents, then 20 and 50 if runs are fast enough.
- [ ] Shapley: compute it exactly up to about 12 agents, and estimate it by sampling above that.
- [ ] Plot walk-forward IC and the effective number of agents against the agent count.

Done when: the plot exists, and the best agent count has been chosen on walk-forward.

## Step 5: Speed (only when runs get slow)

- [ ] Profile first.
- [ ] Run all agents as one batched network (torch.func.stack_module_state + vmap) instead of a loop.
- [ ] Train several seeds and configs in one batched run.
- [ ] Keep all data on the GPU.
- [ ] Revisit change 3 (float64): it is slow on most GPUs. Try float32 with a fix for the shrinking output scale.

## Step 6: More data

- [ ] Bigger universes: CSI500 and CSI1000.
- [ ] Qlib's Alpha158 features (158 instead of 34).
- [ ] Other horizons: 5-day and 60-day labels.
- [ ] S&P 500. This needs a data source decision first: WRDS/CRSP if available, otherwise free data with its gaps documented.

## Step 7: Smarter models

- [ ] Let agents compare stocks with each other (attention across stocks).
- [ ] Give agents recent price history (an encoder over the last 60 days).
- [ ] Apply the diversity penalty to the stocks each agent would actually pick (soft top 10%), not to all scores.
- [ ] Add a turnover penalty to the training loss.
- [ ] Learn the regimes during training, instead of using frozen K-means.

## Later / optional

- [ ] Run the paper's other baselines on our data: AlphaGen (https://github.com/RL-MLDM/alphagen) and AlphaForge (https://github.com/DulyHao/AlphaForge).
- [ ] Apply a multiple-testing correction (deflated Sharpe ratio) across everything logged in EXPERIMENTS.md.