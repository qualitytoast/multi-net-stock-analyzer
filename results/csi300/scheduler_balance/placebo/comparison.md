# Variant comparison: csi300, validation split

- Reference: `placebo_paper` (run group `placebo_paper`). Advantage = reference minus variant (drawdown: variant minus reference), so a positive number means the reference did better.
- Evaluated dates: 2023-01-03 .. 2024-02-28 (278 trading days; IC uses 258). The test period is not used.
- Data tag `2026-11-04`, data hash `8e81084fb70adad6` (shared by every run group).
- Bootstrap: 10,000 draws; days redrawn in circular 20-day blocks (one draw shared by all variants); seeds redrawn per variant from its own random stream; RNG seed 0.
- Rule fixed in advance (Step 1 decision D3): the reference is better if the 95% range lies entirely above 0, the variant is better if entirely below 0, otherwise no detectable difference. IC is primary; Sharpe is the second opinion.
- **Placebo**: every run group was trained and evaluated on shuffled answers (placebo seed 0, 20-day blocks); there is nothing real to find.

## Run groups

| Variant | Config | Run group | Config hash | Model | Weights | Seeds |
|---|---|---|---|---|---|---|
| `placebo_paper` | `configs/scheduler_balance/placebo/paper.yaml` | `placebo_paper` | `51776060e2b66202` | detmaso | 26,634 | 0 1 2 3 4 |
| `placebo_single_mlp` | `configs/scheduler_balance/placebo/single_mlp.yaml` | `placebo_single_mlp` | `7f42c52118e8b556` | single_mlp | 4,865 | 0 1 2 3 4 |
| `placebo_bal_c3e-6` | `configs/scheduler_balance/placebo/bal_c3e-6.yaml` | `placebo_bal_c3e-6` | `9c7f9ec4b5240b5f` | detmaso | 26,634 | 0 1 2 3 4 |
| `placebo_bal_c1e-5` | `configs/scheduler_balance/placebo/bal_c1e-5.yaml` | `placebo_bal_c1e-5` | `e201b78893871eb0` | detmaso | 26,634 | 0 1 2 3 4 |
| `placebo_bal_c3e-5` | `configs/scheduler_balance/placebo/bal_c3e-5.yaml` | `placebo_bal_c3e-5` | `d46f59aa957375c5` | detmaso | 26,634 | 0 1 2 3 4 |
| `placebo_bal_c1e-4` | `configs/scheduler_balance/placebo/bal_c1e-4.yaml` | `placebo_bal_c1e-4` | `1120aba02d4e9674` | detmaso | 26,634 | 0 1 2 3 4 |
| `placebo_bal_c3e-4` | `configs/scheduler_balance/placebo/bal_c3e-4.yaml` | `placebo_bal_c3e-4` | `8a1c5f9629b29578` | detmaso | 26,634 | 0 1 2 3 4 |
| `placebo_bal_c1e-3` | `configs/scheduler_balance/placebo/bal_c1e-3.yaml` | `placebo_bal_c1e-3` | `3335ace9d46f9baf` | detmaso | 26,634 | 0 1 2 3 4 |

## Metrics (mean +- std over seeds)

| Variant | IC | ICIR | Sharpe | Max drawdown | Annual return | Excess AR | Excess Sharpe |
|---|---|---|---|---|---|---|---|
| `placebo_paper` | 0.0055 +- 0.0042 | 0.095 +- 0.074 | -0.01 +- 0.26 | 22.2% +- 2.4% | -1.2% +- 4.0% | 4.8% | 0.95 |
| `placebo_single_mlp` | 0.0096 +- 0.0039 | 0.147 +- 0.066 | 0.01 +- 0.09 | 23.0% +- 1.8% | -1.0% +- 1.4% | 5.0% | 0.99 |
| `placebo_bal_c3e-6` | 0.0038 +- 0.0068 | 0.066 +- 0.114 | -0.14 +- 0.39 | 24.4% +- 2.4% | -3.0% +- 5.8% | 2.9% | 0.56 |
| `placebo_bal_c1e-5` | 0.0102 +- 0.0057 | 0.176 +- 0.098 | -0.02 +- 0.16 | 23.1% +- 2.1% | -1.4% +- 2.5% | 4.7% | 0.93 |
| `placebo_bal_c3e-5` | -0.0007 +- 0.0048 | -0.016 +- 0.091 | -0.08 +- 0.33 | 25.3% +- 2.3% | -2.3% +- 5.0% | 3.7% | 0.69 |
| `placebo_bal_c1e-4` | -0.0008 +- 0.0050 | -0.009 +- 0.087 | -0.18 +- 0.17 | 26.7% +- 1.7% | -3.9% +- 2.5% | 2.0% | 0.40 |
| `placebo_bal_c3e-4` | 0.0050 +- 0.0045 | 0.086 +- 0.076 | -0.22 +- 0.35 | 24.4% +- 4.0% | -4.4% +- 5.3% | 1.4% | 0.22 |
| `placebo_bal_c1e-3` | 0.0056 +- 0.0075 | 0.094 +- 0.122 | -0.28 +- 0.42 | 26.4% +- 1.1% | -5.1% +- 6.0% | 0.8% | 0.13 |

## Uncertainty and luck

| Variant | IC [95% range] | Sharpe [95% range] |
|---|---|---|
| `placebo_paper` | 0.0055 [-0.0063, +0.0175] | -0.01 [-1.84, +1.90] |
| `placebo_single_mlp` | 0.0096 [-0.0098, +0.0301] | 0.01 [-1.81, +1.92] |
| `placebo_bal_c3e-6` | 0.0038 [-0.0069, +0.0149] | -0.14 [-2.03, +1.74] |
| `placebo_bal_c1e-5` | 0.0102 [-0.0016, +0.0208] | -0.02 [-1.86, +1.83] |
| `placebo_bal_c3e-5` | -0.0007 [-0.0131, +0.0122] | -0.08 [-1.95, +1.89] |
| `placebo_bal_c1e-4` | -0.0008 [-0.0172, +0.0146] | -0.18 [-2.17, +1.88] |
| `placebo_bal_c3e-4` | 0.0050 [-0.0075, +0.0172] | -0.22 [-2.13, +1.73] |
| `placebo_bal_c1e-3` | 0.0056 [-0.0095, +0.0217] | -0.28 [-2.25, +1.74] |

95% range = the 2.5-97.5% percentiles of the seed-mean level over the bootstrap draws above.

## Advantage of `placebo_paper` over each variant (95% bootstrap range)

| Variant | IC adv. [95%] | P(>0) | ICIR adv. [95%] | P(>0) | Sharpe adv. [95%] | P(>0) | IC rule | Sharpe rule | MDD adv. |
|---|---|---|---|---|---|---|---|---|---|
| `placebo_single_mlp` | -0.0041 [-0.0193, +0.0111] | 29% | -0.052 [-0.291, +0.185] | 32% | -0.02 [-0.54, +0.48] | 47% | no detectable difference | no detectable difference | +0.8% |
| `placebo_bal_c3e-6` | +0.0018 [-0.0130, +0.0157] | 60% | +0.029 [-0.243, +0.269] | 59% | +0.14 [-0.44, +0.67] | 70% | no detectable difference | no detectable difference | +2.1% |
| `placebo_bal_c1e-5` | -0.0047 [-0.0179, +0.0083] | 22% | -0.081 [-0.319, +0.139] | 22% | +0.01 [-0.48, +0.46] | 52% | no detectable difference | no detectable difference | +0.8% |
| `placebo_bal_c3e-5` | +0.0062 [-0.0056, +0.0188] | 84% | +0.111 [-0.102, +0.336] | 84% | +0.08 [-0.48, +0.60] | 61% | no detectable difference | no detectable difference | +3.1% |
| `placebo_bal_c1e-4` | +0.0063 [-0.0094, +0.0228] | 78% | +0.104 [-0.161, +0.374] | 78% | +0.18 [-0.33, +0.67] | 76% | no detectable difference | no detectable difference | +4.4% |
| `placebo_bal_c3e-4` | +0.0005 [-0.0112, +0.0133] | 52% | +0.009 [-0.197, +0.226] | 52% | +0.22 [-0.34, +0.75] | 80% | no detectable difference | no detectable difference | +2.2% |
| `placebo_bal_c1e-3` | -0.0001 [-0.0155, +0.0153] | 48% | +0.001 [-0.273, +0.258] | 49% | +0.27 [-0.30, +0.80] | 85% | no detectable difference | no detectable difference | +4.2% |

P(>0) = share of bootstrap draws in which the reference did better.

## Early-stopping check

| Variant | IC at the picked epoch (headline) | Validation IC averaged over every epoch | Gain from picking | Mean epochs trained |
|---|---|---|---|---|
| `placebo_paper` | 0.0055 | 0.0014 | +0.0041 | 15.0 |
| `placebo_single_mlp` | 0.0096 | 0.0019 | +0.0078 | 16.8 |
| `placebo_bal_c3e-6` | 0.0038 | 0.0021 | +0.0017 | 15.4 |
| `placebo_bal_c1e-5` | 0.0102 | 0.0047 | +0.0055 | 18.8 |
| `placebo_bal_c3e-5` | -0.0007 | -0.0013 | +0.0006 | 14.6 |
| `placebo_bal_c1e-4` | -0.0008 | -0.0007 | -0.0002 | 22.8 |
| `placebo_bal_c3e-4` | 0.0050 | 0.0013 | +0.0037 | 19.0 |
| `placebo_bal_c1e-3` | 0.0056 | 0.0006 | +0.0049 | 14.8 |

Each model's epoch was picked by the smoothed validation IC, i.e. on the same days compared here, so the headline IC is
optimistic, most of all for runs whose validation IC swings from epoch to epoch. The all-epoch average involves no picking.

## Agents and scheduler (multi-agent variants)

| Variant | Mean abs agent rank corr | Scheduler entropy (max ln n) | Effective agents per day | Effective agents over time | Mean agent weights |
|---|---|---|---|---|---|
| `placebo_paper` | 0.096 | 1.102 (max 1.61) | 3.55 | 3.92 | [0.4, 0.122, 0.241, 0.118, 0.118] |
| `placebo_bal_c3e-6` | 0.095 | 0.768 (max 1.61) | 2.75 | 3.17 | [0.35, 0.083, 0.409, 0.078, 0.08] |
| `placebo_bal_c1e-5` | 0.095 | 1.162 (max 1.61) | 3.76 | 4.08 | [0.364, 0.13, 0.252, 0.121, 0.133] |
| `placebo_bal_c3e-5` | 0.077 | 1.244 (max 1.61) | 3.98 | 4.21 | [0.279, 0.133, 0.325, 0.132, 0.13] |
| `placebo_bal_c1e-4` | 0.086 | 1.157 (max 1.61) | 3.68 | 3.89 | [0.26, 0.114, 0.387, 0.124, 0.114] |
| `placebo_bal_c3e-4` | 0.082 | 1.002 (max 1.61) | 3.31 | 3.74 | [0.388, 0.1, 0.316, 0.096, 0.1] |
| `placebo_bal_c1e-3` | 0.109 | 0.957 (max 1.61) | 3.15 | 3.71 | [0.365, 0.111, 0.327, 0.102, 0.096] |

Validation days. Effective agents per day = mean over days of exp(scheduler entropy): 1 = one agent takes all the weight, n = equal weights.
Effective agents over time = exp(entropy of each seed's mean weights), averaged over seeds: how many agents are used across the period.

## Training days (restored model; from `detmaso usage`)

| Variant | Usage per agent, % (seed mean) | Effective agents over time | Dead agent-days, training (all epochs) | ... up to the restored epoch | Restored model, training days | Restored model, validation days |
|---|---|---|---|---|---|---|
| `placebo_paper` | 35.0, 14.8, 21.3, 15.1, 13.7 | 4.33 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| `placebo_bal_c3e-6` | 31.3, 10.8, 40.9, 9.2, 7.8 | 3.45 | [0, 0, 0, 0, 35] | [0, 0, 0, 0, 1] | [0, 0, 0, 0, 5] | [0, 0, 0, 0, 0] |
| `placebo_bal_c1e-5` | 34.9, 13.9, 25.9, 13.9, 11.4 | 4.17 | [0, 0, 0, 0, 14] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| `placebo_bal_c3e-5` | 27.0, 14.1, 32.1, 14.0, 12.7 | 4.29 | [0, 0, 0, 1, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| `placebo_bal_c1e-4` | 24.2, 12.3, 38.2, 13.8, 11.4 | 4.08 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| `placebo_bal_c3e-4` | 32.2, 12.4, 31.6, 12.6, 11.1 | 4.17 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| `placebo_bal_c1e-3` | 30.6, 13.2, 33.6, 12.5, 10.2 | 4.07 | [0, 0, 0, 0, 17] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |

Usage counts an agent's dead days (constant output, score 0; DEVIATIONS 4) as 0. Dead agent-days are summed over seeds, per agent.

## Rules (docs/RULES.md, scheduler balance)

Target usage (regime shares of the training days), %: 52.6, 7.9, 32.2, 5.1, 2.2. Rule M must hold in every seed, for the restored model and the last epoch trained.

| Strength c_bal | Variant | M0 all seeds trained | M1 usage >= half target | M2 own agent leads its mood | Lowest usage / target | Rule M |
|---|---|---|---|---|---|---|
| 3e-06 | `placebo_bal_c3e-6` | yes | **no** | **no** | 0.00 | fail |
| 1e-05 | `placebo_bal_c1e-5` | yes | **no** | **no** | 0.00 | fail |
| 3e-05 | `placebo_bal_c3e-5` | yes | **no** | **no** | 0.23 | fail |
| 0.0001 | `placebo_bal_c1e-4` | yes | **no** | **no** | 0.13 | fail |
| 0.0003 | `placebo_bal_c3e-4` | yes | **no** | **no** | 0.36 | fail |
| 0.001 | `placebo_bal_c1e-3` | yes | **no** | **no** | 0.23 | fail |

- **Rule S:** not fixed at any tested strength.
- **Outcome:** not fixed at any tested strength.
- **Rule P (placebo):** average placebo IC over the 8 run groups +0.0048 [-0.0056, +0.0157] (standard error 0.0054).
  - P1 range not entirely above 0: yes. P2 chosen strength not significant and not "better": yes.
  - **Placebo passed.**

## Per seed

| Variant | Seed | IC | ICIR | Sharpe | Max drawdown | Annual return | Best epoch | Epochs |
|---|---|---|---|---|---|---|---|---|
| `placebo_paper` | 0 | 0.0002 | 0.003 | 0.03 | 23.8% | -0.8% | 10 | 18 |
| `placebo_paper` | 1 | 0.0079 | 0.114 | 0.21 | 18.3% | 2.1% | 3 | 11 |
| `placebo_paper` | 2 | 0.0019 | 0.033 | 0.27 | 22.7% | 3.0% | 3 | 11 |
| `placebo_paper` | 3 | 0.0083 | 0.154 | -0.28 | 24.5% | -5.6% | 6 | 14 |
| `placebo_paper` | 4 | 0.0093 | 0.171 | -0.26 | 21.9% | -5.0% | 13 | 21 |
| `placebo_single_mlp` | 0 | 0.0161 | 0.260 | -0.02 | 22.9% | -1.5% | 11 | 19 |
| `placebo_single_mlp` | 1 | 0.0106 | 0.151 | 0.16 | 20.6% | 1.2% | 7 | 15 |
| `placebo_single_mlp` | 2 | 0.0066 | 0.095 | 0.01 | 22.5% | -1.0% | 7 | 15 |
| `placebo_single_mlp` | 3 | 0.0073 | 0.114 | -0.08 | 25.6% | -2.4% | 6 | 14 |
| `placebo_single_mlp` | 4 | 0.0075 | 0.116 | -0.01 | 23.6% | -1.4% | 13 | 21 |
| `placebo_bal_c3e-6` | 0 | 0.0104 | 0.183 | -0.10 | 22.3% | -2.6% | 17 | 25 |
| `placebo_bal_c3e-6` | 1 | 0.0029 | 0.051 | -0.68 | 27.4% | -10.7% | 10 | 18 |
| `placebo_bal_c3e-6` | 2 | -0.0007 | -0.012 | -0.07 | 24.9% | -2.3% | 3 | 11 |
| `placebo_bal_c3e-6` | 3 | -0.0046 | -0.073 | -0.28 | 25.9% | -5.0% | 4 | 12 |
| `placebo_bal_c3e-6` | 4 | 0.0108 | 0.179 | 0.41 | 21.5% | 5.4% | 3 | 11 |
| `placebo_bal_c1e-5` | 0 | 0.0136 | 0.209 | -0.16 | 24.1% | -3.5% | 5 | 13 |
| `placebo_bal_c1e-5` | 1 | 0.0109 | 0.192 | 0.03 | 23.6% | -0.7% | 30 | 38 |
| `placebo_bal_c1e-5` | 2 | 0.0154 | 0.264 | -0.19 | 21.3% | -4.1% | 3 | 11 |
| `placebo_bal_c1e-5` | 3 | 0.0105 | 0.207 | 0.02 | 20.8% | -0.8% | 13 | 21 |
| `placebo_bal_c1e-5` | 4 | 0.0006 | 0.009 | 0.22 | 25.7% | 2.2% | 3 | 11 |
| `placebo_bal_c3e-5` | 0 | -0.0018 | -0.033 | -0.24 | 27.2% | -4.8% | 6 | 14 |
| `placebo_bal_c3e-5` | 1 | 0.0018 | 0.035 | -0.22 | 23.2% | -4.3% | 18 | 26 |
| `placebo_bal_c3e-5` | 2 | 0.0004 | 0.006 | 0.29 | 24.9% | 3.5% | 3 | 11 |
| `placebo_bal_c3e-5` | 3 | 0.0043 | 0.075 | 0.23 | 23.2% | 2.4% | 3 | 11 |
| `placebo_bal_c3e-5` | 4 | -0.0083 | -0.162 | -0.47 | 28.2% | -8.3% | 3 | 11 |
| `placebo_bal_c1e-4` | 0 | -0.0006 | -0.008 | -0.10 | 24.3% | -2.7% | 4 | 12 |
| `placebo_bal_c1e-4` | 1 | 0.0067 | 0.126 | -0.36 | 28.2% | -6.3% | 61 | 69 |
| `placebo_bal_c1e-4` | 2 | -0.0046 | -0.078 | 0.01 | 25.5% | -1.1% | 3 | 11 |
| `placebo_bal_c1e-4` | 3 | 0.0005 | 0.009 | -0.10 | 27.7% | -2.8% | 3 | 11 |
| `placebo_bal_c1e-4` | 4 | -0.0061 | -0.094 | -0.36 | 27.7% | -6.7% | 3 | 11 |
| `placebo_bal_c3e-4` | 0 | 0.0091 | 0.154 | -0.73 | 29.1% | -11.9% | 7 | 15 |
| `placebo_bal_c3e-4` | 1 | 0.0068 | 0.119 | 0.04 | 24.3% | -0.6% | 4 | 12 |
| `placebo_bal_c3e-4` | 2 | -0.0026 | -0.043 | 0.00 | 24.6% | -1.1% | 3 | 11 |
| `placebo_bal_c3e-4` | 3 | 0.0069 | 0.115 | -0.47 | 25.8% | -8.1% | 15 | 23 |
| `placebo_bal_c3e-4` | 4 | 0.0047 | 0.086 | 0.04 | 18.2% | -0.6% | 26 | 34 |
| `placebo_bal_c1e-3` | 0 | 0.0039 | 0.065 | -0.60 | 26.9% | -9.8% | 3 | 11 |
| `placebo_bal_c1e-3` | 1 | 0.0045 | 0.078 | -0.71 | 28.2% | -11.0% | 16 | 24 |
| `placebo_bal_c1e-3` | 2 | 0.0172 | 0.280 | 0.18 | 25.3% | 1.6% | 8 | 16 |
| `placebo_bal_c1e-3` | 3 | -0.0038 | -0.058 | 0.14 | 25.9% | 1.0% | 4 | 12 |
| `placebo_bal_c1e-3` | 4 | 0.0061 | 0.105 | -0.42 | 26.0% | -7.2% | 3 | 11 |

## Figures

![metrics.png](figures/metrics.png)
![advantage.png](figures/advantage.png)
![cumulative_return.png](figures/cumulative_return.png)
![strength_sweep.png](figures/strength_sweep.png)
![usage_vs_target.png](figures/usage_vs_target.png)
![routing.png](figures/routing.png)
