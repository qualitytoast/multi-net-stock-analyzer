# Variant comparison: csi300, validation split

- Reference: `single_mlp` (run group `single_mlp`). Advantage = reference minus variant (drawdown: variant minus reference), so a positive number means the reference did better.
- Evaluated dates: 2023-01-03 .. 2024-02-28 (278 trading days; IC uses 258). The test period is not used.
- Data tag `2026-11-04`, data hash `8e81084fb70adad6` (shared by every run group).
- Bootstrap: 10,000 draws; days redrawn in circular 20-day blocks (one draw shared by all variants); seeds redrawn per variant; RNG seed 0.
- Rule fixed in advance (Step 1 decision D3): the reference is better if the 95% range lies entirely above 0, the variant is better if entirely below 0, otherwise no detectable difference. IC is primary; Sharpe is the second opinion.

## Run groups

| Variant | Config | Run group | Config hash | Model | Weights | Seeds |
|---|---|---|---|---|---|---|
| `single_mlp` | `configs/csi300_single_mlp.yaml` | `single_mlp` | `ef0560a807c0b155` | single_mlp | 4,865 | 0 1 2 3 4 |
| `single_mlp_state` | `configs/csi300_single_mlp_state.yaml` | `single_mlp_state` | `d38afb377e8a652b` | single_mlp | 5,633 | 0 1 2 3 4 |

## Metrics (mean +- std over seeds)

| Variant | IC | ICIR | Sharpe | Max drawdown | Annual return | Excess AR | Excess Sharpe |
|---|---|---|---|---|---|---|---|
| `single_mlp` | 0.1028 +- 0.0024 | 0.668 +- 0.026 | 0.66 +- 0.29 | 16.7% +- 3.8% | 8.3% +- 4.0% | 14.4% | 2.04 |
| `single_mlp_state` | 0.1404 +- 0.0142 | 0.729 +- 0.092 | 0.69 +- 0.26 | 20.6% +- 3.3% | 10.6% +- 4.7% | 17.0% | 1.52 |

## Advantage of `single_mlp` over each variant (95% bootstrap range)

| Variant | IC adv. [95%] | P(>0) | ICIR adv. [95%] | P(>0) | Sharpe adv. [95%] | P(>0) | IC rule | Sharpe rule | MDD adv. |
|---|---|---|---|---|---|---|---|---|---|
| `single_mlp_state` | -0.0376 [-0.1008, +0.0254] | 12% | -0.062 [-0.540, +0.371] | 39% | -0.02 [-1.34, +1.12] | 48% | no detectable difference | no detectable difference | +3.8% |

P(>0) = share of bootstrap draws in which the reference did better.

## Early-stopping check

| Variant | IC at the picked epoch (headline) | Validation IC averaged over every epoch | Gain from picking | Mean epochs trained |
|---|---|---|---|---|
| `single_mlp` | 0.1028 | 0.0932 | +0.0097 | 18.6 |
| `single_mlp_state` | 0.1404 | 0.0987 | +0.0417 | 13.0 |

Each model's epoch was picked by the smoothed validation IC, i.e. on the same days compared here, so the headline IC is
optimistic, most of all for runs whose validation IC swings from epoch to epoch. The all-epoch average involves no picking.

## Per seed

| Variant | Seed | IC | ICIR | Sharpe | Max drawdown | Annual return | Best epoch | Epochs |
|---|---|---|---|---|---|---|---|---|
| `single_mlp` | 0 | 0.1068 | 0.707 | 0.35 | 19.9% | 4.0% | 11 | 19 |
| `single_mlp` | 1 | 0.1026 | 0.682 | 0.90 | 10.7% | 11.0% | 4 | 12 |
| `single_mlp` | 2 | 0.1005 | 0.646 | 0.34 | 20.2% | 3.8% | 10 | 18 |
| `single_mlp` | 3 | 0.1017 | 0.647 | 0.88 | 16.8% | 11.3% | 14 | 22 |
| `single_mlp` | 4 | 0.1026 | 0.656 | 0.86 | 16.1% | 11.2% | 14 | 22 |
| `single_mlp_state` | 0 | 0.1497 | 0.833 | 0.52 | 20.5% | 7.7% | 4 | 12 |
| `single_mlp_state` | 1 | 0.1214 | 0.622 | 0.31 | 26.1% | 4.0% | 5 | 13 |
| `single_mlp_state` | 2 | 0.1474 | 0.749 | 0.77 | 17.7% | 11.6% | 5 | 13 |
| `single_mlp_state` | 3 | 0.1292 | 0.649 | 0.90 | 20.1% | 14.4% | 5 | 13 |
| `single_mlp_state` | 4 | 0.1541 | 0.795 | 0.93 | 18.3% | 15.3% | 6 | 14 |

## Figures

![metrics.png](figures/metrics.png)
![advantage.png](figures/advantage.png)
![cumulative_return.png](figures/cumulative_return.png)
