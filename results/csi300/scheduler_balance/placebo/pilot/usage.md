# Training-day scheduler usage (docs/RULES.md, Rules G and M)

Target (regime shares of the 2653 training days), %: 52.6, 7.9, 32.2, 5.1, 2.2

| Run group | c_bal | Seed | Model | Usage per agent, % | Effective agents | M1 | M2 |
|---|---|---|---|---|---|---|---|
| `placebo_bal_pilot_c1e-5` | 1e-05 | 0 | best | 49.68, 13.77, 24.99, 4.81, 6.75 | 3.65 | yes | no |
| `placebo_bal_pilot_c1e-5` | 1e-05 | 0 | last | 49.68, 13.77, 24.99, 4.81, 6.75 | 3.65 | yes | no |
| `placebo_bal_pilot_c1e-5` | 1e-05 | 1 | best | 38.98, 16.54, 23.25, 13.55, 7.68 | 4.36 | yes | no |
| `placebo_bal_pilot_c1e-5` | 1e-05 | 1 | last | 38.98, 16.54, 23.25, 13.55, 7.68 | 4.36 | yes | no |
| `placebo_bal_pilot_c1e-4` | 0.0001 | 0 | best | 32.21, 15.10, 29.52, 14.37, 8.80 | 4.50 | yes | no |
| `placebo_bal_pilot_c1e-4` | 0.0001 | 0 | last | 32.21, 15.10, 29.52, 14.37, 8.80 | 4.50 | yes | no |
| `placebo_bal_pilot_c1e-4` | 0.0001 | 1 | best | 46.96, 11.10, 28.76, 7.98, 5.19 | 3.72 | yes | yes |
| `placebo_bal_pilot_c1e-4` | 0.0001 | 1 | last | 46.96, 11.10, 28.76, 7.98, 5.19 | 3.72 | yes | yes |
| `placebo_bal_pilot_c1e-3` | 0.001 | 0 | best | 49.88, 15.05, 14.47, 11.32, 9.28 | 3.97 | no | yes |
| `placebo_bal_pilot_c1e-3` | 0.001 | 0 | last | 49.88, 15.05, 14.47, 11.32, 9.28 | 3.97 | no | yes |
| `placebo_bal_pilot_c1e-3` | 0.001 | 1 | best | 47.66, 13.25, 21.81, 10.64, 6.64 | 3.94 | yes | yes |
| `placebo_bal_pilot_c1e-3` | 0.001 | 1 | last | 47.66, 13.25, 21.81, 10.64, 6.64 | 3.94 | yes | yes |
| `placebo_bal_pilot_c1e-2` | 0.01 | 0 | best | 44.51, 12.31, 31.38, 5.97, 5.83 | 3.73 | yes | no |
| `placebo_bal_pilot_c1e-2` | 0.01 | 0 | last | 44.51, 12.31, 31.38, 5.97, 5.83 | 3.73 | yes | no |
| `placebo_bal_pilot_c1e-2` | 0.01 | 1 | best | 48.23, 12.15, 28.82, 8.42, 2.39 | 3.54 | yes | no |
| `placebo_bal_pilot_c1e-2` | 0.01 | 1 | last | 48.23, 12.15, 28.82, 8.42, 2.39 | 3.54 | yes | no |
| `placebo_bal_pilot_c1e-1` | 0.1 | 0 | best | 51.40, 8.84, 31.56, 5.63, 2.56 | 3.24 | yes | no |
| `placebo_bal_pilot_c1e-1` | 0.1 | 0 | last | 51.40, 8.84, 31.56, 5.63, 2.56 | 3.24 | yes | no |
| `placebo_bal_pilot_c1e-1` | 0.1 | 1 | best | 52.53, 8.49, 29.69, 6.85, 2.43 | 3.26 | yes | no |
| `placebo_bal_pilot_c1e-1` | 0.1 | 1 | last | 52.53, 8.49, 29.69, 6.85, 2.43 | 3.26 | yes | no |
| `placebo_bal_pilot_c1e0` | 1 | 0 | best | 52.44, 8.02, 32.18, 5.10, 2.26 | 3.14 | yes | no |
| `placebo_bal_pilot_c1e0` | 1 | 0 | last | 52.44, 8.02, 32.18, 5.10, 2.26 | 3.14 | yes | no |
| `placebo_bal_pilot_c1e0` | 1 | 1 | best | 52.60, 7.98, 32.07, 5.11, 2.24 | 3.13 | yes | no |
| `placebo_bal_pilot_c1e0` | 1 | 1 | last | 52.60, 7.98, 32.07, 5.11, 2.24 | 3.13 | yes | no |

## Rule G

- M1 at the last pilot epoch, both seeds: {'1e-05': True, '0.0001': True, '0.001': False, '0.01': True, '0.1': True, '1': True}
- Anchor g0: 1e-05
- Grid: [3e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.001]
- Stop: False
