# Training-day scheduler usage (docs/RULES.md, Rules G and M)

Target (regime shares of the 2653 training days), %: 52.6, 7.9, 32.2, 5.1, 2.2

| Run group | c_bal | Seed | Model | Usage per agent, % | Effective agents | M1 | M2 |
|---|---|---|---|---|---|---|---|
| `bal_pilot_c1e-5` | 1e-05 | 0 | best | 57.63, 0.16, 38.39, 3.64, 0.18 | 2.29 | no | no |
| `bal_pilot_c1e-5` | 1e-05 | 0 | last | 57.63, 0.16, 38.39, 3.64, 0.18 | 2.29 | no | no |
| `bal_pilot_c1e-5` | 1e-05 | 1 | best | 56.23, 0.49, 42.67, 0.32, 0.29 | 2.11 | no | no |
| `bal_pilot_c1e-5` | 1e-05 | 1 | last | 56.23, 0.49, 42.67, 0.32, 0.29 | 2.11 | no | no |
| `bal_pilot_c1e-4` | 0.0001 | 0 | best | 54.47, 0.48, 34.52, 10.26, 0.27 | 2.65 | no | no |
| `bal_pilot_c1e-4` | 0.0001 | 0 | last | 54.47, 0.48, 34.52, 10.26, 0.27 | 2.65 | no | no |
| `bal_pilot_c1e-4` | 0.0001 | 1 | best | 50.51, 0.20, 48.97, 0.16, 0.16 | 2.07 | no | no |
| `bal_pilot_c1e-4` | 0.0001 | 1 | last | 50.51, 0.20, 48.97, 0.16, 0.16 | 2.07 | no | no |
| `bal_pilot_c1e-3` | 0.001 | 0 | best | 54.30, 1.53, 37.71, 6.11, 0.35 | 2.60 | no | no |
| `bal_pilot_c1e-3` | 0.001 | 0 | last | 54.30, 1.53, 37.71, 6.11, 0.35 | 2.60 | no | no |
| `bal_pilot_c1e-3` | 0.001 | 1 | best | 55.06, 2.44, 36.27, 5.79, 0.44 | 2.65 | no | no |
| `bal_pilot_c1e-3` | 0.001 | 1 | last | 55.06, 2.44, 36.27, 5.79, 0.44 | 2.65 | no | no |
| `bal_pilot_c1e-2` | 0.01 | 0 | best | 57.07, 8.22, 25.74, 7.66, 1.31 | 3.09 | yes | no |
| `bal_pilot_c1e-2` | 0.01 | 0 | last | 57.07, 8.22, 25.74, 7.66, 1.31 | 3.09 | yes | no |
| `bal_pilot_c1e-2` | 0.01 | 1 | best | 55.29, 9.74, 32.12, 1.98, 0.86 | 2.82 | no | no |
| `bal_pilot_c1e-2` | 0.01 | 1 | last | 55.29, 9.74, 32.12, 1.98, 0.86 | 2.82 | no | no |
| `bal_pilot_c1e-1` | 0.1 | 0 | best | 53.12, 7.74, 31.82, 4.40, 2.92 | 3.12 | yes | yes |
| `bal_pilot_c1e-1` | 0.1 | 0 | last | 53.12, 7.74, 31.82, 4.40, 2.92 | 3.12 | yes | yes |
| `bal_pilot_c1e-1` | 0.1 | 1 | best | 51.30, 8.51, 33.04, 4.61, 2.54 | 3.17 | yes | yes |
| `bal_pilot_c1e-1` | 0.1 | 1 | last | 51.30, 8.51, 33.04, 4.61, 2.54 | 3.17 | yes | yes |
| `bal_pilot_c1e0` | 1 | 0 | best | 53.26, 7.73, 31.51, 5.18, 2.32 | 3.12 | yes | no |
| `bal_pilot_c1e0` | 1 | 0 | last | 53.26, 7.73, 31.51, 5.18, 2.32 | 3.12 | yes | no |
| `bal_pilot_c1e0` | 1 | 1 | best | 53.11, 7.62, 31.58, 5.39, 2.29 | 3.13 | yes | no |
| `bal_pilot_c1e0` | 1 | 1 | last | 53.11, 7.62, 31.58, 5.39, 2.29 | 3.13 | yes | no |

## Rule G

- M1 at the last pilot epoch, both seeds: {'1e-05': False, '0.0001': False, '0.001': False, '0.01': False, '0.1': True, '1': True}
- Anchor g0: 0.1
- Grid: [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
- Stop: False
