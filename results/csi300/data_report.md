# Data QA report: csi300

- Data tag: `2026-11-04` (sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`)
- Calendar loaded: 2011-01-04 .. 2026-04-30 (3721 days)
- Model dates: 2012-01-04 .. 2026-02-27 (3434 days); stocks ever in the index: 782

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2012-01-04 | 2022-12-30 | 2673 |
| train_fit | 2012-01-04 | 2022-12-02 | 2653 |
| valid | 2023-01-03 | 2024-02-28 | 278 |
| valid_fit | 2023-01-03 | 2024-01-23 | 258 |
| test | 2024-03-01 | 2026-02-27 | 482 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 204, median 296, max 300
- Stocks with a 20-day label per day: min 192, median 292
- Smallest universe on 2015-07-10

## Spot prices (raw close = $close / $factor)

| Stock | Date | Ours | Expected |
|---|---|---|---|
| SH600519 | 2024-03-01 | 1685.06 | 1685.06 |
| SZ000001 | 2024-03-01 | 10.49 | 10.49 |
| SH601318 | 2024-03-01 | 42.99 | 42.99 |

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.1%, vol_5 0.3%, vol_10 0.6%, vol_20 1.0%, vol_60 1.5%, rturn_5 0.3%, rturn_10 0.6%, rturn_20 1.0%, rturn_60 1.5%, pvcorr_5 0.3%, pvcorr_10 0.6%, pvcorr_20 1.0%, upfrac_5 0.3%, upfrac_10 0.6%, upfrac_20 1.0%, madev_5 0.3%, madev_20 1.0%, madev_60 1.5%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.3%, skew60 1.5%, ac1_60 1.6%, momsharpe 1.5%, range20 1.0%, logvol20 1.0%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.482, K=3: 0.290, K=4: 0.229, K=5: 0.247, K=6: 0.218, K=7: 0.200, K=8: 0.201, K=9: 0.203, K=10: 0.202

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days | Paper (Table 4) |
|---|---|---|
| 1 Low-Vol Bull | 52.5% | 35% |
| 2 High-Vol Bull | 7.9% | 12% |
| 3 Sideways | 32.4% | 32% |
| 4 Low-Vol Bear | 5.1% | 16% |
| 5 Tail-Risk | 2.2% | 5% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0098 | 0.0029 | 0.0164 | 0.0107 | 0.0099 | 0.0165 | 0.0118 | 0.0074 | 0.0168 | 0.0126 | 0.0001 | 0.0169 |
| High-Vol Bull | 0.0177 | 0.0261 | 0.0269 | 0.0183 | 0.114 | 0.0263 | 0.0162 | 0.2824 | 0.024 | 0.0156 | 0.4149 | 0.0225 |
| Sideways | 0.0124 | -0.0039 | 0.0223 | 0.0129 | -0.0146 | 0.0222 | 0.0126 | -0.0065 | 0.0219 | 0.0124 | 0.0033 | 0.0216 |
| Low-Vol Bear | 0.0231 | 0.0032 | 0.021 | 0.0261 | 0.0049 | 0.0218 | 0.0303 | -0.0639 | 0.0235 | 0.0338 | -0.1118 | 0.0262 |
| Tail-Risk | 0.0457 | -0.0359 | 0.033 | 0.0475 | -0.1448 | 0.0332 | 0.0419 | -0.1678 | 0.0328 | 0.0325 | 0.1879 | 0.0299 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): the cluster named Low-Vol Bear is more volatile than High-Vol Bull.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2012-2022 |
|---|---|
| mom_20 | -0.0385 |
| vol_20 | -0.0445 |
| mom_5 | -0.0131 |
| rturn_20 | -0.0006 |
| logvol20 | -0.0201 |
