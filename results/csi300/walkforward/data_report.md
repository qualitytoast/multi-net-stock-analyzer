# Data QA report: csi300

- Data tag: `2026-11-04` (sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`)
- Calendar loaded: 2007-01-04 .. 2026-04-30 (4695 days)
- Model dates: 2008-01-02 .. 2026-02-27 (4410 days); stocks ever in the index: 895

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2008-01-02 | 2022-12-30 | 3649 |
| train_fit | 2008-01-02 | 2022-12-02 | 3629 |
| valid | 2023-01-03 | 2024-02-28 | 278 |
| valid_fit | 2023-01-03 | 2024-01-23 | 258 |
| test | 2024-03-01 | 2026-02-27 | 482 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 204, median 293, max 300
- Stocks with a 20-day label per day: min 192, median 287
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

K=2: 0.469, K=3: 0.312, K=4: 0.231, K=5: 0.235, K=6: 0.197, K=7: 0.189, K=8: 0.193, K=9: 0.196, K=10: 0.182

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days | Paper (Table 4) |
|---|---|---|
| 1 Low-Vol Bull | 11.1% | 35% |
| 2 High-Vol Bull | 7.5% | 12% |
| 3 Sideways | 47.8% | 32% |
| 4 Low-Vol Bear | 28.6% | 16% |
| 5 Tail-Risk | 5.0% | 5% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0177 | 0.0274 | 0.0256 | 0.0185 | 0.1126 | 0.0249 | 0.0185 | 0.2657 | 0.0237 | 0.0196 | 0.407 | 0.0232 |
| High-Vol Bull | 0.0257 | 0.0191 | 0.0233 | 0.0307 | 0.0288 | 0.0246 | 0.0324 | -0.0394 | 0.0253 | 0.0326 | -0.1327 | 0.0263 |
| Sideways | 0.0106 | 0.0017 | 0.0163 | 0.0115 | 0.0057 | 0.0164 | 0.0125 | -0.0031 | 0.0168 | 0.0132 | -0.0118 | 0.017 |
| Low-Vol Bear | 0.0131 | -0.0051 | 0.0218 | 0.0138 | -0.0153 | 0.0218 | 0.0138 | -0.0034 | 0.0217 | 0.0136 | 0.0245 | 0.0214 |
| Tail-Risk | 0.0402 | -0.0614 | 0.0282 | 0.0379 | -0.1899 | 0.0273 | 0.035 | -0.2799 | 0.0269 | 0.0322 | -0.2843 | 0.0266 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2008-2022 |
|---|---|
| mom_20 | -0.0440 |
| vol_20 | -0.0394 |
| mom_5 | -0.0178 |
| rturn_20 | -0.0024 |
| logvol20 | -0.0287 |

## Per year (walk-forward history check)

Universe = index members that traded the day before; missing = average share of universe stock-days with a missing
feature (before filling with 0); moves > 21% = universe stock-days whose one-day return exceeds every price limit
(possible data errors, or days without a limit such as a stock's first trading days).

| Year | Days | Universe min | Universe median | Labelled median | Missing features | Moves > 21% |
|---|---|---|---|---|---|---|
| 2008 | 246 | 272 | 288 | 279 | 0.76% | 0 |
| 2009 | 244 | 275 | 293 | 285 | 0.68% | 1 |
| 2010 | 242 | 280 | 291 | 282 | 0.58% | 0 |
| 2011 | 244 | 278 | 292 | 284 | 0.50% | 0 |
| 2012 | 243 | 278 | 296 | 292 | 0.36% | 0 |
| 2013 | 238 | 287 | 293 | 289 | 0.69% | 0 |
| 2014 | 245 | 280 | 289 | 282 | 1.05% | 0 |
| 2015 | 244 | 204 | 281 | 271 | 2.18% | 0 |
| 2016 | 244 | 276 | 288 | 282 | 1.27% | 0 |
| 2017 | 244 | 275 | 284 | 279 | 0.98% | 0 |
| 2018 | 243 | 281 | 288 | 286 | 0.68% | 0 |
| 2019 | 244 | 296 | 299 | 298 | 0.17% | 0 |
| 2020 | 243 | 298 | 300 | 299 | 0.08% | 0 |
| 2021 | 243 | 297 | 300 | 299 | 0.07% | 0 |
| 2022 | 242 | 298 | 300 | 300 | 0.04% | 0 |
| 2023 | 242 | 299 | 300 | 300 | 0.00% | 0 |
| 2024 | 37 | 300 | 300 | 300 | 0.00% | 0 |
