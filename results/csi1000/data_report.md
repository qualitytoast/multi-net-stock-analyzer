# Data QA report: csi1000

- Data tag: `2026-11-04` (sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`)
- Calendar loaded: 2015-01-05 .. 2026-04-30 (2751 days)
- Model dates: 2016-01-04 .. 2026-02-27 (2464 days); stocks that are members on a model date before the test period: 2396

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2016-01-04 | 2022-12-30 | 1703 |
| train_fit | 2016-01-04 | 2022-12-02 | 1683 |
| valid | 2023-01-03 | 2024-02-28 | 278 |
| valid_fit | 2023-01-03 | 2024-01-23 | 258 |
| test | 2024-03-01 | 2026-02-27 | 482 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 885, median 998, max 1000
- Stocks with a 20-day label per day: min 855, median 995
- Smallest universe on 2018-02-09
- Label counts stop at 2024-01-24: the 20-day answers of later dates end inside the test period

## Spot prices

Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data
check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.0%, vol_5 0.3%, vol_10 0.6%, vol_20 1.1%, vol_60 1.6%, rturn_5 0.3%, rturn_10 0.6%, rturn_20 1.1%, rturn_60 1.6%, pvcorr_5 0.3%, pvcorr_10 0.6%, pvcorr_20 1.1%, upfrac_5 0.3%, upfrac_10 0.6%, upfrac_20 1.1%, madev_5 0.3%, madev_20 1.1%, madev_60 1.6%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.3%, skew60 1.6%, ac1_60 1.7%, momsharpe 1.6%, range20 1.1%, logvol20 1.1%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.255, K=3: 0.279, K=4: 0.254, K=5: 0.247, K=6: 0.218, K=7: 0.225, K=8: 0.202, K=9: 0.222, K=10: 0.205

The silhouette-optimal count is **K = 3**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days |
|---|---|
| 1 Low-Vol Bull | 32.8% |
| 2 High-Vol Bull | 13.9% |
| 3 Sideways | 30.5% |
| 4 Low-Vol Bear | 19.1% |
| 5 Tail-Risk | 3.8% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0105 | 0.0025 | 0.0266 | 0.0119 | 0.0118 | 0.0267 | 0.0132 | 0.0275 | 0.0269 | 0.0137 | 0.0621 | 0.0269 |
| High-Vol Bull | 0.0188 | 0.0128 | 0.0261 | 0.0204 | 0.0448 | 0.026 | 0.0201 | 0.129 | 0.0251 | 0.0208 | 0.145 | 0.0246 |
| Sideways | 0.0094 | 0.0054 | 0.0192 | 0.0111 | 0.0065 | 0.0194 | 0.0123 | 0.0017 | 0.0196 | 0.0137 | -0.0113 | 0.02 |
| Low-Vol Bear | 0.0175 | -0.021 | 0.023 | 0.0176 | -0.0547 | 0.0228 | 0.0173 | -0.1167 | 0.0227 | 0.0168 | -0.1634 | 0.0228 |
| Tail-Risk | 0.0358 | -0.0176 | 0.0255 | 0.0366 | -0.0686 | 0.0259 | 0.0347 | -0.1367 | 0.0277 | 0.0385 | 0.0939 | 0.0291 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2016-2022 |
|---|---|
| mom_20 | -0.0596 |
| vol_20 | -0.0960 |
| mom_5 | -0.0296 |
| rturn_20 | -0.0175 |
| logvol20 | -0.0697 |

## Membership, pricing and survivorship (docs/RULES.md, Rule ND)

Dates checked: the 1982 model dates before the test period, 2016-01-04 .. 2024-02-29; no later date is read.

- Member = in the `csi1000` list on that day.
- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.
- A stock's data ends early if its last close on the dates checked falls before 2024-02-29 (the last date checked), or it
  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading
  would show 0 everywhere. A stock that is only suspended on 2024-02-29 is counted too (nothing later is read).
- Ended members = stocks that are members on at least one date of the year and whose data ends early.
- Last close in year = stocks whose data ends early with their last close in that year.

| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |
|---|---|---|---|---|---|---|---|---|---|
| 2016 | 244 | 1000 | 1000 | 1000 | 244000 | 223731 | 91.69% | 71 | 0 |
| 2017 | 244 | 1000 | 1000 | 1000 | 244000 | 226484 | 92.82% | 61 | 1 |
| 2018 | 243 | 1000 | 1000 | 1000 | 243000 | 230180 | 94.72% | 55 | 3 |
| 2019 | 244 | 1000 | 1000 | 1000 | 244000 | 243447 | 99.77% | 45 | 5 |
| 2020 | 243 | 1000 | 1000 | 1000 | 243000 | 242543 | 99.81% | 31 | 11 |
| 2021 | 243 | 1000 | 1000 | 1000 | 243000 | 242664 | 99.86% | 11 | 11 |
| 2022 | 242 | 1000 | 1000 | 1001 | 242001 | 241752 | 99.90% | 6 | 24 |
| 2023 | 242 | 1000 | 1001 | 1001 | 242241 | 241858 | 99.84% | 3 | 31 |
| 2024 | 37 | 1000 | 1000 | 1000 | 37000 | 36974 | 99.93% | 0 | 3 |

- All dates checked: members per day min 1000, median 1000, max 1001; 1982242 member-days, 1929633 priced (97.35%), 52609 unpriced.
- Survivorship: 2396 stocks are members on at least one date checked; the data of 90 of them ends early (1 of those 90 with no close on any date checked).
- Warm-up (the 244 dates loaded before the first model date, 2015-01-05 .. 2015-12-31): members per day min 3, median 1000, max 1002.
- Market-state warm-up: the first model date (2016-01-04) reads the member returns of the 120 dates 2015-07-08 .. 2015-12-31 (the longest window is 120 days). Members per day on those dates: min 1000, median 1000, max 1000; members with a one-day return: min 384, median 822.5.

### Unpriced member spans

A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced
again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.

| Span length (trading days) | Spans | Unpriced member-days |
|---|---|---|
| 1-5 | 808 | 2337 |
| 6-20 | 604 | 6152 |
| 21-60 | 234 | 9953 |
| 61-250 | 334 | 33734 |
| 251 or more | 1 | 433 |

The 10 longest:

| Stock | First unpriced member date | Last | Trading days | Note |
|---|---|---|---|---|
| SZ000029 | 2016-09-14 | 2018-06-28 | 433 | - |
| SZ302132 | 2022-12-30 | 2023-12-28 | 242 | - |
| SH600084 | 2017-07-10 | 2018-06-12 | 227 | - |
| SH600226 | 2017-11-28 | 2018-10-30 | 224 | - |
| SZ002260 | 2016-12-05 | 2017-11-03 | 224 | - |
| SZ000035 | 2017-08-22 | 2018-07-06 | 213 | - |
| SH600209 | 2016-02-24 | 2016-12-27 | 209 | - |
| SH600577 | 2016-05-03 | 2017-03-01 | 202 | - |
| SZ300109 | 2017-03-27 | 2018-01-11 | 198 | - |
| SZ002389 | 2016-02-03 | 2016-11-23 | 195 | - |
