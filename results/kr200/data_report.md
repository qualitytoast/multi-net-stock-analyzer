# Data QA report: kr200

- Data tag: `marcap-d3b7036` (sha256 `12b24b74dd15cddadeb713606c9da6dba18a380f7f9deab225f6fb0f905ec802`)
- Calendar loaded: 2011-01-03 .. 2026-04-30 (3769 days)
- Model dates: 2012-01-02 .. 2026-02-27 (3478 days); stocks that are members on a model date before the test period: 378

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2012-01-02 | 2022-12-29 | 2709 |
| train_fit | 2012-01-02 | 2022-12-01 | 2689 |
| valid | 2023-01-02 | 2024-02-28 | 285 |
| valid_fit | 2023-01-02 | 2024-01-29 | 265 |
| test | 2024-03-04 | 2026-02-27 | 483 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 193, median 199, max 200
- Stocks with a 20-day label per day: min 192, median 199
- Smallest universe on 2017-10-13
- Label counts stop at 2024-01-30: the 20-day answers of later dates end inside the test period

## Spot prices

Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data
check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.0%, vol_5 0.0%, vol_10 0.1%, vol_20 0.2%, vol_60 0.4%, rturn_5 0.0%, rturn_10 0.1%, rturn_20 0.2%, rturn_60 0.4%, pvcorr_5 0.1%, pvcorr_10 0.1%, pvcorr_20 0.2%, upfrac_5 0.0%, upfrac_10 0.1%, upfrac_20 0.2%, madev_5 0.0%, madev_20 0.2%, madev_60 0.4%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.0%, skew60 0.4%, ac1_60 0.5%, momsharpe 0.4%, range20 0.2%, logvol20 0.2%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.396, K=3: 0.310, K=4: 0.233, K=5: 0.238, K=6: 0.192, K=7: 0.166, K=8: 0.172, K=9: 0.173, K=10: 0.179

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days |
|---|---|
| 1 Low-Vol Bull | 18.0% |
| 2 High-Vol Bull | 3.8% |
| 3 Sideways | 50.2% |
| 4 Low-Vol Bear | 27.2% |
| 5 Tail-Risk | 0.7% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0087 | 0.0074 | 0.0249 | 0.0096 | 0.0276 | 0.0251 | 0.0104 | 0.0659 | 0.025 | 0.0109 | 0.1381 | 0.025 |
| High-Vol Bull | 0.0131 | 0.0136 | 0.0273 | 0.0152 | 0.0705 | 0.0275 | 0.0223 | 0.1083 | 0.0278 | 0.0246 | 0.0273 | 0.0257 |
| Sideways | 0.006 | 0.0037 | 0.0191 | 0.0068 | 0.0109 | 0.0193 | 0.0074 | 0.0187 | 0.0195 | 0.0079 | 0.0199 | 0.0198 |
| Low-Vol Bear | 0.0112 | -0.0122 | 0.0215 | 0.0112 | -0.0355 | 0.0212 | 0.0105 | -0.0591 | 0.0211 | 0.0103 | -0.0717 | 0.0212 |
| Tail-Risk | 0.0394 | 0.0364 | 0.0382 | 0.0483 | -0.107 | 0.0353 | 0.0305 | -0.2747 | 0.0244 | 0.0222 | -0.2467 | 0.0203 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2012-2022 |
|---|---|
| mom_20 | -0.0220 |
| vol_20 | -0.0246 |
| mom_5 | -0.0044 |
| rturn_20 | +0.0055 |
| logvol20 | -0.0100 |

## Membership, pricing and survivorship (docs/RULES.md, Rule ND)

Dates checked: the 2995 model dates before the test period, 2012-01-02 .. 2024-02-29; no later date is read.

- Member = in the `kr200` list on that day.
- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.
- A stock's data ends early if its last close on the dates checked falls before 2024-02-29 (the last date checked), or it
  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading
  would show 0 everywhere. A stock that is only suspended on 2024-02-29 is counted too (nothing later is read).
- Ended members = stocks that are members on at least one date of the year and whose data ends early.
- Last close in year = stocks whose data ends early with their last close in that year.

| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |
|---|---|---|---|---|---|---|---|---|---|
| 2012 | 248 | 200 | 200 | 200 | 49600 | 49536 | 99.87% | 24 | 2 |
| 2013 | 247 | 197 | 200 | 200 | 49191 | 48976 | 99.56% | 23 | 2 |
| 2014 | 245 | 198 | 200 | 200 | 48899 | 48719 | 99.63% | 19 | 4 |
| 2015 | 248 | 196 | 199 | 200 | 49181 | 49057 | 99.75% | 16 | 4 |
| 2016 | 246 | 199 | 200 | 200 | 49178 | 48980 | 99.60% | 10 | 4 |
| 2017 | 243 | 198 | 199 | 200 | 48335 | 47857 | 99.01% | 9 | 3 |
| 2018 | 244 | 199 | 200 | 200 | 48750 | 48593 | 99.68% | 6 | 1 |
| 2019 | 246 | 199 | 200 | 200 | 49125 | 49080 | 99.91% | 5 | 1 |
| 2020 | 248 | 199 | 200 | 200 | 49528 | 49396 | 99.73% | 5 | 3 |
| 2021 | 248 | 200 | 200 | 200 | 49600 | 49428 | 99.65% | 3 | 2 |
| 2022 | 246 | 200 | 200 | 200 | 49200 | 49184 | 99.97% | 2 | 1 |
| 2023 | 245 | 198 | 200 | 200 | 48908 | 48811 | 99.80% | 2 | 2 |
| 2024 | 41 | 200 | 200 | 200 | 8200 | 8200 | 100.00% | 0 | 2 |

- All dates checked: members per day min 196, median 200, max 200; 597695 member-days, 595817 priced (99.69%), 1878 unpriced.
- Survivorship: 378 stocks are members on at least one date checked; the data of 31 of them ends early (0 of those 31 with no close on any date checked).
- Warm-up (the 248 dates loaded before the first model date, 2011-01-03 .. 2011-12-29): members per day min 198, median 199, max 200.
- Market-state warm-up: the first model date (2012-01-02) reads the member returns of the 120 dates 2011-07-11 .. 2011-12-29 (the longest window is 120 days). Members per day on those dates: min 198, median 199, max 200; members with a one-day return: min 197, median 198.

### Unpriced member spans

A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced
again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.

| Span length (trading days) | Spans | Unpriced member-days |
|---|---|---|
| 1-5 | 38 | 78 |
| 6-20 | 59 | 812 |
| 21-60 | 25 | 673 |
| 61-250 | 0 | 0 |
| 251 or more | 1 | 315 |

The 10 longest:

| Stock | First unpriced member date | Last | Trading days | Note |
|---|---|---|---|---|
| KR042660 | 2016-07-15 | 2017-10-27 | 315 | - |
| KR020560 | 2021-05-27 | 2021-07-15 | 36 | - |
| KR002620 | 2017-05-30 | 2017-07-14 | 33 | - |
| KR003490 | 2013-07-30 | 2013-09-13 | 33 | - |
| KR005850 | 2020-05-21 | 2020-07-03 | 32 | - |
| KR037560 | 2015-11-02 | 2015-12-14 | 31 | - |
| KR004800 | 2018-05-30 | 2018-07-12 | 30 | - |
| KR007860 | 2014-06-27 | 2014-08-07 | 30 | - |
| KR027410 | 2017-10-30 | 2017-12-07 | 29 | - |
| KR012630 | 2018-04-27 | 2018-06-11 | 28 | - |
