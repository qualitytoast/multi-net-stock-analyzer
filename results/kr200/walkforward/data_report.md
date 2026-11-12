# Data QA report: kr200

- Data tag: `marcap-d3b7036` (sha256 `12b24b74dd15cddadeb713606c9da6dba18a380f7f9deab225f6fb0f905ec802`)
- Calendar loaded: 2007-01-02 .. 2026-04-30 (4767 days)
- Model dates: 2008-01-02 .. 2026-02-27 (4478 days); stocks that are members on a model date before the test period: 429

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2008-01-02 | 2022-12-29 | 3709 |
| train_fit | 2008-01-02 | 2022-12-01 | 3689 |
| valid | 2023-01-02 | 2024-02-28 | 285 |
| valid_fit | 2023-01-02 | 2024-01-29 | 265 |
| test | 2024-03-04 | 2026-02-27 | 483 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 193, median 199, max 200
- Stocks with a 20-day label per day: min 192, median 198
- Smallest universe on 2010-05-04
- Label counts stop at 2024-01-30: the 20-day answers of later dates end inside the test period

## Spot prices

Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data
check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.0%, vol_5 0.0%, vol_10 0.1%, vol_20 0.2%, vol_60 0.4%, rturn_5 0.0%, rturn_10 0.1%, rturn_20 0.2%, rturn_60 0.4%, pvcorr_5 0.1%, pvcorr_10 0.1%, pvcorr_20 0.2%, upfrac_5 0.0%, upfrac_10 0.1%, upfrac_20 0.2%, madev_5 0.0%, madev_20 0.2%, madev_60 0.4%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.0%, skew60 0.4%, ac1_60 0.5%, momsharpe 0.4%, range20 0.2%, logvol20 0.2%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.527, K=3: 0.363, K=4: 0.288, K=5: 0.241, K=6: 0.214, K=7: 0.189, K=8: 0.183, K=9: 0.181, K=10: 0.178

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days |
|---|---|
| 1 Low-Vol Bull | 25.1% |
| 2 High-Vol Bull | 7.0% |
| 3 Sideways | 53.9% |
| 4 Low-Vol Bear | 12.3% |
| 5 Tail-Risk | 1.7% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0085 | 0.0088 | 0.0238 | 0.0094 | 0.0297 | 0.0239 | 0.0105 | 0.0671 | 0.024 | 0.0113 | 0.1295 | 0.0243 |
| High-Vol Bull | 0.0157 | 0.0168 | 0.0284 | 0.019 | 0.0699 | 0.0294 | 0.0247 | 0.1035 | 0.0304 | 0.0274 | 0.0273 | 0.0299 |
| Sideways | 0.0072 | -0.0001 | 0.0195 | 0.0079 | 0.0003 | 0.0197 | 0.0083 | -0.0004 | 0.02 | 0.0085 | 0.0028 | 0.0202 |
| Low-Vol Bear | 0.0172 | -0.0223 | 0.0258 | 0.0163 | -0.0631 | 0.0248 | 0.0145 | -0.0939 | 0.0242 | 0.014 | -0.1058 | 0.0242 |
| Tail-Risk | 0.0465 | 0.0046 | 0.0442 | 0.0496 | -0.1191 | 0.0419 | 0.0392 | -0.3068 | 0.0348 | 0.0295 | -0.43 | 0.0292 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2008-2022 |
|---|---|
| mom_20 | -0.0243 |
| vol_20 | -0.0207 |
| mom_5 | -0.0150 |
| rturn_20 | +0.0124 |
| logvol20 | -0.0094 |

## Per year (walk-forward history check)

Universe = index members that traded the day before; missing = average share of universe stock-days with a missing
feature (before filling with 0); moves > 31% = universe stock-days whose one-day return exceeds every price limit
(possible data errors, or days without a limit such as a stock's first trading days).

| Year | Days | Universe min | Universe median | Labelled median | Missing features | Moves > 31% |
|---|---|---|---|---|---|---|
| 2008 | 248 | 196 | 199 | 198 | 0.25% | 0 |
| 2009 | 253 | 197 | 198 | 198 | 0.16% | 0 |
| 2010 | 251 | 193 | 199 | 198 | 0.20% | 0 |
| 2011 | 248 | 196 | 198 | 198 | 0.10% | 0 |
| 2012 | 248 | 198 | 200 | 200 | 0.07% | 0 |
| 2013 | 247 | 195 | 198 | 198 | 0.15% | 0 |
| 2014 | 245 | 197 | 199 | 198 | 0.22% | 0 |
| 2015 | 248 | 195 | 198 | 197 | 0.04% | 0 |
| 2016 | 246 | 197 | 199 | 199 | 0.10% | 0 |
| 2017 | 243 | 193 | 197 | 196 | 0.27% | 0 |
| 2018 | 244 | 195 | 199 | 199 | 0.21% | 0 |
| 2019 | 246 | 197 | 200 | 199 | 0.05% | 0 |
| 2020 | 248 | 198 | 199 | 199 | 0.15% | 0 |
| 2021 | 248 | 198 | 199 | 199 | 0.22% | 0 |
| 2022 | 246 | 198 | 200 | 200 | 0.03% | 0 |
| 2023 | 245 | 196 | 200 | 200 | 0.08% | 0 |
| 2024 | 41 | 200 | 200 | 200 | 0.00% | 0 |

## Membership, pricing and survivorship (docs/RULES.md, Rule ND)

Dates checked: the 3995 model dates before the test period, 2008-01-02 .. 2024-02-29; no later date is read.

- Member = in the `kr200` list on that day.
- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.
- A stock's data ends early if its last close on the dates checked falls before 2024-02-29 (the last date checked), or it
  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading
  would show 0 everywhere. A stock that is only suspended on 2024-02-29 is counted too (nothing later is read).
- Ended members = stocks that are members on at least one date of the year and whose data ends early.
- Last close in year = stocks whose data ends early with their last close in that year.

| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |
|---|---|---|---|---|---|---|---|---|---|
| 2008 | 248 | 199 | 200 | 200 | 49564 | 49361 | 99.59% | 36 | 1 |
| 2009 | 253 | 198 | 200 | 200 | 50389 | 50206 | 99.64% | 38 | 4 |
| 2010 | 251 | 197 | 200 | 200 | 49982 | 49802 | 99.64% | 36 | 4 |
| 2011 | 248 | 198 | 199 | 200 | 49338 | 49183 | 99.69% | 29 | 4 |
| 2012 | 248 | 200 | 200 | 200 | 49600 | 49536 | 99.87% | 24 | 4 |
| 2013 | 247 | 197 | 200 | 200 | 49191 | 48976 | 99.56% | 23 | 3 |
| 2014 | 245 | 198 | 200 | 200 | 48899 | 48719 | 99.63% | 19 | 4 |
| 2015 | 248 | 196 | 199 | 200 | 49181 | 49057 | 99.75% | 16 | 5 |
| 2016 | 246 | 199 | 200 | 200 | 49178 | 48980 | 99.60% | 10 | 4 |
| 2017 | 243 | 198 | 199 | 200 | 48335 | 47857 | 99.01% | 9 | 4 |
| 2018 | 244 | 199 | 200 | 200 | 48750 | 48593 | 99.68% | 6 | 2 |
| 2019 | 246 | 199 | 200 | 200 | 49125 | 49080 | 99.91% | 5 | 1 |
| 2020 | 248 | 199 | 200 | 200 | 49528 | 49396 | 99.73% | 5 | 3 |
| 2021 | 248 | 200 | 200 | 200 | 49600 | 49428 | 99.65% | 3 | 2 |
| 2022 | 246 | 200 | 200 | 200 | 49200 | 49184 | 99.97% | 2 | 1 |
| 2023 | 245 | 198 | 200 | 200 | 48908 | 48811 | 99.80% | 2 | 3 |
| 2024 | 41 | 200 | 200 | 200 | 8200 | 8200 | 100.00% | 0 | 2 |

- All dates checked: members per day min 196, median 200, max 200; 796968 member-days, 794369 priced (99.67%), 2599 unpriced.
- Survivorship: 429 stocks are members on at least one date checked; the data of 51 of them ends early (0 of those 51 with no close on any date checked).
- Warm-up (the 246 dates loaded before the first model date, 2007-01-02 .. 2007-12-28): members per day min 198, median 200, max 200.
- Market-state warm-up: the first model date (2008-01-02) reads the member returns of the 120 dates 2007-07-04 .. 2007-12-28 (the longest window is 120 days). Members per day on those dates: min 198, median 200, max 200; members with a one-day return: min 197, median 199.

### Unpriced member spans

A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced
again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.

| Span length (trading days) | Spans | Unpriced member-days |
|---|---|---|
| 1-5 | 66 | 113 |
| 6-20 | 91 | 1236 |
| 21-60 | 36 | 935 |
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
| KR000070 | 2011-10-26 | 2011-12-02 | 28 | - |
