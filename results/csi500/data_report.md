# Data QA report: csi500

- Data tag: `2026-11-04` (sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`)
- Calendar loaded: 2011-01-04 .. 2026-04-30 (3721 days)
- Model dates: 2012-01-04 .. 2026-02-27 (3434 days); stocks that are members on a model date before the test period: 1420

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2012-01-04 | 2022-12-30 | 2673 |
| train_fit | 2012-01-04 | 2022-12-02 | 2653 |
| valid | 2023-01-03 | 2024-02-28 | 278 |
| valid_fit | 2023-01-03 | 2024-01-23 | 258 |
| test | 2024-03-01 | 2026-02-27 | 482 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 269, median 490, max 500
- Stocks with a 20-day label per day: min 246, median 483
- Smallest universe on 2015-07-10
- Label counts stop at 2024-01-24: the 20-day answers of later dates end inside the test period

## Spot prices

Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data
check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.0%, vol_5 0.4%, vol_10 0.8%, vol_20 1.4%, vol_60 2.1%, rturn_5 0.4%, rturn_10 0.8%, rturn_20 1.4%, rturn_60 2.1%, pvcorr_5 0.4%, pvcorr_10 0.8%, pvcorr_20 1.4%, upfrac_5 0.4%, upfrac_10 0.8%, upfrac_20 1.4%, madev_5 0.4%, madev_20 1.4%, madev_60 2.1%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.4%, skew60 2.1%, ac1_60 2.3%, momsharpe 2.1%, range20 1.4%, logvol20 1.4%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.567, K=3: 0.277, K=4: 0.245, K=5: 0.254, K=6: 0.211, K=7: 0.198, K=8: 0.200, K=9: 0.203, K=10: 0.189

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days |
|---|---|
| 1 Low-Vol Bull | 5.3% |
| 2 High-Vol Bull | 5.0% |
| 3 Sideways | 33.4% |
| 4 Low-Vol Bear | 54.0% |
| 5 Tail-Risk | 2.3% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.021 | 0.0319 | 0.0293 | 0.02 | 0.1563 | 0.0283 | 0.017 | 0.3437 | 0.0253 | 0.0171 | 0.4251 | 0.024 |
| High-Vol Bull | 0.0254 | 0.0268 | 0.0244 | 0.0321 | 0.0565 | 0.0252 | 0.0379 | -0.0111 | 0.0268 | 0.0404 | 0.0281 | 0.0283 |
| Sideways | 0.0118 | 0.0026 | 0.024 | 0.0129 | 0.0066 | 0.024 | 0.0133 | 0.0365 | 0.0239 | 0.0131 | 0.0819 | 0.0235 |
| Low-Vol Bear | 0.0119 | 0.0006 | 0.0186 | 0.0128 | 0.0024 | 0.0186 | 0.0136 | -0.0007 | 0.0188 | 0.0144 | -0.0152 | 0.019 |
| Tail-Risk | 0.056 | -0.0784 | 0.0303 | 0.0521 | -0.2288 | 0.0303 | 0.0449 | -0.1697 | 0.031 | 0.0379 | 0.2011 | 0.0298 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2012-2022 |
|---|---|
| mom_20 | -0.0480 |
| vol_20 | -0.0620 |
| mom_5 | -0.0215 |
| rturn_20 | -0.0082 |
| logvol20 | -0.0679 |

## Membership, pricing and survivorship (docs/RULES.md, Rule ND)

Dates checked: the 2952 model dates before the test period, 2012-01-04 .. 2024-02-29; no later date is read.

- Member = in the `csi500` list on that day.
- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.
- A stock's data ends early if its last close on the dates checked falls before 2024-02-29 (the last date checked), or it
  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading
  would show 0 everywhere. A stock that is only suspended on 2024-02-29 is counted too (nothing later is read).
- Ended members = stocks that are members on at least one date of the year and whose data ends early.
- Last close in year = stocks whose data ends early with their last close in that year.

| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |
|---|---|---|---|---|---|---|---|---|---|
| 2012 | 243 | 500 | 500 | 500 | 121500 | 118253 | 97.33% | 34 | 3 |
| 2013 | 238 | 500 | 500 | 500 | 119000 | 116307 | 97.74% | 33 | 1 |
| 2014 | 245 | 500 | 500 | 500 | 122500 | 116425 | 95.04% | 37 | 1 |
| 2015 | 244 | 500 | 500 | 500 | 122000 | 107876 | 88.42% | 34 | 1 |
| 2016 | 244 | 500 | 500 | 500 | 122000 | 114517 | 93.87% | 35 | 1 |
| 2017 | 244 | 500 | 500 | 500 | 122000 | 115352 | 94.55% | 35 | 0 |
| 2018 | 243 | 500 | 500 | 500 | 121500 | 115555 | 95.11% | 33 | 5 |
| 2019 | 244 | 500 | 500 | 500 | 122000 | 121657 | 99.72% | 27 | 7 |
| 2020 | 243 | 500 | 500 | 500 | 121500 | 121269 | 99.81% | 12 | 7 |
| 2021 | 243 | 500 | 500 | 500 | 121500 | 121245 | 99.79% | 11 | 11 |
| 2022 | 242 | 500 | 500 | 500 | 121000 | 120702 | 99.75% | 6 | 7 |
| 2023 | 242 | 500 | 500 | 501 | 121001 | 120658 | 99.72% | 3 | 21 |
| 2024 | 37 | 501 | 501 | 501 | 18537 | 18466 | 99.62% | 1 | 1 |

- All dates checked: members per day min 500, median 500, max 501; 1476038 member-days, 1428282 priced (96.76%), 47756 unpriced.
- Survivorship: 1420 stocks are members on at least one date checked; the data of 67 of them ends early (1 of those 67 with no close on any date checked).
- Warm-up (the 244 dates loaded before the first model date, 2011-01-04 .. 2011-12-30): members per day min 500, median 500, max 500.
- Market-state warm-up: the first model date (2012-01-04) reads the member returns of the 120 dates 2011-07-08 .. 2011-12-30 (the longest window is 120 days). Members per day on those dates: min 500, median 500, max 500; members with a one-day return: min 460, median 480.

### Unpriced member spans

A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced
again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.

| Span length (trading days) | Spans | Unpriced member-days |
|---|---|---|
| 1-5 | 2031 | 3980 |
| 6-20 | 603 | 6366 |
| 21-60 | 275 | 10719 |
| 61-250 | 272 | 26149 |
| 251 or more | 2 | 542 |

The 10 longest:

| Stock | First unpriced member date | Last | Trading days | Note |
|---|---|---|---|---|
| SH600780 | 2012-01-18 | 2013-03-29 | 289 | - |
| SH600673 | 2016-11-15 | 2017-11-24 | 253 | - |
| SH600575 | 2014-09-09 | 2015-09-15 | 249 | - |
| SZ002219 | 2015-07-08 | 2016-06-21 | 233 | - |
| SZ300088 | 2016-09-12 | 2017-08-08 | 219 | - |
| SH600655 | 2016-12-20 | 2017-11-08 | 216 | - |
| SZ000982 | 2014-08-25 | 2015-06-29 | 205 | - |
| SZ002025 | 2016-09-12 | 2017-06-29 | 191 | - |
| SZ002745 | 2016-07-18 | 2017-04-28 | 190 | - |
| SH600086 | 2018-01-19 | 2018-11-01 | 189 | - |
