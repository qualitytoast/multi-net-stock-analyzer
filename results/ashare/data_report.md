# Data QA report: ashare

- Data tag: `2026-11-04` (sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`)
- Calendar loaded: 2011-01-04 .. 2026-04-30 (3721 days)
- Model dates: 2012-01-04 .. 2026-02-27 (3434 days); stocks that are members on a model date before the test period: 5427

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2012-01-04 | 2022-12-30 | 2673 |
| train_fit | 2012-01-04 | 2022-12-02 | 2653 |
| valid | 2023-01-03 | 2024-02-28 | 278 |
| valid_fit | 2023-01-03 | 2024-01-23 | 258 |
| test | 2024-03-01 | 2026-02-27 | 482 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 1201, median 3075, max 5255
- Stocks with a 20-day label per day: min 1116, median 2975
- Smallest universe on 2015-07-10
- Label counts stop at 2024-01-24: the 20-day answers of later dates end inside the test period

## Spot prices

Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data
check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.0%, vol_5 0.4%, vol_10 0.8%, vol_20 1.4%, vol_60 2.3%, rturn_5 0.4%, rturn_10 0.8%, rturn_20 1.4%, rturn_60 2.3%, pvcorr_5 0.4%, pvcorr_10 0.8%, pvcorr_20 1.4%, upfrac_5 0.4%, upfrac_10 0.8%, upfrac_20 1.4%, madev_5 0.4%, madev_20 1.4%, madev_60 2.3%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.4%, skew60 2.3%, ac1_60 2.4%, momsharpe 2.3%, range20 1.4%, logvol20 1.4%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.575, K=3: 0.284, K=4: 0.283, K=5: 0.291, K=6: 0.236, K=7: 0.216, K=8: 0.218, K=9: 0.199, K=10: 0.178

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days |
|---|---|
| 1 Low-Vol Bull | 2.4% |
| 2 High-Vol Bull | 4.8% |
| 3 Sideways | 36.8% |
| 4 Low-Vol Bear | 53.4% |
| 5 Tail-Risk | 2.5% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0229 | 0.0424 | 0.0354 | 0.022 | 0.2103 | 0.0339 | 0.0184 | 0.5303 | 0.03 | 0.0179 | 0.6725 | 0.0277 |
| High-Vol Bull | 0.0258 | 0.0341 | 0.0278 | 0.0324 | 0.0869 | 0.0288 | 0.0388 | 0.0451 | 0.0304 | 0.0411 | 0.1113 | 0.032 |
| Sideways | 0.013 | 0.0031 | 0.0267 | 0.014 | 0.0179 | 0.0266 | 0.0144 | 0.057 | 0.0263 | 0.0143 | 0.1081 | 0.0259 |
| Low-Vol Bear | 0.0121 | 0.0023 | 0.0205 | 0.0132 | 0.0041 | 0.0205 | 0.0139 | 0.0039 | 0.0206 | 0.0147 | -0.0046 | 0.0208 |
| Tail-Risk | 0.0539 | -0.077 | 0.0339 | 0.0521 | -0.237 | 0.0336 | 0.0447 | -0.1653 | 0.0345 | 0.0383 | 0.2511 | 0.0327 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2012-2022 |
|---|---|
| mom_20 | -0.0653 |
| vol_20 | -0.0797 |
| mom_5 | -0.0301 |
| rturn_20 | -0.0189 |
| logvol20 | -0.0778 |

## Membership, pricing and survivorship (docs/RULES.md, Rule ND)

Dates checked: the 2952 model dates before the test period, 2012-01-04 .. 2024-02-29; no later date is read.

- Member = in the `all` list on that day, after the membership filters below.
- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.
- A stock's data ends early if its last close on the dates checked falls before 2024-02-29 (the last date checked), or it
  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading
  would show 0 everywhere. A stock that is only suspended on 2024-02-29 is counted too (nothing later is read).
- Ended members = stocks that are members on at least one date of the year and whose data ends early.
- Last close in year = stocks whose data ends early with their last close in that year.

| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |
|---|---|---|---|---|---|---|---|---|---|
| 2012 | 243 | 2161 | 2287 | 2404 | 555299 | 540500 | 97.33% | 149 | 3 |
| 2013 | 238 | 2406 | 2452 | 2462 | 582875 | 561070 | 96.26% | 149 | 4 |
| 2014 | 245 | 2462 | 2464 | 2522 | 608257 | 559770 | 92.03% | 150 | 3 |
| 2015 | 244 | 2522 | 2581 | 2772 | 635306 | 542126 | 85.33% | 153 | 5 |
| 2016 | 244 | 2772 | 2808.5 | 2869 | 685059 | 623042 | 90.95% | 150 | 2 |
| 2017 | 244 | 2870 | 3034 | 3280 | 744600 | 690482 | 92.73% | 153 | 4 |
| 2018 | 243 | 3281 | 3465 | 3527 | 835638 | 796368 | 95.30% | 149 | 8 |
| 2019 | 244 | 3525 | 3566 | 3625 | 871524 | 866258 | 99.40% | 141 | 10 |
| 2020 | 243 | 3625 | 3754 | 3870 | 910927 | 905269 | 99.38% | 133 | 18 |
| 2021 | 243 | 3872 | 4129 | 4375 | 1002584 | 999343 | 99.68% | 115 | 19 |
| 2022 | 242 | 4377 | 4643 | 4820 | 1118124 | 1115079 | 99.73% | 96 | 42 |
| 2023 | 242 | 4823 | 5026.5 | 5199 | 1213776 | 1211778 | 99.84% | 54 | 48 |
| 2024 | 37 | 5201 | 5229 | 5259 | 193543 | 193391 | 99.92% | 9 | 6 |

- All dates checked: members per day min 2161, median 3308, max 5259; 9957512 member-days, 9604476 priced (96.45%), 353036 unpriced.
- Survivorship: 5427 stocks are members on at least one date checked; the data of 172 of them ends early (0 of those 172 with no close on any date checked).
- Membership filters: the `all` list has 10378047 member-days on the dates checked; 9957512 are kept.
  - Never members (not stocks): SH000300, SH000852, SH000905, SH000906, SH000985, SZ399300; 17712 member-days removed.
  - New-listing filter: a stock is a member only once it has 120 traded days in the loaded data up to the day before (traded day 121 on); 402823 member-days removed (3.89% of the stocks' listed member-days).
- Warm-up (the 244 dates loaded before the first model date, 2011-01-04 .. 2011-12-30): members per day min 0, median 1098.5, max 2158.
- Market-state warm-up: the first model date (2012-01-04) reads the member returns of the 120 dates 2011-07-08 .. 2011-12-30 (the longest window is 120 days). Members per day on those dates: min 1710, median 2066.5, max 2158; members with a one-day return: min 1686, median 1990.

### Unpriced member spans

A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced
again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.

| Span length (trading days) | Spans | Unpriced member-days |
|---|---|---|
| 1-5 | 11504 | 23539 |
| 6-20 | 3769 | 38514 |
| 21-60 | 1708 | 67249 |
| 61-250 | 2062 | 202760 |
| 251 or more | 57 | 20974 |

The 10 longest:

| Stock | First unpriced member date | Last | Trading days | Note |
|---|---|---|---|---|
| SH600145 | 2015-12-07 | 2020-06-29 | 1110 | - |
| SZ000029 | 2016-09-14 | 2020-11-06 | 1006 | - |
| SZ002260 | 2019-04-29 | 2022-04-29 | 730 | - |
| SH600485 | 2016-12-26 | 2019-07-11 | 619 | - |
| SZ000670 | 2020-03-20 | 2022-08-19 | 589 | - |
| SZ002075 | 2016-09-19 | 2018-11-15 | 526 | - |
| SZ000033 | 2015-04-30 | 2017-05-23 | 504 | - |
| SZ000693 | 2016-03-01 | 2018-03-20 | 503 | - |
| SZ000520 | 2013-12-27 | 2015-12-17 | 482 | - |
| SZ300028 | 2018-08-08 | 2020-06-17 | 451 | - |
