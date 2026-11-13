# Data QA report: csi500

- Data tag: `2026-11-04` (sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`)
- Calendar loaded: 2007-01-04 .. 2026-04-30 (4695 days)
- Model dates: 2008-01-02 .. 2026-02-27 (4410 days); stocks that are members on a model date before the test period: 1648

## Splits

| Split | First | Last | Days |
|---|---|---|---|
| train | 2008-01-02 | 2022-12-30 | 3649 |
| train_fit | 2008-01-02 | 2022-12-02 | 3629 |
| valid | 2023-01-03 | 2024-02-28 | 278 |
| valid_fit | 2023-01-03 | 2024-01-23 | 258 |
| test | 2024-03-01 | 2026-02-27 | 482 |

## Cross-section size (train and validation dates only)

- Universe per day (members that traded the day before): min 269, median 488, max 500
- Stocks with a 20-day label per day: min 246, median 476
- Smallest universe on 2015-07-10
- Label counts stop at 2024-01-24: the 20-day answers of later dates end inside the test period

## Spot prices

Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data
check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).

## Missing values per feature (share of universe stock-days before the test period, before filling with 0)

mom_1 0.0%, mom_5 0.0%, mom_10 0.0%, mom_20 0.0%, mom_60 0.0%, mom_120 0.0%, vol_5 0.4%, vol_10 0.8%, vol_20 1.3%, vol_60 2.0%, rturn_5 0.4%, rturn_10 0.8%, rturn_20 1.3%, rturn_60 2.0%, pvcorr_5 0.4%, pvcorr_10 0.8%, pvcorr_20 1.3%, upfrac_5 0.4%, upfrac_10 0.8%, upfrac_20 1.3%, madev_5 0.4%, madev_20 1.3%, madev_60 2.0%, macd 0.0%, overnight 0.0%, intraday 0.0%, closepos 0.0%, hl_1 0.0%, hl_5 0.4%, skew60 2.0%, ac1_60 2.1%, momsharpe 2.0%, range20 1.3%, logvol20 1.3%

## Regimes

Silhouette score by number of clusters (training period):

K=2: 0.466, K=3: 0.282, K=4: 0.234, K=5: 0.240, K=6: 0.190, K=7: 0.183, K=8: 0.185, K=9: 0.187, K=10: 0.185

The silhouette-optimal count is **K = 2**; per decision D2 the model uses n = 5 regardless.

| Regime (agent) | Our coverage of training days |
|---|---|
| 1 Low-Vol Bull | 12.1% |
| 2 High-Vol Bull | 6.9% |
| 3 Sideways | 27.3% |
| 4 Low-Vol Bear | 49.0% |
| 5 Tail-Risk | 4.7% |

Centroids in raw units (daily returns):

|  | sigma5 | mu5 | delta5 | sigma20 | mu20 | delta20 | sigma60 | mu60 | delta60 | sigma120 | mu120 | delta120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low-Vol Bull | 0.0193 | 0.0297 | 0.0257 | 0.0204 | 0.114 | 0.0253 | 0.0208 | 0.2724 | 0.0245 | 0.0224 | 0.3901 | 0.0244 |
| High-Vol Bull | 0.0298 | 0.0316 | 0.0258 | 0.0359 | 0.0604 | 0.0266 | 0.0383 | -0.0253 | 0.0271 | 0.0377 | -0.0637 | 0.0274 |
| Sideways | 0.0126 | -0.0036 | 0.0239 | 0.0137 | -0.002 | 0.0239 | 0.0139 | 0.0355 | 0.0237 | 0.0139 | 0.0831 | 0.0233 |
| Low-Vol Bear | 0.0123 | 0.0016 | 0.0186 | 0.0134 | 0.0023 | 0.0186 | 0.0142 | -0.0066 | 0.0189 | 0.0149 | -0.0101 | 0.0192 |
| Tail-Risk | 0.0442 | -0.0787 | 0.0273 | 0.0408 | -0.2244 | 0.0271 | 0.0378 | -0.2852 | 0.0271 | 0.0355 | -0.2386 | 0.0271 |

Naming-rule check (plan A16 is fixed in advance; names are labels only): names are consistent with the centroid volatilities.

## Single-feature IC on the training period (pipeline sanity check)

| Feature | Mean daily IC 2008-2022 |
|---|---|
| mom_20 | -0.0537 |
| vol_20 | -0.0604 |
| mom_5 | -0.0278 |
| rturn_20 | -0.0112 |
| logvol20 | -0.0704 |

## Per year (walk-forward history check)

Universe = index members that traded the day before; missing = average share of universe stock-days with a missing
feature (before filling with 0); moves > 21% = universe stock-days whose one-day return exceeds every price limit
(possible data errors, or days without a limit such as a stock's first trading days).

| Year | Days | Universe min | Universe median | Labelled median | Missing features | Moves > 21% |
|---|---|---|---|---|---|---|
| 2008 | 246 | 455 | 487 | 472 | 0.67% | 0 |
| 2009 | 244 | 462 | 488 | 472 | 0.54% | 0 |
| 2010 | 242 | 465 | 487 | 471 | 0.54% | 0 |
| 2011 | 244 | 459 | 484 | 469 | 0.65% | 0 |
| 2012 | 243 | 459 | 487 | 478 | 0.59% | 0 |
| 2013 | 238 | 479 | 488 | 480 | 0.86% | 0 |
| 2014 | 245 | 455 | 476 | 461 | 1.48% | 0 |
| 2015 | 244 | 269 | 449 | 419 | 3.44% | 0 |
| 2016 | 244 | 456 | 467 | 457 | 1.67% | 0 |
| 2017 | 244 | 462 | 472 | 463 | 1.11% | 0 |
| 2018 | 243 | 452 | 472 | 471 | 1.09% | 0 |
| 2019 | 244 | 495 | 499 | 498 | 0.10% | 0 |
| 2020 | 243 | 496 | 499 | 498 | 0.09% | 0 |
| 2021 | 243 | 495 | 499 | 498 | 0.07% | 0 |
| 2022 | 242 | 496 | 499 | 498 | 0.08% | 0 |
| 2023 | 242 | 496 | 499 | 499 | 0.04% | 0 |
| 2024 | 37 | 498 | 499 | 499 | 0.01% | 0 |

## Membership, pricing and survivorship (docs/RULES.md, Rule ND)

Dates checked: the 3928 model dates before the test period, 2008-01-02 .. 2024-02-29; no later date is read.

- Member = in the `csi500` list on that day.
- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.
- A stock's data ends early if its last close on the dates checked falls before 2024-02-29 (the last date checked), or it
  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading
  would show 0 everywhere. A stock that is only suspended on 2024-02-29 is counted too (nothing later is read).
- Ended members = stocks that are members on at least one date of the year and whose data ends early.
- Last close in year = stocks whose data ends early with their last close in that year.

| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |
|---|---|---|---|---|---|---|---|---|---|
| 2008 | 246 | 500 | 500 | 500 | 123000 | 119504 | 97.16% | 43 | 0 |
| 2009 | 244 | 500 | 500 | 500 | 122000 | 119058 | 97.59% | 44 | 2 |
| 2010 | 242 | 500 | 500 | 500 | 121000 | 117758 | 97.32% | 34 | 1 |
| 2011 | 244 | 500 | 500 | 500 | 122000 | 117906 | 96.64% | 32 | 1 |
| 2012 | 243 | 500 | 500 | 500 | 121500 | 118253 | 97.33% | 34 | 3 |
| 2013 | 238 | 500 | 500 | 500 | 119000 | 116307 | 97.74% | 33 | 3 |
| 2014 | 245 | 500 | 500 | 500 | 122500 | 116425 | 95.04% | 37 | 1 |
| 2015 | 244 | 500 | 500 | 500 | 122000 | 107876 | 88.42% | 34 | 1 |
| 2016 | 244 | 500 | 500 | 500 | 122000 | 114517 | 93.87% | 35 | 1 |
| 2017 | 244 | 500 | 500 | 500 | 122000 | 115352 | 94.55% | 35 | 2 |
| 2018 | 243 | 500 | 500 | 500 | 121500 | 115555 | 95.11% | 33 | 7 |
| 2019 | 244 | 500 | 500 | 500 | 122000 | 121657 | 99.72% | 27 | 8 |
| 2020 | 243 | 500 | 500 | 500 | 121500 | 121269 | 99.81% | 12 | 9 |
| 2021 | 243 | 500 | 500 | 500 | 121500 | 121245 | 99.79% | 11 | 12 |
| 2022 | 242 | 500 | 500 | 500 | 121000 | 120702 | 99.75% | 6 | 12 |
| 2023 | 242 | 500 | 500 | 501 | 121001 | 120658 | 99.72% | 3 | 22 |
| 2024 | 37 | 501 | 501 | 501 | 18537 | 18466 | 99.62% | 1 | 1 |

- All dates checked: members per day min 500, median 500, max 501; 1964038 member-days, 1902508 priced (96.87%), 61530 unpriced.
- Survivorship: 1648 stocks are members on at least one date checked; the data of 87 of them ends early (1 of those 87 with no close on any date checked).
- Warm-up (the 242 dates loaded before the first model date, 2007-01-04 .. 2007-12-28): members per day min 0, median 500, max 500.
- Market-state warm-up: the first model date (2008-01-02) reads the member returns of the 120 dates 2007-07-09 .. 2007-12-28 (the longest window is 120 days). Members per day on those dates: min 500, median 500, max 500; members with a one-day return: min 463, median 475.

### Unpriced member spans

A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced
again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.

| Span length (trading days) | Spans | Unpriced member-days |
|---|---|---|
| 1-5 | 8245 | 11069 |
| 6-20 | 699 | 7424 |
| 21-60 | 356 | 12980 |
| 61-250 | 307 | 29515 |
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
| SH600816 | 2010-07-01 | 2011-04-25 | 197 | - |
| SZ002025 | 2016-09-12 | 2017-06-29 | 191 | - |
| SZ002745 | 2016-07-18 | 2017-04-28 | 190 | - |
