# Free US coverage check (Rule UC)

Question: if the S&P 500 were built from free data (members from fja05680/sp500, prices from Tiingo), how much
of its history could be priced? This check compares the member list with Tiingo's public list of supported
tickers. It fetches no prices. The rule and its details were fixed in `docs/RULES.md` (Rule UC) before the
match was first computed.

## Verdict

**FAIL: 91.06% of member-days are matched (1,926,082 of 2,115,048); the bar is 95.0%.**

By Rule UC the free S&P 500 route is recorded as not good enough as it stands, and the owner is told.

What a matched member-day shows: Tiingo's list has a US stock row under that ticker whose first and last date
cover the day. What it does not show:

- that Tiingo's free tier serves those prices. That stays unverified until a fetch;
- that the row is the same company. A reused ticker whose dates happen to cover the day counts as matched;
- that a price exists on every day in between. The list gives a first and a last date only;
- that the member list is right. fja05680/sp500 is an unofficial list.

## Numbers

- Window: 2008-01-01 to 2024-02-29, every Monday to Friday (4,218 days; weekday holidays count).
- Member-days: 2,115,048 (501.4 members per day on average).
- Matched member-days (the rule): 1,926,082 = 91.06%. Unmatched: 188,966.
- Distinct tickers that were members in the window: 848. With at least one unmatched member-day: 141, of which 112 have no matched member-day at all.
- Not part of the rule, ticker only (a stock row exists under the ticker, whatever its dates): 1,985,279 = 93.86%. The gap to the rule's number, 59,197 member-days, is what the date ranges remove (reused tickers and missing dates).
- Not part of the rule, an upper bound for any reading of "Tiingo lists that ticker as a stock" (any row exists under the ticker, whatever its asset type, currency, exchange and dates): 2,003,289 = 94.71%. This is below the bar too, so the call does not hang on how a stock row is told apart.

## Tickers that left the index before 2024-03-01

A ticker has left if it has a member-day in the window and is not in the member list of 2024-02-29. This is
the survivorship question: free sources tend to drop companies that no longer trade.

| Tickers | Count | With no matched day | Member-days | Matched | Matched share (the rule) | Ticker only (extra) |
|---|---|---|---|---|---|---|
| Left before 2024-03-01 | 345 | 107 | 565,716 | 398,121 | 70.37% | 79.59% |
| Members on 2024-02-29 | 503 | 5 | 1,549,332 | 1,527,961 | 98.62% | 99.07% |

## Per year

| Year | Weekdays | Member-days | Matched | Matched share (the rule) | Ticker only (extra) |
|---|---|---|---|---|---|
| 2008 | 262 | 130,317 | 105,349 | 80.84% | 88.41% |
| 2009 | 261 | 130,178 | 108,663 | 83.47% | 89.74% |
| 2010 | 261 | 130,022 | 111,376 | 85.65% | 91.09% |
| 2011 | 260 | 129,218 | 112,077 | 86.73% | 91.98% |
| 2012 | 261 | 129,709 | 114,541 | 88.30% | 92.60% |
| 2013 | 261 | 129,714 | 116,737 | 89.99% | 93.71% |
| 2014 | 261 | 130,016 | 117,847 | 90.64% | 93.80% |
| 2015 | 261 | 130,649 | 118,775 | 90.91% | 93.76% |
| 2016 | 261 | 131,898 | 122,293 | 92.71% | 94.20% |
| 2017 | 260 | 131,549 | 122,681 | 93.25% | 94.51% |
| 2018 | 261 | 131,973 | 122,840 | 93.07% | 94.26% |
| 2019 | 261 | 131,805 | 123,505 | 93.70% | 94.59% |
| 2020 | 262 | 132,312 | 125,836 | 95.10% | 95.95% |
| 2021 | 261 | 131,805 | 126,048 | 95.63% | 96.34% |
| 2022 | 260 | 130,971 | 127,140 | 97.07% | 97.35% |
| 2023 | 260 | 130,780 | 128,529 | 98.27% | 98.47% |
| 2024 | 44 | 22,132 | 21,845 | 98.70% | 98.90% |

The last year in the table ends on 2024-02-29.

## Unmatched member-days by likely reason

The likely reason is worked out by code from the two files only, so it is a hint, not a finding. It is not
part of the rule.

| Likely reason | Tickers | Unmatched member-days | Share of all member-days | What it means |
|---|---|---|---|---|
| not listed | 71 | 111,759 | 5.28% | Tiingo's list has no row under this ticker. The company may trade under another ticker now (Tiingo files a company under one ticker, so a rename hides the old one), or Tiingo may not cover it. |
| reused ticker | 41 | 48,813 | 2.30% | Tiingo has rows under this ticker, but none is a US stock whose dates touch the membership dates: a fund, or a stock listed only before or after. The letters belong to something else in Tiingo's list. |
| missing dates | 29 | 28,394 | 1.34% | Tiingo lists the ticker as a US stock, but its first and last date cover only part of the membership dates (or it has no dates). |
| class share | 0 | 0 | 0.00% | The ticker has a share-class suffix and Tiingo's list has no ticker written that way. |

## The 30 tickers with the most unmatched member-days

The full list is in `coverage_unmatched.csv`. "Possible rename" means only that another ticker joined the index on
the day this one left; the files do not say whether it is the same company.

| Ticker | Unmatched member-days | Member-days | First member day | Last member day | Likely reason |
|---|---|---|---|---|---|
| BK | 4,218 | 4,218 | 2008-01-01 | 2024-02-29 | not listed: no row under this ticker; still a member at the end of the window |
| EQR | 4,218 | 4,218 | 2008-01-01 | 2024-02-29 | not listed: no row under this ticker; still a member at the end of the window |
| MMC | 4,218 | 4,218 | 2008-01-01 | 2024-02-29 | not listed: no row under this ticker; still a member at the end of the window |
| ABC | 4,086 | 4,086 | 2008-01-01 | 2023-08-29 | not listed: no row under this ticker; left the index on 2023-08-30, when COR joined (possible rename) |
| PKI | 4,010 | 4,010 | 2008-01-01 | 2023-05-15 | not listed: no row under this ticker; left the index on 2023-05-16, when RVTY joined (possible rename) |
| ANTM | 3,780 | 3,780 | 2008-01-01 | 2022-06-27 | not listed: no row under this ticker; left the index on 2022-06-28, when ELV joined (possible rename) |
| BLL | 3,745 | 3,745 | 2008-01-01 | 2022-05-09 | not listed: no row under this ticker; left the index on 2022-05-10, when BALL joined (possible rename) |
| GPS | 3,676 | 3,676 | 2008-01-01 | 2022-02-01 | not listed: no row under this ticker; left the index on 2022-02-02, when CEG joined (possible rename) |
| LB | 3,545 | 3,545 | 2008-01-01 | 2021-08-02 | reused ticker: Tiingo lists it for 2024-06-28..2026-11-11 only, outside the membership dates; left the index on 2021-08-03, when BBWI joined (possible rename) |
| COG | 3,465 | 3,465 | 2008-06-23 | 2021-10-01 | not listed: no row under this ticker; left the index on 2021-10-04, when CTRA joined (possible rename) |
| CTL | 3,318 | 3,318 | 2008-01-01 | 2020-09-17 | not listed: no row under this ticker; left the index on 2020-09-18, when LUMN joined (possible rename) |
| UTX | 3,198 | 3,198 | 2008-01-01 | 2020-04-02 | not listed: no row under this ticker; left the index on 2020-04-03, when CARR, OTIS, RTX joined (possible rename) |
| ARNC | 3,196 | 3,199 | 2008-01-01 | 2020-04-03 | missing dates: Tiingo lists it for 2020-04-01..2023-08-17 only; left the index on 2020-04-06, when HWM joined (possible rename) |
| DISCA | 3,157 | 3,160 | 2010-03-01 | 2022-04-08 | missing dates: Tiingo lists it for 2022-04-06..2022-09-16 only; left the index on 2022-04-11, when WBD joined (possible rename) |
| JEC | 3,115 | 3,115 | 2008-01-01 | 2019-12-09 | not listed: no row under this ticker; left the index on 2019-12-10, when J joined (possible rename) |
| CBS | 3,112 | 3,112 | 2008-01-01 | 2019-12-04 | not listed: no row under this ticker; left the index on 2019-12-05, when VIAC, WRB joined (possible rename) |
| SYMC | 3,090 | 3,090 | 2008-01-01 | 2019-11-04 | not listed: no row under this ticker; left the index on 2019-11-05, when NLOK, PEAK joined (possible rename) |
| BHGE | 3,078 | 3,078 | 2008-01-01 | 2019-10-17 | not listed: no row under this ticker; left the index on 2019-10-18, when BKR joined (possible rename) |
| TSS | 3,058 | 3,058 | 2008-01-02 | 2019-09-20 | reused ticker: Tiingo lists it only as ETF (USD, NYSE); left the index on 2019-09-23, when CDW joined (possible rename) |
| TMK | 3,027 | 3,027 | 2008-01-01 | 2019-08-07 | not listed: no row under this ticker; left the index on 2019-08-08, when GL joined (possible rename) |
| HCP | 3,026 | 3,026 | 2008-03-31 | 2019-11-04 | reused ticker: Tiingo lists it for 2021-12-09..2025-02-27 only, outside the membership dates; left the index on 2019-11-05, when NLOK, PEAK joined (possible rename) |
| LLL | 2,999 | 2,999 | 2008-01-01 | 2019-06-28 | reused ticker: Tiingo lists it only as ETF (USD, NYSE); left the index on 2019-07-01, when MKTX joined (possible rename) |
| FOXA | 2,920 | 4,218 | 2008-01-01 | 2024-02-29 | missing dates: Tiingo lists it for 2019-03-12..2026-11-11 only; still a member at the end of the window |
| HRS | 2,790 | 2,790 | 2008-09-22 | 2019-05-31 | not listed: no row under this ticker; left the index on 2019-06-01, when LHX joined (possible rename) |
| WYND | 2,717 | 2,717 | 2008-01-01 | 2018-05-30 | not listed: no row under this ticker; left the index on 2018-05-31, when ABMD joined (possible rename) |
| BBBY | 2,496 | 2,496 | 2008-01-01 | 2017-07-25 | not listed: no row under this ticker; left the index on 2017-07-26, when AOS, DRE, MGM, PKG and others joined (possible rename) |
| HOT | 2,278 | 2,278 | 2008-01-01 | 2016-09-22 | reused ticker: Tiingo lists it only as ETF (USD, NYSE); left the index on 2016-09-23, when COO joined (possible rename) |
| TE | 2,218 | 2,218 | 2008-01-01 | 2016-06-30 | reused ticker: Tiingo lists it for 2021-07-08..2026-11-11 only, outside the membership dates; left the index on 2016-07-01, when ALB, LNT joined (possible rename) |
| FB | 2,208 | 2,208 | 2013-12-23 | 2022-06-08 | reused ticker: Tiingo lists it only as ETF (USD, BATS); left the index on 2022-06-09, when META joined (possible rename) |
| CCE | 2,195 | 2,195 | 2008-01-01 | 2016-05-30 | not listed: no row under this ticker; left the index on 2016-05-31, when AJG joined (possible rename) |

## Inputs

- Membership: fja05680/sp500 at commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`, file `S&P 500 Historical Components & Changes (Updated).csv`, sha256
  `36326709d46d6cd25834de5df457b16f5f96fad3a06b9beac28f7b88aa0b0d54`.
- Tiingo: `supported_tickers.zip` from `https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip`,
  downloaded 2026-11-12T22:24:27Z, sha256 `3d33d870111072d8a87de8a3f73cb3b30f00ec928fa517846ae29b614822f800`.
- Both stay outside the repo. `scripts/get_us_coverage_data.sh` fetches and verifies them and prints their paths;
  `python -m detmaso.data.ingest.us_coverage --membership <csv> --tiingo <zip>` checks the hashes again and
  writes this report.

## How a member-day is matched

- Members on a day: the list of the last fja05680 row dated on or before that day.
- Tickers are compared in upper case with "." and "/" written as "-".
- A Tiingo row counts as a stock if its asset type is `Stock`, its currency is `USD` and its exchange is
  not outside the US (LSE, SHE, SHEB, SHG, SHGB). Over-the-counter venues count.
- The day must lie between the row's first and last date, both included. With several rows for one ticker,
  one covering row is enough. A row without dates never matches.
- Shares are rounded down to 2 decimals. The bar is applied to the exact counts.
