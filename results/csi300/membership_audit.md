# Membership timing audit: CSI300 and CSI500 (Rule MA)

Written by `scripts/audit_membership.py` (docs/RULES.md, Rule MA). A report only: it changes no data and no
decision depends on it. Every date that was queried or counted is before 2024-03-01, the first day of the
locked test period. No model was trained, so this adds no tries.

## In short

- **CSI300, month-end lists:** 0.42% of names differ between the archive and BaoStock (258 stocks in exactly one list out of 61928 in either, 206 month ends; identical lists at 179 of them). Leaving out the 19 month ends where BaoStock's own list was out of date: 0.11%.
- **CSI500, month-end lists:** 0.56% of names differ between the archive and BaoStock (582 stocks in exactly one list out of 103292 in either, 206 month ends; identical lists at 161 of them). Leaving out the 19 month ends where BaoStock's own list was out of date: 0.08%.
- **CSI300, delay:** the archive shows an index change a median of 9 trading days after BaoStock (90th percentile 14, longest 102), over the 1330 changes that BaoStock dates to the week. From 2012-01 on, the dates the model uses: median 12, 90th percentile 14, longest 89 (899 changes). Over all 1522 paired changes, including those where BaoStock skipped updates: median 9, 90th percentile 14, longest 102. The archive never shows 2 of BaoStock's 1524 changes (0.1%).
- **CSI500, delay:** the archive shows an index change a median of 13 trading days after BaoStock (90th percentile 21, longest 98), over the 3025 changes that BaoStock dates to the week. From 2012-01 on, the dates the model uses: median 13, 90th percentile 18, longest 92 (2116 changes). Over all 3428 paired changes, including those where BaoStock skipped updates: median 12, 90th percentile 18, longest 114. The archive never shows 6 of BaoStock's 3434 changes (0.2%).
- **Where the delay comes from** (section 4): at the 17 twice-yearly reviews since 2013-12-16 that BaoStock dates to the week, the archive is 9 to 14 trading days late. Single stocks that move between reviews wait for the archive's next change day; the longest waits, in trading days: 102 (CSI300), 98 (CSI500).
- **Look-ahead check: in none of the 4950 paired changes does the archive show the change clearly EARLIER than BaoStock** (on or before the date of a BaoStock list that still lacked it).
- Another 1202 changes (24.3%) show up in the archive before BaoStock's new list but after BaoStock's last old one. BaoStock had no list in between, so these say nothing either way about look-ahead. For 767 of them the two BaoStock lists are a week apart and the archive is at most 4 trading days ahead, which is what an archive that is exactly on time looks like next to a weekly list; the other 435 fall where BaoStock skipped updates (section 3).
- BaoStock is an unofficial source, not ground truth, and it was itself out of date at 38 of the 412 month-end lists compared. See "Limits".

## Why this audit exists

On every day the model ranks the stocks that the pinned archive lists as index members on that day
(`~/.qlib/qlib_data/cn_data_2026-11-04`, files `instruments/csi300.txt` and `instruments/csi500.txt`). The project's notes say
the archive's maintainer rebuilt these lists from snapshots of the index's weights, not from the index
provider's change notices. A list rebuilt that way can show an index change on the wrong day:

- **late** is stale but honest: for a while the model ranks a slightly outdated list;
- **early** is look-ahead: the model would know that a stock joins or leaves the index before it happened.

The audit measures the direction and the size of the error against a second source.

## Method

**The archive.** Each line of a membership file is a stock and a span of dates; a stock is a member on a day
if one of its spans covers that day. Spans that touch are joined. The list can only change on the days on
which spans start, and there are few of them:

| Index | Archive's first day | Days on which its list can change, 2007-01-31 to 2024-02-29 | Typical distance (trading days, median) | Largest distance |
|---|---|---|---|---|
| CSI300 | 2005-04-08 | 41 | 118.5 | 126 |
| CSI500 | 2007-01-31 | 40 | 119 | 139 |

**BaoStock.** A free data service (Python package `baostock`, anonymous login, no account). It was asked for
the member list on one date at a time with `query_hs300_stocks(date=...)` (CSI300) and
`query_zz500_stocks(date=...)` (CSI500). Each answer carries the stock codes and an "update date".

**Step 1, month ends.** The last trading day of every month from 2007-01 to 2024-02 in the archive's
own trading calendar: 206 month ends, 2007-01-31 to 2024-02-29. On each one the two lists are compared.
Mismatch share = stocks in exactly one of the two lists / stocks in either list. The overall and yearly
figures pool the month ends (sum of the first count / sum of the second). A BaoStock answer that is empty or
more than 2% away from the index's size (300 or 500 stocks) is counted as unusable and left out.

**Step 2, the day of each change in BaoStock.** Whenever BaoStock's lists at two consecutive month ends
differ, the days in between are searched by halving: ask for the list in the middle, keep the half (or halves)
whose two ends differ, repeat until two neighbouring trading days carry different lists. The later of the two
is the day the change takes effect in BaoStock. The update date was not trusted in advance; it was checked
against these directly found days afterwards (section 2).

**Step 3, the day of the same change in the archive.** For an add, the first day of a membership run of that
stock; for a drop, the first trading day after a run ends. A BaoStock change is paired with the archive change
of the same stock and the same direction that is nearest in time, at most 250 trading days (about one
year) away, each archive change used once. The window was fixed before any result was seen.

**Lag** = the archive's day minus BaoStock's day, in trading days. Above 0 the archive is late, below 0 it is
early. A BaoStock change without a partner is:

- *never shown*, if at least 250 trading days of archive history follow it;
- *too close to the cut-off*, if fewer follow (the archive may still show it after 2024-02-29, which this audit does not look at);
- *in the archive from its first day*, if the archive already lists the stock on its very first day, so the
  archive's own add day cannot be known.

**Early changes come in two kinds**, because BaoStock only has a list for certain dates (section 2):

- *early, between two BaoStock lists*: the archive shows the change after the date of BaoStock's last old list and
  before BaoStock's first new list. BaoStock had no list in between, so it cannot tell who is right.
- *early, clear*: the archive shows the change on or before the date of a BaoStock list that still lacked it.
  The two sources contradict each other on that day. This is the look-ahead signal.

**"Dated to the week".** BaoStock normally stores a new list every Monday, so its last old list and its first
new list are at most 5 trading days apart and the change day is known to the week. Where BaoStock skipped
updates the two lists are further apart, BaoStock's change day is too late by an unknown amount, and the lag
says more about BaoStock than about the archive. Section 3 gives the lags both ways.

**What was fixed when.** The 250-day pairing window, the 2% size rule, the percentile method and the two
kinds of "early" were fixed before any BaoStock list was compared with the archive (a first probe of seven
dates had only shown that answers carry a Monday update date). The splits by the age of BaoStock's list
("at most 6 days old", "dated to the week") and by review days were added after the first comparisons
showed that BaoStock skips updates. The all-inclusive numbers are always shown next to them.

## 1. Month-end lists

| Index | Month ends | Compared | Stocks in exactly one list | Stocks in either list | Mismatch share | Stocks in one list only, per month end | Month ends with identical lists | Worst month end |
|---|---|---|---|---|---|---|---|---|
| CSI300 | all | 206 | 258 | 61928 | **0.42%** | 1.3 | 179 | 17.07% (2021-12-31) |
| CSI300 | BaoStock's list at most 6 days old | 187 | 62 | 56130 | **0.11%** | 0.3 | 165 | 3.28% (2007-06-29) |
| CSI300 | BaoStock's list older (out of date) | 19 | 196 | 5798 | **3.38%** | 10.3 | 14 | 17.07% (2021-12-31) |
| CSI300 | 2012-01 on | 146 | 206 | 43903 | **0.47%** | 1.4 | 136 | 17.07% (2021-12-31) |
| CSI300 | 2012-01 on, BaoStock's list at most 6 days old | 127 | 10 | 38105 | **0.03%** | 0.1 | 122 | 0.66% (2017-05-31) |
| CSI500 | all | 206 | 582 | 103292 | **0.56%** | 2.8 | 161 | 18.18% (2023-02-28) |
| CSI500 | BaoStock's list at most 6 days old | 187 | 74 | 93538 | **0.08%** | 0.4 | 149 | 0.60% (2021-09-30) |
| CSI500 | BaoStock's list older (out of date) | 19 | 508 | 9754 | **5.21%** | 26.7 | 12 | 18.18% (2023-02-28) |
| CSI500 | 2012-01 on | 146 | 542 | 73272 | **0.74%** | 3.7 | 121 | 18.18% (2023-02-28) |
| CSI500 | 2012-01 on, BaoStock's list at most 6 days old | 127 | 34 | 63518 | **0.05%** | 0.3 | 109 | 0.60% (2021-09-30) |

"2012-01 on" is the part the model uses (training starts 2012-01-01). A BaoStock list is "out of date"
when its update date is more than 6 days before the month end (section 2); a mismatch there is BaoStock's
delay, not the archive's.

**Month ends flatter the archive.** Its list mostly changes on a month end (CSI300: 24 of its 41 change days are month ends; CSI500: 34 of its 40 change days are month ends), so a
month-end comparison lands just after the archive has caught up. The days in between are measured in
section 3, and section 4 turns them into a share of the universe.

**Unusable BaoStock answers at month ends.** CSI300: 0 empty, 0 with an obviously wrong size, 0 where the archive has no list; CSI500: 0 empty, 0 with an obviously wrong size, 0 where the archive has no list.

BaoStock list sizes at the month ends: CSI300: 300 stocks at 206 month ends; CSI500: 500 stocks at 205 month ends, 499 stocks at 1 month end.
Archive list sizes at the month ends: CSI300: 300 stocks at 205 month ends, 298 stocks at 1 month end; CSI500: 500 stocks at 203 month ends, 501 stocks at 3 month ends.

### Per year

| Year | CSI300 mismatch share | In one list / in either | Month ends | Of which BaoStock out of date | CSI500 mismatch share | In one list / in either | Month ends | Of which BaoStock out of date |
|---|---|---|---|---|---|---|---|---|
| 2007 | 0.83% | 30 / 3615 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2008 | 0.00% | 0 / 3600 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2009 | 0.06% | 2 / 3600 | 12 | 0 | 0.17% | 10 / 6005 | 12 | 0 |
| 2010 | 0.28% | 10 / 3605 | 12 | 0 | 0.17% | 10 / 6005 | 12 | 0 |
| 2011 | 0.28% | 10 / 3605 | 12 | 0 | 0.33% | 20 / 6010 | 12 | 0 |
| 2012 | 0.00% | 0 / 3600 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2013 | 0.00% | 0 / 3600 | 12 | 0 | 0.03% | 2 / 6001 | 12 | 0 |
| 2014 | 0.00% | 0 / 3600 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2015 | 0.06% | 2 / 3601 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2016 | 0.00% | 0 / 3600 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2017 | 0.22% | 8 / 3604 | 12 | 0 | 0.13% | 8 / 6004 | 12 | 0 |
| 2018 | 0.00% | 0 / 3600 | 12 | 0 | 0.00% | 0 / 6000 | 12 | 0 |
| 2019 | 0.00% | 0 / 3600 | 12 | 0 | 0.17% | 10 / 6005 | 12 | 0 |
| 2020 | 0.00% | 0 / 3600 | 12 | 0 | 0.13% | 8 / 6004 | 12 | 0 |
| 2021 | 2.90% | 106 / 3653 | 12 | 5 | 3.46% | 211 / 6105 | 12 | 5 |
| 2022 | 0.83% | 30 / 3615 | 12 | 12 | 1.65% | 100 / 6050 | 12 | 12 |
| 2023 | 1.65% | 60 / 3630 | 12 | 2 | 3.29% | 201 / 6101 | 12 | 2 |
| 2024 | 0.00% | 0 / 600 | 2 | 0 | 0.20% | 2 / 1002 | 2 | 0 |

### Worst month ends

**CSI300** (the 10 largest):

| Month end | In the archive only | In BaoStock only | In either | Mismatch share | BaoStock's list is dated | Days old |
|---|---|---|---|---|---|---|
| 2021-12-31 | 28 | 28 | 328 | 17.07% | 2021-11-15 | 46 |
| 2021-06-30 | 25 | 25 | 325 | 15.38% | 2021-06-14 | 16 |
| 2022-12-30 | 15 | 15 | 315 | 9.52% | 2022-08-01 | 151 |
| 2023-01-31 | 15 | 15 | 315 | 9.52% | 2022-08-01 | 183 |
| 2023-02-28 | 15 | 15 | 315 | 9.52% | 2022-08-01 | 211 |
| 2007-06-29 | 5 | 5 | 305 | 3.28% | 2007-06-25 | 4 |
| 2007-05-31 | 4 | 4 | 304 | 2.63% | 2007-05-28 | 3 |
| 2007-04-30 | 3 | 3 | 303 | 1.98% | 2007-04-30 | 0 |
| 2009-12-31 | 0 | 2 | 300 | 0.67% | 2009-12-28 | 3 |
| 2007-01-31 | 1 | 1 | 301 | 0.66% | 2007-01-29 | 2 |

**CSI500** (the 10 largest):

| Month end | In the archive only | In BaoStock only | In either | Mismatch share | BaoStock's list is dated | Days old |
|---|---|---|---|---|---|---|
| 2021-06-30 | 50 | 50 | 550 | 18.18% | 2021-06-14 | 16 |
| 2021-12-31 | 50 | 50 | 550 | 18.18% | 2021-11-15 | 46 |
| 2022-12-30 | 50 | 50 | 550 | 18.18% | 2022-08-01 | 151 |
| 2023-01-31 | 50 | 50 | 550 | 18.18% | 2022-08-01 | 183 |
| 2023-02-28 | 50 | 50 | 550 | 18.18% | 2022-08-01 | 211 |
| 2021-10-29 | 2 | 2 | 502 | 0.80% | 2021-10-18 | 11 |
| 2021-11-30 | 2 | 2 | 502 | 0.80% | 2021-11-15 | 15 |
| 2021-09-30 | 2 | 1 | 501 | 0.60% | 2021-09-27 | 3 |
| 2009-08-31 | 1 | 1 | 501 | 0.40% | 2009-08-31 | 0 |
| 2009-09-30 | 1 | 1 | 501 | 0.40% | 2009-09-28 | 2 |

## 2. How BaoStock dates its lists

Measured on every answer the audit used (month ends and the extra days of step 2):

| Index | Answers with stocks | One update date for all rows | Update date on or before the asked date | Distinct update dates seen | Their weekdays | Update dates that always come with the same list | Change days where the new list starts on its update date | Answers at most 6 days old | Oldest answer |
|---|---|---|---|---|---|---|---|---|---|
| CSI300 | 417 | 417 of 417 | 417 of 417 | 295 | Monday 295 | 295 of 295 | 50 of 50 | 390 of 417 | 228 days (asked 2023-03-17, dated 2022-08-01) |
| CSI500 | 421 | 421 of 421 | 421 of 421 | 292 | Monday 292 | 292 of 292 | 50 of 50 | 387 of 421 | 228 days (asked 2023-03-17, dated 2022-08-01) |

Reading: an answer is the latest list BaoStock stored on or before the asked date, and the update date is the
day that list was stored. "Starts on its update date" means the update date lies after the last trading day
with the old list and on or before the first trading day with the new one (it can be a holiday in between).

- **Every update date seen is a Monday: BaoStock stores at most one list per week.** A change that takes
  effect on a Wednesday appears in BaoStock on the following Monday at the earliest, up to 4 trading days
  later. So a negative lag of a few days does not prove that the archive is early.
- **BaoStock skipped updates in places.** At 38 month ends its list was more than 6 days old
  (table below). In such a stretch BaoStock shows an index change weeks or months after it happened.
- **Which day was used, and why:** the day found by direct queries. The update date agrees with it in every
  case, so it proved reliable, but the change day does not depend on it. The update date is used for two
  things: the date of BaoStock's last old list, which separates the two kinds of "early", and the age of
  a list, which shows where BaoStock itself was out of date.
- Stock moves that BaoStock undid again within the same month (not counted as changes): CSI300 0, CSI500 0. Changes
  whose BaoStock day could not be pinned to one trading day because of unusable answers: 0.

Month ends at which BaoStock's list was more than 6 days old:

| Index | Month end | BaoStock's list is dated | Days old | Stocks in one list only | Mismatch share |
|---|---|---|---|---|---|
| CSI300 | 2021-06-30 | 2021-06-14 | 16 | 50 | 15.38% |
| CSI300 | 2021-08-31 | 2021-08-09 | 22 | 0 | 0.00% |
| CSI300 | 2021-10-29 | 2021-10-18 | 11 | 0 | 0.00% |
| CSI300 | 2021-11-30 | 2021-11-15 | 15 | 0 | 0.00% |
| CSI300 | 2021-12-31 | 2021-11-15 | 46 | 56 | 17.07% |
| CSI300 | 2022-01-28 | 2022-01-17 | 11 | 0 | 0.00% |
| CSI300 | 2022-02-28 | 2022-01-17 | 42 | 0 | 0.00% |
| CSI300 | 2022-03-31 | 2022-03-21 | 10 | 0 | 0.00% |
| CSI300 | 2022-04-29 | 2022-03-21 | 39 | 0 | 0.00% |
| CSI300 | 2022-05-31 | 2022-03-21 | 71 | 0 | 0.00% |
| CSI300 | 2022-06-30 | 2022-06-13 | 17 | 0 | 0.00% |
| CSI300 | 2022-07-29 | 2022-06-13 | 46 | 0 | 0.00% |
| CSI300 | 2022-08-31 | 2022-08-01 | 30 | 0 | 0.00% |
| CSI300 | 2022-09-30 | 2022-08-01 | 60 | 0 | 0.00% |
| CSI300 | 2022-10-31 | 2022-08-01 | 91 | 0 | 0.00% |
| CSI300 | 2022-11-30 | 2022-08-01 | 121 | 0 | 0.00% |
| CSI300 | 2022-12-30 | 2022-08-01 | 151 | 30 | 9.52% |
| CSI300 | 2023-01-31 | 2022-08-01 | 183 | 30 | 9.52% |
| CSI300 | 2023-02-28 | 2022-08-01 | 211 | 30 | 9.52% |
| CSI500 | 2021-06-30 | 2021-06-14 | 16 | 100 | 18.18% |
| CSI500 | 2021-08-31 | 2021-08-09 | 22 | 0 | 0.00% |
| CSI500 | 2021-10-29 | 2021-10-18 | 11 | 4 | 0.80% |
| CSI500 | 2021-11-30 | 2021-11-15 | 15 | 4 | 0.80% |
| CSI500 | 2021-12-31 | 2021-11-15 | 46 | 100 | 18.18% |
| CSI500 | 2022-01-28 | 2022-01-17 | 11 | 0 | 0.00% |
| CSI500 | 2022-02-28 | 2022-01-17 | 42 | 0 | 0.00% |
| CSI500 | 2022-03-31 | 2022-03-21 | 10 | 0 | 0.00% |
| CSI500 | 2022-04-29 | 2022-03-21 | 39 | 0 | 0.00% |
| CSI500 | 2022-05-31 | 2022-03-21 | 71 | 0 | 0.00% |
| CSI500 | 2022-06-30 | 2022-06-13 | 17 | 0 | 0.00% |
| CSI500 | 2022-07-29 | 2022-06-13 | 46 | 0 | 0.00% |
| CSI500 | 2022-08-31 | 2022-08-01 | 30 | 0 | 0.00% |
| CSI500 | 2022-09-30 | 2022-08-01 | 60 | 0 | 0.00% |
| CSI500 | 2022-10-31 | 2022-08-01 | 91 | 0 | 0.00% |
| CSI500 | 2022-11-30 | 2022-08-01 | 121 | 0 | 0.00% |
| CSI500 | 2022-12-30 | 2022-08-01 | 151 | 100 | 18.18% |
| CSI500 | 2023-01-31 | 2022-08-01 | 183 | 100 | 18.18% |
| CSI500 | 2023-02-28 | 2022-08-01 | 211 | 100 | 18.18% |

## 3. How late the archive shows index changes

Lags are in trading days (positive = the archive is late). The 90th percentile is the smallest lag that at
least 90% of the changes do not exceed. "Never shown (share)" = never shown / (paired + never shown).

**Changes that BaoStock dates to the week** (its last old list and its first new list are at most 5 trading
days apart). This is the cleaner measure of the archive's delay:

| Index | Period | Change | BaoStock changes | With a lag | Median lag | 90th percentile | Longest | Shortest | Late | Same day | Early, between two BaoStock lists | Early, clear | Never shown (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CSI300 | all | adds | 666 | 665 | 9 | 14 | 102 | -4 | 383 | 101 | 181 | 0 | 1 (0.2%) |
| CSI300 | all | drops | 666 | 665 | 9 | 14 | 102 | -4 | 384 | 99 | 182 | 0 | 1 (0.2%) |
| CSI300 | all | both | 1332 | 1330 | 9 | 14 | 102 | -4 | 767 | 200 | 363 | 0 | 2 (0.2%) |
| CSI300 | 2012-01 on | adds | 450 | 450 | 12 | 14 | 89 | -3 | 376 | 34 | 40 | 0 | 0 (0.0%) |
| CSI300 | 2012-01 on | drops | 450 | 449 | 12 | 14 | 89 | -3 | 376 | 34 | 39 | 0 | 1 (0.2%) |
| CSI300 | 2012-01 on | both | 900 | 899 | 12 | 14 | 89 | -3 | 752 | 68 | 79 | 0 | 1 (0.1%) |
| CSI500 | all | adds | 1516 | 1513 | 13 | 21 | 98 | -4 | 1261 | 50 | 202 | 0 | 3 (0.2%) |
| CSI500 | all | drops | 1514 | 1512 | 13 | 21 | 92 | -4 | 1260 | 50 | 202 | 0 | 2 (0.1%) |
| CSI500 | all | both | 3030 | 3025 | 13 | 21 | 98 | -4 | 2521 | 100 | 404 | 0 | 5 (0.2%) |
| CSI500 | 2012-01 on | adds | 1061 | 1058 | 13 | 18 | 92 | -1 | 1057 | 0 | 1 | 0 | 3 (0.3%) |
| CSI500 | 2012-01 on | drops | 1059 | 1058 | 13 | 18 | 92 | -1 | 1057 | 0 | 1 | 0 | 1 (0.1%) |
| CSI500 | 2012-01 on | both | 2120 | 2116 | 13 | 18 | 92 | -1 | 2114 | 0 | 2 | 0 | 4 (0.2%) |

The same week-dated changes, split by what kind of day it was in BaoStock. The index is reviewed twice a
year, when many stocks move at once; in between, single stocks move now and then (a merger, a delisting, a
very large new listing):

| Index | BaoStock change day | Days | Changes | With a lag | Median lag | 90th percentile | Longest | Shortest | Never shown (share) |
|---|---|---|---|---|---|---|---|---|---|
| CSI300 | review days (10 or more stocks move) | 30 | 1296 | 1295 | 9 | 14 | 14 | -4 | 1 (0.1%) |
| CSI300 | other days (fewer than 10) | 16 | 36 | 35 | 5 | 89 | 102 | -4 | 1 (2.8%) |
| CSI500 | review days (10 or more stocks move) | 30 | 3000 | 2998 | 13 | 21 | 22 | -4 | 2 (0.1%) |
| CSI500 | other days (fewer than 10) | 14 | 30 | 27 | 19 | 92 | 98 | -3 | 3 (10.0%) |

Where those review days fall in the calendar:

| Index | Review day | Days | BaoStock's days | Changes | Median lag | Shortest | Longest |
|---|---|---|---|---|---|---|---|
| CSI300 | BaoStock's day is the first trading day after the second Friday of June or December | 17 | 2013-12-16 to 2023-12-11 | 742 | 13 | 9 | 14 |
| CSI300 | the archive's day is the first trading day of January or July | 13 | 2007-07-02 to 2013-07-01 | 553 | -2 | -4 | 0 |
| CSI500 | BaoStock's day is the first trading day after the second Friday of June or December | 17 | 2013-12-16 to 2023-12-11 | 1699 | 12 | 9 | 14 |
| CSI500 | the archive's day is the first trading day of January or July | 5 | 2009-07-06 to 2011-07-04 | 500 | -2 | -4 | 0 |
| CSI500 | neither of the two | 8 | 2007-07-02 to 2013-07-01 | 799 | 18 | 11 | 22 |

A reading aid, not a finding: the index provider's twice-yearly reviews are generally described as taking
effect at the start of January and July until mid-2013, and on the trading day after the second Friday of
June and December since then. This audit did not check that against official notices. If it is right, a
row of the first kind is BaoStock on the true day with the archive behind it, a row of the second kind is
the archive on the true day with BaoStock's weekly list a few days behind, and a row of the third kind is a
review that the archive shows late by more than BaoStock's weekly step.

**All changes**, including those where BaoStock skipped updates and its change day is too late:

| Index | Period | Change | BaoStock changes | With a lag | Median lag | 90th percentile | Longest | Shortest | Late | Same day | Early, between two BaoStock lists | Early, clear | Never shown (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CSI300 | all | adds | 762 | 761 | 9 | 14 | 102 | -50 | 411 | 101 | 249 | 0 | 1 (0.1%) |
| CSI300 | all | drops | 762 | 761 | 9 | 14 | 102 | -50 | 412 | 99 | 250 | 0 | 1 (0.1%) |
| CSI300 | all | both | 1524 | 1522 | 9 | 14 | 102 | -50 | 823 | 200 | 499 | 0 | 2 (0.1%) |
| CSI300 | 2012-01 on | adds | 546 | 546 | 11 | 14 | 89 | -50 | 404 | 34 | 108 | 0 | 0 (0.0%) |
| CSI300 | 2012-01 on | drops | 546 | 545 | 11 | 14 | 89 | -50 | 404 | 34 | 107 | 0 | 1 (0.2%) |
| CSI300 | 2012-01 on | both | 1092 | 1091 | 11 | 14 | 89 | -50 | 808 | 68 | 215 | 0 | 1 (0.1%) |
| CSI500 | all | adds | 1717 | 1714 | 12 | 18 | 98 | -50 | 1312 | 50 | 352 | 0 | 3 (0.2%) |
| CSI500 | all | drops | 1717 | 1714 | 12 | 18 | 114 | -50 | 1313 | 50 | 351 | 0 | 3 (0.2%) |
| CSI500 | all | both | 3434 | 3428 | 12 | 18 | 114 | -50 | 2625 | 100 | 703 | 0 | 6 (0.2%) |
| CSI500 | 2012-01 on | adds | 1262 | 1259 | 12 | 18 | 92 | -50 | 1108 | 0 | 151 | 0 | 3 (0.2%) |
| CSI500 | 2012-01 on | drops | 1262 | 1260 | 12 | 18 | 114 | -50 | 1110 | 0 | 150 | 0 | 2 (0.2%) |
| CSI500 | 2012-01 on | both | 2524 | 2519 | 12 | 18 | 114 | -50 | 2218 | 0 | 301 | 0 | 5 (0.2%) |

Not paired and left out of the never-shown share: CSI300: 0 too close to the cut-off, 0 in the archive from its first day; CSI500: 0 too close to the cut-off, 0 in the archive from its first day.

### Changes the archive shows EARLIER than BaoStock

**CSI300, early, clear: 0 of 1522 paired changes (0.0%); 0 adds, 0 drops.**

**CSI300, early, between two BaoStock lists: 499 of 1522 paired changes (32.8%); 249 adds, 250 drops.** Of these, 363 are dated to the week by BaoStock (the archive is at most a few days ahead of the next Monday list) and 136 fall where BaoStock skipped updates.

| BaoStock's new list starts on | BaoStock's last old list is dated | Archive shows it on | Lag (trading days) | Outcome | Adds | Drops | Stocks |
|---|---|---|---|---|---|---|---|
| 2008-01-07 | 2007-12-31 | 2008-01-02 | -3 | early, between two BaoStock lists | 30 | 30 | SH600089, SH600109, SH600111, SH600115 ... |
| 2008-03-24 | 2008-03-17 | 2008-03-18 | -4 | early, between two BaoStock lists | 1 | 1 | SZ000301, SH600786 |
| 2008-07-07 | 2008-06-30 | 2008-07-01 | -4 | early, between two BaoStock lists | 19 | 19 | SH600169, SH600376, SH600804, SH600859 ... |
| 2009-07-06 | 2009-06-29 | 2009-07-01 | -3 | early, between two BaoStock lists | 24 | 24 | SH600169, SH600176, SH600216, SH600276 ... |
| 2010-01-04 | 2009-12-28 | 2009-12-29 | -3 | early, between two BaoStock lists | 0 | 2 | SH600001, SH600357 |
| 2010-07-05 | 2010-06-28 | 2010-07-01 | -2 | early, between two BaoStock lists | 18 | 18 | SH600300, SH600863, SH600999, SH601003 ... |
| 2011-01-10 | 2011-01-03 | 2011-01-04 | -4 | early, between two BaoStock lists | 26 | 26 | SH600115, SH600276, SH600406, SH600481 ... |
| 2011-07-04 | 2011-06-27 | 2011-07-01 | -1 | early, between two BaoStock lists | 23 | 23 | SH600267, SH600859, SH600971, SH600998 ... |
| 2012-01-09 | 2012-01-02 | 2012-01-04 | -3 | early, between two BaoStock lists | 24 | 23 | SH600098, SH600160, SH600252, SH600259 ... |
| 2013-01-07 | 2012-12-31 | 2013-01-04 | -1 | early, between two BaoStock lists | 15 | 15 | SH600027, SH600157, SH600340, SH600582 ... |
| 2016-01-04 | 2015-12-28 | 2015-12-31 | -1 | early, between two BaoStock lists | 1 | 1 | SZ001979, SZ000024 |
| 2021-07-19 | 2021-06-14 | 2021-06-30 | -13 | early, between two BaoStock lists | 25 | 25 | SH600079, SH600132, SH600143, SH600426 ... |
| 2022-01-17 | 2021-11-15 | 2021-12-31 | -10 | early, between two BaoStock lists | 28 | 28 | SH600905, SH601728, SH601865, SH601868 ... |
| 2023-03-20 | 2022-08-01 | 2022-12-30 | -50 | early, between two BaoStock lists | 15 | 15 | SH600039, SH600233, SH600674, SH600803 ... |

**CSI500, early, clear: 0 of 3428 paired changes (0.0%); 0 adds, 0 drops.**

**CSI500, early, between two BaoStock lists: 703 of 3428 paired changes (20.5%); 352 adds, 351 drops.** Of these, 404 are dated to the week by BaoStock (the archive is at most a few days ahead of the next Monday list) and 299 fall where BaoStock skipped updates.

| BaoStock's new list starts on | BaoStock's last old list is dated | Archive shows it on | Lag (trading days) | Outcome | Adds | Drops | Stocks |
|---|---|---|---|---|---|---|---|
| 2009-05-11 | 2009-05-04 | 2009-05-06 | -3 | early, between two BaoStock lists | 1 | 1 | SH600436, SZ000515 |
| 2009-07-06 | 2009-06-29 | 2009-07-01 | -3 | early, between two BaoStock lists | 50 | 50 | SH600163, SH600409, SH600425, SH600435 ... |
| 2010-07-05 | 2010-06-28 | 2010-07-01 | -2 | early, between two BaoStock lists | 50 | 50 | SH600020, SH600139, SH600158, SH600252 ... |
| 2011-01-10 | 2011-01-03 | 2011-01-04 | -4 | early, between two BaoStock lists | 50 | 50 | SH600021, SH600176, SH600223, SH600236 ... |
| 2011-07-04 | 2011-06-27 | 2011-07-01 | -1 | early, between two BaoStock lists | 50 | 50 | SH600006, SH600017, SH600022, SH600033 ... |
| 2013-05-02 | 2013-04-22 | 2013-04-26 | -1 | early, between two BaoStock lists | 1 | 1 | SZ002093, SZ000522 |
| 2021-07-19 | 2021-06-14 | 2021-06-30 | -13 | early, between two BaoStock lists | 50 | 50 | SH600027, SH600066, SH600068, SH600095 ... |
| 2022-01-17 | 2021-11-15 | 2021-12-31 | -10 | early, between two BaoStock lists | 50 | 49 | SH600118, SH600157, SH600233, SH600236 ... |
| 2023-03-20 | 2022-08-01 | 2022-12-30 | -50 | early, between two BaoStock lists | 50 | 50 | SH600032, SH600161, SH600352, SH600378 ... |

### Per year

By the year of BaoStock's change day. The three lag columns use only the changes that BaoStock dates to the
week; "Early" counts both kinds over all changes, with the clear ones in brackets.

| Year | CSI300 changes | Dated to the week | Median lag | Shortest | Longest | Early (clear) | Never shown | CSI500 changes | Dated to the week | Median lag | Shortest | Longest | Early (clear) | Never shown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2007 | 72 | 72 | 0 | 0 | 70 | 0 (0) | 0 | 100 | 100 | 21 | 21 | 21 | 0 (0) | 0 |
| 2008 | 102 | 102 | -3 | -4 | 0 | 100 (0) | 0 | 200 | 200 | 18 | 18 | 18 | 0 (0) | 0 |
| 2009 | 84 | 84 | -3 | -3 | 0 | 48 (0) | 0 | 204 | 204 | 5.5 | -3 | 83 | 102 (0) | 0 |
| 2010 | 74 | 74 | -2 | -3 | 102 | 38 (0) | 0 | 202 | 202 | 0 | -2 | 88 | 100 (0) | 0 |
| 2011 | 100 | 100 | -4 | -4 | 84 | 98 (0) | 1 | 204 | 204 | -1 | -4 | 98 | 200 (0) | 1 |
| 2012 | 84 | 84 | -3 | -3 | 0 | 47 (0) | 1 | 206 | 206 | 19 | 4 | 21 | 0 (0) | 1 |
| 2013 | 106 | 106 | 0 | -1 | 11 | 30 (0) | 0 | 302 | 302 | 18 | -1 | 22 | 2 (0) | 0 |
| 2014 | 96 | 96 | 10 | 10 | 12 | 0 (0) | 0 | 200 | 200 | 11 | 10 | 12 | 0 (0) | 0 |
| 2015 | 82 | 82 | 10 | 4 | 13 | 0 (0) | 0 | 200 | 200 | 11.5 | 10 | 13 | 0 (0) | 0 |
| 2016 | 110 | 110 | 14 | -1 | 14 | 2 (0) | 0 | 200 | 200 | 13.5 | 13 | 14 | 0 (0) | 0 |
| 2017 | 110 | 110 | 14 | 14 | 89 | 0 (0) | 0 | 204 | 204 | 14 | 3 | 89 | 0 (0) | 0 |
| 2018 | 102 | 102 | 13 | 9 | 13 | 0 (0) | 0 | 200 | 200 | 11 | 9 | 13 | 0 (0) | 0 |
| 2019 | 70 | 70 | 9 | 9 | 11 | 0 (0) | 0 | 202 | 201 | 11 | 9 | 11 | 0 (0) | 2 |
| 2020 | 94 | 94 | 13 | 9 | 13 | 0 (0) | 0 | 202 | 202 | 13 | 9 | 92 | 0 (0) | 0 |
| 2021 | 50 | 0 | n/a | n/a | n/a | 50 (0) | 0 | 108 | 5 | 12 | 12 | 12 | 100 (0) | 1 |
| 2022 | 112 | 0 | n/a | n/a | n/a | 56 (0) | 0 | 200 | 0 | n/a | n/a | n/a | 99 (0) | 1 |
| 2023 | 76 | 46 | 14 | 12 | 14 | 30 (0) | 0 | 300 | 200 | 13 | 12 | 14 | 100 (0) | 0 |

### Changes the archive never shows

**CSI300: 2** (1 add, 1 drop).

| BaoStock's new list starts on | BaoStock's last old list is dated | Archive shows it on | Lag (trading days) | Outcome | Adds | Drops | Stocks |
|---|---|---|---|---|---|---|---|
| 2011-08-29 | 2011-08-22 | never | n/a | never shown | 1 | 0 | SH600827 |
| 2012-01-09 | 2012-01-02 | never | n/a | never shown | 0 | 1 | SH600827 |

**CSI500: 6** (3 adds, 3 drops).

| BaoStock's new list starts on | BaoStock's last old list is dated | Archive shows it on | Lag (trading days) | Outcome | Adds | Drops | Stocks |
|---|---|---|---|---|---|---|---|
| 2011-08-29 | 2011-08-22 | never | n/a | never shown | 0 | 1 | SH600827 |
| 2012-01-09 | 2012-01-02 | never | n/a | never shown | 1 | 0 | SH600827 |
| 2019-01-21 | 2019-01-14 | never | n/a | never shown | 1 | 0 | SH601598 |
| 2019-06-17 | 2019-06-10 | never | n/a | never shown | 0 | 1 | SH601598 |
| 2021-10-08 | 2021-09-27 | never | n/a | never shown | 1 | 0 | SH601868 |
| 2022-01-17 | 2021-11-15 | never | n/a | never shown | 0 | 1 | SH601868 |

### The other direction: archive changes that BaoStock never shows

- CSI300: 2 (1 add, 1 drop) after 2007-01-31. Days with the most: 2007-02-26 (2).
- CSI500: 1 (1 add, 0 drops) after 2007-01-31. Days with the most: 2023-12-29 (1).

These are stock moves in the archive with no BaoStock change of the same stock and direction within 250
trading days. Some are the other half of a "never shown" case above (the two sources swapped different
stocks). One shortly after 2007-01-31 can also belong to a BaoStock change from before the audit window.

### Every day on which BaoStock's list changes

One row per BaoStock change day, archive day and outcome; "Stocks" names up to four of them.

**CSI300:**

| BaoStock's new list starts on | BaoStock's last old list is dated | Archive shows it on | Lag (trading days) | Outcome | Adds | Drops | Stocks |
|---|---|---|---|---|---|---|---|
| 2007-02-26 | 2007-02-12 | 2007-02-26 | 0 | same day | 1 | 1 | SH601166, SZ000939 |
| 2007-03-19 | 2007-03-12 | 2007-07-02 | 70 | late | 1 | 1 | SH601318, SH600231 |
| 2007-04-30 | 2007-04-23 | 2007-07-02 | 40 | late | 2 | 2 | SH600501, SH601600, SH600205, SH600296 |
| 2007-05-21 | 2007-05-14 | 2007-07-02 | 30 | late | 1 | 1 | SH601998, SH600724 |
| 2007-06-04 | 2007-05-28 | 2007-07-02 | 20 | late | 1 | 1 | SH601328, SH600333 |
| 2007-07-02 | 2007-06-25 | 2007-07-02 | 0 | same day | 28 | 28 | SH600003, SH600017, SH600208, SH600219 ... |
| 2007-10-29 | 2007-10-22 | 2007-11-19 | 15 | late | 1 | 1 | SH601088, SH600200 |
| 2007-11-19 | 2007-11-12 | 2007-11-19 | 0 | same day | 1 | 1 | SH601857, SZ002025 |
| 2008-01-02 | 2007-12-24 | 2008-01-02 | 0 | same day | 1 | 1 | SH600380, SH600472 |
| 2008-01-07 | 2007-12-31 | 2008-01-02 | -3 | early, between two BaoStock lists | 30 | 30 | SH600089, SH600109, SH600111, SH600115 ... |
| 2008-03-24 | 2008-03-17 | 2008-03-18 | -4 | early, between two BaoStock lists | 1 | 1 | SZ000301, SH600786 |
| 2008-07-07 | 2008-06-30 | 2008-07-01 | -4 | early, between two BaoStock lists | 19 | 19 | SH600169, SH600376, SH600804, SH600859 ... |
| 2009-01-05 | 2008-12-29 | 2009-01-05 | 0 | same day | 18 | 18 | SH600062, SH600251, SH600299, SH600426 ... |
| 2009-07-06 | 2009-06-29 | 2009-07-01 | -3 | early, between two BaoStock lists | 24 | 24 | SH600169, SH600176, SH600216, SH600276 ... |
| 2010-01-04 | 2009-12-28 | 2009-12-29 | -3 | early, between two BaoStock lists | 0 | 2 | SH600001, SH600357 |
| 2010-01-04 | 2009-12-28 | 2010-01-04 | 0 | same day | 18 | 16 | SH600062, SH600161, SH600166, SH600239 ... |
| 2010-07-05 | 2010-06-28 | 2010-07-01 | -2 | early, between two BaoStock lists | 18 | 18 | SH600300, SH600863, SH600999, SH601003 ... |
| 2010-08-02 | 2010-07-26 | 2011-01-04 | 102 | late | 1 | 1 | SH601288, SH600685 |
| 2011-01-10 | 2011-01-03 | 2011-01-04 | -4 | early, between two BaoStock lists | 26 | 26 | SH600115, SH600276, SH600406, SH600481 ... |
| 2011-07-04 | 2011-06-27 | 2011-07-01 | -1 | early, between two BaoStock lists | 23 | 23 | SH600267, SH600859, SH600971, SH600998 ... |
| 2011-08-29 | 2011-08-22 | 2012-01-04 | 84 | late | 0 | 1 | SH600631 |
| 2011-08-29 | 2011-08-22 | never | n/a | never shown | 1 | 0 | SH600827 |
| 2012-01-09 | 2012-01-02 | 2012-01-04 | -3 | early, between two BaoStock lists | 24 | 23 | SH600098, SH600160, SH600252, SH600259 ... |
| 2012-01-09 | 2012-01-02 | never | n/a | never shown | 0 | 1 | SH600827 |
| 2012-07-02 | 2012-06-25 | 2012-07-02 | 0 | same day | 18 | 18 | SH600011, SH600022, SH600315, SH600827 ... |
| 2013-01-07 | 2012-12-31 | 2013-01-04 | -1 | early, between two BaoStock lists | 15 | 15 | SH600027, SH600157, SH600340, SH600582 ... |
| 2013-07-01 | 2013-06-24 | 2013-07-01 | 0 | same day | 16 | 16 | SH600060, SH600332, SH600403, SH600436 ... |
| 2013-09-23 | 2013-09-16 | 2013-09-30 | 5 | late | 1 | 1 | SZ000333, SZ000527 |
| 2013-12-16 | 2013-12-09 | 2013-12-31 | 11 | late | 21 | 21 | SH600008, SH600018, SH600079, SH600597 ... |
| 2014-06-16 | 2014-06-09 | 2014-06-30 | 10 | late | 26 | 26 | SH600023, SH600277, SH600633, SH600867 ... |
| 2014-12-15 | 2014-12-08 | 2014-12-31 | 12 | late | 22 | 22 | SH600038, SH600373, SH600398, SH600485 ... |
| 2015-01-26 | 2015-01-19 | 2015-01-30 | 4 | late | 1 | 1 | SZ000166, SZ000562 |
| 2015-05-25 | 2015-05-18 | 2015-05-29 | 4 | late | 2 | 2 | SZ000738, SZ300003, SH600832, SH601299 |
| 2015-06-15 | 2015-06-08 | 2015-06-30 | 10 | late | 18 | 18 | SH600005, SH600317, SH600717, SH600958 ... |
| 2015-12-14 | 2015-12-07 | 2015-12-31 | 13 | late | 20 | 20 | SH600021, SH600350, SH600820, SH600895 ... |
| 2016-01-04 | 2015-12-28 | 2015-12-31 | -1 | early, between two BaoStock lists | 1 | 1 | SZ001979, SZ000024 |
| 2016-06-13 | 2016-06-06 | 2016-06-30 | 13 | late | 24 | 24 | SH600022, SH600037, SH600061, SH600074 ... |
| 2016-12-12 | 2016-12-05 | 2016-12-30 | 14 | late | 30 | 30 | SH600297, SH600482, SH600498, SH600654 ... |
| 2017-02-20 | 2017-02-13 | 2017-06-30 | 89 | late | 1 | 1 | SH600549, SH600005 |
| 2017-06-12 | 2017-06-05 | 2017-06-30 | 14 | late | 30 | 30 | SH600233, SH600436, SH600522, SH600682 ... |
| 2017-12-11 | 2017-12-04 | 2017-12-29 | 14 | late | 24 | 24 | SH600011, SH600219, SH600390, SH601012 ... |
| 2018-06-11 | 2018-06-04 | 2018-06-29 | 13 | late | 27 | 27 | SH600025, SH600176, SH600339, SH600346 ... |
| 2018-12-17 | 2018-12-10 | 2018-12-28 | 9 | late | 24 | 24 | SH600004, SH600027, SH600566, SH600760 ... |
| 2019-06-17 | 2019-06-10 | 2019-06-28 | 9 | late | 19 | 19 | SH600299, SH600663, SH600733, SH601162 ... |
| 2019-12-16 | 2019-12-09 | 2019-12-31 | 11 | late | 16 | 16 | SH600183, SH600655, SH600848, SH600928 ... |
| 2020-06-15 | 2020-06-08 | 2020-06-30 | 9 | late | 21 | 21 | SH600745, SH601077, SH601100, SH601231 ... |
| 2020-12-14 | 2020-12-07 | 2020-12-31 | 13 | late | 26 | 26 | SH600150, SH600161, SH600584, SH600600 ... |
| 2021-07-19 | 2021-06-14 | 2021-06-30 | -13 | early, between two BaoStock lists | 25 | 25 | SH600079, SH600132, SH600143, SH600426 ... |
| 2022-01-17 | 2021-11-15 | 2021-12-31 | -10 | early, between two BaoStock lists | 28 | 28 | SH600905, SH601728, SH601865, SH601868 ... |
| 2022-06-13 | 2022-03-21 | 2022-06-30 | 13 | late | 28 | 28 | SH600089, SH600188, SH600219, SH600460 ... |
| 2023-03-20 | 2022-08-01 | 2022-12-30 | -50 | early, between two BaoStock lists | 15 | 15 | SH600039, SH600233, SH600674, SH600803 ... |
| 2023-06-12 | 2023-06-05 | 2023-06-30 | 12 | late | 9 | 9 | SH600732, SH600754, SH600875, SH601607 ... |
| 2023-12-11 | 2023-12-04 | 2023-12-29 | 14 | late | 14 | 14 | SH600023, SH600372, SH600489, SH600515 ... |

**CSI500:**

| BaoStock's new list starts on | BaoStock's last old list is dated | Archive shows it on | Lag (trading days) | Outcome | Adds | Drops | Stocks |
|---|---|---|---|---|---|---|---|
| 2007-07-02 | 2007-06-25 | 2007-07-31 | 21 | late | 50 | 50 | SH600073, SH600121, SH600166, SH600186 ... |
| 2008-01-07 | 2007-12-31 | 2008-01-31 | 18 | late | 50 | 50 | SH600060, SH600062, SH600078, SH600160 ... |
| 2008-07-07 | 2008-06-30 | 2008-07-31 | 18 | late | 50 | 50 | SH600035, SH600038, SH600053, SH600064 ... |
| 2009-01-05 | 2008-12-29 | 2009-01-23 | 14 | late | 50 | 50 | SH600012, SH600052, SH600072, SH600107 ... |
| 2009-05-11 | 2009-05-04 | 2009-05-06 | -3 | early, between two BaoStock lists | 1 | 1 | SH600436, SZ000515 |
| 2009-07-06 | 2009-06-29 | 2009-07-01 | -3 | early, between two BaoStock lists | 50 | 50 | SH600163, SH600409, SH600425, SH600435 ... |
| 2009-08-31 | 2009-08-24 | 2010-01-04 | 83 | late | 1 | 1 | SZ002230, SH600840 |
| 2010-01-04 | 2009-12-28 | 2010-01-04 | 0 | same day | 50 | 50 | SH600110, SH600112, SH600117, SH600198 ... |
| 2010-02-22 | 2010-02-08 | 2010-07-01 | 88 | late | 1 | 1 | SZ002121, SH600607 |
| 2010-07-05 | 2010-06-28 | 2010-07-01 | -2 | early, between two BaoStock lists | 50 | 50 | SH600020, SH600139, SH600158, SH600252 ... |
| 2011-01-10 | 2011-01-03 | 2011-01-04 | -4 | early, between two BaoStock lists | 50 | 50 | SH600021, SH600176, SH600223, SH600236 ... |
| 2011-02-21 | 2011-02-14 | 2011-07-01 | 90 | late | 1 | 1 | SZ002428, SH600553 |
| 2011-07-04 | 2011-06-27 | 2011-07-01 | -1 | early, between two BaoStock lists | 50 | 50 | SH600006, SH600017, SH600022, SH600033 ... |
| 2011-08-29 | 2011-08-22 | 2012-01-31 | 98 | late | 1 | 0 | SZ002154 |
| 2011-08-29 | 2011-08-22 | never | n/a | never shown | 0 | 1 | SH600827 |
| 2012-01-09 | 2012-01-02 | 2012-01-31 | 11 | late | 49 | 50 | SH600004, SH600157, SH600184, SH600220 ... |
| 2012-01-09 | 2012-01-02 | never | n/a | never shown | 1 | 0 | SH600827 |
| 2012-03-05 | 2012-02-27 | 2012-03-30 | 19 | late | 2 | 2 | SH600819, SH601011, SH600102, SH600263 |
| 2012-03-26 | 2012-03-19 | 2012-03-30 | 4 | late | 1 | 1 | SZ000418, SH600991 |
| 2012-07-02 | 2012-06-25 | 2012-07-31 | 21 | late | 50 | 50 | SH600008, SH600151, SH600161, SH600200 ... |
| 2013-01-07 | 2012-12-31 | 2013-01-31 | 18 | late | 50 | 50 | SH600094, SH600183, SH600260, SH600397 ... |
| 2013-05-02 | 2013-04-22 | 2013-04-26 | -1 | early, between two BaoStock lists | 1 | 1 | SZ002093, SZ000522 |
| 2013-07-01 | 2013-06-24 | 2013-07-31 | 22 | late | 50 | 50 | SH600006, SH600026, SH600037, SH600039 ... |
| 2013-12-16 | 2013-12-09 | 2013-12-31 | 11 | late | 50 | 50 | SH600120, SH600125, SH600132, SH600199 ... |
| 2014-06-16 | 2014-06-09 | 2014-06-30 | 10 | late | 50 | 50 | SH600160, SH600169, SH600175, SH600187 ... |
| 2014-12-15 | 2014-12-08 | 2014-12-31 | 12 | late | 50 | 50 | SH600062, SH600098, SH600216, SH600219 ... |
| 2015-06-15 | 2015-06-08 | 2015-06-30 | 10 | late | 50 | 50 | SH600058, SH600079, SH600143, SH600267 ... |
| 2015-12-14 | 2015-12-07 | 2015-12-31 | 13 | late | 50 | 50 | SH600022, SH600074, SH600094, SH600108 ... |
| 2016-06-13 | 2016-06-06 | 2016-06-30 | 13 | late | 50 | 50 | SH600053, SH600166, SH600266, SH600277 ... |
| 2016-12-12 | 2016-12-05 | 2016-12-30 | 14 | late | 50 | 50 | SH600022, SH600098, SH600291, SH600307 ... |
| 2017-01-23 | 2017-01-16 | 2017-01-26 | 3 | late | 1 | 1 | SZ002670, SZ000748 |
| 2017-02-20 | 2017-02-13 | 2017-06-30 | 89 | late | 1 | 1 | SH600848, SH600549 |
| 2017-06-12 | 2017-06-05 | 2017-06-30 | 14 | late | 50 | 50 | SH600126, SH600566, SH600582, SH600642 ... |
| 2017-12-11 | 2017-12-04 | 2017-12-29 | 14 | late | 50 | 50 | SH600004, SH600037, SH600060, SH600155 ... |
| 2018-06-11 | 2018-06-04 | 2018-06-29 | 13 | late | 50 | 50 | SH600021, SH600167, SH600258, SH600350 ... |
| 2018-12-17 | 2018-12-10 | 2018-12-28 | 9 | late | 50 | 50 | SH600008, SH600373, SH600376, SH600507 ... |
| 2019-01-07 | 2018-12-24 | 2019-06-28 | 114 | late | 0 | 1 | SH600270 |
| 2019-01-21 | 2019-01-14 | never | n/a | never shown | 1 | 0 | SH601598 |
| 2019-06-17 | 2019-06-10 | 2019-06-28 | 9 | late | 50 | 49 | SH600317, SH600486, SH600497, SH600515 ... |
| 2019-06-17 | 2019-06-10 | never | n/a | never shown | 0 | 1 | SH601598 |
| 2019-12-16 | 2019-12-09 | 2019-12-31 | 11 | late | 50 | 50 | SH600132, SH600150, SH600256, SH600273 ... |
| 2020-06-15 | 2020-06-08 | 2020-06-30 | 9 | late | 50 | 50 | SH600153, SH600529, SH600535, SH600566 ... |
| 2020-08-17 | 2020-08-10 | 2020-12-31 | 92 | late | 1 | 1 | SH603708, SZ002470 |
| 2020-12-14 | 2020-12-07 | 2020-12-31 | 13 | late | 50 | 50 | SH600038, SH600089, SH600131, SH600170 ... |
| 2021-03-15 | 2021-03-08 | 2021-03-31 | 12 | late | 2 | 2 | SH600764, SH600956, SH600515, SZ000564 |
| 2021-07-19 | 2021-06-14 | 2021-06-30 | -13 | early, between two BaoStock lists | 50 | 50 | SH600027, SH600066, SH600068, SH600095 ... |
| 2021-09-13 | 2021-08-09 | 2021-12-31 | 72 | late | 1 | 2 | SH600297, SH600068, SH600260 |
| 2021-10-08 | 2021-09-27 | never | n/a | never shown | 1 | 0 | SH601868 |
| 2022-01-17 | 2021-11-15 | 2021-12-31 | -10 | early, between two BaoStock lists | 50 | 49 | SH600118, SH600157, SH600233, SH600236 ... |
| 2022-01-17 | 2021-11-15 | never | n/a | never shown | 0 | 1 | SH601868 |
| 2022-06-13 | 2022-03-21 | 2022-06-30 | 13 | late | 50 | 50 | SH600079, SH600096, SH600109, SH600143 ... |
| 2023-03-20 | 2022-08-01 | 2022-12-30 | -50 | early, between two BaoStock lists | 50 | 50 | SH600032, SH600161, SH600352, SH600378 ... |
| 2023-06-12 | 2023-06-05 | 2023-06-30 | 12 | late | 50 | 50 | SH600416, SH600481, SH600995, SH601136 ... |
| 2023-12-11 | 2023-12-04 | 2023-12-29 | 14 | late | 50 | 50 | SH600004, SH600129, SH600583, SH600884 ... |

## 4. What this means for the project

- **CSI300, 2012-01 on (the dates the model uses):** an index change reaches the model's universe a median of 12 trading days after BaoStock shows it (90th percentile 14, longest 89; 899 changes dated to the week). At a month end where BaoStock is up to date, 0.1 of the 300 names differ on average (0.03%).
- **CSI500, 2012-01 on (the dates the model uses):** an index change reaches the model's universe a median of 13 trading days after BaoStock shows it (90th percentile 18, longest 92; 2116 changes dated to the week). At a month end where BaoStock is up to date, 0.3 of the 500 names differ on average (0.05%).
- **CSI300, where the delay comes from** (changes that BaoStock dates to the week): at the 17 reviews from 2013-12-16 to 2023-12-11 that BaoStock dates to the first trading day after the second Friday of June or December, the lag is 9 to 14 trading days (median 13); at the 13 reviews from 2007-07-02 to 2013-07-01 that the archive dates to the first trading day of January or July, the lag is -4 to 0 trading days (median -2); the 35 changes on other days (single stocks moving between reviews) wait for the archive's next change day: median 5, 90th percentile 89, longest 102 trading days.
- **CSI500, where the delay comes from** (changes that BaoStock dates to the week): at the 17 reviews from 2013-12-16 to 2023-12-11 that BaoStock dates to the first trading day after the second Friday of June or December, the lag is 9 to 14 trading days (median 12); at the 5 reviews from 2009-07-06 to 2011-07-04 that the archive dates to the first trading day of January or July, the lag is -4 to 0 trading days (median -2); at the 8 reviews from 2007-07-02 to 2013-07-01 that fit neither pattern, the lag is 11 to 22 trading days (median 18); the 27 changes on other days (single stocks moving between reviews) wait for the archive's next change day: median 19, 90th percentile 92, longest 98 trading days.
- **What a delay does.** For that many days after an index change, the universe still holds the stocks that
  left the index and lacks the ones that joined. That makes the universe slightly out of date. It uses no
  information from the future, so it cannot inflate a backtest the way look-ahead does.
- **How much of the CSI300 universe is out of date, 2012-01 on.** Stocks that BaoStock had already dropped but the archive still listed add up to 4898 stock-days, 0.55% of the archive's 885600 member-days (one stock on one trading day). About as many stock-days are missing on the other side (stocks that had joined). Where BaoStock skipped updates the delay cannot be measured and counts as zero here, so the true share is somewhat higher.
- **How much of the CSI500 universe is out of date, 2012-01 on.** Stocks that BaoStock had already dropped but the archive still listed add up to 14799 stock-days, 1.00% of the archive's 1476038 member-days (one stock on one trading day). About as many stock-days are missing on the other side (stocks that had joined). Where BaoStock skipped updates the delay cannot be measured and counts as zero here, so the true share is somewhat higher.
- **Look-ahead.** No change is clearly earlier in the archive than in BaoStock.
- **Not settled by this audit.** 1202 changes appear in the archive between two BaoStock lists. Where the
  two lists are a week apart, the archive is at most a few days ahead of BaoStock's next Monday list, which
  is what an archive that is exactly on time looks like. Where BaoStock skipped updates, nothing can be
  said. Settling these needs the index provider's own effective dates, which this audit did not use.
- No data and no config was changed. Rule MA makes this a report, not a decision.

## 5. Cache and hashes

- Cache: `~/data/baostock_membership/2026-11-12/<index>/<asked date>.csv`, one file per BaoStock answer, exactly as returned (columns
  `updateDate, code, code_name`; an empty answer would be a file with only the header). Outside the repo, not committed.
- CSI300: 417 answers (206 month ends, 211 extra days), downloaded 2026-11-12T22:32:49Z to 2026-11-12T22:53:19Z (UTC) with baostock 00.9.40; 0 needed more than one try.
- CSI500: 421 answers (206 month ends, 215 extra days), downloaded 2026-11-12T22:32:49Z to 2026-11-12T22:58:55Z (UTC) with baostock 00.9.40; 0 needed more than one try.
- Hash list: `~/data/baostock_membership/2026-11-12/SHA256SUMS` holds the sha256 of every answer file (838 lines, checkable with
  `shasum -a 256 -c SHA256SUMS` inside the cache folder). The sha256 of that list itself is
  `20fcbe284013c3a0fc74c8b2762412e8a5e6008dacf60ffc1cb52a65a57a3ef7`.
- Archive: tag `2026-11-04`; `instruments/csi300.txt` sha256 `29913856f1d81a734cf531c1c5e4584218b1383840ed80f9dbffd3e538de2adf`,
  `instruments/csi500.txt` sha256 `68e0f2baabcb49ee0867dc64d779ec38c237a9716c8cfba12866e04c59c2abbf`,
  `calendars/day.txt` sha256 `6fd735297189e5fbfcf382320b5e7e1765b1c6686d2f6b12974cce5d6d9d4737`.
- Rebuild this report without any network (it checks every file against the hash list first):
  `python3 -m uv run python scripts/audit_membership.py --offline --cache-dir ~/data/baostock_membership/2026-11-12`
- Requests went out one at a time per index, with a pause before each and up to 6 tries with growing waits.
  An empty answer would have been asked for a second time before it was believed.

## 6. Limits of this audit

- **BaoStock is not ground truth.** It is a free third-party service with no stated link to the index provider.
  Where the two sources differ, this audit cannot say which one is wrong. The index provider's own change
  notices and effective dates were not used.
- **BaoStock's lists are weekly at best.** Its change days are known only to the week (section 2). If its
  Monday list is right for that Monday, BaoStock shows a change up to 4 trading days after it took effect, so
  a lag here can understate the archive's true delay by that much, and an archive that is exactly on time can
  look a few days early.
- **BaoStock was out of date for whole stretches** (section 2). There its change days are too late, the
  month-end mismatch is its own delay, and a clearly early archive change could hide among the "between two
  BaoStock lists" cases without this audit being able to see it.
- **The two sources may share an origin.** If BaoStock's history was rebuilt from the same kind of weight
  snapshots as the archive, both can be late in the same way and agree with each other while both are wrong.
  Agreement here is weaker evidence than disagreement.
- **Only month ends were compared in step 1**, and step 2 only searched months whose two ends differ. A stock
  that BaoStock added and removed again within one month is invisible unless the same month had another change.
- **The pairing window is a choice.** 250 trading days; a change the archive shows later than that counts as never shown.
- **Nothing from 2024-03-01 on was looked at**, so changes in the last months before it may still be shown
  later by the archive ("too close to the cut-off").
- **Only CSI300 and CSI500 membership was audited.** Prices, adjustment factors and the other universes
  (CSI1000, all A-shares) are not covered; the BaoStock package has list queries for CSI300, CSI500 and SSE50 only.
