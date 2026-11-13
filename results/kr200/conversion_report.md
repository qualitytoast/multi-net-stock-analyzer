# South Korea conversion report (docs/RULES.md, Rules ND, KG and KU)

- Source: FinanceData/marcap at commit `d3b703628e5a73112111d8bc43439f950762f67e` (manifest sha256 `12b24b74dd15cddadeb713606c9da6dba18a380f7f9deab225f6fb0f905ec802`).
- Qlib files written with Qlib's own `scripts/dump_bin.py` at commit `b87a2c294d364a33fb739359886acffe8ec907d1` (sha256 `b8f34c57ce1ef4b1772f3909735e66058f21b25bd7ab8a5f16318822401fe53f`); sha256 over the whole output tree: `c63038a4596a5f2d989757103fbf58f82ec3e409683f49086a576ec27268d953`.
- Calendar: 1995-05-02 .. 2026-11-11 (7942 trading days). The conversion is mechanical and covers all dates.
- **Every statistic below uses only dates before 2024-03-01** (last date checked: 2024-02-29), except where rows dated from 2024-03-01 on are consulted: (1) one conversion integrity count marked "(all dates)", the number of converted rows without a usable base price, a check that has to cover every converted row; (2) whether a stock's data "ends early" and whether a member "stopped for good" are judged from the stock's last row over all source dates (checked for this build: no member has a gap that spans 2024-03-01, so both counts are the same as with rows before that date only); (3) the factor table covers the stocks picked on any selection date, 10 of them picked only on selection dates on or after 2024-03-01, using only their rows before 2024-03-01.
- Universe `kr200`; index series `KRTOP200SYN` (synthetic, 1000 on 1995-11-30); stock symbols are `KR` + the 6-character code.

## Universe (Rule KU)

- Selection dates before 2024-03-01: 57, from 1995-11-30 to 2023-11-30; picks per selection date: min 200, max 200.
- Stocks ever picked on those dates: **648**; of these, 183 have data that ends before 2024-02-29 (the survivorship check: stocks that were delisted or merged away are kept). The conversion covers all dates and holds 668 stocks, the picks of later selection dates included.
- Members that stopped for good while a member (last row inside a membership span), before 2024-03-01: 76.

Per year: members per day, the share of member-days that are priced (have a close that day), and the members
whose last row falls in that year while they are members.

| Year | Days with members | Members min | Members median | Members max | Member-days | Priced share | Members that stopped |
|---|---|---|---|---|---|---|---|
| 1995 | 22 | 200 | 200 | 200 | 4,400 | 98.73% | 0 |
| 1996 | 293 | 199 | 200 | 200 | 58,510 | 98.81% | 1 |
| 1997 | 292 | 199 | 200 | 200 | 58,395 | 98.85% | 1 |
| 1998 | 292 | 200 | 200 | 200 | 58,400 | 97.17% | 0 |
| 1999 | 249 | 193 | 197 | 200 | 48,963 | 98.58% | 13 |
| 2000 | 241 | 199 | 200 | 200 | 48,119 | 98.63% | 2 |
| 2001 | 246 | 195 | 200 | 200 | 48,966 | 97.20% | 10 |
| 2002 | 244 | 196 | 200 | 200 | 48,560 | 98.09% | 4 |
| 2003 | 247 | 197 | 199 | 200 | 49,121 | 98.77% | 3 |
| 2004 | 249 | 197 | 199 | 200 | 49,435 | 98.91% | 4 |
| 2005 | 249 | 198 | 199 | 200 | 49,543 | 99.31% | 4 |
| 2006 | 247 | 199 | 199 | 200 | 49,271 | 99.72% | 1 |
| 2007 | 246 | 198 | 200 | 200 | 49,117 | 99.61% | 4 |
| 2008 | 248 | 199 | 200 | 200 | 49,564 | 99.59% | 1 |
| 2009 | 253 | 198 | 200 | 200 | 50,389 | 99.64% | 2 |
| 2010 | 251 | 197 | 200 | 200 | 49,982 | 99.64% | 3 |
| 2011 | 248 | 198 | 199 | 200 | 49,338 | 99.69% | 4 |
| 2012 | 248 | 200 | 200 | 200 | 49,600 | 99.87% | 0 |
| 2013 | 247 | 197 | 200 | 200 | 49,191 | 99.56% | 3 |
| 2014 | 245 | 198 | 200 | 200 | 48,899 | 99.63% | 2 |
| 2015 | 248 | 196 | 199 | 200 | 49,181 | 99.75% | 5 |
| 2016 | 246 | 199 | 200 | 200 | 49,178 | 99.60% | 1 |
| 2017 | 243 | 198 | 199 | 200 | 48,335 | 99.01% | 3 |
| 2018 | 244 | 199 | 200 | 200 | 48,750 | 99.68% | 1 |
| 2019 | 246 | 199 | 200 | 200 | 49,125 | 99.91% | 1 |
| 2020 | 248 | 199 | 200 | 200 | 49,528 | 99.73% | 1 |
| 2021 | 248 | 200 | 200 | 200 | 49,600 | 99.65% | 0 |
| 2022 | 246 | 200 | 200 | 200 | 49,200 | 99.97% | 0 |
| 2023 | 245 | 198 | 200 | 200 | 48,908 | 99.80% | 2 |
| 2024 | 41 | 200 | 200 | 200 | 8,200 | 100.00% | 0 |

## Factor (Rule KG2)

Stocks ever picked (658 with rows before 2024-03-01), all their rows on any board. A step is a row whose factor differs from the previous row's, because the exchange's base price differs from the previous close: above 1 for a split, bonus issue or rights issue (base price below the last close), below 1 for a reverse split or capital reduction. A re-listing after a spin-off, merger or long halt can step either way.

| Year | Rows | Steps | Steps > 1 | Steps < 1 | Largest | Smallest | Beyond 100x | Rows without a base price |
|---|---|---|---|---|---|---|---|---|
| 1995 | 72,275 | 112 | 111 | 1 | 1.496 | 0.9467 | 0 | 0 |
| 1996 | 116,107 | 480 | 463 | 17 | 4.864 | 0.931 | 0 | 0 |
| 1997 | 123,140 | 473 | 467 | 6 | 6.929 | 0.8813 | 0 | 0 |
| 1998 | 123,209 | 400 | 382 | 18 | 50 | 0.07405 | 0 | 771 |
| 1999 | 100,687 | 353 | 321 | 32 | 10 | 0.05 | 0 | 31 |
| 2000 | 97,061 | 127 | 114 | 13 | 10 | 0.1 | 0 | 2 |
| 2001 | 99,216 | 86 | 62 | 24 | 72 | 0.03333 | 0 | 0 |
| 2002 | 98,437 | 72 | 49 | 23 | 10 | 0.03333 | 0 | 0 |
| 2003 | 99,740 | 72 | 44 | 28 | 10 | 0.02452 | 0 | 0 |
| 2004 | 99,569 | 55 | 40 | 15 | 10 | 0.02618 | 0 | 0 |
| 2005 | 99,376 | 38 | 29 | 9 | 10 | 0.06176 | 0 | 0 |
| 2006 | 100,394 | 60 | 50 | 10 | 10 | 0.2 | 0 | 0 |
| 2007 | 100,044 | 67 | 58 | 9 | 10 | 0.1 | 0 | 0 |
| 2008 | 102,243 | 64 | 54 | 10 | 10 | 0.1 | 0 | 0 |
| 2009 | 106,488 | 62 | 54 | 8 | 10 | 0.03741 | 0 | 0 |
| 2010 | 107,033 | 58 | 50 | 8 | 24.93 | 0.03793 | 0 | 0 |
| 2011 | 107,670 | 54 | 47 | 7 | 10 | 0.1 | 0 | 0 |
| 2012 | 108,229 | 44 | 35 | 9 | 5 | 0.01389 | 0 | 0 |
| 2013 | 107,108 | 51 | 35 | 16 | 10 | 0.025 | 0 | 0 |
| 2014 | 106,912 | 59 | 45 | 14 | 10 | 0.05617 | 0 | 0 |
| 2015 | 108,633 | 53 | 41 | 12 | 10 | 0.06662 | 0 | 0 |
| 2016 | 107,960 | 45 | 32 | 13 | 10 | 0.04 | 0 | 0 |
| 2017 | 107,925 | 53 | 45 | 8 | 20.36 | 0.05 | 0 | 0 |
| 2018 | 110,460 | 54 | 50 | 4 | 50 | 0.129 | 0 | 0 |
| 2019 | 112,250 | 37 | 32 | 5 | 10 | 0.0102 | 0 | 0 |
| 2020 | 113,945 | 46 | 39 | 7 | 10 | 0.02107 | 0 | 0 |
| 2021 | 116,116 | 53 | 45 | 8 | 10 | 0.1997 | 0 | 0 |
| 2022 | 116,283 | 31 | 29 | 2 | 5 | 0.3162 | 0 | 0 |
| 2023 | 115,498 | 46 | 36 | 10 | 10 | 0.06667 | 0 | 0 |
| 2024 | 19,437 | 6 | 5 | 1 | 2 | 0.3332 | 0 | 0 |

- Steps beyond 100x in either direction (listed as implausible): **0**.
- Rows without a base price (factor carried over; docs/RULES.md, Rule KU details): 804 before 2024-03-01, from 1998-01-09 to 2000-01-25; on or after 2007-01-01 (all dates): **0**.
- Traded rows with an open, high or low missing in the source (close and volume kept): 178.
- Steps on a member's first traded day after a halt whose base price equals its opening price (the exchange set the base price at the opening auction; see the known gaps): 118, of which 82 from 2007-01-01 on. Plain splits whose first trade is at the base price are counted too.

The 10 largest and 10 smallest steps:

| Stock | Name | Date | Step |
|---|---|---|---|
| KR000200 | 대우중공업 | 2001-02-02 | 72 |
| KR005930 | 삼성전자 | 2018-05-04 | 50 |
| KR025560 | 미래산업 | 1998-03-02 | 50 |
| KR030000 | 제일기획 | 2010-05-10 | 24.9286 |
| KR001800 | 오리온홀딩스 | 2017-07-07 | 20.3571 |
| KR011300 | 성안 | 2010-05-14 | 10.012 |
| KR000050 | 경방 | 2016-10-28 | 10 |
| KR000060 | 메리츠화재 | 2006-02-23 | 10 |
| KR000180 | 성창기업지주 | 2016-05-11 | 10 |
| KR000240 | 한국타이어 | 1998-08-03 | 10 |
| KR013890 | 지누스 | 2019-10-30 | 0.0102041 |
| KR002410 | 범양건영 | 2012-12-05 | 0.0138889 |
| KR025560 | 미래산업 | 2020-07-02 | 0.0210667 |
| KR009270 | 신원 | 2003-05-19 | 0.0245245 |
| KR001260 | 남광토건 | 2013-02-15 | 0.025 |
| KR032710 | LG카드 | 2004-05-19 | 0.0261845 |
| KR003560 | 라보라 | 2001-12-21 | 0.0333333 |
| KR005870 | 휴니드 | 2002-04-25 | 0.0333333 |
| KR017300 | KDS | 2003-03-07 | 0.0337209 |
| KR000080 | 진로 | 2009-10-19 | 0.0374065 |

## Known gaps

- No cash dividends: the factor covers share changes only, so returns are price returns.
- No delisting returns: a delisted stock's history simply ends on its last row.
- Base prices set by an opening auction: when a stock resumes after a capital reduction, a spin-off or merger, or a
  long halt, the exchange sets that day's base price at the opening price. The move from the last close to that
  opening price is then part of the factor step, so it is missing from returns (example: KR042660 re-opened on
  2017-10-30 at half its pre-halt price, a step of 2).
- A code that is delisted and later re-listed under the same number stays a member through its dead period: a
  span ends at the code's last row anywhere in the source, not at its last row before the gap. One case, KR000030
  (24 member-days without a source row, 2002-04-29 to 2002-05-31), before every configured load window. To be
  fixed when the dataset is next rebuilt (end the span at the last row before the gap).
- The index series is synthetic (the members' market-cap-weighted return), not the official KOSPI 200.
