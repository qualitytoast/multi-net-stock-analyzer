# Free stock data: what exists, what we use, and what to avoid

A survey of free data the model can be tested on beyond CSI300, and the status of each source in this repo.
The survey was done on 2026-11-08 (five web searches: China and Hong Kong, the US, other Asia-Pacific, Europe and
the Americas, ready-made datasets). Facts marked "checked locally" were verified on the pinned archive on disk;
everything else is as reported by its source on that date and may have changed since. The rules for every dataset
built from these sources are in [RULES.md](RULES.md), "More data: new universes and markets".

## Terms

- **Survivorship bias:** a dataset holding only today's stocks leaves out every company that failed or was bought,
  so backtests look better than reality.
- **Point-in-time membership:** which stocks were in the index on each past date. Using today's list for the past
  is the same bias in disguise.
- **Adjustment factor:** a multiplier that removes the price jumps caused by splits and dividends, so returns line
  up over time.
- **Delisting return:** the last price move when a stock stops trading, often a big loss. Free data usually lacks it.
- **Synthetic universe:** our own fixed rule for picking stocks on each date (for example "the 200 largest by
  market cap, re-picked every 6 months, using only past data"). Used when no free official membership history exists.
- **Pinned snapshot:** a downloaded file whose hash is recorded, so a result can be reproduced. Live APIs can
  silently rewrite history.
- **Member-day:** one stock on one trading day on which it is in the universe. A **priced** member-day has a close.

## What a dataset must provide

- Per stock and day: open, high, low, close, volume and an adjustment factor (`RAW_FIELDS` in
  `src/detmaso/data/qlib_io.py`); no price on a day without trading.
- Point-in-time membership spans, plus an index series (used in plots only).
- Each stock's full price history, also after it leaves the index (labels need it).
- For walk-forward: history from 2008 (loaded from 2007 for warm-up), test years 2013-2023 ending 2024-02-28, and
  a locked test period from 2024-03-01.

## Status in this repo

| Dataset | Config | Source | State |
|---|---|---|---|
| CSI300 | `configs/csi300.yaml` | pinned China archive | in use since the MVP |
| CSI500 | `configs/csi500.yaml`, `configs/walkforward/csi500/` | pinned China archive | prepared 2026-11-12 (validation-period and walk-forward data); [QA](../results/csi500/data_report.md) |
| CSI1000 | `configs/csi1000.yaml`, `configs/walkforward/csi1000/` | pinned China archive | prepared 2026-11-12; history from 2016, 6 walk-forward folds (2018-2023); [QA](../results/csi1000/data_report.md) |
| Whole A-share market | `configs/ashare.yaml`, `configs/walkforward/ashare/` | pinned China archive | validation-period data prepared 2026-11-12 ([QA](../results/ashare/data_report.md)); the walk-forward data does not fit in 24 GB of memory with the current pipeline, so it is not prepared (Rule CU) |
| South Korea, top 200 | `configs/kr200.yaml`, `configs/walkforward/kr200/` | FinanceData/marcap at a pinned commit, converted by `detmaso.data.ingest.kr` | built and prepared 2026-11-12 (validation-period and walk-forward data); go/no-go passed; [QA](../results/kr200/data_report.md) |
| S&P 500 | none | fja05680/sp500 + Tiingo | free coverage check **failed** 2026-11-12 (91.06% against a 95.0% bar); no data built |

No model has been trained or evaluated on any of the new datasets. Each has its own try ledger
(`results/<market>/trials.csv`) and test lock (`locks/<market>_test.json`), created on first use.

## 1. China: more universes in the pinned archive (checked locally)

The pinned archive (chenditc/investment_data, tag 2026-11-04, https://github.com/chenditc/investment_data) holds
more than CSI300. Daily member counts from its membership files:

| Universe | Members per day | Usable from | Index series | Notes |
|---|---|---|---|---|
| csi300 | 300 | 2005-04 | SH000300 | the paper's universe |
| csi500 | 500 (501 on some days since 2023-12-29) | 2007-01-31 | SH000905 | fits all 11 walk-forward years |
| csi1000 | 1,000 (1,001-1,002 on some days) | 2015-05-29 (only 3 members before) | SH000852 | walk-forward test years 2018-2023 at most |
| csi800 | 800 | 2012-01-31 | SH000906 | before 2012 the file flips between 200 and 800 members and the 200-member lists are wrong; rebuild as csi300 + csi500 if ever needed. Not used |
| all | 1,363 (2005) to 5,577 (2026) | 2000 | SH000985 (CSI All Share) | delisted stocks are kept |

Fields: open, high, low, close, volume, factor (used), plus amount, vwap, change and adjclose (unused; Alpha158
needs vwap).

Caveats:

- **Index changes appear late.** The survey expected each change about 2-4 weeks late (membership rebuilt from
  monthly index-weight snapshots). The membership audit below measured it: 9 to 14 trading days late for the
  twice-yearly reviews since 2013-12, and no clear case of a change shown early. This also applies to the CSI300
  data used so far.
- **Some members have no prices.** For example, the archive's GitHub issue #15 reports a CSI500 member with no
  prices from 2012 to 2019. Since 2019 the data comes from a single upstream source (Tushare). Each QA report
  shows the priced share of member-days per year.
- **Releases can be replaced after publishing.** The maintainer swapped out the 2026-11-08 archive. The hash of the
  2026-11-04 archive used here is unchanged; the local copy is kept.
- **No ST flags.** ST marks a stock under a risk warning (5% daily price limit). The archive does not carry it, so
  the whole-market universe cannot filter ST stocks yet.
- **Newer indices are unusable for folds from 2008.** CSI A500 (launched 2024) and CSI 2000 (launched 2023) have
  only back-calculated history before launch.
- **Free cross-check: BaoStock** (http://baostock.com, no account). CSI300 and CSI500 membership on any date from
  about 2006, delisted stocks, daily trading-status and ST flags.

### Measured in this build (2026-11-12, dates before 2024-03-01 only; EXPERIMENTS E20)

| Dataset | Dates checked | Members per day | Priced share of member-days | Stocks; data ends early |
|---|---|---|---|---|
| CSI500 | 2012-01-04 to 2024-02-29 | 500 (501 on 38 days) | 96.76% | 1,420; 67 |
| CSI500, walk-forward history | 2008-01-02 to 2024-02-29 | 500 (501 on 38 days) | 96.87% | 1,648; 87 |
| CSI1000 | 2016-01-04 to 2024-02-29 | 1,000 (1,001 on 242 days) | 97.35% | 2,396; 90 |
| Whole A-share market | 2012-01-04 to 2024-02-29 | 2,161 to 5,259 (median 3,308) | 96.45% | 5,427; 172 |

- "Data ends early" = the stock's last close falls before the last date checked. A survivors-only dataset would
  show 0, so these counts confirm that delisted stocks are kept.
- The priced share is lowest in 2015 and 2016 (the 2015 mass trading halts): 88.4% for CSI500 in 2015, 91.7% for
  CSI1000 in 2016, 85.3% for the whole market in 2015. It is 99.4% or higher from 2019.
- The 501st and 1,001st member is one phantom code, SZ302132: on the CSI1000 list from 2022-12-30 and the CSI500
  list from 2023-12-29, with no price before the test period (it looks like a renumbered code listed without its
  history). It never enters the model's universe, which needs a trade the day before.
- The reported "CSI500 member with no prices from 2012 to 2019" does not show up as a member-day gap in this
  archive. The longest unpriced member spans are 289 trading days (CSI500: SH600780, 2012-01 to 2013-03), 433
  (CSI1000: SZ000029, 2016-09 to 2018-06) and 1,110 (whole market: SH600145, 2015-12 to 2020-06); these are long
  trading halts.
- The archive's `all` list also contains 6 index series (SH000300, SH000852, SH000905, SH000906, SH000985,
  SZ399300). They are not stocks, so `configs/ashare.yaml` excludes them (`data.exclude_codes`).
- Whole-market filters: new listings enter from their 121st traded day (`data.min_traded_days: 120`), which removes
  3.89% of member-days. Known gaps: no ST filter; the filter counts traded days in the loaded data, which starts one
  year before the first model date, so the market state on the first model dates sees about 1,700 to 2,200 members
  a day instead of the full list.

### Membership timing audit against BaoStock (2026-11-12; Rule MA; EXPERIMENTS E21)

Full report: [results/csi300/membership_audit.md](../results/csi300/membership_audit.md). It compares the archive's
CSI300 and CSI500 member lists with BaoStock's from 2007-01-31 to 2024-02-29. It is a report only: it changes no
data and no decision depends on it.

- **No look-ahead found.** In 0 of 4,950 paired index changes (1,522 CSI300, 3,428 CSI500) does the archive show a
  change on or before the date of a BaoStock list that still lacked it. "Early" would mean the model knows a stock
  joins the index before it happened.
- **The archive is late.** For the 17 twice-yearly reviews from 2013-12 to 2023-12 it shows the change 9 to 14
  trading days after BaoStock (median 13 for CSI300, 12 for CSI500): on the last trading day of June and December
  instead of the day after the second Friday. A single stock that moves between reviews waits for the archive's
  next change day (longest 102 trading days for CSI300, 114 for CSI500).
- **The archive's list changes only about twice a year,** on 41 days (CSI300) and 40 days (CSI500) in 17 years, not
  monthly.
- **Out-of-date share, 2012-01 on:** stocks BaoStock had already dropped but the archive still listed make up 0.55%
  of CSI300 member-days and 1.00% of CSI500 member-days (a lower bound).
- **Month-end mismatch** (stocks in exactly one list / stocks in either): 0.42% for CSI300 and 0.56% for CSI500 over
  all 206 month ends; 0.11% and 0.08% on the 187 month ends where BaoStock's own list is fresh.
- **Limits.** BaoStock is a third-party source, not ground truth. It updates once a week at best and skipped
  updates between 2021-06 and 2023-03 (at 19 of 206 month ends its list was 10 to 211 days old). Because of that,
  1,202 changes sit "between two BaoStock lists" with the archive ahead: 767 by at most 4 trading days (what an
  on-time archive looks like next to a weekly list, mostly CSI300 reviews before mid-2013) and 435 inside
  BaoStock's gaps. Those cannot be settled without the index provider's official effective dates.
- The BaoStock answers are cached outside the repo (838 files, hash list sha256 `20fcbe28...`), and
  `scripts/audit_membership.py --offline` rebuilds the report from the cache byte for byte.

## 2. Independent markets with free, survivorship-aware data (ranked)

| Rank | Market | Free source(s) | History | Delisted kept | Membership | Adjustment | Terms | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | South Korea | FinanceData/marcap on GitHub, updated daily (OHLCV, trading value, market cap, shares); KRX Open API (free key, from 2010) as a cross-check | 1995 on | yes (daily snapshots) | KOSPI 200 only from 2014-05 (needs a free KRX login); synthetic before | built from the exchange's daily base price; no cash dividends | non-commercial, no redistribution | great |
| 2 | US | fja05680/sp500 membership plus Tiingo free tier | 2005 on | Tiingo lists about 8,000 delisted tickers; whether the free tier serves their prices is unverified | fja05680, from 1996 (unofficial) | Tiingo gives raw and adjusted prices, dividends and splits | personal use only | usable; great if coverage passes |
| 3 | Taiwan | TWSE daily all-stock files (from 2004-02-11), TPEx (from 2007-07), FinMind free tier | 2004 on | yes | none free; synthetic | built from the reference-price change (to verify) | site terms unverified | usable to great |
| 4 | Brazil | B3 COTAHIST official yearly files | 1986 on | yes | none free; synthetic by liquidity | raw only; corporate actions from an undocumented B3 endpoint | manual download only (B3 bans automated collection) | usable |
| 5 | India | NSE daily bhavcopy files (the official end-of-day price file); Nifty 50 changes on Wikipedia (from 2005) | about 1994 on (unverified) | yes | Nifty 50 only | corporate actions spotty before 2010 | unverified; NSE blocks some IPs | usable |
| 6 | Poland | GPW official daily archive; WIG20 changes on Wikipedia (from 2005) | 1994 on | yes | WIG20 only (20 stocks) | raw only | unread | usable |

### South Korea: built 2026-11-12 (Rules KG and KU; EXPERIMENTS E23)

Reports: [go/no-go](../results/kr200/go_no_go.md), [conversion](../results/kr200/conversion_report.md),
[QA](../results/kr200/data_report.md), [QA, walk-forward](../results/kr200/walkforward/data_report.md).

- **Source:** FinanceData/marcap (https://github.com/FinanceData/marcap) at commit
  `d3b703628e5a73112111d8bc43439f950762f67e`: 32 yearly parquet files, 15.7 million rows, every trading day from
  1995-05-02. `scripts/get_kr_data.sh` fetches it and checks every file against `scripts/kr_data.sha256`; the
  commit and the other pins are written once, in `scripts/kr_data.pins`. The repo states no license; it says the
  data comes from the Korea Exchange's public data portal. No data is committed here.
- **Go/no-go, all three passed** (rows before 2024-03-01):
  - delisted stocks are present: 1,151 KOSPI codes end more than a year before 2024-03-01 (bar: 200), and all 3
    named delistings are there;
  - the split adjustment is right: Samsung Electronics' 50-for-1 split gives a factor step of exactly 50, and the
    3 other named splits come out at 9.997, 4.993 and 4.982 (the exchange rounds the base price to a price tick);
  - suspended days are identified: Samsung has no price on its three halt days and a price on the days around them.
- **Universe:** a synthetic top 200 (Rule KU), re-picked at the end of every May and November from the average
  market cap of the past 6 months. 648 stocks were picked at some time before 2024-03-01; 183 of them have data
  that ends early, and 76 stopped trading for good while a member. Without the name patterns, 16 listed funds
  (infrastructure funds, REITs) would have been picked on some date.
- **QA** (dates before 2024-03-01): 196 to 200 members per day; 99.69% of member-days priced from 2012 (99.67%
  from 2008); no one-day move above 31% in any year.
- **Trading settings** (`configs/kr200.yaml`, each with its source): sell-side transaction tax including the rural
  development tax, 30 bps until 2019-05-29, then 25, 23 (2021), 20 (2023), 18 (2024), 15 (2025), 20 (2026); daily
  price limits 15%, then 30% from 2015-06-15. The 2021 and 2023 totals are derived from the reported cuts, not
  quoted directly by a source.
- **Known gaps:**
  - No cash dividends, so returns are price returns.
  - No delisting returns: a delisted stock's history simply ends.
  - **Re-opening moves are missing from returns.** When a stock resumes after a capital reduction, a spin-off or
    merger, or a long halt, the exchange sets that day's base price at the opening auction price, so the move from
    the last close to that opening price lands in the factor instead of in the return. Example: Daewoo
    Shipbuilding re-opened on 2017-10-30 at half its pre-halt price; the true one-day move was -56.7%, the adjusted
    data shows -13.4%. There are 82 factor steps of this kind on member-days from 2007 on (1 to 10 a year; the
    count also includes plain splits). Fixing it needs corporate-action ratios from another source.
  - On 804 rows of picked stocks between 1998-01-09 and 2000-01-25 the source has no usable base price (mostly
    failed banks under a halt); the factor is carried over there. None is later than 2000, so no config (all load
    from 2007 or later) contains one.
  - The index series is synthetic (the members' market-cap-weighted return), not the official KOSPI 200.

### US details

| Item | Details |
|---|---|
| Membership | fja05680/sp500 on GitHub (https://github.com/fja05680/sp500): MIT license, from 1996. Unofficial; its author cannot verify it. About 900 tickers were members at some point since 2008. |
| Prices | Tiingo free tier (https://www.tiingo.com): raw and adjusted OHLCV plus dividends and splits, 30+ years. Limits: 500 new tickers per month, 1,000 requests per day, so a full fetch takes about 2 months. Personal use. Its public ticker list can be checked without a key. |
| Fallbacks | yfinance for current names (Yahoo alone misses about 24% of 2010-2024 S&P 500 members). Alpha Vantage's free LISTING_STATUS gives delisting dates. |
| Complete option | CRSP via WRDS: permanent IDs, adjustment factors and delisting returns. Needs a university subscription; the owner has none (2026-11-12). |
| Smoke test only | Qlib's prebuilt US data still downloads (450 MB, 2000 to 2020-11-10), but its S&P 500 membership has errors and about 11% of members have no prices. |
| Paper link | US data would also unblock the deferred M10: the paper reports S&P 500 results (IC 0.072, Sharpe 1.58). |

### US: the free coverage check (2026-11-12): failed

The check ([results/sp500/coverage_check.md](../results/sp500/coverage_check.md); Rule UC; EXPERIMENTS E22) matched
the fja05680 member list against Tiingo's public list of supported tickers, on ticker plus date range, for every
weekday from 2008-01-01 to 2024-02-29. It fetched no prices. The pass bar, written beforehand, was 95.0% of
member-days.

- **Result: 91.06% matched (1,926,082 of 2,115,048 member-days). Failed.** Even the loosest reading (any Tiingo row
  under the ticker, dates ignored) gives 94.71%, still under the bar.
- **The gap is survivorship.** Tickers that left the index before 2024-03-01 are matched on 70.37% of their
  member-days; tickers still in the index on 2024-02-29 on 98.62%. By year the share climbs from 80.84% (2008) to
  98.27% (2023); only 2020 onward clears 95% in a single year.
- **Why:** Tiingo's list carries a company under one ticker only (its current or last one), while the member list
  uses the ticker of the day, so renamed companies are missing under their old ticker (5.28% of member-days);
  reused tickers account for 2.30% and partly covered date ranges for 1.34%. Tiingo's list is also thin for stocks
  that stopped trading before 2016.
- **Consequence:** the free S&P 500 route is not good enough as it stands. No Tiingo fetch was started. The owner
  has no WRDS access (2026-11-12), so CRSP is out. Remaining routes: a rename map (old ticker to new) followed by
  a new check under a new rule written beforehand, or a paid source (section 5).
- Pins: fja05680/sp500 commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`; Tiingo's list downloaded
  2026-11-12T22:24:27Z, sha256 `3d33d870111072d8a87de8a3f73cb3b30f00ec928fa517846ae29b614822f800`. They are
  written once, in `scripts/us_coverage_data.pins`; `scripts/get_us_coverage_data.sh` fetches and verifies both.

## 3. Different kinds of test beds (documented only, not built)

- **Numerai v6 "Unity"** (released 2026-10-16; https://numer.ai). Free, real stock identifiers, 1,040 ready-made
  features, 20-day and 60-day targets, scored by per-date correlation (like our IC), benchmark predictions on the
  same rows, training data to 2018-12-28 and validation from 2019. It has no raw prices, so our 34 features and
  12 market-mood inputs would have to be rebuilt from its features: it would test the 5-agent design, not our
  features. Its terms of use could not be read during the survey.
- **Binance public data** (https://data.binance.vision). Free daily price bars for every trading pair, delisted
  pairs kept, license CC BY-NC-SA 4.0 (allows academic research). Crypto trades 7 days a week and coins move
  together, so it is a stress test, not a stock market. No published baselines.

## 4. Reference numbers (sanity checks, not test data)

- **Qlib benchmark tables** (official data, 1-day label, test 2017-01 to 2020-08; compare with their "Rank IC"
  column, which matches our IC): CSI500 with Alpha158: LightGBM 0.048, MLP 0.036. CSI300 with Alpha158: MLP 0.043.
- **MASTER** (AAAI 2024; 5-day label; test 2020Q3-2022Q4; built from the same chenditc data): Rank IC 0.076 on
  CSI300 and 0.066 on CSI800.
- **The same model gives very different IC under different setups.** HIST reports IC 0.131 in its own paper but
  0.052 in Qlib's rerun. None of these uses our 20-day label, so treat them as order-of-magnitude checks, and only
  outside the locked test period.
- **Factor check:** JKP Global Factor Data (https://jkpfactors.com; free daily factor returns for 93 countries,
  CC BY-NC) and the Kenneth French data library. Regressing the model's daily long-short returns on momentum and
  reversal factors shows whether it only rediscovers known effects.

## 5. Not worth it (free versions), and why

| Source | Why not |
|---|---|
| Kaggle competition data (Ubiquant, JPX, Jane Street, Two Sigma, Hull) | rules allow competition use only; some say delete afterwards; Ubiquant is no longer downloadable |
| Japan (J-Quants free tier, Kaggle JPX) | free tier is 2 years, 12 weeks delayed; terms appear to bar academic research; JPX Kaggle rules as above |
| Hong Kong, Canada, Australia, most of Europe, Mexico | no free source with delisted stocks and membership history |
| Free API tiers (Alpha Vantage, EODHD, Marketstack, Massive (formerly Polygon), FMP, Twelve Data, Finnhub, Databento credits) | 20-250 calls per day, or 1-2 years of history, or only recent years |
| Stooq | personal use is free, but bulk files now need a manual captcha, and whether delisted stocks are kept is undocumented |
| Nasdaq Data Link WIKI prices | removed |
| Kaggle US price datasets, FNSPID | survivors only, or no membership dates; stale |
| StockNet, KDD17, Feng NASDAQ/NYSE, CMIN | tiny, short, or filtered with hindsight (survivors only) |
| Not free, cheapest complete fallbacks | EODHD All-World $19.99/month (50% student discount); Norgate trial (only 2 years) |

## 6. Traps that apply to every source

1. **Survivorship.** Most free data holds only today's stocks. Every dataset here reports, per year, the share of
   member-days that have a price and the number of stocks whose data ends early.
2. **Late membership.** The China archive shows index changes late. Measured by the membership audit.
3. **US tickers get reused and renamed,** and class shares such as BRK.B break matching. Match on ticker plus date
   range.
4. **No delisting returns.** In free data, a stock that delists inside the 20-day label window gets no label and
   silently drops out.
5. **Self-built adjustments can be wrong.** Check them against known splits (Rule KG2 does this for Korea).
6. **Releases can be replaced.** Keep local archives, and pin each by sha256 or git commit.
7. **Licenses are personal or non-commercial, and this repo is public.** Commit scripts and hashes, never data.
8. **Lock each new test period before any run.** Every dataset's config locks 2024-03-01 to 2026-03-01, so
   walk-forward stops at 2024-02-28 everywhere, and data-check statistics use only earlier dates (Rule ND).
9. **Trading rules differ per market.** Costs and price limits are per-market settings in each config
   (`trading.stamp_duty_bps`, `trading.price_limits`).
