# Decision rules, fixed before testing

Every change to the model or its training is judged by rules written here **before its first run**, dated,
and applied by code rather than by eye (CLAUDE.md, "Guards"). A rule is never changed after seeing
results; if one turns out to be wrong, the owner is asked first, and the change and its reason are logged
in "Rule changes" at the end of this file.

Words used below: a **mood** is one of the 5 K-means market regimes (agent i is tied to mood i); an
agent's **usage** is its average scheduler weight over a set of days; the **95% range** is the 2.5-97.5%
percentile range of the paired block resampling in `detmaso compare` (validation days redrawn in 20-day
blocks shared by every model, seeds redrawn per model, 10,000 draws).

## General keep rule (from CLAUDE.md, 2026-11-07)

Walk-forward IC is the main metric. A change counts as an improvement only if all of these hold:

- its average gain across all walk-forward years has a 95% range entirely above 0;
- it wins in at least 8 of 11 years. For a change aimed at specific markets (for example crash years), it
  must instead win in most of those years and be no worse (95% range not entirely below 0) in the rest. Say
  which years it targets before running;
- at least 4 of 5 seeds show a gain on average;
- if the change has a tunable setting, the next setting over also shows a gain;
- (added 2026-11-08 with the owner's OK, before any walk-forward result; see "Rule changes") its after-cost Sharpe
  gain has a 95% range that is not entirely below 0.

Until walk-forward exists, a change can only be "kept for re-check" or "dropped". The exact walk-forward
definitions are in "Walk-forward evaluation" below (Rule WK).

## Single network vs Det-MASO (fixed 2026-11-06, applied in Step 1)

- Primary: the IC difference, Det-MASO minus the single network, on validation.
  - "Det-MASO beats a single network" = the 95% range lies entirely above 0.
  - "The single network beats Det-MASO" = the range lies entirely below 0.
  - Otherwise "no detectable difference on this validation period".
- Second opinion: Sharpe, with its own 95% range; disagreements are stated.
- Max drawdown: averages and seed ranges only (resampling breaks the day order it depends on).

## Scheduler balance (fixed 2026-11-07, before any run of the change)

The change: a load-balancing penalty `loss.c_bal × KL(target ‖ usage)`, where usage is the scheduler's
average weight per agent over all training days and the target is each mood's share of training days
(`loss.balance_target: regime`). Plan: `give-me-a-plan-abundant-beacon.md` (approved 2026-11-07), decisions
D1 a, D2 b, D3 a, D4 a, D5 a, D6 a, D7 a, D8 a.

Target shares on the training days (2,653 days, 2012-01-04 to 2022-12-02): Low-Vol Bull 52.6%,
High-Vol Bull 7.9%, Sideways 32.2%, Low-Vol Bear 5.1%, Tail-Risk 2.2%.

### Rule G: where the strength grid goes (training-only pilot)

- Pilot runs:
  - strengths 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 and 1;
  - seeds 0 and 1 (seed 1 is the worst collapse);
  - 8 epochs each, with validation switched off.
- The pilot's anchor g₀: the smallest pilot strength at which, in both seeds, every agent's usage on the
  epoch-8 model is at least half its target (the M1 part of Rule M).
- Grid: 6 half-decade steps: g₀/√10, g₀, √10·g₀, 10·g₀, √10·10·g₀, 100·g₀ (rounded to 1 significant digit).
  One step sits below the pilot's anchor and four above it.
- If no pilot strength passes: stop and ask.
- On shuffled answers (Rule P) the pilot runs the same way and places the placebo grid. If it finds no
  anchor there, the placebo grid is simply the 6 pilot strengths, so the rest of the placebo process still
  runs.
- How often gradient clipping is active is logged at each strength.

### Rule M: the collapse is fixed (at one strength)

- M0: all 5 seeds finish training (no aborts).
- M1 usage: in every seed, every agent's usage on the training days (dead agent-days excluded) is at least
  half its target share.
- M2 routing (mood targets): in every seed, on each mood's training days, that mood's own agent has the
  largest average weight.
- M1 and M2 must hold both for the restored best-epoch model and for the last epoch trained.

### Rule S: the strength

- c* = the smallest grid strength that satisfies Rule M.
- If M1 fails at every strength: report "not fixed at any tested strength" and stop.
- If M1 passes somewhere but M2 fails everywhere: report "all agents used, but no specialists" and stop.
- No grid extension and no other fix without the owner's OK.

### Rule K: keep or drop (c* vs the paper Det-MASO `final`, on validation)

Uses the paired block resampling (20-day blocks, 10,000 draws, RNG seed 0), with each model's random
stream independent of row order. IC is primary.

- "Better" only if all three hold:
  1. the 95% range of the IC difference (c* minus paper) lies entirely above 0;
  2. in at least 4 of 5 seeds, c*'s IC is above the paper's IC for the same seed;
  3. nearby setting: the next grid strength above c* also satisfies Rule M and also has mean IC above the
     paper's. If c* is the top of the grid, "better" is not possible.
- "Worse": the 95% range lies entirely below 0.
- Otherwise "no detectable difference".
- Sharpe is the second opinion, with the same resampling; disagreements with IC are stated.

Outcome:
- If Rule M holds and the verdict is not "worse": kept for re-check (the balanced Det-MASO becomes the
  Det-MASO that the walk-forward step re-checks).
- Otherwise: dropped, and listed in the README's "What didn't work".
- It counts as an improvement only if it passes the general keep rule at the walk-forward re-check, in its
  targeted form. The validation verdict is descriptive and not required.
- Reported, but not part of the rule: c* vs the single network, with the same statistics.

Targeted years, declared now: the years in which rare moods (High-Vol Bull, Low-Vol Bear, Tail-Risk) make
up at least 10% of trading days: 2015 (all days), 2016 (27%), 2019 (16%) and 2020 (13%). At the re-check
the change must win in at least 3 of those 4 years and be no worse (95% range not entirely below 0) in the
other years; the conditions on the average gain, the seeds and the next setting apply too.

### Rule P: placebo (runs before any real run of the change; must pass before any real result is produced)

Shuffled answers:
- Within each block of 20 trading days, stocks are paired at random; each stock gets its partner's 20-day
  returns and next-day returns for that block.
- Only stocks with a 20-day return on every day of the block take part; the others are left out of
  training, IC and the portfolio for that block.
- Placebo results are reported against their own uncertainty (shuffled answers are 2-4 times calmer than
  real ones).

What runs: the whole procedure on the shuffled answers, unchanged: the pilot and Rule G, then the sweep,
the paper config and the single network, then Rules M, S and K.

Pass if both:
- P1: the average placebo IC over all 8 placebo run groups has a 95% range that is not entirely above 0.
- P2: if a placebo strength is chosen, its IC range is not entirely above 0, and Rule K's verdict is not
  "better".

If either check fails: stop, report, propose a fix, and ask before continuing.

### Outcome (2026-11-07)

Rule P passed; Rule G set the grid below; Rule S stopped with "all agents used, but no specialists". The owner
kept the rules as written (no rule change), so the change was dropped (EXPERIMENTS E12-E17).

## Walk-forward evaluation (fixed 2026-11-08, before any walk-forward run)

Plan: `make-a-plan-for-melodic-ocean.md` (approved 2026-11-08), decisions D1-D8 all "a". These rules fix how
every later change is judged. They are implemented in code (`detmaso.walkforward`, `detmaso.eval.trading`,
`detmaso.eval.wf_compare`, `detmaso.eval.guard.keep_rule` and `rule_wp`).

### Rule WL: the fold layout

- History starts 2008-01-01 (data loaded from 2007-01-01, so the 120-day features and market state have their
  warm-up). Configs: `configs/walkforward/`, run folder `runs/csi300_wf` (the paper config's data is untouched).
- 11 folds, named by their test year Y = 2013 ... 2023:
  - training: every model date from the history start to the end of year Y-2;
  - early stopping: calendar year Y-1;
  - test window: calendar year Y; the last fold's runs from 2023-01-01 to 2024-02-28.
- Purges (20 trading days, the answer horizon): the last 20 training dates are not trained on; IC in the
  early-stopping year and in the test window uses only dates whose 20-day answers end inside that window. The
  portfolio uses every test date, so the 11 test windows join into one continuous daily series.
- Every fold trains from scratch with the paper recipe (one step per date, cosine learning rate, early stopping
  on the 3-epoch smoothed early-stopping-year IC with patience 8, best epoch restored), seeds 0-4. "Seed s" means
  the 11 fold models trained with seed s. No second training after early stopping.
- Moods per fold: the K-means moods and the scheduler's input scaling are refit on every model date before the
  fold's test window and named by the fixed rule A16. Each fold's report shows how far the moods moved.
- The locked test period is never used: code refuses any fold whose test window, answers or next-day returns
  reach 2024-03-01.

### Rule WT: trading, costs and turnover

- Portfolio: the paper's top 10% of the eligible stocks by score, equal weights, rebalanced at every close.
- Turnover: each day's buys and sells as shares of the portfolio, measured against the previous holdings after
  they drifted with the day's returns. One-way turnover = the share of the portfolio replaced.
- Cost of a day = buys x buy rate + sells x sell rate, paid at the rebalance: net return = (1 - cost) x
  (1 + gross return) - 1.
- Flat cost levels: 0, 10 and 20 bps on every buy and every sell.
- Realistic costs: buys 8 bps (3 bps commission and exchange fees + 5 bps slippage); sells 18 bps (the same 8 +
  10 bps stamp duty), 13 bps from 2023-08-28 (stamp duty halved to 5 bps).
- Blocked trades (used by every "realistic" number): at the close of day t,
  - a stock that did not trade on t, or closed at its lower price limit, cannot be sold or reduced (it stays held);
  - a stock that closed at its upper price limit cannot be bought or increased;
  - price limits: 20% for ChiNext (codes SZ300, SZ301) from 2020-08-24 and for STAR (SH688), otherwise 10%;
    "closed at the limit" = the close-to-previous-traded-close return is within 0.5 percentage points of the limit
    and the close equals the day's high (upper) or low (lower);
  - weight that cannot be placed stays in cash (return 0); the remaining target stocks share the rest equally;
  - a held stock's value follows its last traded close (a suspended stock earns its whole move when it trades again).
- Without blocking, the gross returns are exactly the paper backtest's (a held stock with no next-day price earns 0
  that day).
- Placebo runs (shuffled answers) value holdings with the shuffled next-day returns (a stock outside the day's
  universe, or outside that block's shuffle, earns 0), so no real return enters a placebo result.
- Long-short spread: top 10% minus bottom 10% of the eligible stocks, equal weights, gross, no blocking. A signal
  measure only (shorting is restricted in China), never a tradable result.
- "After-cost Sharpe" = Sharpe of the daily net return with realistic costs and blocking.

### Rule WK: the keep rule, applied by code

Statistics (`detmaso compare --split walkforward`):

- A model's yearly IC = the mean daily IC in that fold's test window; its seed-mean IC averages the seeds.
- Yearly gain = the candidate's seed-mean yearly IC minus the reference's. Average gain = the mean of the 11
  yearly gains (each fold weighs the same; the 14-month last fold counts as one year).
- 95% ranges: paired, stratified block resampling. Within each test window the days are redrawn as circular
  20-day blocks (one draw shared by every model, so the comparison is paired by day); each model's seeds are
  redrawn with replacement from its own random stream (seeded by RNG seed 0 and the model's name), once per draw
  for all folds; 10,000 draws. The range is the 2.5-97.5% percentile range of the draws. Sharpe ranges use the
  same day draws, joined across the folds.

"Improvement" only if all of these hold:

- K1: the average gain's 95% range lies entirely above 0;
- K2 (general form): the yearly gain is above 0 in at least 8 of the 11 years;
- K2 (targeted form, when target years are declared here and on the command line before the run): the yearly
  gain is above 0 in more than half of the target years, and in every other year that year's 95% range is not
  entirely below 0;
- K3: at least 4 of 5 seeds gain on average (seed s's gain = the mean over the years of its yearly IC minus the
  reference's seed s yearly IC);
- K4: if the change has a tunable setting, the next setting over (named on the command line) also has an average
  gain above 0. The command line must say either which run is the next setting or that there is no tunable setting;
- K5: the 95% range of the after-cost Sharpe difference (candidate minus reference) is not entirely below 0.

Otherwise: "worse" if the average gain's 95% range lies entirely below 0, else "no improvement". Every condition
is reported with the numbers behind it, together with each model's try number, deflated Sharpe and deflated IC.

### Rule WP: the walk-forward placebo (runs before any real walk-forward run)

The walk-forward paper config and the walk-forward single network, trained and evaluated through every fold on
shuffled answers (the existing placebo: 20-day blocks, shuffle seed 0). Pass only if both hold:

- P1: for each of the two placebo groups, the 95% range of its average IC (the mean over the 11 years of its
  seed-mean yearly IC) is not entirely above 0;
- P2: Rule WK, applied in both directions between the two groups (no tunable setting), never says "improvement".

If either check fails: stop, report, and ask the owner before any real walk-forward run.

Outcome (2026-11-08, EXPERIMENTS E19): **passed**. P1: average IC +0.0010 [-0.0031, +0.0051] (paper config) and
+0.0028 [-0.0025, +0.0081] (single network); P2: "no improvement" in both directions. Reported to the owner: the
after-cost Sharpe difference between the two placebo groups is detectable (+0.29 [+0.11, +0.48]) because their
turnover differs (37% vs 26% of the portfolio a day), not their gross returns; K5 responds to turnover.

Note (2026-11-11): re-evaluated with the placebo valuation fix logged in "Rule changes" (evaluation only, same
models). Only a few 2015 numbers moved, in the second decimal; every figure above and the verdict are unchanged.

### Deflated metrics for walk-forward results

- Number of tries: every counted try in the ledger (validation and walk-forward alike).
- Spread of the tries' results: from the walk-forward tries once at least 3 have walk-forward metrics; until
  then, from the validation tries, whose spread is larger, so the deflated numbers come out more cautious. The
  report says which was used.
- Deflated Sharpe uses the daily after-cost excess return (realistic costs with blocking, minus the equal-weighted
  eligible stocks); deflated IC uses the standard error of the average IC from the resampling above.

## More data: new universes and markets (fixed 2026-11-12, before any data check or prepare run)

Plan: `i-want-more-free-mutable-barto.md`. Owner's go-ahead 2026-11-12: record the findings, add all the extra China
data, build the Korea dataset, run the free US coverage check. Decisions: D1 c, D2 a, D3 a, D4 b (coverage check
only, no price fetch), D5 a, D6 a, D7 a, D8 a, D9 a, D10 a. This build trains no model, so it adds no tries.

Words used below: a **member-day** is one stock on one trading day on which it is in the universe; a **priced**
member-day has a close on that day; the **factor** is the adjustment factor (adjusted price = raw price x factor,
adjusted volume = raw volume / factor).

### Rule ND: every new dataset

- Locked test period: 2024-03-01 to 2026-03-01 in every new config, the same as CSI300. Walk-forward ends
  2024-02-28. No model has been evaluated on any new dataset's test period, and none is in this build.
- Every data-check statistic (QA reports, coverage, audits, go/no-go checks) uses only dates before 2024-03-01.
  The conversion itself is mechanical and runs over all dates.
- Each dataset's QA report shows, per year: members per day, the share of member-days that are priced, and the
  number of stocks whose data ends before the last date checked (a survivorship check).
- Role in decisions (D2 a, D5 a): CSI500, CSI1000, the whole A-share market and Korea are reported next to CSI300
  as extra checks only. The general keep rule and the second-market rule (the US) stay exactly as written.
- Downloads are pinned by git commit or sha256 and kept outside the repo; the repo holds scripts and hashes only.

### Rule CU: the China universes (from the pinned archive already on disk)

- CSI500 (`csi500`, index series SH000905): the same splits and the same 11 walk-forward folds as CSI300.
- CSI1000 (`csi1000`, index series SH000852): membership is usable from 2015-05-29, and the 120-day market state
  needs a full member list, so its history starts 2016-01-01. Walk-forward test years 2018-2023 (6 folds; the first
  fold trains on 2016 and early-stops on 2017).
- The whole A-share market (`all`, index series SH000985, delisted stocks kept), with these filters:
  - suspensions: the existing universe rule (a stock must have traded the day before);
  - new listings: a stock enters the universe only from its 121st traded day in the loaded data (120 = the longest
    feature window). Behind a config flag that is off for every existing config;
  - ST stocks (under a risk warning): the pinned archive carries no ST flags, so no ST filter is applied in this
    build. ST stocks stay in the universe and their 5% price limit is not modelled. This gap is reported with the
    dataset, not hidden.
  - Price limits for blocked trades: the existing China rules, plus 30% for Beijing Stock Exchange codes (BJ).
- If preparing the whole A-share market does not fit in this machine's memory with the existing pipeline, the build
  stops there and reports the numbers; the pipeline is not rewritten in this build.

### Rule KG: South Korea go/no-go (written before the data was opened)

Source: FinanceData/marcap at a pinned commit. All three checks must pass; if any fails, the Korea build stops and
the owner is told (Taiwan is the fallback, and that is the owner's call).

- KG1, delisted stocks are present. Both must hold, on rows before 2024-03-01:
  - at least 200 KOSPI stock codes have their last row more than one year before 2024-03-01;
  - at least 2 of these 3 known delistings are present, with their last traded row in the stated year: Hanjin
    Shipping (117930, 2017), Korea Exchange Bank (004940, 2013), the old Samsung C&T (000830, 2015).
- KG2, the split adjustment comes out right. The factor is chained from the exchange's own base price: base price
  of day t = close - change, and the factor is multiplied by (previous row's close / base price) on every day.
  Both must hold:
  - Samsung Electronics (005930), 50-for-1 split: the factor after the split (2018-05-04) is 50 times the factor
    before it (its last traded day in April 2018), within 1%;
  - at least 2 of these 3 other splits come out within 1% of their ratio, in the stated month: Amorepacific
    (090430) 10-for-1 in 2015-05, Naver (035420) 5-for-1 in 2018-10, Kakao (035720) 5-for-1 in 2021-04.
- KG3, suspended days can be identified. In the converted data, Samsung Electronics has no price on its split halt
  (2018-04-30, 2018-05-02, 2018-05-03) and has a price on 2018-04-27 and 2018-05-04.
- Cash dividends are not in this source, so the factor does not include them. Reported with the dataset.

### Rule KU: the Korea stock universe (D6 a: a synthetic top 200)

- Eligible: stocks listed on KOSPI that are common shares. Preferred shares (codes not ending in 0), ETFs, ETNs,
  SPACs, REITs and other listed funds are excluded. The exact code and name patterns are fixed in the converter
  from stock codes, names and markets only, never from prices or returns, and are listed under "Details" below
  before the universe is first computed.
- Re-picked twice a year. Selection date = the last trading day of May and of November. Ranking = the stock's
  average daily market cap over the 6 calendar months ending on the selection date; a stock needs a market cap on
  at least 80% of that window's trading days. The 200 largest are picked (fewer if fewer are eligible).
- The picks are members from the first trading day of June or December until the day before the next effective
  date. A member that stops trading for good leaves after its last row and is not replaced until the next re-pick,
  so some days have fewer than 200 members.
- Only data up to the selection date is used, so membership on any day depends only on earlier days.
- The index series for plots is synthetic too: the member stocks' market-cap-weighted daily return, chained.
- The official KOSPI 200 is not used (free history starts only in 2014-05 and needs a login).
- Trading settings (D9 a): Korea's sell-side transaction tax steps and its daily price limits (15%, then 30% from
  2015-06-15) are set in the Korea config, each with its source cited in the config.

Details (appended by the Korea build before the universe is first computed):

- **2026-11-12 16:42, exclusion patterns.** Fixed from stock codes, names and market labels only; no price, return
  or market-cap value was looked at to choose them, and the universe had not been computed. Source: FinanceData/marcap
  at commit `d3b703628e5a73112111d8bc43439f950762f67e`. The code that applies them is
  `src/detmaso/data/ingest/kr.py`.
  - Stock code: the source's `Code`, left-padded with zeros to 6 characters. Rows up to 2000-05-26 store codes
    without their leading zeros; the padded codes line up with the later 6-character codes (across the switch,
    1452 of 1460 codes continue with the same market label; the 8 that do not are 7 preferred shares and one stock
    that left the list).
  - KOSPI: the row's `Market` label is exactly `KOSPI` (the same rows as `MarketId` = `STK`). Rows labelled
    `KOSDAQ`, `KOSDAQ GLOBAL` or `KONEX` are not KOSPI.
  - Common share: the padded code's last character is `0`. Every other code is excluded (preferred shares end in
    5, 7, 9, K, L or another letter; before 2000-05-29 some end in other digits).
  - Name patterns (regular expressions), tested on the stock's name on the selection date with all spaces removed
    and letters in upper case. A stock whose name matches any of them is excluded:
    - SPACs: `스팩|SPAC|기업인수목적`
    - REITs and listed real-estate funds:
      `(?<!메)리츠|REIT|부동산투자|코크렙|리얼티|^교보메리츠$|^유레스메리츠|^맥쿼리센트럴|^케이알제\d+호$|^트러스제\d+호$`
      ("리츠" counts only when it does not follow "메", so the Meritz companies, 메리츠화재, 메리츠증권, 메리츠종금 and
      메리츠금융지주, stay eligible.)
    - ETFs and ETNs: `ETF|ETN|^(KODEX|TIGER|KOSEF|KINDEX|ARIRANG|KBSTAR|HANARO)` (a safety net: the source lists
      stocks only, and no name in it matches).
    - Ship funds: `^(동북아|아시아|코리아|바다로|거북선|하이골드)\d+호$|^아시아퍼시픽\d+$`
    - Infrastructure funds: `인프라$` (맥쿼리인프라, KB발해인프라; a name such as 두산인프라코어 does not end in it).
    - Resource, overseas and auction funds: `유전|니켈\d+호$|베트남|경매|^한국패러랠$|^한국아태특별$`
    - Mutual funds and restructuring funds listed in 1999-2000:
      `펀드|기금|^박현주|^플래티넘|^플러스\d+호$|^장인환\d+호$|^알바트로스$|^코스파이더$|^SEI에셋$|^트윈스비전일$|^엠피챌린지$|^마이다스성일$|^옵티멈전환일$`
      (알바트로스, 코스파이더, 엠피챌린지 and SEI에셋 are judged to be funds from their names and from their codes,
      which sit in the same 1999 run of codes as the other funds; they stopped being listed by 2001.)
    - Check on the pinned data, names only: among KOSPI codes ending in 0, the patterns match 145 of 2372 distinct
      names, and no code has both a matching and a non-matching name.
  - A stock that moves between KOSDAQ and KOSPI keeps its code. It is eligible on a selection date only if its row
    on that date is labelled KOSPI; its average market cap uses all of its rows in the window, whichever board
    they are on. Once picked it stays a member until the next effective date whatever its board label, and its
    whole price history on every board is converted.
- **2026-11-12 16:42, how the rule's wording is read by the code (fixed at the same time, before the universe was
  computed).** These used the source's layout and counts of unusable rows only, never prices or returns.
  - A source row counts only if its close is a positive number (rows with a zero or missing close are ignored
    everywhere; among codes ending in 0 there are 6 such rows, all funds). "Listed on a date" = has a row that day.
  - Trading days = the dates present in the source. A selection date is the last trading day of a May or November
    that is followed in the source by a trading day of the next month (the effective date), and whose whole window
    lies inside the source (first selection date 1995-11-30).
  - The 6 calendar months ending on the selection date = December to May, or June to November: from the first day
    of the month five months before the selection month up to the selection date.
  - A stock has a market cap on a day if it has a row that day with a positive `Marcap`. The average is over the
    days it has one. The 80% test is exact: 5 x (days with a market cap) >= 4 x (trading days in the window).
  - A stock must be listed on the selection date. Ties in average market cap are broken by the lower code.
  - A member's spans end on its last row in the source. A pick with no row on or after the effective date never
    becomes a member.
  - Index series: starts at 1000 on the first selection date. Each day's return is the sum over that day's members
    of weight x return, where the weight is the stock's market cap on its latest earlier row (weights sum to 1
    over members that have one) and the return is its adjusted close over its previous traded adjusted close,
    minus 1 (0 on a day it does not trade).
  - Factor (Rule KG2): "row" means a source row with a positive close, traded or not. On a row whose base price
    (close - change) is not a positive number the source gives no base price, so the factor is carried over
    unchanged on that row. Among codes ending in 0 there are 884 such rows: 876 from 1998-01-06 to 2000-01-25
    (63 stocks, mostly failed banks under a trading halt; 867 of the rows have zero volume, and on the 9 traded
    ones a share change that day would be missed) and 8 in 2009 (ship funds, which are excluded). None is on or
    after 2007-01-01 in a stock that can be picked, so no configured load window (from 2007) contains one; the
    converter reports the count and a test fails if one ever appears from 2007 on.
  - A traded row whose open, high or low is zero in the source (all rows of 1995-06-10 and a few rows a year
    after) keeps its close and volume; the zero fields are written as missing.

### Rule UC: the free US coverage check (D4; a go/no-go for the free S&P 500 route)

- Membership: fja05680/sp500 at a pinned commit (unofficial; S&P 500 members by date).
- Price coverage: Tiingo's public list of supported tickers (no API key), downloaded once; its sha256 and download
  time are recorded, and the file stays outside the repo.
- Window: 2008-01-01 to 2024-02-29 (the walk-forward history up to the locked test period). Days = weekdays.
- A member-day is **matched** if Tiingo lists that ticker as a stock with a start date on or before the day and an
  end date on or after it. Tickers are compared in upper case with "." and "/" written as "-" (class shares). The
  date range must cover the day because tickers get reused. A listing without dates never matches.
- Pass bar: at least 95.0% of all member-days in the window are matched.
- Reported whatever the outcome: the per-year matched share, the matched share among tickers that left the index
  before 2024-03-01, and the unmatched tickers with their member-day counts.
- What a pass means: Tiingo lists the ticker for those dates. Whether the free tier serves those prices is still
  unverified until a fetch. A pass makes the Tiingo fetch a TODO item; it is not started in this build.
- If it fails: the free S&P 500 route is recorded as not good enough as it stands, and the owner is told.

Details (appended by the US check before the match is first computed):

- **2026-11-12 16:27 local time (22:27 UTC). Fixed from the two files' formats only (column names, a few rows, the
  lists of values in Tiingo's columns, the fja05680 README); no match had been computed.**
  - Pins (fetched and verified by `scripts/get_us_coverage_data.sh`):
    - fja05680/sp500 at commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7` (its newest commit, dated 2026-09-07).
      File used: `S&P 500 Historical Components & Changes (Updated).csv`, sha256
      `36326709d46d6cd25834de5df457b16f5f96fad3a06b9beac28f7b88aa0b0d54`.
    - Tiingo: `supported_tickers.zip` (holds one file, `supported_tickers.csv`, with the columns ticker, exchange,
      assetType, priceCurrency, startDate, endDate), downloaded 2026-11-12T22:24:27Z from
      `https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip`, 810,580 bytes, sha256
      `3d33d870111072d8a87de8a3f73cb3b30f00ec928fa517846ae29b614822f800`.
  - Membership. Each row of the fja05680 file is a date and the full member list from that date on (columns `date`
    and `tickers`, the list comma-separated; rows sorted by date, one per change date, 1996-01-02 to 2026-08-18).
    The members on a day are the list of the last row dated on or before that day (the repo's own
    `sp500_by_date.ipynb` reads the file the same way). Rows dated after 2024-02-29 are not read.
  - Days. Every Monday to Friday from 2008-01-01 to 2024-02-29, both ends included: 4,218 days. US market holidays
    that fall on a weekday are included (the rule says weekdays; no trading calendar is used). One member-day = one
    ticker in that day's list. A ticker that appears twice in one list after normalization counts once.
  - Ticker normalization, applied the same way to both files: surrounding spaces removed, upper case, "." and "/"
    replaced by "-". Nothing else is changed (no suffix is dropped, no other character is replaced).
  - "Tiingo lists that ticker as a stock" = a row of the Tiingo file with all three of:
    - assetType exactly `Stock` (the file's other values, `ETF` and `Mutual Fund`, do not count);
    - priceCurrency exactly `USD` (the file's other values, CNY, HKD and AUD, do not count);
    - exchange not one of the exchanges outside the US: `LSE`, `SHE`, `SHEB`, `SHG`, `SHGB` (London, Shenzhen,
      Shanghai; some of their rows are labelled USD, and the same letters there are another company). Every other
      exchange value in the pinned file counts: `NYSE`, `NASDAQ`, `AMEX`, `NYSE MKT`, `NYSE ARCA`, `NYSE NAT`,
      `BATS`, `PINK`, `OTCMKTS`, `OTCGREY`, `OTCBB`, `OTCQB`, `OTCQX`, `OTCCE`, `OTCD`, `EXPM`, `NMFQS` and a blank
      exchange. Over-the-counter venues count because a row carries one exchange only, not a history, so it cannot
      say where the ticker traded on a past day, and the membership file names bankrupt members by their later
      over-the-counter tickers (for example `AAMRQ`). An exchange value in neither list stops the check with an
      error (this can only happen if the pin is changed).
  - Several Tiingo rows for one ticker (2,306 tickers have more than one row, up to 6). Rows are neither merged nor
    de-duplicated. A member-day is matched if at least one stock row with the same normalized ticker has
    startDate <= day <= endDate, both ends included. Days that fall between two rows of one ticker stay unmatched.
    Rows whose tickers differ only by ".", "/" or "-" are rows of the same ticker.
  - A row with an empty startDate or an empty endDate never matches (in the pinned file every row has both). A
    date that is not written YYYY-MM-DD stops the check with an error.
  - The bar is applied to the exact counts, not to a rounded percentage: pass if matched x 1000 >= all x 950.
  - "Tickers that left the index before 2024-03-01" = tickers with at least one member-day in the window that are
    not in the member list of 2024-02-29. Their matched share = their matched member-days / their member-days.
  - Per-year shares go by the calendar year of the day; 2024 is January and February only.
  - Reported as an extra, not part of the rule: the share of member-days whose ticker has at least one stock row
    whatever its dates (ticker only).

### Rule MA: membership timing audit (D7 a; a report, not a decision)

- Compares the pinned archive's CSI300 and CSI500 membership with BaoStock's, from 2007-01 to 2024-02.
- Reports the mismatch share (stocks in exactly one of the two lists / stocks in either list) and how late the
  archive shows index changes. It changes no data and no decision depends on it.

### Outcome (2026-11-12, EXPERIMENTS E20-E23)

- Rule CU: CSI500 and CSI1000 prepared. The whole A-share market prepared for the validation-period split only;
  its walk-forward data was stopped at 24.5 GiB of memory on a 24 GB machine, so it is not prepared.
- Rule KG: **go**, all three checks passed. The Korea dataset is built (Rule KU: 57 selection dates before
  2024-03-01, 200 picks each).
- Rule UC: **failed**, 91.06% of member-days matched against the 95.0% bar (94.71% under the loosest reading). The
  free S&P 500 route is not good enough as it stands; no Tiingo fetch was started.
- Rule MA: report written. No clear case of the archive showing an index change early; reviews since 2013-12 appear
  9 to 14 trading days late.
- Two points were not covered by the rules as written and are logged under "Rule changes"; the owner approved both on 2026-11-12.

## Grids set by Rule G

(Appended, timestamped, as each pilot finishes and before any run that looks at validation.)

- **2026-11-07 15:41, placebo pilot (shuffled answers):** M1 at epoch 8 in both seeds: 1e-5 yes, 1e-4 yes,
  1e-3 no, 1e-2 yes, 1e-1 yes, 1 yes. Anchor g₀ = 1e-5. **Placebo grid: 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3.**
  Details: `results/csi300/scheduler_balance/placebo/pilot/usage.md`.
- **2026-11-07 16:35, real pilot (after the placebo passed):** M1 at epoch 8 in both seeds: 1e-5 no, 1e-4 no,
  1e-3 no, 1e-2 no, 1e-1 yes, 1 yes. Anchor g₀ = 0.1. **Grid: 0.03, 0.1, 0.3, 1, 3, 10.** Gradient clipping was
  active on 1-3% of steps at every pilot strength. Details: `results/csi300/scheduler_balance/pilot/usage.md`.

## Rule changes

- **2026-11-07, how `detmaso compare` redraws seeds (Step 1 rule; logged with the owner's OK).**
  - Before (Step 1's results): after the shared day draws, every model's seeds were redrawn from that same
    random stream, one model after another, so a model's draws depended on which other models were in the
    comparison and in what order.
  - Now: each model redraws its seeds from a random stream of its own (RNG seed 0 and the model's name).
    Unchanged: the shared 20-day day blocks, 10,000 draws, RNG seed 0, and the 95%-range thresholds.
  - Reason: Rule K (fixed 2026-11-07, before its runs) needs each model's random stream independent of row
    order, so that the same pair of models gets the same range in every report.
  - Made after Step 1's results were seen. Every Step 1 range moves (IC, ICIR and Sharpe, in both Step 1
    reports), by about 0.01, and no verdict changes. The Step 1 write-ups keep the ranges from the earlier
    method and say so. Two examples, from the scheduler-balance report, which repeats two Step 1 comparisons:
    Det-MASO minus the single network, IC [-0.0878, +0.0114] becomes [-0.0882, +0.0102]; minus the single
    network with mood inputs, [-0.1435, -0.0166] becomes [-0.1423, -0.0177].
- **2026-11-08, general keep rule: after-cost Sharpe condition added (owner's OK, walk-forward plan decision D6 a).**
  - Before: the keep rule judged walk-forward IC only.
  - Now: it also requires that the 95% range of the after-cost Sharpe gain is not entirely below 0 (Rule WK, K5).
  - Reason: TODO.md says every later step is judged on walk-forward IC and after-cost Sharpe; a change that raises
    IC but detectably loses money after trading costs is not an improvement.
  - Made before any walk-forward run, so no result had been seen. The matching bullet is in CLAUDE.md's keep rule,
    added at the owner's request.
- **2026-11-11, walk-forward rules as applied by code: three corrections (approved by the owner during the review of
  the walk-forward change, before any real-answer walk-forward run).**
  - Placebo valuation (Rule WT). Before: a held stock outside that block's shuffle (held on only because it could
    not be sold) earned its real next-day return. Now: it earns 0, like a stock outside the day's universe. Reason:
    the code now matches the rule's existing sentence "no real return enters a placebo result".
  - Rules WK and WP. Before: a condition whose gain, range bound or Sharpe difference was not finite could count
    as met. Now: the rule refuses to give a verdict. Reason: a number that could not be computed is no evidence.
  - Walk-forward resampling (Rule WK statistics). Before: one day without an IC made that seed's yearly IC, and
    every range built on it, not finite. Now: such days are left out. Reason: consistent with the point estimates
    and with the validation-period resampling. Results whose IC days are all finite do not change.
  - Made after the walk-forward placebo results (E19) were seen. The Rule WP verdict is unaffected: P1 uses IC
    only, P2 fails on K1 in both directions, and every committed range is finite. The committed placebo results
    were then re-evaluated with the correction (evaluation only, no retraining): only a few 2015 numbers moved, in
    the second decimal (EXPERIMENTS E19).
- **2026-11-12, Rule CU (the whole A-share market): index series taken out of the `all` list. Reported to the owner
  the same day and approved by the owner.**
  - Before (Rule CU as written): the universe is the archive's `all` list, with the suspension and new-listing filters.
  - Now: the 6 index series that the archive keeps in that list (SH000300, SH000852, SH000905, SH000906, SH000985,
    SZ399300) are also left out, with the new flag `data.exclude_codes` in `configs/ashare.yaml`. A test checks that
    these 6 are exactly the non-stock codes in the list, judged by code prefix only.
  - Reason: an index is not a stock. Left in, the model would score and "buy" index levels.
  - Made during the build, when the list was first read, before the whole-market data was prepared and before any
    model run on it (none exists). No price or return informed it. It removes 17,712 of 10,378,047 member-days.
- **2026-11-12, Rule KG2 (Korea factor chain): rows without a usable base price. Reported to the owner the same day
  and approved by the owner.**
  - Before (Rule KG2 as written): the factor is multiplied by (previous row's close / base price) on every day, and
    a rule that cannot be applied as written stops the build.
  - Now: on a row where the source gives change = close (so the base price is 0), the factor is carried over
    unchanged. The build went on.
  - Reason: 804 such rows exist among picked stocks, all between 1998-01-09 and 2000-01-25 (mostly failed banks
    under a trading halt). None is dated 2007 or later, and no config loads data before 2007; a data test fails if
    one ever appears there.
  - Written into the Rule KU details before the universe was first computed, from a count of such rows only. The
    alternative, if the owner prefers it, is to start the Korea dataset after 2000-01-25.

