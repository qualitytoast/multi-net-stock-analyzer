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

### Deflated metrics for walk-forward results

- Number of tries: every counted try in the ledger (validation and walk-forward alike).
- Spread of the tries' results: from the walk-forward tries once at least 3 have walk-forward metrics; until
  then, from the validation tries, whose spread is larger, so the deflated numbers come out more cautious. The
  report says which was used.
- Deflated Sharpe uses the daily after-cost excess return (realistic costs with blocking, minus the equal-weighted
  eligible stocks); deflated IC uses the standard error of the average IC from the resampling above.

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
  - Made before any walk-forward run, so no result had been seen. CLAUDE.md's wording is the owner's to update.
