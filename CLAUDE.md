- Ask me for permission before shipping a code change (not a test run)

Testing:
- Never use the test set (not validation set) to choose, tune or compare anything.
- Compare changes only on validation or walk-forward folds
- Always ask before running on the locked test set. The contents of which are defined in the ranges in the files of configs/.

Code Changes:
- Every new idea goes behind a config flag.
- The paper config's results must not change.

Guards (use whenever the model is changed or tuned):
- Count tries: every trained configuration goes in the try ledger (results/csi300/trials.csv), including failed and aborted runs. Every reported result shows its try number, deflated Sharpe and deflated IC.
- Rules first: before the first run of a change, write its keep/drop rule in docs/RULES.md, dated. Apply rules with code, not by eye. Never change a rule after seeing results; if a rule turns out to be wrong, ask me, then log the change and the reason.
- Placebo: whenever a new choosing or tuning procedure is introduced, run it on shuffled answers (a placebo config) before any real run. If the placebo "finds" a result, stop and tell me.
- Report everything: every number comes with its uncertainty (± std across seeds, or a 95% range) and its try count. Dropped ideas go in the README's "What didn't work" section, with their result and why they were dropped.
- Keep rule (walk-forward IC is the main metric). A change counts as an improvement only if all of these hold:
  - its average gain across all walk-forward years has a 95% range entirely above 0;
  - it wins in at least 8 of 11 years. For a change aimed at specific markets (for example crash years), it must instead win in most of those years and be no worse (95% range not entirely below 0) in the rest. Say which years it targets before running;
  - at least 4 of 5 seeds show a gain on average;
  - if the change has a tunable setting, the next setting over also shows a gain;
  - its after-cost Sharpe gain (realistic costs, with blocked trades) has a 95% range that is not entirely below 0.
  Until walk-forward exists, a change can only be "kept for re-check" or "dropped".

Bookkeeping:
- Log every experiment run in EXPERIMENTS.md: date, config, data used, results, kept or dropped
- Log every major milestone in ROADMAP.md, and remove that major milestone from TODO.md.
- Check README.md after each major milestone and tell me when it has fallen behind