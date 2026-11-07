- Ask me for permission before shipping a code change (not a test run)

Testing:
- Never use the test set (not validation set) to choose, tune or compare anything.
- Compare changes only on validation or walk-forward folds
- Always ask before running on the locked test set. The contents of which are defined in the ranges in the files of configs/.

Code Changes:
- Every new idea goes behind a config flag.
- The paper config's results must not change.

Bookkeeping:
- Log every experiment run in EXPERIMENTS.md: date, config, data used, results, kept or dropped
- Log every major milestone in ROADMAP.md, and remove that major milestone from TODO.md.
- Check READMD.md after each major milestone and tell me when it has fallen behind