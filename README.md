# Det-MASO reproduction (MVP, CSI300)

A faithful reproduction of **Det-MASO** (Wu and Yang, arXiv 2610.34936, Oct 2026): five MLP agents
tied to K-means market regimes, an attention scheduler that mixes their per-stock scores, a
log-determinant diversity penalty, a top-decile long-only backtest and a Shapley interaction
analysis. Scope is the paper's method only: no baselines, ablations or improvements.

- The full design is in [docs/PLAN.md](docs/PLAN.md): a spec table tying every component to the paper,
  about 45 documented defaults for details the paper leaves out (IDs A1-A45, each a value in
  `src/detmaso/config.py` / `configs/csi300.yaml`), the test list, success criteria, and 27 verified
  internal inconsistencies in the paper.
- Changes made to those planned defaults during the build are logged with evidence in
  [DEVIATIONS.md](DEVIATIONS.md).

## Status

Work in progress. The data pipeline, model, loss, training loop, backtest, metrics, Shapley analysis
and 49 tests are complete, and development runs on the training and validation periods are done.
The final 5-seed evaluation on the test period (2024-03-01 to 2026-03-01) has **not** been run yet;
it is opened exactly once, with a frozen configuration.

Early findings (validation period, details in DEVIATIONS.md): the paper's objective makes the
scheduler collapse onto the two agents tied to the most common regimes, and the agents tied to rare
regimes get almost no learning signal. Making training run at all required four documented changes
to planned defaults (GELU, decoupled weight decay, float64, a defined score for constant agent output).

The market data is not included in this repository: it is downloaded by `scripts/get_cn_data.sh`
from the community Qlib dataset (chenditc/investment_data), whose data has no explicit license, so
the dataset is not redistributed here.

## Setup

```bash
# Python 3.12 is required (pyqlib 0.9.7 has no 3.13 wheels); uv installs it for you.
uv sync
./scripts/get_cn_data.sh        # pinned community Qlib CN dataset, verified by sha256 (about 570 MB)
uv run pytest                   # 49 tests; data-dependent ones skip until `prepare` has run
```

## Run

```bash
uv run detmaso prepare  --config configs/csi300.yaml                  # features, regimes, panel, QA report
uv run detmaso train    --config configs/csi300.yaml --seeds 0 1 2 3 4 --tag final
uv run detmaso evaluate --config configs/csi300.yaml --seeds 0 1 2 3 4 --tag final --split valid
uv run detmaso evaluate --config configs/csi300.yaml --seeds 0 1 2 3 4 --tag final --split test --unlock-test
```

The test period (2024-03-01 to 2026-03-01) is locked: evaluating it needs `--unlock-test`, and the
first such run records the config hash in `runs/csi300/test_lock.json`; a later test evaluation with a
different configuration is refused. Development uses the training and validation periods only.

## Outputs

| Path | Content |
|---|---|
| `runs/csi300/prepared/data_report.md` | data QA: splits, universe sizes, spot prices, missing values, silhouette curve, regime coverage vs the paper, training-period single-feature ICs |
| `runs/csi300/<tag>/seed<k>/` | `model.pt` (best epoch), `history.jsonl` (per-epoch diagnostics), `summary.json` |
| `runs/csi300/<tag>/report_<split>.md` | metrics vs the paper, per-seed table, diversity, Shapley matrix, qualitative checks, verdict (test only), figures |

## Layout

```
src/detmaso/
  config.py            typed config; every assumption is a named value
  data/                qlib_io, returns, labels, features (34), market_state (12), regimes, splits, panel, qa
  model/               agent MLP, scheduler (window encoders + attention + head), DetMASO
  losses/              soft rank (Blondel et al. L2, exact PAV), masked statistics, the full objective
  train/trainer.py     Algorithm 1: one Adam(W) step per date, cosine LR, clipping, early stopping
  eval/                metrics, top-decile backtest, Shapley interaction, report
  cli.py               prepare | train | evaluate
tests/                 the plan's 30 checks (look-ahead, gradients, soft rank, metrics, Shapley, ...)
```
