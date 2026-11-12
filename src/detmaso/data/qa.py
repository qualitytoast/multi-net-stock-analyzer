"""Data QA report (plan Section 5 step 13). Label and missing-value statistics use only dates before the test period; the spot prices are data-integrity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config
from detmaso.data.panel import PreparedData, artifacts_dir
from detmaso.losses.stats import daily_spearman
from detmaso.mdtable import md_frame

PAPER_COVERAGE = {  # Table 4: share of training days per regime
    "csi300": [35, 12, 32, 16, 5],
    "sp500": [40, 15, 28, 13, 4],
}


def write_data_report(data: PreparedData, cfg: Config) -> Path:
    p = data.panel
    lines = [f"# Data QA report: {cfg.market}", ""]
    lines += [
        f"- Data tag: `{cfg.data.data_tag}` (sha256 `{cfg.data.data_sha256}`)",
        f"- Calendar loaded: {data.raw.calendar[0].date()} .. {data.raw.calendar[-1].date()} ({len(data.raw.calendar)} days)",
        f"- Model dates: {p.dates[0].date()} .. {p.dates[-1].date()} ({len(p.dates)} days); stocks ever in the index: {len(p.code_list)}",
        "",
        "## Splits",
        "",
        "| Split | First | Last | Days |",
        "|---|---|---|---|",
    ]
    for name, idx in vars(data.splits).items():
        lines.append(f"| {name} | {idx[0].date()} | {idx[-1].date()} | {len(idx)} |")

    pre_test = (p.dates < pd.Timestamp(cfg.splits.test[0]))
    uni = p.universe.sum(1).numpy()[pre_test]
    lab = p.label_ok.sum(1).numpy()[pre_test]
    lines += [
        "",
        "## Cross-section size (train and validation dates only)",
        "",
        f"- Universe per day (members that traded the day before): min {uni.min()}, median {np.median(uni):.0f}, max {uni.max()}",
        f"- Stocks with a 20-day label per day: min {lab.min()}, median {np.median(lab):.0f}",
        f"- Smallest universe on {p.dates[pre_test][uni.argmin()].date()}",
    ]

    close, factor = data.raw.fields["close"], data.raw.fields["factor"]
    spot = [("SH600519", "2024-03-01", 1685.06), ("SZ000001", "2024-03-01", 10.49), ("SH601318", "2024-03-01", 42.99)]
    lines += ["", "## Spot prices (raw close = $close / $factor)", "", "| Stock | Date | Ours | Expected |", "|---|---|---|---|"]
    for code, d, expected in spot:
        if code in close.columns:
            val = close.loc[d, code] / factor.loc[d, code]
            lines.append(f"| {code} | {d} | {val:.2f} | {expected} |")

    universe = data.masks["universe"]
    lines += ["", "## Missing values per feature (share of universe stock-days before the test period, before filling with 0)", ""]
    model_rows = universe.index.isin(p.dates[pre_test])
    uni_model = universe.loc[model_rows]
    nan_rates = {
        name: float((frame.loc[model_rows].isna() & uni_model).to_numpy().sum() / uni_model.to_numpy().sum())
        for name, frame in data.features_raw.items()
    }
    lines.append(", ".join(f"{k} {v:.1%}" for k, v in nan_rates.items()))

    reg = data.regimes
    lines += ["", "## Regimes", "", "Silhouette score by number of clusters (training period):", ""]
    lines.append(", ".join(f"K={k}: {v:.3f}" for k, v in reg.silhouette.items()))
    best_k = max(reg.silhouette, key=reg.silhouette.get)
    lines += [
        "",
        f"The silhouette-optimal count is **K = {best_k}**; per decision D2 the model uses n = {cfg.regimes.n_agents} regardless.",
        "",
        "| Regime (agent) | Our coverage of training days | Paper (Table 4) |",
        "|---|---|---|",
    ]
    tr = p.date_pos(data.splits.train).numpy()
    cov = np.bincount(p.regime[tr].numpy(), minlength=len(reg.names)) / len(tr)
    paper = PAPER_COVERAGE.get(cfg.market, [None] * len(reg.names))
    for i, name in enumerate(reg.names):
        lines.append(f"| {i + 1} {name} | {cov[i]:.1%} | {paper[i]}% |")
    cent = pd.DataFrame(reg.centroids * reg.std + reg.mean, index=reg.names, columns=reg.state_names)
    lines += ["", "Centroids in raw units (daily returns):", "", md_frame(cent.round(4))]
    v_vol = (cent["sigma20"] + cent["sigma60"]) / 2
    flags = []
    if v_vol["Low-Vol Bear"] > v_vol["High-Vol Bull"]:
        flags.append("the cluster named Low-Vol Bear is more volatile than High-Vol Bull")
    if v_vol["Low-Vol Bull"] > v_vol["High-Vol Bull"]:
        flags.append("the cluster named Low-Vol Bull is more volatile than High-Vol Bull")
    lines += [
        "",
        "Naming-rule check (plan A16 is fixed in advance; names are labels only): "
        + ("; ".join(flags) + "." if flags else "names are consistent with the centroid volatilities."),
    ]

    # Pipeline sanity check (test 30): single-feature ICs on the TRAINING period only.
    lines += ["", "## Single-feature IC on the training period (pipeline sanity check)", ""]
    pos = p.date_pos(data.splits.train_fit)
    y = p.y[pos].numpy()
    m = p.label_ok[pos].numpy()
    rows = []
    for name in ("mom_20", "vol_20", "mom_5", "rturn_20", "logvol20"):
        j = p.feature_names.index(name)
        ic = daily_spearman(p.X[pos, :, j].numpy(), y, m)
        rows.append(f"| {name} | {np.nanmean(ic):+.4f} |")
    fit = data.splits.train_fit
    lines += [f"| Feature | Mean daily IC {fit[0].year}-{fit[-1].year} |", "|---|---|", *rows]
    if cfg.is_walkforward:
        lines += _per_year_history(data, cfg)

    path = artifacts_dir(cfg) / "data_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _per_year_history(data: PreparedData, cfg: Config) -> list[str]:
    """Walk-forward configs: per-year coverage and data-error checks over the whole history (before the test period)."""
    from detmaso.data.returns import returns_one_day

    p = data.panel
    universe = data.masks["universe"]
    ret = returns_one_day(data.raw.fields["close"])
    rows = []
    pre_test = p.dates[p.dates < pd.Timestamp(cfg.splits.test[0])]
    for year in sorted(set(pre_test.year)):
        days = pre_test[pre_test.year == year]
        pos = p.date_pos(days).numpy()
        uni = p.universe[pos].sum(1).numpy()
        lab = p.label_ok[pos].sum(1).numpy()
        u = universe.loc[days]
        missing = np.mean([float((frame.loc[days].isna() & u).to_numpy().sum()) for frame in data.features_raw.values()]) / max(1, u.to_numpy().sum())
        big = int((ret.loc[days].abs().where(u) > 0.21).to_numpy().sum())
        rows.append(f"| {year} | {len(days)} | {uni.min()} | {np.median(uni):.0f} | {np.median(lab):.0f} | {missing:.2%} | {big} |")
    return [
        "",
        "## Per year (walk-forward history check)",
        "",
        "Universe = index members that traded the day before; missing = average share of universe stock-days with a missing",
        "feature (before filling with 0); moves > 21% = universe stock-days whose one-day return exceeds every price limit",
        "(possible data errors, or days without a limit such as a stock's first trading days).",
        "",
        "| Year | Days | Universe min | Universe median | Labelled median | Missing features | Moves > 21% |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]
