"""Data QA report (plan Section 5 step 13). Never touches the test period's labels."""

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

    uni = p.universe.sum(1).numpy()
    lab = p.label_ok.sum(1).numpy()
    lines += [
        "",
        "## Cross-section size",
        "",
        f"- Universe per day (members that traded the day before): min {uni.min()}, median {np.median(uni):.0f}, max {uni.max()}",
        f"- Stocks with a 20-day label per day: min {lab.min()}, median {np.median(lab):.0f}",
        f"- Smallest universe on {p.dates[uni.argmin()].date()}",
    ]

    close, factor = data.raw.fields["close"], data.raw.fields["factor"]
    spot = [("SH600519", "2024-03-01", 1685.06), ("SZ000001", "2024-03-01", 10.49), ("SH601318", "2024-03-01", 42.99)]
    lines += ["", "## Spot prices (raw close = $close / $factor)", "", "| Stock | Date | Ours | Expected |", "|---|---|---|---|"]
    for code, d, expected in spot:
        if code in close.columns:
            val = close.loc[d, code] / factor.loc[d, code]
            lines.append(f"| {code} | {d} | {val:.2f} | {expected} |")

    universe = data.masks["universe"]
    lines += ["", "## Missing values per feature (share of universe stock-days, before filling with 0)", ""]
    model_rows = universe.index.isin(p.dates)
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
    lines += ["| Feature | Mean daily IC 2012-2022 |", "|---|---|", *rows]

    path = artifacts_dir(cfg) / "data_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
