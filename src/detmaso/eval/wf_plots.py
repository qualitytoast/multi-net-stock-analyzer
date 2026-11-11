"""Figures for the walk-forward reports (static PNGs, one y-axis each, a fixed categorical order)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from detmaso.eval.wf_stats import WFRun, pct_range  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, always in this order
NEUTRAL, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"


def _style(ax) -> None:
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def run_figures(run: WFRun, sm: dict, boot: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    dates = np.concatenate([run.ret[y] for y in run.years])
    net = run.stitched(run.realistic)
    gross = run.stitched(lambda y: run.gross(y, False))
    bench = run.stitched(lambda y: run.cols[y]["bench"])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for x, color, label in ((gross, SERIES[1], "Paper backtest (no costs)"), (net, SERIES[0], "After costs")):
        curves = (np.cumprod(1 + x, axis=1) - 1) * 100
        for c in curves:
            ax.plot(dates, c, color=color, lw=0.6, alpha=0.35)
        ax.plot(dates, curves.mean(axis=0), color=color, lw=2, label=f"{label}, seed mean (thin: each seed)")
    ax.plot(dates, (np.cumprod(1 + bench[0]) - 1) * 100, color=NEUTRAL, lw=1.4, ls="--", label="Equal-weighted eligible stocks")
    ax.set_ylabel("Cumulative return (%)", color=MUTED, fontsize=9)
    ax.set_title(f"Top-10% portfolio, walk-forward test years {run.years[0]}-{run.years[-1]}", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    files.append(_save(fig, out / "cumulative_return.png"))

    fig, ax = plt.subplots(figsize=(10, 3.6))
    mean = sm["ic_year"].mean(axis=0)
    lo_hi = np.array([pct_range(boot["ic_year"][i]) for i in range(len(run.years))])
    xs = np.arange(len(run.years))
    ax.bar(xs, mean, width=0.6, color=SERIES[0], label="Seed-mean IC")
    ax.errorbar(xs, mean, yerr=[mean - lo_hi[:, 0], lo_hi[:, 1] - mean], fmt="none", ecolor=INK, elinewidth=1, capsize=3, label="95% range")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(xs, [str(y) for y in run.years])
    ax.set_ylabel("IC", color=MUTED, fontsize=9)
    ax.set_title("IC by test year", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    files.append(_save(fig, out / "ic_by_year.png"))
    return files


def compare_figures(years: list[int], gains: dict[str, dict], out: Path) -> list[str]:
    """gains[variant][year] = {"gain", "low", "high"}: yearly IC gain over the reference (at most 3 variants)."""
    out.mkdir(parents=True, exist_ok=True)
    names = list(gains)[: len(SERIES)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = np.arange(len(years))
    width = 0.8 / max(1, len(names))
    for k, name in enumerate(names):
        g = np.array([gains[name][y]["gain"] for y in years])
        lo = np.array([gains[name][y]["low"] for y in years])
        hi = np.array([gains[name][y]["high"] for y in years])
        pos = xs - 0.4 + width * (k + 0.5)
        ax.bar(pos, g, width=width * 0.9, color=SERIES[k], label=name)
        ax.errorbar(pos, g, yerr=[g - lo, hi - g], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(xs, [str(y) for y in years])
    ax.set_ylabel("IC gain over the reference", color=MUTED, fontsize=9)
    ax.set_title("Yearly IC gain with 95% ranges (positive = the variant did better)", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    return [_save(fig, out / "ic_gain_by_year.png")]
