"""Figures for eval.compare: static PNGs for the markdown report (its tables are the table view).

Style follows the project's charting rules: thin marks, hairline solid grids, text in ink colors,
one hue where the axis already names the rows, and a validated blue/orange pair where two series
must be told apart (dataviz validator, light surface: all checks pass).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from detmaso.eval.compare import _seed_values  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.labelsize": 9, "axes.titlelocation": "left",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9,
}  # fmt: skip

LEVELS = [("ic", "IC", False), ("icir", "ICIR", False), ("sharpe", "Sharpe", False), ("mdd", "Max drawdown (lower is better)", True)]
ADVANTAGE = [("ic", "IC"), ("icir", "ICIR"), ("sharpe", "Sharpe")]


def _label(v) -> str:
    """Display name: the paper config is Det-MASO itself."""
    return "Det-MASO (paper config)" if v.name == "paper" else v.name


def _title(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11.5, color=INK)


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _levels_figure(variants, ref, out: Path) -> str:
    order = [ref] + [v for v in variants if v is not ref]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(12.5, 0.5 * len(order) + 1.5), sharey=True, layout="constrained")
    for ax, (metric, label, pct) in zip(axes, LEVELS):
        if not pct:
            ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        for yi, v in zip(y, order):
            vals = _seed_values(v, metric)
            ax.plot(vals, np.full(len(vals), yi), "o", ms=5.5, color=MUTED, mec=SURFACE, mew=1.0, zorder=2)
            ax.plot([vals.mean()], [yi], "o", ms=8.5, color=SERIES_1, mec=SURFACE, mew=1.5, zorder=3)
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.12)
        if pct:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_yticks(y, [f"{_label(v)}, reference" if v is ref else _label(v) for v in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    _title(fig, "Validation metrics by variant: gray dots are the training seeds, blue is their mean")
    return _save(fig, out / "metrics.png")


def _advantage_figure(variants, ref, stats: dict, out: Path) -> str:
    others = [v for v in variants if v is not ref]
    y = np.arange(len(others))[::-1]
    fig, axes = plt.subplots(1, len(ADVANTAGE), figsize=(12.5, 0.55 * len(others) + 1.6), sharey=True, layout="constrained")
    for ax, (metric, label) in zip(axes, ADVANTAGE):
        ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        for yi, v in zip(y, others):
            a = stats["advantage"][v.name][metric]
            ax.plot([a["ci_low"], a["ci_high"]], [yi, yi], color=SERIES_1, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot([a["observed"]], [yi], "o", ms=8.5, color=SERIES_1, mec=SURFACE, mew=1.5, zorder=3)
        lo = min(stats["advantage"][v.name][metric]["ci_low"] for v in others)
        hi = max(stats["advantage"][v.name][metric]["ci_high"] for v in others)
        span = max(abs(lo), abs(hi)) * 1.15
        ax.set_xlim(-span, span)  # 0 in the middle: left = the variant did better, right = the reference did
        ax.set_title(f"{label} advantage")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [_label(v) for v in others])
    axes[0].set_ylim(-0.6, len(others) - 0.4)
    for ax in axes:
        ax.set_xlabel(f"right of 0: {_label(ref)} better", color=MUTED)
    _title(fig, f"{_label(ref)} minus each variant on validation, with the 95% bootstrap range (dot = observed)")
    return _save(fig, out / "advantage.png")


def _cumulative_figure(variants, ref, out: Path) -> str:
    others = [v for v in variants if v is not ref]
    ncols = 2 if len(others) > 1 else 1
    nrows = math.ceil(len(others) / ncols)
    dates = ref.ret_dates

    def curve(r: np.ndarray) -> np.ndarray:
        return np.cumprod(1.0 + r) - 1.0

    bench = curve(ref.bench[0])  # the eligible universe does not depend on the model or seed
    ref_curve = curve(ref.port.mean(axis=0))  # money split equally across the seeds' portfolios
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.0 * nrows + 0.9), sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, v in zip(axes.flat, others):
        ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
        ax.plot(dates, bench, color=MUTED, lw=1.5, label="Equal-weight eligible stocks", zorder=2)
        ax.plot(dates, ref_curve, color=SERIES_2, lw=1.8, label=f"{_label(ref)}, reference", zorder=3)
        ax.plot(dates, curve(v.port.mean(axis=0)), color=SERIES_1, lw=1.8, label="The panel's variant", zorder=4)
        ax.set_title(_label(v))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for ax in list(axes.flat)[len(others) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center", ncol=3)
    _title(fig, "Cumulative validation return of the daily top-10% portfolio (mean of the seeds' daily returns)")
    return _save(fig, out / "cumulative_return.png")


def comparison_figures(variants, ref, stats: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        files = [_levels_figure(variants, ref, out)]
        if len(variants) > 1:
            files += [_advantage_figure(variants, ref, stats, out), _cumulative_figure(variants, ref, out)]
    return files
