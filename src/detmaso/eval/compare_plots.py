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


# Scheduler-balance rule figures (only with `compare --rule scheduler_balance`) -----------------------
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
AGENT_NAMES = ("Low-Vol Bull", "High-Vol Bull", "Sideways", "Low-Vol Bear", "Tail-Risk")


def _seed_best(diag: dict, key: str) -> list:
    return [s["best"][key] for s in diag["seeds"].values() if "best" in s]


def _sweep_figure(variants, ref, stats: dict, rules: dict, usage: dict, out: Path) -> str:
    from detmaso.losses.balance import effective_agents

    by_name = {v.name: v for v in variants}
    names = list(rules["rule_m"])  # sweep order (ascending strength)
    x = np.asarray(rules["strengths"], dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")

    r = stats["variants"][ref.name]["ic"]
    ax1.axhspan(r["ci_low"], r["ci_high"], color=SERIES_2, alpha=0.10, lw=0, zorder=0)
    ax1.axhline(r["mean"], color=SERIES_2, lw=1.6, zorder=1, label=f"{_label(ref)}, no penalty: mean {r['mean']:.3f} (shaded: 95% range)")
    for xi, name in zip(x, names):
        s = stats["variants"][name]["ic"]
        vals = _seed_values(by_name[name], "ic")
        ax1.plot([xi, xi], [s["ci_low"], s["ci_high"]], color=SERIES_1, lw=2.0, solid_capstyle="round", zorder=2)
        ax1.plot(np.full(len(vals), xi), vals, "o", ms=5.0, color=MUTED, mec=SURFACE, mew=1.0, zorder=3)
        ax1.plot([xi], [s["mean"]], "o", ms=8.5, color=SERIES_1, mec=SURFACE, mew=1.5, zorder=4)
    ax1.set_xscale("log")
    ax1.set_xlabel("Penalty strength c_bal (log scale)")

    target_eff = effective_agents(rules["target"])
    ax2.axhline(target_eff, color=INK2, lw=1.2, ls="--", zorder=1, label=f"target ({rules['target_kind']} shares): {target_eff:.2f}")
    if usage.get(ref.name):
        ref_eff = float(np.mean(_seed_best(usage[ref.name], "effective_agents")))
        ax2.axhline(ref_eff, color=SERIES_2, lw=1.6, zorder=1, label=f"{_label(ref)}, no penalty: {ref_eff:.2f}")
    for xi, name in zip(x, names):
        eff = _seed_best(rules["diagnostics"][name], "effective_agents")
        ax2.plot(np.full(len(eff), xi), eff, "o", ms=5.0, color=MUTED, mec=SURFACE, mew=1.0, zorder=3)
        ax2.plot([xi], [np.mean(eff)], "o", ms=8.5, color=SERIES_1, mec=SURFACE, mew=1.5, zorder=4)
        m = rules["rule_m"][name]
        label = "Rule M: pass" if m["pass"] else f"usage {'ok' if m['m1'] else 'fail'}\nrouting {'ok' if m['m2'] else 'fail'}"
        ax2.text(xi, 1.02, label, transform=ax2.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5,
                 color=INK if m["pass"] else MUTED, fontweight="bold" if m["pass"] else "normal")  # fmt: skip
    ax2.set_xscale("log")
    ax2.set_ylim(0.9, max(5.1, ax2.get_ylim()[1]))
    ax2.set_xlabel("Penalty strength c_bal (log scale)")
    ax2.set_title("Effective agents over time, training days (restored models)", pad=28)
    ax1.set_title("Validation IC: seed dots (gray), seed mean and 95% range (blue)", pad=28)
    c_star = rules["rule_s"]["c_star"]
    for ax in (ax1, ax2):
        if c_star is not None:
            ax.axvline(c_star, color=AXIS, lw=1.0, ls=":", zorder=0)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    _title(fig, "Load-balancing strength sweep" + (f" (dotted line: c* = {c_star:g})" if c_star is not None else " (no strength passed Rule M)"))
    return _save(fig, out / "strength_sweep.png")


def _focus(rules: dict) -> tuple[str, str]:
    """The balanced variant shown next to the reference: c*, or the largest strength if none was chosen."""
    if rules["c_star_variant"] is not None:
        return rules["c_star_variant"], f"c* = {rules['rule_s']['c_star']:g}"
    name = list(rules["rule_m"])[-1]
    return name, f"c_bal = {rules['strengths'][-1]:g} (largest tested; no c*)"


def _usage_figure(ref, rules: dict, usage: dict, out: Path) -> str:
    target = np.asarray(rules["target"], dtype=float)
    focus, focus_label = _focus(rules)
    rows = [(f"{_label(ref)}", usage.get(ref.name), SERIES_2), (focus_label, rules["diagnostics"][focus], SERIES_1)]
    agents = np.flatnonzero(target > 0)  # an agent with a zero target has no usage bar to clear
    n = len(agents)
    y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * n + 1.7), layout="constrained")
    ax.axvline(1.0, color=INK2, lw=1.2, zorder=1)
    ax.axvline(0.5, color=INK2, lw=1.0, ls="--", zorder=1)
    floor = 1e-4
    for k, (label, diag, color) in enumerate(rows):
        if not diag:
            continue
        usage_seeds = np.asarray(_seed_best(diag, "usage"), dtype=float)[:, agents]  # [seeds, agents]
        ratios = np.maximum(usage_seeds / target[agents], floor)
        offset = 0.14 if k == 0 else -0.14
        for s_ratios in ratios:
            ax.plot(s_ratios, y + offset, "o", ms=6.5, color=color, mec=SURFACE, mew=1.2, alpha=0.9, zorder=3, label=label)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="outside lower center", ncol=2)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.7, max(3.0, ax.get_xlim()[1]))
    ax.set_yticks(y, [f"{AGENT_NAMES[i] if i < len(AGENT_NAMES) else i} (target {100 * target[i]:.1f}%)" for i in agents])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Usage on training days / target share (log scale; one dot per seed; floored at 0.0001)")
    ax.text(1.0, n - 0.45, " target", color=INK2, fontsize=8)
    ax.text(0.5, n - 0.45, " Rule M bar (half the target) ", color=INK2, fontsize=8, ha="right")
    _title(fig, "Each agent's use relative to its target share (restored models)")
    return _save(fig, out / "usage_vs_target.png")


def _routing_figure(ref, rules: dict, usage: dict, out: Path) -> str:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    focus, focus_label = _focus(rules)
    panels = [(f"{_label(ref)}", usage.get(ref.name)), (focus_label, rules["diagnostics"][focus])]
    panels = [(t, d) for t, d in panels if d]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), layout="constrained", squeeze=False)
    for ax, (title, diag) in zip(axes[0], panels):
        moods = [k for k, d in enumerate(diag["mood_days"]) if d > 0]  # rows only for moods that occur
        mat = np.mean(np.asarray(_seed_best(diag, "routing"), dtype=float)[:, moods], axis=0)
        n = mat.shape[1]
        ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0)
        names = [AGENT_NAMES[i] if i < len(AGENT_NAMES) else str(i) for i in range(n)]
        ax.set_xticks(range(n), names, rotation=30, ha="right")
        ax.set_yticks(range(len(moods)), [f"{names[k]} days ({diag['mood_days'][k]})" for k in moods])
        ax.grid(False)
        for r in range(len(moods)):
            for i in range(n):
                v = mat[r, i]
                ax.text(i, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color=SURFACE if v > 0.55 else INK)
        ax.set_xlabel("Agent")
        ax.set_title(title)
    _title(fig, "Mean scheduler weight (%) on each mood's training days: the diagonal is each mood's own agent")
    return _save(fig, out / "routing.png")


def rule_figures(variants, ref, stats: dict, rules: dict, usage: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        return [_sweep_figure(variants, ref, stats, rules, usage, out), _usage_figure(ref, rules, usage, out), _routing_figure(ref, rules, usage, out)]
