"""Evaluation and report (plan Sections 7, 8, 10; Milestones M7-M9).

For the validation split (development), IC and Shapley use the purged validation dates, so no
label reaches into the test period. The test split can only be evaluated with a record returned
by eval.lock.open_test_period (the test-period lock).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from detmaso.config import Config  # noqa: E402
from detmaso.data.panel import Panel, load_prepared  # noqa: E402
from detmaso.data.qlib_io import cache_dir  # noqa: E402
from detmaso.eval.backtest import top_decile_backtest  # noqa: E402
from detmaso.eval.metrics import ic_summary, portfolio_metrics  # noqa: E402
from detmaso.eval.shapley import coalition_values, interaction_index  # noqa: E402
from detmaso.losses.stats import daily_spearman  # noqa: E402
from detmaso.mdtable import md_table  # noqa: E402
from detmaso.train.trainer import load_model, predict  # noqa: E402

PAPER = {  # Table 5 (Det-MASO rows), Tables 6-7 (ICIR); floors = Table 5 Single-MLP rows
    "csi300": {"ic": 0.081, "icir": 0.62, "sharpe": 1.85, "ar": 0.315, "mdd": 0.085, "floor_ic": 0.055, "floor_sharpe": 1.05},
    "sp500": {"ic": 0.072, "icir": None, "sharpe": 1.58, "ar": 0.268, "mdd": 0.082, "floor_ic": 0.048, "floor_sharpe": 0.92},
}
PAPER_FIG3 = {"mean_abs_corr": 0.49, "logdet": -0.28}
BANDS = {"ic": (0.015, 0.030), "icir": (0.15, 0.30), "sharpe": (0.40, 0.80), "ar": (0.07, 0.14), "mdd": (0.04, 0.08)}
LOW_VOL_BULL, TAIL_RISK = 0, 4


@dataclass
class SeedResult:
    seed: int
    ic: dict
    port: dict
    excess: dict
    sub: dict | None
    shapley: np.ndarray
    v: np.ndarray
    shapley_degenerate_days: int
    diversity_abs_corr: float
    diversity_logdet: float
    entropy: float
    dead_days: list[int]
    train_fig3: dict
    summary: dict
    series: dict


def agent_rank_corr(alphas: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per day over live agents: mean |off-diagonal| Spearman correlation, log det, and live count."""
    T = alphas.shape[0]
    abs_corr = np.full(T, np.nan)
    logdet = np.full(T, np.nan)
    n_live = np.zeros(T, dtype=int)
    for t in range(T):
        x = alphas[t][:, mask[t]]
        live = x.std(axis=1) > 0  # agents whose score is not constant today
        k = int(live.sum())
        n_live[t] = k
        if k < 2:
            continue
        c = np.corrcoef(rankdata(x[live], axis=1))
        abs_corr[t] = np.abs(c[~np.eye(k, dtype=bool)]).mean()
        sign, ld = np.linalg.slogdet(c)
        logdet[t] = ld if sign > 0 else -np.inf
    return abs_corr, logdet, n_live


def training_fig3(history: list[dict], last: int) -> dict:
    """Figure 3 protocol: averages over the last min(last, E) training epochs, ending at the stop epoch."""
    tail = history[-min(last, len(history)) :]
    return {
        "epochs_averaged": len(tail),
        "mean_abs_corr": float(np.mean([h["mean_abs_corr"] for h in tail])),
        "logdet": float(np.mean([h["logdet_raw"] for h in tail])),
        "mean_live_agents": float(np.mean([h.get("n_live", float("nan")) for h in tail])),
    }


def next_dates(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The trading day after each date (the day each holding-period return is realized)."""
    pos = calendar.get_indexer(dates)
    if (pos < 0).any() or (pos + 1 >= len(calendar)).any():
        raise KeyError("dates outside the loaded calendar")
    return calendar[pos + 1]


def evaluate_seed(cfg: Config, panel: Panel, seed_dir: Path, ic_dates, bt_dates, calendar, sub_window) -> SeedResult:
    model = load_model(cfg, panel, seed_dir)
    bt_pos = panel.date_pos(bt_dates)
    out = predict(model, panel, bt_pos, device=next(model.parameters()).device)
    in_ic = np.isin(bt_dates, ic_dates)
    labels = panel.y[bt_pos].numpy()
    label_ok = panel.label_ok[bt_pos].numpy()
    universe = panel.universe[bt_pos].numpy()

    ic_series = daily_spearman(out["alpha_hat"][in_ic], labels[in_ic], label_ok[in_ic])
    port, bench, _ = top_decile_backtest(out["alpha_hat"], panel.tradable[bt_pos].numpy(), panel.r_next[bt_pos].numpy(), cfg.backtest.top_frac)
    ret_dates = next_dates(bt_dates, calendar)
    start, end = bt_dates[0], ret_dates[-1]
    port_m = portfolio_metrics(port, start, end, cfg.backtest.qlib_days)
    excess_m = portfolio_metrics(port - bench, start, end, cfg.backtest.qlib_days)
    sub = None
    if sub_window is not None:
        k = (bt_dates >= pd.Timestamp(sub_window[0])) & (bt_dates <= pd.Timestamp(sub_window[1]))
        sub_ic = daily_spearman(out["alpha_hat"][k & in_ic], labels[k & in_ic], label_ok[k & in_ic])
        s0, s1 = bt_dates[k][0], ret_dates[k][-1]
        sub = {
            **ic_summary(sub_ic),
            **portfolio_metrics(port[k], s0, s1, cfg.backtest.qlib_days),
            "excess": portfolio_metrics(port[k] - bench[k], s0, s1, cfg.backtest.qlib_days),
        }

    v, degenerate = coalition_values(out["alphas"][in_ic], out["logits"][in_ic], labels[in_ic], label_ok[in_ic])
    shap = interaction_index(v, out["alphas"].shape[1])
    abs_corr, logdet, n_live = agent_rank_corr(out["alphas"], universe)
    w = np.clip(out["w"], 1e-300, 1.0)
    entropy = -(w * np.log(w)).sum(axis=1)
    dead_days = (out["raw_std"] < cfg.train.dead_std).sum(axis=0).astype(int).tolist()
    history = [json.loads(line) for line in (seed_dir / "history.jsonl").read_text().splitlines()]
    summary = json.loads((seed_dir / "summary.json").read_text())
    return SeedResult(
        seed=int(seed_dir.name.removeprefix("seed")),
        ic=ic_summary(ic_series),
        port=port_m,
        excess=excess_m,
        sub=sub,
        shapley=shap,
        v=v,
        shapley_degenerate_days=int(degenerate.sum()),
        diversity_abs_corr=float(np.nanmean(abs_corr)),
        diversity_logdet=float(np.nanmean(logdet[np.isfinite(logdet)])),
        entropy=float(entropy.mean()),
        dead_days=dead_days,
        train_fig3=training_fig3(history, cfg.train.diversity_last_epochs),
        summary=summary,
        series={
            "decision_dates": bt_dates,
            "ret_dates": ret_dates,
            "port": port,
            "bench": bench,
            "w": out["w"],
            "a": out["a"],
            "abs_corr": abs_corr,
            "n_live": n_live,
            "ic": ic_series,
            "val_curve": [h["val_ic"] for h in history],
            "val_smoothed": [h["val_ic_smoothed"] for h in history],
            "best_epoch": summary["best_epoch"],
        },
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def band_status(metric: str, ours: float, paper: float | None) -> str:
    if paper is None or not np.isfinite(ours):
        return "n/a"
    tight, loose = BANDS[metric]
    better = (paper - ours) if metric == "mdd" else (ours - paper)  # positive = better than the paper
    if better > tight:
        return "exceeds paper"
    if abs(better) <= tight:
        return "reproduced band"
    if abs(better) <= loose:
        return "close band"
    return "outside"


def qualitative(results: list[SeedResult], panel: Panel, train_dates) -> dict:
    tr = panel.date_pos(train_dates).numpy()
    tail_cov = float(np.mean(panel.regime[tr].numpy() == TAIL_RISK))
    i15 = [r.shapley[LOW_VOL_BULL, TAIL_RISK] for r in results]
    largest = []
    for r in results:
        m = np.where(np.isnan(r.shapley), -np.inf, r.shapley)
        largest.append(bool(m[LOW_VOL_BULL, TAIL_RISK] == m.max()))
    need = math.ceil(0.8 * len(results))
    q1 = float(np.mean([r.diversity_abs_corr for r in results]))
    q2 = float(np.mean([r.entropy for r in results]))
    return {
        "q1": (q1, "< 0.7"),
        "q2": (q2, "> 0.80"),
        "q3": (tail_cov, "3%-7%"),
        "q4": (sum(x > 0 for x in i15), f">= {need} of {len(results)} seeds"),
        "q1_ok": q1 < 0.7,
        "q2_ok": q2 > 0.80,
        "q3_ok": 0.03 <= tail_cov <= 0.07,
        "q4_ok": sum(x > 0 for x in i15) >= need,
        "q4_largest": sum(largest),
    }


def verdict(market: str, means: dict, q: dict) -> tuple[str, list[str]]:
    """Plan Section 10, applied in order. Returns (label, list of criteria that failed)."""
    p = PAPER[market]
    failed = []
    if means["ic"] < p["floor_ic"]:
        failed.append(f"IC {means['ic']:.3f} below the paper's single-MLP floor {p['floor_ic']}")
    if means["sharpe"] < p["floor_sharpe"]:
        failed.append(f"Sharpe {means['sharpe']:.2f} below the paper's single-MLP floor {p['floor_sharpe']}")
    if failed:
        return "Not reproduced", failed
    checks = {
        f"IC >= {p['ic'] - BANDS['ic'][0]:.3f}": means["ic"] >= p["ic"] - BANDS["ic"][0],
        f"Sharpe >= {p['sharpe'] - BANDS['sharpe'][0]:.2f}": means["sharpe"] >= p["sharpe"] - BANDS["sharpe"][0],
        f"annual return >= {p['ar'] - BANDS['ar'][1]:.1%}": means["ar"] >= p["ar"] - BANDS["ar"][1],
        f"max drawdown <= {p['mdd'] + BANDS['mdd'][1]:.1%}": means["mdd"] <= p["mdd"] + BANDS["mdd"][1],
        **{f"q{i}": q[f"q{i}_ok"] for i in range(1, 5)},
    }
    failed = [name for name, ok in checks.items() if not ok]
    if not failed:
        return "Reproduced", []
    n_q = sum(q[f"q{i}_ok"] for i in range(1, 5))
    if means["ic"] >= p["ic"] - BANDS["ic"][1] and means["sharpe"] >= p["sharpe"] - BANDS["sharpe"][1] and n_q >= 3:
        return "Partially reproduced", failed
    return "Inconclusive", failed


def _plots(results: list[SeedResult], regime_names: list[str], index_close: pd.Series, fig_dir: Path, split: str) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = []
    s0 = results[0].series
    dec, ret = s0["decision_dates"], s0["ret_dates"]
    idx_ret = index_close.reindex(dec[:1].append(ret)).pct_change().iloc[1:].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for r in results:
        ax.plot(ret, (np.cumprod(1 + r.series["port"]) - 1) * 100, lw=1, alpha=0.8, label=f"Det-MASO seed {r.seed}")
    ax.plot(ret, (np.cumprod(1 + s0["bench"]) - 1) * 100, color="gray", lw=1.5, ls="--", label="Equal-weight eligible members")
    ax.plot(ret, (np.cumprod(1 + idx_ret) - 1) * 100, color="black", lw=1.2, label="CSI300 index (buy and hold)")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(f"Top-decile long portfolio ({split})")
    ax.legend(fontsize=7, ncol=2)
    files.append(_save(fig, fig_dir / "cumulative_return.png"))

    w = np.mean([r.series["w"] for r in results], axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ax1.plot(dec, index_close.reindex(dec).to_numpy(), color="black", lw=1)
    ax1.set_ylabel("CSI300 index (scaled level)")
    ax2.stackplot(dec, (w * 100).T, labels=regime_names)
    ax2.set_ylabel("Agent weight (%), seed mean")
    ax2.legend(fontsize=7, loc="upper left", ncol=5)
    fig.suptitle(f"Scheduler weights by decision date ({split}); index is the real SH000300 series")
    files.append(_save(fig, fig_dir / "scheduler_weights.png"))

    a = np.mean([r.series["a"] for r in results], axis=0)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.stackplot(dec, (a * 100).T, labels=["5-day", "20-day", "60-day", "120-day"])
    ax.set_ylabel("Window attention (%), seed mean")
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    ax.set_title(f"Scheduler attention over time windows a_t ({split})")
    files.append(_save(fig, fig_dir / "window_attention.png"))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for r in results:
        ax.plot(dec, r.series["abs_corr"], lw=0.8, alpha=0.8, label=f"seed {r.seed}")
    ax.set_ylabel("Mean |rank corr|, live agents")
    ax.set_title(f"Agent diversity by decision date ({split})")
    ax.legend(fontsize=7, ncol=5)
    files.append(_save(fig, fig_dir / "agent_diversity.png"))

    shap = np.mean([r.shapley for r in results], axis=0)
    vmax = float(np.nanmax(np.abs(shap))) or 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.nan_to_num(shap), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(regime_names)), regime_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(regime_names)), regime_names, fontsize=8)
    for i in range(len(regime_names)):
        for j in range(len(regime_names)):
            if i != j:
                ax.text(j, i, f"{shap[i, j]:+.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title(f"Shapley interaction I(i,j), seed mean ({split})")
    files.append(_save(fig, fig_dir / "shapley.png"))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for r in results:
        epochs = range(1, len(r.series["val_curve"]) + 1)
        line = ax.plot(epochs, [np.nan if x is None else x for x in r.series["val_smoothed"]], lw=1.4, label=f"seed {r.seed} (smoothed)")
        ax.plot(epochs, r.series["val_curve"], lw=0.6, alpha=0.4, color=line[0].get_color())
        ax.axvline(r.series["best_epoch"], color=line[0].get_color(), lw=0.8, ls=":")
    ax.set_xlabel("Epoch (dotted line = restored best epoch)")
    ax.set_ylabel("Validation IC")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Early stopping signal: 3-epoch smoothed validation IC (raw IC faint)")
    files.append(_save(fig, fig_dir / "validation_ic.png"))
    return files


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def evaluate_and_report(cfg: Config, tag: str, seeds: list[int], split: str, lock_record: dict | None = None) -> Path:
    if split == "test" and lock_record is None:
        raise PermissionError("the test period can only be evaluated through eval.lock.open_test_period")
    panel, regimes, splits = load_prepared(cfg)
    raw = cache_dir(cfg)
    calendar = pd.read_parquet(raw / "member.parquet").index
    index_close = pd.read_parquet(raw / "index_close.parquet")["index_close"]
    if split == "valid":
        ic_dates, bt_dates, sub_window = splits.valid_fit, splits.valid, None
    else:
        ic_dates, bt_dates, sub_window = splits.test, splits.test, cfg.splits.sub_window
    run = cfg.run_path / tag
    results = [evaluate_seed(cfg, panel, run / f"seed{s}", ic_dates, bt_dates, calendar, sub_window) for s in seeds]

    means = {
        "ic": _mean_std([r.ic["ic"] for r in results]),
        "icir": _mean_std([r.ic["icir"] for r in results]),
        "sharpe": _mean_std([r.port["sharpe"] for r in results]),
        "ar": _mean_std([r.port["ar"] for r in results]),
        "mdd": _mean_std([r.port["mdd"] for r in results]),
    }
    paper = PAPER[cfg.market]
    q = qualitative(results, panel, splits.train)
    figs = _plots(results, regimes.names, index_close, run / f"figures_{split}", split)

    def fmt(metric: str, value: float) -> str:
        return f"{value:.1%}" if metric in ("ar", "mdd") else f"{value:.3f}"

    lines = [f"# Det-MASO reproduction: {cfg.market}, {split} split", ""]
    lines += [
        f"- Run group `{tag}`, seeds {seeds}, config hash `{cfg.hash()}`, data tag `{cfg.data.data_tag}`",
        f"- Evaluated dates: {bt_dates[0].date()} .. {bt_dates[-1].date()} ({len(bt_dates)} trading days; IC uses {len(ic_dates)})",
        "- Changes from the approved plan: see DEVIATIONS.md",
    ]
    if lock_record is not None:
        lines.append(
            f"- Test-period lock: opening {lock_record['openings']} recorded in locks/{cfg.market}_test.json "
            f"({lock_record['reason']}); code {lock_record['code_sha256']}"
        )
    lines += ["", "## Headline metrics (mean +- std over seeds)", ""]
    rows = []
    for metric, label in [("ic", "IC"), ("icir", "ICIR"), ("sharpe", "Sharpe"), ("ar", "Annual return"), ("mdd", "Max drawdown")]:
        m, s = means[metric]
        pv = paper.get(metric)
        rows.append([label, f"{fmt(metric, m)} +- {fmt(metric, s)}", "-" if pv is None else fmt(metric, pv), band_status(metric, m, pv)])
    lines.append(md_table(["Metric", "Ours", "Paper (Table 5/7)", "Band status"], rows))

    lines += ["", "## Per seed", ""]
    lines.append(md_table(
        ["Seed", "IC", "ICIR", "Sharpe", "AR", "MDD", "Qlib AR", "Qlib IR", "Qlib MDD", "Excess AR", "Excess Sharpe", "Excess MDD", "Best epoch", "Epochs"],
        [
            [r.seed, f"{r.ic['ic']:.4f}", f"{r.ic['icir']:.3f}", f"{r.port['sharpe']:.2f}", f"{r.port['ar']:.1%}", f"{r.port['mdd']:.1%}",
             f"{r.port['qlib_ar']:.1%}", f"{r.port['qlib_ir']:.2f}", f"{r.port['qlib_mdd']:.1%}", f"{r.excess['ar']:.1%}",
             f"{r.excess['sharpe']:.2f}", f"{r.excess['mdd']:.1%}", r.summary["best_epoch"], r.summary["epochs_run"]]
            for r in results
        ],
    ))  # fmt: skip
    lines += [
        "",
        "Headline = compounded annual growth, Sharpe with observed trading days per year, drawdown of the value curve.",
        "Qlib = risk_analysis sum mode with 238 days. Excess = minus the equal-weighted average of the eligible members",
        "(index members that traded on t-1 and t; see DEVIATIONS.md).",
    ]
    if results[0].sub is not None:
        lines += ["", f"## Sub-window {cfg.splits.sub_window[0]} .. {cfg.splits.sub_window[1]} (Figure 4's actual span; comparison only)", ""]
        lines.append(md_table(
            ["Seed", "IC", "ICIR", "Sharpe", "AR", "MDD", "Excess AR", "Excess Sharpe"],
            [[r.seed, f"{r.sub['ic']:.4f}", f"{r.sub['icir']:.3f}", f"{r.sub['sharpe']:.2f}", f"{r.sub['ar']:.1%}", f"{r.sub['mdd']:.1%}",
              f"{r.sub['excess']['ar']:.1%}", f"{r.sub['excess']['sharpe']:.2f}"] for r in results],
        ))  # fmt: skip

    lines += ["", "## Diversity, scheduler and dead agents", ""]
    lines.append(md_table(
        ["Quantity", "Ours (mean over seeds)", "Paper"],
        [
            ["Mean abs pairwise corr, last training epochs (Fig. 3 protocol)", f"{np.mean([r.train_fig3['mean_abs_corr'] for r in results]):.3f}", f"{PAPER_FIG3['mean_abs_corr']} (suspect, see docs/PLAN.md Sec 12)"],
            ["Mean log det Sigma, last training epochs", f"{np.mean([r.train_fig3['logdet'] for r in results]):.3f}", f"{PAPER_FIG3['logdet']} (impossible for 5 agents)"],
            ["Mean live agents per day, last training epochs", f"{np.mean([r.train_fig3['mean_live_agents'] for r in results]):.2f}", "5"],
            [f"Mean abs pairwise rank corr over the {split} period (live agents)", f"{np.mean([r.diversity_abs_corr for r in results]):.3f}", "-"],
            [f"Mean log det over the {split} period (live agents)", f"{np.mean([r.diversity_logdet for r in results]):.3f}", "-"],
            [f"Mean scheduler entropy over the {split} period (max ln 5 = 1.609)", f"{np.mean([r.entropy for r in results]):.3f}", "-"],
            ["Mean agent weights", str(np.round(np.mean([r.series['w'].mean(0) for r in results], axis=0), 3).tolist()), "-"],
        ],
    ))  # fmt: skip
    lines += ["", f"Dead agent-days over the {split} period (agent output constant across stocks; DEVIATIONS 4):", ""]
    lines.append(md_table(["Seed", *regimes.names], [[r.seed, *r.dead_days] for r in results]))

    shap = np.mean([r.shapley for r in results], axis=0)
    shap_sd = np.std([r.shapley for r in results], axis=0, ddof=1) if len(results) > 1 else np.zeros_like(shap)
    names = regimes.names
    lines += ["", "## Shapley interaction I(i,j): mean +- std over seeds", ""]
    lines.append(md_table(
        ["", *names],
        [[names[i], *["-" if i == j else f"{shap[i, j]:+.4f} +- {shap_sd[i, j]:.4f}" for j in range(len(names))]] for i in range(len(names))],
    ))  # fmt: skip
    lines += [
        "",
        f"v(all agents) = {np.mean([r.v[-1] for r in results]):.4f}; single agents: "
        + ", ".join(f"{names[i]} {np.mean([r.v[1 << i] for r in results]):.4f}" for i in range(len(names))),
        f"Coalition-days with a constant score (IC set to 0): {sum(r.shapley_degenerate_days for r in results)}",
    ]

    lines += ["", "## Qualitative checks (docs/PLAN.md Section 10)", ""]
    lines.append(md_table(
        ["Check", "Ours", "Target", "Holds?"],
        [
            ["q1 agents stay different (mean abs corr)", f"{q['q1'][0]:.3f}", q["q1"][1], q["q1_ok"]],
            ["q2 scheduler does not collapse (mean entropy)", f"{q['q2'][0]:.3f}", q["q2"][1], q["q2_ok"]],
            ["q3 Tail-Risk share of training days", f"{q['q3'][0]:.1%}", q["q3"][1], q["q3_ok"]],
            ["q4 I(Low-Vol Bull, Tail-Risk) > 0", f"{q['q4'][0]} of {len(results)} seeds", q["q4"][1], q["q4_ok"]],
            ["q4b ... and it is the largest pair", f"{q['q4_largest']} of {len(results)} seeds", "reported", "-"],
        ],
    ))  # fmt: skip
    if split == "test":
        label, failed = verdict(cfg.market, {k: v[0] for k, v in means.items()}, q)
        lines += ["", f"## Verdict: **{label}**", "", "Rules applied in order (docs/PLAN.md Section 10): floor, reproduced, partially reproduced, inconclusive."]
        if failed:
            lines += ["", "Criteria not met:", ""] + [f"- {f}" for f in failed]
    else:
        lines += ["", "Validation is for development only; the verdict is computed on the test period (Milestone 9)."]
    lines += ["", "## Figures", ""] + [f"![{f}](figures_{split}/{f})" for f in figs]

    path = run / f"report_{split}.md"
    path.write_text("\n".join(lines) + "\n")
    payload = {
        "means": means,
        "qualitative": q,
        "lock": lock_record,
        "per_seed": [
            {
                "seed": r.seed, "ic": r.ic, "port": r.port, "excess": r.excess, "sub": r.sub, "train_fig3": r.train_fig3,
                "entropy": r.entropy, "diversity_abs_corr": r.diversity_abs_corr, "dead_days": r.dead_days,
                "shapley": r.shapley, "v": r.v,
            }
            for r in results
        ],
    }  # fmt: skip
    (run / f"results_{split}.json").write_text(json.dumps(payload, indent=2, default=_json_default))
    return path
