"""Walk-forward evaluation of a run group (`detmaso evaluate --split walkforward`; docs/RULES.md, Rules WL and WT).

Every (fold, seed) model scores its own test window. Per seed the 11 windows join into one continuous daily
series, traded three ways: the paper backtest (gross, no blocking), the same portfolio with its turnover and
costs, and the portfolio that respects blocked trades. Saved: series_wf.parquet (the daily series, input to
`compare --split walkforward`), results_wf.json and report_wf.md (+ figures_wf/). The locked test period is never
read: every fold is checked by detmaso.walkforward.folds.check_fold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config
from detmaso.data.qlib_io import cache_dir
from detmaso.eval import guard, ledger
from detmaso.eval.backtest import top_decile_backtest
from detmaso.eval.report import next_dates
from detmaso.eval.trading import (
    code_space_growth, cost_settings, long_short, panel_growth, simulate_blocked, simulate_unblocked, trade_flags,
)  # fmt: skip
from detmaso.eval.wf_stats import (
    AFTER_COSTS, RESULTS_FILE, SERIES_FILE, WFRun, bootstrap, frame_to_run, mean_std, metric_values, pct, pct_range, seed_metrics,
    write_json,
)  # fmt: skip
from detmaso.losses.stats import daily_spearman
from detmaso.mdtable import md_table
from detmaso.walkforward.folds import mood_summary
from detmaso.walkforward.runner import group_hash, job_dir, job_done, load_folds, select_folds

log = logging.getLogger(__name__)


def _scores(cfg: Config, fs, folds, seeds: list[int], tag: str, calendar: pd.DatetimeIndex) -> tuple[dict, dict]:
    """Per seed, per fold: daily IC, paper backtest, picks and long-short; and per fold the training facts."""
    from detmaso.train.trainer import load_model, predict

    per_seed: dict[int, list[dict]] = {s: [] for s in seeds}
    facts: dict[int, dict] = {}
    top = cfg.backtest.top_frac
    for fold in folds:
        fp = fs.panels[fold.year]
        dates = fold.splits.test
        pos = fp.date_pos(dates)
        in_ic = np.isin(dates, fold.test_ic)
        y, lab = fp.y[pos].numpy(), fp.label_ok[pos].numpy()
        trd, rn, code = fp.tradable[pos].numpy(), fp.r_next[pos].numpy(), fp.code_idx[pos].numpy()
        ret_dates = next_dates(dates, calendar)
        seed_facts = {}
        for s in seeds:
            d = job_dir(cfg, tag, fold.year, s)
            model = load_model(cfg, fp, d)
            out = predict(model, fp, pos, device=next(model.parameters()).device)
            alpha = out["alpha_hat"]
            ic = np.full(len(dates), np.nan)
            ic[in_ic] = daily_spearman(alpha[in_ic], y[in_ic], lab[in_ic])
            port, bench, picks = top_decile_backtest(alpha, trd, rn, top)
            w = np.clip(out["w"], 1e-300, 1.0)
            per_seed[s].append({
                "year": fold.year, "dates": dates, "ret_dates": ret_dates, "in_ic": in_ic, "ic": ic, "port": port, "bench": bench,
                "ls": long_short(alpha, trd, rn, top), "pick_codes": [code[t, p] for t, p in enumerate(picks)],
                "pick_returns": [rn[t, p] for t, p in enumerate(picks)], "code": code, "r_next": rn, "label_ok": lab,
                "entropy": -(w * np.log(w)).sum(axis=1), "w": out["w"],
            })  # fmt: skip
            summary = json.loads((d / "summary.json").read_text())
            curve = np.asarray(summary["val_ic_curve"], dtype=float)
            seed_facts[s] = {"best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
                             "early_stop_ic": summary["best_val_ic_smoothed"], "early_stop_ic_all_epochs": float(curve.mean())}  # fmt: skip
        facts[fold.year] = {**fold.describe(), "seeds": seed_facts}
    return per_seed, facts


def _trade(cfg: Config, chunks: list[dict], flags, n_codes: int) -> dict[str, np.ndarray]:
    """One seed's joined series traded without and with blocked trades."""
    dates = pd.DatetimeIndex(np.concatenate([c["dates"] for c in chunks]))
    picks = [p for c in chunks for p in c["pick_codes"]]
    unblocked = simulate_unblocked(picks, [r for c in chunks for r in c["pick_returns"]], n_codes)
    paper = np.concatenate([c["port"] for c in chunks])
    if not np.array_equal(unblocked.port, paper, equal_nan=True):
        raise AssertionError("the unblocked portfolio does not reproduce the paper backtest")
    if cfg.is_placebo:  # shuffled answers: value holdings with the shuffled next-day returns only
        growth = np.concatenate([panel_growth(c["code"], c["r_next"], c["label_ok"], n_codes) for c in chunks])
    else:
        growth = code_space_growth(flags, dates)
    rows = flags.dates.get_indexer(dates)
    if (rows < 0).any():
        raise KeyError("decision dates outside the loaded calendar")
    blocked = simulate_blocked(picks, growth, flags.can_buy[rows], flags.can_sell[rows])
    return {
        "buy": unblocked.buy, "sell": unblocked.sell, "port_blk": blocked.port, "buy_blk": blocked.buy, "sell_blk": blocked.sell,
        "cash_blk": blocked.cash, "blocked_buys": blocked.blocked_buys, "blocked_sells": blocked.blocked_sells,
    }  # fmt: skip


def evaluate_walkforward(cfg: Config, config_path: str, tag: str, seeds: list[int], years: list[int] | None = None,
                         ledger_path: Path | None = None) -> Path:  # fmt: skip
    if group_hash(cfg, tag) != cfg.hash():
        raise SystemExit(f"run group '{tag}' was trained with config {group_hash(cfg, tag)}, this config is {cfg.hash()}; pass the matching --config")
    fs = load_folds(cfg)
    folds = select_folds(fs.folds, years)
    missing = [f"fold {f.year} seed {s}" for f in folds for s in seeds if not job_done(cfg, tag, f.year, s)]
    if missing:
        raise SystemExit(f"run group '{tag}' has untrained jobs: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    raw = cache_dir(cfg)
    calendar = pd.read_parquet(raw / "member.parquet").index
    flags = trade_flags(*(pd.read_parquet(raw / f"{f}.parquet") for f in ("close", "high", "low")), cfg.trading)
    if flags.codes != fs.panel.code_list:
        raise AssertionError("raw price columns do not match the panel's stock codes")
    per_seed, facts = _scores(cfg, fs, folds, seeds, tag, calendar)

    frames = []
    for s in seeds:
        chunks = per_seed[s]
        traded = _trade(cfg, chunks, flags, len(fs.panel.code_list))
        n_agents = chunks[0]["w"].shape[1]
        frame = pd.DataFrame({
            "seed": s, "year": np.concatenate([np.full(len(c["dates"]), c["year"]) for c in chunks]),
            "decision_date": np.concatenate([c["dates"] for c in chunks]), "ret_date": np.concatenate([c["ret_dates"] for c in chunks]),
            **{k: np.concatenate([c[k] for c in chunks]) for k in ("in_ic", "ic", "port", "bench", "ls", "entropy")},
            **traded, **{f"w{i}": np.concatenate([c["w"][:, i] for c in chunks]) for i in range(n_agents)},
        })  # fmt: skip
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    run_dir = cfg.run_path / tag
    daily.to_parquet(run_dir / SERIES_FILE, index=False)

    moods, prev = {}, None
    for fold in fs.folds:
        moods[fold.year] = mood_summary(fs.regimes[fold.year], fs.state, fold, prev)
        prev = (fs.regimes[fold.year], fold)
    info = {"folds": {y: facts[y] for y in sorted(facts)}, "moods": {f.year: moods[f.year] for f in folds}, "mood_names": fs.regimes[folds[0].year].names}
    run = frame_to_run(daily, cfg, config_path, tag, info)
    complete = run.years == cfg.walkforward.years and run.seeds == sorted(cfg.train.seeds)
    sm = seed_metrics(run)
    boot = bootstrap([run])[run.name]
    if complete and ledger_path is not None:
        if not ledger.read(ledger_path)["config_hash"].eq(cfg.hash()).any():
            raise SystemExit(f"config {cfg.hash()} is not in the try ledger; it must be registered by `train`")
        ledger.record_wf_metrics(ledger_path, cfg, ledger_wf_metrics(run, sm))
        log.info("walk-forward metrics recorded in the try ledger %s", ledger_path)
    elif not complete:
        log.warning("evaluated folds %s and seeds %s only: not recorded in the try ledger, and `compare` will refuse this group", run.years, run.seeds)
    deflated = run_deflation(run, sm, boot, ledger_path) if (ledger_path is not None and ledger_path.exists() and not cfg.is_placebo) else None
    from detmaso.eval.wf_plots import run_figures

    figs = run_figures(run, sm, boot, run_dir / "figures_wf")
    summary = run_summary(run, sm, boot)
    write_json(run_dir / RESULTS_FILE, {**info, "complete": complete, "summary": summary, "deflated": deflated,
                                        "config_hash": cfg.hash(), "data_hash": cfg.data_hash(), "seeds": run.seeds, "years": run.years})  # fmt: skip
    (run_dir / "report_wf.md").write_text(run_report(run, sm, boot, summary, deflated, complete, figs))
    return run_dir / "report_wf.md"


# Statistics ----------------------------------------------------------------------------------------------
def ledger_wf_metrics(run: WFRun, sm: dict) -> dict:
    net = run.stitched(run.realistic)
    bench = run.stitched(lambda y: run.cols[y]["bench"])
    return {
        "wf_years": len(run.years),
        "wf_ic": float(sm["ic_avg"].mean()),
        "wf_ic_sd_seeds": float(sm["ic_avg"].std(ddof=1)) if len(sm["ic_avg"]) > 1 else 0.0,
        "wf_sharpe_net": float(metric_values(sm["joined"][AFTER_COSTS], "sharpe").mean()),
        "wf_excess_sr_daily_net": guard.excess_sharpe_inputs(net, bench)["sr"],
    }


def run_deflation(run: WFRun, sm: dict, boot: dict, ledger_path: Path) -> dict:
    """Try number, deflated Sharpe (daily after-cost excess return) and deflated IC (docs/RULES.md)."""
    d = ledger.deflation_inputs(ledger_path)
    x = guard.excess_sharpe_inputs(run.stitched(run.realistic), run.stitched(lambda y: run.cols[y]["bench"]))
    dsr = guard.deflated_sharpe(x["sr"], x["n_obs"], x["skew"], x["kurt"], d["sr_sd"], d["n_tries"])
    dic = guard.deflated_ic(float(sm["ic_avg"].mean()), float(np.std(boot["ic_avg"], ddof=1)), d["ic_sd"], d["n_tries"])
    return {"try_no": ledger.try_number(ledger_path, run.cfg.hash()), **d, **x, **dsr, **dic}


def run_summary(run: WFRun, sm: dict, boot: dict) -> dict:
    """Headline numbers with their seed spread and 95% ranges (the JSON behind the report)."""
    j = sm["joined"]
    out = {
        "ic_avg": {**mean_std(sm["ic_avg"]), "range": pct_range(boot["ic_avg"]), "se": float(np.std(boot["ic_avg"], ddof=1))},
        "sharpe_gross": {**mean_std(metric_values(j["gross_unblocked"], "sharpe")), "range": pct_range(boot["sharpe_gross"])},
        "sharpe_net": {**mean_std(metric_values(j[AFTER_COSTS], "sharpe")), "range": pct_range(boot["sharpe_net"])},
        "ar_net": mean_std(metric_values(j[AFTER_COSTS], "ar")),
        "mdd_net": mean_std(metric_values(j[AFTER_COSTS], "mdd")),
        "excess_ar_net": mean_std(metric_values(j["excess_realistic"], "ar")),
        "turnover": mean_std(sm["turnover"]),
        "turnover_blk": mean_std(sm["turnover_blk"]),
        "ls_mean": mean_std(sm["ls_mean"]),
        "ls_sharpe": mean_std(sm["ls_sharpe"]),
        "blocked_per_day": mean_std(sm["blocked_per_day"]),
        "cash_blk": mean_std(sm["cash_blk"]),
        "costs": {k: {"sharpe": mean_std(metric_values(v, "sharpe")), "ar": mean_std(metric_values(v, "ar"))} for k, v in j.items()},
        "yearly": {},
    }
    for i, y in enumerate(run.years):
        yr = sm["yearly"][y]
        out["yearly"][y] = {
            "days": len(run.dec[y]), "ic_days": int(run.in_ic[y].sum()),
            "ic": {**mean_std(sm["ic_year"][:, i]), "range": pct_range(boot["ic_year"][i])},
            "icir": mean_std(sm["icir_year"][:, i]),
            "ls_mean": mean_std(yr["ls_mean"]), "ls_sharpe": mean_std(yr["ls_sharpe"]),
            "sharpe_gross": mean_std(metric_values(yr["gross"], "sharpe")), "sharpe_net": mean_std(metric_values(yr["realistic"], "sharpe")),
            "excess_ar_net": mean_std(metric_values(yr["excess_realistic"], "ar")),
            "turnover": mean_std(yr["turnover"]), "turnover_blk": mean_std(yr["turnover_blk"]), "blocked_per_day": mean_std(yr["blocked_per_day"]),
        }  # fmt: skip
    return out


# Report --------------------------------------------------------------------------------------------------
def _pm(d: dict, fmt: str) -> str:
    return f"{d['mean']:{fmt}} +- {d['std']:{fmt}}"


def _rng(r, fmt: str) -> str:
    return f"[{r[0]:+{fmt}}, {r[1]:+{fmt}}]"


def run_report(run: WFRun, sm: dict, boot: dict, s: dict, deflated: dict | None, complete: bool, figs: list[str]) -> str:
    cfg = run.cfg
    tc = cfg.trading
    lines = [f"# Walk-forward evaluation: `{run.name}` ({cfg.market})", ""]
    lines += [
        f"- Run group `{run.tag}`, config `{run.config_path}` (hash `{cfg.hash()}`), data tag `{cfg.data.data_tag}`, data hash `{cfg.data_hash()}`.",
        f"- {len(run.years)} folds (test years {run.years[0]}-{run.years[-1]}), seeds {run.seeds}: {run.n_days} test days from "
        f"{run.start.date()} to {run.end.date()}. Each fold trains from {cfg.splits.train[0]} to two years before its test year and "
        "early-stops on the year before (docs/RULES.md, Rule WL). The locked test period is never used.",
        "- Every number is the mean +- std over seeds; 95% ranges come from the paired, stratified 20-day block resampling (Rule WK).",
        f"- Costs (Rule WT): flat {', '.join(f'{b:g}' for b in tc.flat_bps)} bps per buy and per sell, and realistic: buys "
        f"{tc.fee_bps + tc.slippage_bps:g} bps, sells {tc.fee_bps + tc.slippage_bps:g} bps + stamp duty "
        f"({'; '.join(f'{b:g} bps from {d}' for d, b in tc.stamp_duty_bps)}). \"After costs\" = realistic costs "
        "with blocked trades (price limits, suspensions).",
    ]
    if not complete:
        lines.append("- **Incomplete**: not every fold and configured seed was evaluated. Not recorded in the try ledger; `compare` refuses it.")
    if cfg.is_placebo:
        lines.append(f"- **Placebo**: trained and evaluated on shuffled answers (seed {cfg.placebo.shuffle_seed}); holdings are valued with the shuffled next-day returns. There is nothing real to find.")

    lines += ["", "## Headline (all test years joined)", ""]
    rows = [
        ["Average IC (mean of the yearly ICs)", _pm(s["ic_avg"], ".4f"), _rng(s["ic_avg"]["range"], ".4f")],
        ["Sharpe, paper backtest (no costs, no blocking)", _pm(s["sharpe_gross"], ".2f"), _rng(s["sharpe_gross"]["range"], ".2f")],
        ["Sharpe after costs", _pm(s["sharpe_net"], ".2f"), _rng(s["sharpe_net"]["range"], ".2f")],
        ["Annual return after costs", f"{s['ar_net']['mean']:.1%} +- {s['ar_net']['std']:.1%}", "-"],
        ["Max drawdown after costs", f"{s['mdd_net']['mean']:.1%} +- {s['mdd_net']['std']:.1%}", "-"],
        ["Excess annual return after costs (vs equal-weighted eligible stocks)", f"{s['excess_ar_net']['mean']:.1%} +- {s['excess_ar_net']['std']:.1%}", "-"],
        ["Daily turnover, one-way (without / with blocking)", f"{s['turnover']['mean']:.1%} +- {s['turnover']['std']:.1%} / {s['turnover_blk']['mean']:.1%}", "-"],
        ["Long-short spread, gross (bps per day; Sharpe)", f"{1e4 * s['ls_mean']['mean']:.1f} +- {1e4 * s['ls_mean']['std']:.1f}; {_pm(s['ls_sharpe'], '.2f')}", "-"],
        ["Blocked trades per day; average cash share (blocking)", f"{s['blocked_per_day']['mean']:.2f}; {s['cash_blk']['mean']:.2%}", "-"],
    ]
    lines.append(md_table(["Metric", "Mean +- std over seeds", "95% range"], rows))
    if deflated is not None:
        try_no = "-" if deflated["try_no"] is None else str(deflated["try_no"])
        lines += [
            "",
            f"Try #{try_no} of {deflated['n_tries']} counted tries. Deflated Sharpe {pct(deflated['dsr'])} (daily after-cost excess return), "
            f"deflated IC {pct(deflated['dic'])}: the probability that the true value beats the best of {deflated['n_tries']} tries of pure luck "
            f"(spread of results from {deflated['source']}). Above about 95% = likely real.",
        ]

    lines += ["", "## By test year", ""]
    rows = []
    for y in run.years:
        v = s["yearly"][y]
        rows.append([str(y), v["days"], _pm(v["ic"], ".4f"), _rng(v["ic"]["range"], ".4f"), _pm(v["icir"], ".2f"),
                     f"{1e4 * v['ls_mean']['mean']:.1f} +- {1e4 * v['ls_mean']['std']:.1f}", _pm(v["sharpe_gross"], ".2f"), _pm(v["sharpe_net"], ".2f"),
                     f"{v['excess_ar_net']['mean']:+.1%} +- {v['excess_ar_net']['std']:.1%}", f"{v['turnover']['mean']:.1%}", f"{v['blocked_per_day']['mean']:.2f}"])  # fmt: skip
    lines.append(md_table(["Year", "Days", "IC", "IC 95% range", "ICIR", "Long-short (bps/day)", "Sharpe, paper backtest", "Sharpe after costs",
                           "Excess return after costs", "Turnover/day", "Blocked trades/day"], rows))  # fmt: skip

    lines += ["", "## Costs and blocked trades (all test years joined)", ""]
    rows = []
    for name in ["gross"] + [st.name for st in cost_settings(tc)]:
        cells = [name if name != "gross" else "none (paper backtest)"]
        for mode in ("unblocked", "blocked"):
            c = s["costs"][f"{name}_{mode}"]
            cells += [_pm(c["sharpe"], ".2f"), f"{c['ar']['mean']:.1%} +- {c['ar']['std']:.1%}"]
        rows.append(cells)
    lines.append(md_table(["Costs", "Sharpe, no blocking", "Annual return, no blocking", "Sharpe, with blocking", "Annual return, with blocking"], rows))

    names = run.info.get("mood_names", [])
    lines += ["", "## Market moods per fold (refit on each fold's pre-test days; rule A16 names)", ""]
    lines.append(f"Moods: {', '.join(f'{i}: {n}' for i, n in enumerate(names))}. Shares are in %. Centre shift = distance between a mood's centre and the previous fold's, in standardized units; relabelled = share of the previous fold's fit days whose mood changed.")
    lines.append("")
    rows = []
    for y in run.years:
        m = run.info["moods"][y] if y in run.info["moods"] else run.info["moods"][str(y)]
        rows.append([str(y), f"{m['fit'][0]} .. {m['fit'][1]}", m["best_k"], ", ".join(f"{100 * x:.1f}" for x in m["train_shares"]),
                     ", ".join(f"{100 * x:.1f}" for x in m["test_shares"]), "-" if m["centre_shift"] is None else ", ".join(f"{x:.2f}" for x in m["centre_shift"]),
                     "-" if m["relabelled"] is None else f"{m['relabelled']:.1%}"])  # fmt: skip
    lines.append(md_table(["Fold", "Fit days", "Best K (silhouette)", "Training-day shares", "Test-day shares", "Centre shift", "Relabelled"], rows))

    lines += ["", "## Early stopping per fold", ""]
    rows = []
    for y in run.years:
        f = run.info["folds"][y] if y in run.info["folds"] else run.info["folds"][str(y)]
        seeds = f["seeds"]
        es = [v["early_stop_ic"] for v in seeds.values()]
        alle = [v["early_stop_ic_all_epochs"] for v in seeds.values()]
        rows.append([str(y), f"{f['train_fit'][0]} .. {f['train_fit'][1]} ({f['train_fit'][2]})", f"{f['early_stop'][0]} .. {f['early_stop'][1]}",
                     " ".join(str(v["best_epoch"]) for v in seeds.values()), f"{np.mean([v['epochs_run'] for v in seeds.values()]):.1f}",
                     _pm(mean_std(es), ".4f"), f"{np.mean(alle):.4f}"])  # fmt: skip
    lines.append(md_table(["Fold", "Training dates (trained on)", "Early-stopping year", "Best epoch per seed", "Epochs run", "Early-stop IC at the picked epoch", "... averaged over every epoch"], rows))
    lines += ["", "The early-stopping IC is the only number used to pick an epoch; the test year plays no part in it."]
    if figs:
        lines += ["", "## Figures", ""] + [f"![{f}](figures_wf/{f})" for f in figs]
    return "\n".join(lines) + "\n"
