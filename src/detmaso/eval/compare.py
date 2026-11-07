"""Side-by-side comparison of trained variants on the validation period (Step 1 in ROADMAP.md; Step 1 decision D3).

Each run group must already be evaluated with ``detmaso evaluate --split valid``, which saves
runs/<market>/<tag>/series_valid.parquet (per seed and decision date: daily IC, portfolio and
benchmark returns, scheduler weights) and results_valid.json. The test period is never used.

Statistics
- Per seed: IC, ICIR, Sharpe, max drawdown, annual return and excess-return metrics, recomputed from
  the saved series with the report's own functions and checked against the report's numbers.
- Per variant: mean +- std over seeds, and the 95% range of the seed-mean IC, ICIR and Sharpe.
- Reference advantage = reference minus variant (for drawdown: variant minus reference, so positive
  always means the reference did better), with a two-level paired block bootstrap. Each draw:
  (1) redraws the evaluation days as circular blocks of ``block`` trading days (20 = the label
  horizon: neighbouring days' 20-day labels overlap, so single days are not independent), one draw
  shared by every variant, which pairs the comparison by day; (2) redraws each variant's seeds with
  replacement (training randomness; seeds are not paired across variants), from a random stream of its
  own (seeded by the RNG seed and the variant's name, so a variant's draws do not depend on which other
  variants are compared, or in which order). The seed-mean metric is recomputed and the advantage
  recorded; the 95% range is the 2.5-97.5% percentiles of the draws.
- Rule fixed before seeing any results (Step 1 decision D3, recorded in
  results/csi300/single_vs_ensemble/README.md): the reference is better if the 95% range of its
  advantage lies entirely above 0, the variant is better if it lies entirely below 0, otherwise there
  is no detectable difference. IC is the primary metric and Sharpe the second opinion. Max drawdown
  is not bootstrapped (block resampling breaks the day order it depends on).
- Early-stopping check: every model's epoch was picked by the same validation IC compared here,
  which flatters runs whose validation IC swings from epoch to epoch. Next to the picked epoch's IC,
  the report shows the validation IC averaged over every epoch trained (no picking).
- Guards (CLAUDE.md): with the try ledger, every variant gets its try number, a deflated Sharpe ratio
  (on its daily excess return) and a deflated IC (eval.guard). Placebo and real runs are never mixed.
- ``rule="scheduler_balance"`` applies docs/RULES.md Rules S, K (and P for placebo groups) to a sweep of
  load-balancing strengths, using each run group's training-day usage (``detmaso usage``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config
from detmaso.data.features import FEATURE_NAMES
from detmaso.eval import guard, ledger
from detmaso.eval.metrics import ic_summary, portfolio_metrics
from detmaso.mdtable import md_table
from detmaso.model.factory import build_model

BOOT_METRICS = ("ic", "icir", "sharpe")
N_BOOT, BLOCK, RNG_SEED = 10_000, 20, 0  # Step 1 decision D3, fixed before seeing results
REFERENCE_BETTER, VARIANT_BETTER, NO_DIFFERENCE = "reference better", "variant better", "no detectable difference"
RULES = ("scheduler_balance",)


class CompareError(RuntimeError):
    pass


@dataclass
class Variant:
    name: str  # cfg.variant
    tag: str
    config_path: str
    cfg: Config
    seeds: list[int]
    ic: np.ndarray  # [S, T_ic] daily IC on the IC dates
    port: np.ndarray  # [S, T] daily portfolio returns
    bench: np.ndarray  # [S, T] daily equal-weighted eligible-universe returns
    ic_dates: pd.DatetimeIndex
    decision_dates: pd.DatetimeIndex
    ret_dates: pd.DatetimeIndex
    per_seed: list[dict]
    agents: dict | None  # scheduler and diversity summaries (multi-agent models only)
    n_weights: int
    dead_days_valid: list[list[int]] | None = field(default=None)  # per seed, per agent

    @property
    def run(self) -> Path:
        return self.cfg.run_path / self.tag


def _seed_frames(series: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(s): g.sort_values("decision_date").reset_index(drop=True) for s, g in series.groupby("seed", sort=True)}


def seed_metrics(ic: np.ndarray, port: np.ndarray, bench: np.ndarray, start, end, qlib_days: int) -> dict:
    """The report's per-seed metrics from one seed's daily series (eval.report.evaluate_seed)."""
    return {"ic": ic_summary(ic), "port": portfolio_metrics(port, start, end, qlib_days), "excess": portfolio_metrics(port - bench, start, end, qlib_days)}


def _effective(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[w > 0] / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def load_variant(cfg: Config, config_path: str, tag: str, split: str = "valid") -> Variant:
    if split != "valid":
        raise CompareError("comparisons use the validation period only")
    run = cfg.run_path / tag
    saved = run / "config.json"
    if not saved.exists():
        raise CompareError(f"run group {run} has no config.json; train it first")
    if json.loads(saved.read_text())["hash"] != cfg.hash():
        raise CompareError(f"run group '{tag}' was not trained with {config_path} (config hash mismatch)")
    series_path, results_path = run / f"series_{split}.parquet", run / f"results_{split}.json"
    if not series_path.exists() or not results_path.exists():
        raise CompareError(f"run group '{tag}' has no saved {split} evaluation; run `detmaso evaluate --split {split}` first")
    frames = _seed_frames(pd.read_parquet(series_path))
    report = json.loads(results_path.read_text())
    seeds = [int(p["seed"]) for p in report["per_seed"]]
    if sorted(frames) != sorted(seeds):
        raise CompareError(f"run group '{tag}': series seeds {sorted(frames)} differ from report seeds {seeds}")
    first = frames[seeds[0]]
    for s in seeds:
        if not first["decision_date"].equals(frames[s]["decision_date"]) or not first["in_ic"].equals(frames[s]["in_ic"]):
            raise CompareError(f"run group '{tag}': seeds were evaluated on different dates")
    decision_dates = pd.DatetimeIndex(first["decision_date"])
    ret_dates = pd.DatetimeIndex(first["ret_date"])
    in_ic = first["in_ic"].to_numpy(bool)
    ic = np.stack([frames[s]["ic"].to_numpy(float)[in_ic] for s in seeds])
    port = np.stack([frames[s]["port"].to_numpy(float) for s in seeds])
    bench = np.stack([frames[s]["bench"].to_numpy(float) for s in seeds])

    per_seed = []
    for k, s in enumerate(seeds):
        m = seed_metrics(ic[k], port[k], bench[k], decision_dates[0], ret_dates[-1], cfg.backtest.qlib_days)
        saved_seed = report["per_seed"][k]
        for section, key in (("ic", "ic"), ("ic", "icir"), ("port", "sharpe"), ("port", "ar"), ("port", "mdd"), ("excess", "ar")):
            if not np.isclose(m[section][key], saved_seed[section][key], rtol=1e-9, atol=1e-12):
                raise CompareError(f"run group '{tag}' seed {s}: recomputed {section}.{key} {m[section][key]} != report {saved_seed[section][key]}")
        summary = json.loads((run / f"seed{s}" / "summary.json").read_text())
        curve = np.asarray(summary["val_ic_curve"], dtype=float)
        per_seed.append({"seed": s, **m, "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
                         "val_ic_all_epochs": float(curve.mean())})  # fmt: skip

    w_cols = sorted((c for c in first.columns if c.startswith("w") and c[1:].isdigit()), key=lambda c: int(c[1:]))
    agents = None
    if len(w_cols) > 1:
        entropy = np.stack([frames[s]["entropy"].to_numpy(float) for s in seeds])
        agents = {
            "n_agents": len(w_cols),
            "mean_abs_corr": float(np.mean([np.nanmean(frames[s]["abs_corr"].to_numpy(float)) for s in seeds])),
            "entropy": float(entropy.mean()),
            "effective_agents_per_day": float(np.exp(entropy).mean()),
            "effective_agents_over_time": float(np.mean([_effective(frames[s][w_cols].to_numpy(float).mean(axis=0)) for s in seeds])),
            "mean_weights": np.mean([frames[s][w_cols].to_numpy(float).mean(axis=0) for s in seeds], axis=0).round(3).tolist(),
        }
    dead_valid = [p.get("dead_days") for p in report["per_seed"]]
    n_weights = sum(p.numel() for p in build_model(cfg, d_in=len(FEATURE_NAMES)).parameters())
    return Variant(cfg.variant, tag, config_path, cfg, seeds, ic, port, bench, decision_dates[in_ic], decision_dates, ret_dates,
                   per_seed, agents, n_weights, dead_valid if all(d is not None for d in dead_valid) else None)  # fmt: skip


def check_comparable(variants: list[Variant]) -> None:
    ref = variants[0]
    for v in variants[1:]:
        if v.cfg.placebo != ref.cfg.placebo:
            raise CompareError(f"'{v.tag}' and '{ref.tag}' differ in placebo settings: placebo and real runs are never compared")
        if v.cfg.data_hash() != ref.cfg.data_hash():
            raise CompareError(f"'{v.tag}' and '{ref.tag}' use different data, splits or regimes (data hash mismatch)")
        if v.cfg.backtest != ref.cfg.backtest:
            raise CompareError(f"'{v.tag}' and '{ref.tag}' use different backtest settings")
        if not (v.decision_dates.equals(ref.decision_dates) and v.ic_dates.equals(ref.ic_dates)):
            raise CompareError(f"'{v.tag}' and '{ref.tag}' were evaluated on different dates")
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise CompareError(f"variant names must be unique, got {names}")


def block_indices(rng: np.random.Generator, n_days: int, block: int, n_boot: int) -> np.ndarray:
    """[n_boot, n_days] day indices: circular blocks of ``block`` consecutive days, uniform starts."""
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_days
    return idx.reshape(n_boot, -1)[:, :n_days]


def variant_rng(rng_seed: int, name: str) -> np.random.Generator:
    """A variant's own random stream for redrawing its seeds (independent of the other variants)."""
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng([rng_seed, key])


def _nanstd(x: np.ndarray, axis: int) -> np.ndarray:
    return np.nanstd(x, axis=axis, ddof=1)


def _per_seed_stat(metric: str, x: np.ndarray, days_per_year: float) -> np.ndarray:
    """Metric over the last axis (days): x is [S, ..., T]."""
    if metric == "ic":
        return np.nanmean(x, axis=-1)
    if metric == "icir":
        return np.nanmean(x, axis=-1) / _nanstd(x, axis=-1)
    if metric == "sharpe":
        return np.nanmean(x, axis=-1) / _nanstd(x, axis=-1) * np.sqrt(days_per_year)
    raise ValueError(metric)


def bootstrap_advantage(variants: list[Variant], n_boot: int, block: int, rng_seed: int, chunk: int = 1000) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap draws of the seed-mean metric for every variant: {variant name: {metric: [n_boot]}}."""
    rng = np.random.default_rng(rng_seed)
    n_ic, n_ret = variants[0].ic.shape[1], variants[0].port.shape[1]
    years = (variants[0].ret_dates[-1] - variants[0].decision_dates[0]).days / 365.25
    days_per_year = n_ret / years
    ic_idx = block_indices(rng, n_ic, block, n_boot)
    ret_idx = block_indices(rng, n_ret, block, n_boot)
    seed_idx = {v.name: variant_rng(rng_seed, v.name).integers(0, len(v.seeds), size=(n_boot, len(v.seeds))) for v in variants}
    draws = {v.name: {m: np.empty(n_boot) for m in BOOT_METRICS} for v in variants}
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        for v in variants:
            for metric in BOOT_METRICS:
                x, idx = (v.ic, ic_idx) if metric in ("ic", "icir") else (v.port, ret_idx)
                per_seed = _per_seed_stat(metric, x[:, idx[lo:hi]], days_per_year)  # [S, b]
                picked = np.take_along_axis(per_seed.T, seed_idx[v.name][lo:hi], axis=1)  # [b, S]
                draws[v.name][metric][lo:hi] = picked.mean(axis=1)
    return draws


def rule(ci_low: float, ci_high: float) -> str:
    """Step 1 decision D3, fixed before seeing results."""
    if ci_low > 0:
        return REFERENCE_BETTER
    if ci_high < 0:
        return VARIANT_BETTER
    return NO_DIFFERENCE


def _seed_values(v: Variant, metric: str) -> np.ndarray:
    section = {"ic": "ic", "icir": "ic", "sharpe": "port", "ar": "port", "mdd": "port"}[metric]
    return np.array([p[section][metric] for p in v.per_seed])


def summarize(
    variants: list[Variant], ref_index: int, n_boot: int = N_BOOT, block: int = BLOCK, rng_seed: int = RNG_SEED, draws: dict | None = None
) -> dict:
    ref = variants[ref_index]
    if draws is None:
        draws = bootstrap_advantage(variants, n_boot, block, rng_seed)
    out = {"variants": {}, "advantage": {}}
    for v in variants:
        stats = {}
        for metric in ("ic", "icir", "sharpe", "mdd", "ar"):
            vals = _seed_values(v, metric)
            stats[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "min": float(vals.min()), "max": float(vals.max())}
            if metric in BOOT_METRICS:  # the 95% range of the seed-mean level itself, and its standard error
                lo, hi = np.percentile(draws[v.name][metric], [2.5, 97.5])
                stats[metric].update(ci_low=float(lo), ci_high=float(hi), se=float(np.std(draws[v.name][metric], ddof=1)))
        for metric in ("ar", "sharpe", "mdd"):
            vals = np.array([p["excess"][metric] for p in v.per_seed])
            stats[f"excess_{metric}"] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        all_epochs = np.array([p["val_ic_all_epochs"] for p in v.per_seed])
        stats["val_ic_all_epochs"] = {"mean": float(all_epochs.mean()), "std": float(all_epochs.std(ddof=1)) if len(all_epochs) > 1 else 0.0}
        stats["epochs_run"] = float(np.mean([p["epochs_run"] for p in v.per_seed]))
        out["variants"][v.name] = stats
    for v in variants:
        if v is ref:
            continue
        adv = {}
        for metric in BOOT_METRICS:
            observed = float(_seed_values(ref, metric).mean() - _seed_values(v, metric).mean())
            d = draws[ref.name][metric] - draws[v.name][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            adv[metric] = {"observed": observed, "ci_low": float(lo), "ci_high": float(hi), "share_positive": float(np.mean(d > 0)), "verdict": rule(lo, hi)}
        adv["mdd"] = {"observed": float(_seed_values(v, "mdd").mean() - _seed_values(ref, "mdd").mean())}
        out["advantage"][v.name] = adv
    return out


def deflation(variants: list[Variant], stats: dict, ledger_path: Path) -> dict:
    """Try numbers, deflated Sharpe (daily excess returns) and deflated IC, from the try ledger."""
    tries = ledger.tries(ledger_path)
    n = len(tries)
    sr_trials = tries["excess_sr_daily"].dropna().to_numpy(float)
    ic_trials = tries["ic"].dropna().to_numpy(float)
    sr_sd = float(np.std(sr_trials, ddof=1)) if len(sr_trials) > 1 else float("nan")
    ic_sd = float(np.std(ic_trials, ddof=1)) if len(ic_trials) > 1 else float("nan")
    out = {"ledger": str(ledger_path), "n_tries": n, "tries_with_metrics": int(len(sr_trials)), "sr_sd_trials": sr_sd,
           "ic_sd_trials": ic_sd, "expected_max_z": guard.expected_max_z(n), "variants": {}}  # fmt: skip
    for v in variants:
        x = guard.excess_sharpe_inputs(v.port, v.bench)
        dsr = guard.deflated_sharpe(x["sr"], x["n_obs"], x["skew"], x["kurt"], sr_sd, n)
        ic = stats["variants"][v.name]["ic"]
        dic = guard.deflated_ic(ic["mean"], ic["se"], ic_sd, n)
        out["variants"][v.name] = {"try_no": ledger.try_number(ledger_path, v.cfg.hash()), **x, **dsr, **dic}
    return out


def training_usage(v: Variant) -> dict | None:
    """Training-day scheduler summary (needs `detmaso usage`), with the variant's own target kind."""
    from detmaso.eval.scheduler_diag import USAGE_FILE, summarize as usage_summary

    if v.agents is None or not (v.run / USAGE_FILE).exists():
        return None
    kind = v.cfg.loss.balance_target if v.cfg.loss.c_bal > 0 else "regime"
    return usage_summary(v.cfg, v.tag, kind)


def apply_scheduler_balance_rules(variants: list[Variant], ref: Variant, stats: dict, draws: dict) -> dict:
    """docs/RULES.md Rules M, S, K (and P when every variant is a placebo run)."""
    from detmaso.eval.scheduler_diag import USAGE_FILE, summarize as usage_summary

    sweep = sorted((v for v in variants if v.cfg.loss.c_bal > 0), key=lambda v: v.cfg.loss.c_bal)
    if not sweep:
        raise CompareError("rule scheduler_balance needs run groups with loss.c_bal > 0")
    kinds = {v.cfg.loss.balance_target for v in sweep}
    if len(kinds) != 1:
        raise CompareError(f"the strengths use different balance targets {sorted(kinds)}")
    kind = kinds.pop()
    missing = [v.tag for v in sweep + [ref] if v.agents is not None and not (v.run / USAGE_FILE).exists()]
    if missing:
        raise CompareError(f"run `detmaso usage` first for {missing}")
    diag = {v.name: usage_summary(v.cfg, v.tag, kind) for v in sweep}
    m_by = {v.cfg.loss.c_bal: diag[v.name]["rule_m"] for v in sweep}
    s = guard.rule_s(m_by)
    out = {"target_kind": kind, "target": diag[sweep[0].name]["target"], "strengths": [v.cfg.loss.c_bal for v in sweep],
           "rule_m": {v.name: _rule_m_brief(diag[v.name]) for v in sweep}, "rule_s": s, "rule_k": None, "outcome": s["outcome"],
           "c_star_variant": None}  # fmt: skip
    k = None
    if s["c_star"] is not None:
        cand = next(v for v in sweep if v.cfg.loss.c_bal == s["c_star"])
        adv = stats["advantage"][cand.name]
        nxt = sweep[sweep.index(cand) + 1] if sweep.index(cand) + 1 < len(sweep) else None
        next_info = None if nxt is None else {
            "strength": nxt.cfg.loss.c_bal, "variant": nxt.name, "passes_m": m_by[nxt.cfg.loss.c_bal]["pass"],
            "mean_ic": stats["variants"][nxt.name]["ic"]["mean"],
        }  # fmt: skip
        k = guard.rule_k(
            -adv["ic"]["ci_high"], -adv["ic"]["ci_low"],
            {p["seed"]: p["ic"]["ic"] for p in cand.per_seed}, {p["seed"]: p["ic"]["ic"] for p in ref.per_seed},
            next_info, stats["variants"][ref.name]["ic"]["mean"],
        )  # fmt: skip
        k.update(
            ic_diff=-adv["ic"]["observed"], ic_diff_low=-adv["ic"]["ci_high"], ic_diff_high=-adv["ic"]["ci_low"],
            sharpe_diff=-adv["sharpe"]["observed"], sharpe_diff_low=-adv["sharpe"]["ci_high"], sharpe_diff_high=-adv["sharpe"]["ci_low"],
        )  # fmt: skip
        sharpe_verdict = guard.BETTER if k["sharpe_diff_low"] > 0 else guard.WORSE if k["sharpe_diff_high"] < 0 else guard.NO_DIFFERENCE
        k["sharpe_second_opinion"] = sharpe_verdict
        out.update(rule_k=k, outcome=guard.outcome(True, k["verdict"]), c_star_variant=cand.name)
    if all(v.cfg.is_placebo for v in variants):
        pooled = np.mean([draws[v.name]["ic"] for v in variants], axis=0)
        lo, hi = np.percentile(pooled, [2.5, 97.5])
        chosen = None
        if k is not None:
            chosen = {"ic_low": stats["variants"][out["c_star_variant"]]["ic"]["ci_low"], "k_verdict": k["verdict"]}
        p = guard.rule_p(float(lo), chosen)
        p.update(pooled_ic=float(np.mean([stats["variants"][v.name]["ic"]["mean"] for v in variants])), pooled_low=float(lo), pooled_high=float(hi),
                 pooled_se=float(np.std(pooled, ddof=1)), groups=[v.name for v in variants])  # fmt: skip
        out["rule_p"] = p
    out["diagnostics"] = diag
    return out


def _rule_m_brief(d: dict) -> dict:
    """Rule M summary for the markdown: pass flags and the binding usage / target ratio."""
    target = np.asarray(d["target"])
    ratios = [
        float(np.min(np.asarray(c["usage"])[target > 0] / target[target > 0]))
        for s in d["seeds"].values() for ck, c in s.items() if ck in ("best", "last")
    ]  # fmt: skip
    m = d["rule_m"]
    return {"m0": m["m0"], "m1": m["m1"], "m2": m["m2"], "pass": m["pass"], "lowest_usage_over_target": min(ratios) if ratios else float("nan")}


def _fmt(metric: str, x: float) -> str:
    return f"{x:.1%}" if metric in ("ar", "mdd", "excess_ar", "excess_mdd") else (f"{x:.2f}" if "sharpe" in metric else f"{x:.3f}" if metric == "icir" else f"{x:.4f}")


def _signed(metric: str, x: float) -> str:
    return ("+" if x >= 0 else "") + _fmt(metric, x)


def _pct_list(xs) -> str:
    return ", ".join(f"{100 * x:.1f}" for x in xs)


def _yes(flag: bool) -> str:
    return "yes" if flag else "**no**"


def write_markdown(variants: list[Variant], ref: Variant, stats: dict, figs: list[str], meta: dict, deflated: dict | None = None,
                   usage: dict | None = None, rules: dict | None = None) -> str:  # fmt: skip
    v0 = variants[0]
    lines = [f"# Variant comparison: {v0.cfg.market}, validation split", ""]
    lines += [
        f"- Reference: `{ref.name}` (run group `{ref.tag}`). Advantage = reference minus variant (drawdown: variant minus reference), so a positive number means the reference did better.",
        f"- Evaluated dates: {v0.decision_dates[0].date()} .. {v0.decision_dates[-1].date()} ({len(v0.decision_dates)} trading days; IC uses {len(v0.ic_dates)}). The test period is not used.",
        f"- Data tag `{v0.cfg.data.data_tag}`, data hash `{v0.cfg.data_hash()}` (shared by every run group).",
        f"- Bootstrap: {meta['n_boot']:,} draws; days redrawn in circular {meta['block']}-day blocks (one draw shared by all variants); seeds redrawn per variant from its own random stream; RNG seed {meta['rng_seed']}.",
        "- Rule fixed in advance (Step 1 decision D3): the reference is better if the 95% range lies entirely above 0, the variant is better if entirely below 0, otherwise no detectable difference. IC is primary; Sharpe is the second opinion.",
    ]
    if v0.cfg.is_placebo:
        lines.append(f"- **Placebo**: every run group was trained and evaluated on shuffled answers (placebo seed {v0.cfg.placebo.shuffle_seed}, {v0.cfg.placebo.block_days}-day blocks); there is nothing real to find.")
    lines += ["", "## Run groups", ""]
    lines.append(md_table(
        ["Variant", "Config", "Run group", "Config hash", "Model", "Weights", "Seeds"],
        [[f"`{v.name}`", f"`{v.config_path}`", f"`{v.tag}`", f"`{v.cfg.hash()}`", v.cfg.model.kind, f"{v.n_weights:,}", " ".join(map(str, v.seeds))] for v in variants],
    ))  # fmt: skip

    lines += ["", "## Metrics (mean +- std over seeds)", ""]
    rows = []
    for v in variants:
        s = stats["variants"][v.name]
        rows.append([f"`{v.name}`"] + [f"{_fmt(m, s[m]['mean'])} +- {_fmt(m, s[m]['std'])}" for m in ("ic", "icir", "sharpe", "mdd", "ar")]
                    + [f"{_fmt('excess_ar', s['excess_ar']['mean'])}", f"{_fmt('excess_sharpe', s['excess_sharpe']['mean'])}"])  # fmt: skip
    lines.append(md_table(["Variant", "IC", "ICIR", "Sharpe", "Max drawdown", "Annual return", "Excess AR", "Excess Sharpe"], rows))

    lines += ["", "## Uncertainty and luck", ""]
    head = ["Variant", "IC [95% range]", "Sharpe [95% range]"]
    if deflated is not None:
        head = ["Variant", "Try #"] + head[1:] + ["Deflated Sharpe (excess)", "Deflated IC"]
    rows = []
    for v in variants:
        s = stats["variants"][v.name]
        cells = [f"`{v.name}`", f"{_fmt('ic', s['ic']['mean'])} [{_signed('ic', s['ic']['ci_low'])}, {_signed('ic', s['ic']['ci_high'])}]",
                 f"{_fmt('sharpe', s['sharpe']['mean'])} [{_signed('sharpe', s['sharpe']['ci_low'])}, {_signed('sharpe', s['sharpe']['ci_high'])}]"]  # fmt: skip
        if deflated is not None:
            d = deflated["variants"][v.name]
            cells = cells[:1] + ["-" if d["try_no"] is None else str(d["try_no"])] + cells[1:] + [f"{d['dsr']:.0%}", f"{d['dic']:.0%}"]
        rows.append(cells)
    lines.append(md_table(head, rows))
    lines += ["", "95% range = the 2.5-97.5% percentiles of the seed-mean level over the bootstrap draws above."]
    if deflated is not None:
        lines += [
            f"Deflated numbers (CLAUDE.md guards): the probability that the true value beats the best of {deflated['n_tries']} tries of pure luck "
            f"(try ledger `{Path(deflated['ledger']).name}`; spread of the tries' daily excess Sharpe {deflated['sr_sd_trials']:.4f} and IC {deflated['ic_sd_trials']:.4f}, "
            f"from the {deflated['tries_with_metrics']} tries with validation metrics). Above about 95% = likely real. "
            "Deflated Sharpe uses the daily excess return over the equal-weighted eligible stocks; deflated IC uses the bootstrap standard error.",
        ]

    lines += ["", f"## Advantage of `{ref.name}` over each variant (95% bootstrap range)", ""]
    rows = []
    for v in variants:
        if v is ref:
            continue
        a = stats["advantage"][v.name]
        cells = [f"`{v.name}`"]
        for metric in BOOT_METRICS:
            x = a[metric]
            cells += [f"{_signed(metric, x['observed'])} [{_signed(metric, x['ci_low'])}, {_signed(metric, x['ci_high'])}]", f"{x['share_positive']:.0%}"]
        cells += [a["ic"]["verdict"], a["sharpe"]["verdict"], _signed("mdd", a["mdd"]["observed"])]
        rows.append(cells)
    lines.append(md_table(
        ["Variant", "IC adv. [95%]", "P(>0)", "ICIR adv. [95%]", "P(>0)", "Sharpe adv. [95%]", "P(>0)", "IC rule", "Sharpe rule", "MDD adv."],
        rows,
    ))  # fmt: skip
    lines += ["", "P(>0) = share of bootstrap draws in which the reference did better."]

    lines += ["", "## Early-stopping check", ""]
    lines.append(md_table(
        ["Variant", "IC at the picked epoch (headline)", "Validation IC averaged over every epoch", "Gain from picking", "Mean epochs trained"],
        [[f"`{v.name}`", f"{stats['variants'][v.name]['ic']['mean']:.4f}", f"{stats['variants'][v.name]['val_ic_all_epochs']['mean']:.4f}",
          _signed("ic", stats["variants"][v.name]["ic"]["mean"] - stats["variants"][v.name]["val_ic_all_epochs"]["mean"]),
          f"{stats['variants'][v.name]['epochs_run']:.1f}"] for v in variants],
    ))  # fmt: skip
    lines += [
        "",
        "Each model's epoch was picked by the smoothed validation IC, i.e. on the same days compared here, so the headline IC is",
        "optimistic, most of all for runs whose validation IC swings from epoch to epoch. The all-epoch average involves no picking.",
    ]

    multi = [v for v in variants if v.agents]
    if multi:
        lines += ["", "## Agents and scheduler (multi-agent variants)", ""]
        lines.append(md_table(
            ["Variant", "Mean abs agent rank corr", "Scheduler entropy (max ln n)", "Effective agents per day", "Effective agents over time", "Mean agent weights"],
            [[f"`{v.name}`", f"{v.agents['mean_abs_corr']:.3f}", f"{v.agents['entropy']:.3f} (max {np.log(v.agents['n_agents']):.2f})",
              f"{v.agents['effective_agents_per_day']:.2f}", f"{v.agents['effective_agents_over_time']:.2f}", str(v.agents["mean_weights"])] for v in multi],
        ))  # fmt: skip
        lines += [
            "",
            "Validation days. Effective agents per day = mean over days of exp(scheduler entropy): 1 = one agent takes all the weight, n = equal weights.",
            "Effective agents over time = exp(entropy of each seed's mean weights), averaged over seeds: how many agents are used across the period.",
        ]
        with_usage = [v for v in multi if usage and usage.get(v.name)]
        if with_usage:
            lines += ["", "## Training days (restored model; from `detmaso usage`)", ""]
            rows = []
            for v in with_usage:
                u = usage[v.name]
                best = [c["best"] for c in u["seeds"].values() if "best" in c]
                mean_usage = np.mean([b["usage"] for b in best], axis=0)
                eff = np.mean([b["effective_agents"] for b in best])
                dead_hist = [c.get("dead_days_training") for c in u["seeds"].values()]
                dead_all = np.sum([d["all_epochs"] for d in dead_hist if d], axis=0).tolist() if all(dead_hist) else "-"
                dead_upto = np.sum([d["to_restored_epoch"] for d in dead_hist if d], axis=0).tolist() if all(dead_hist) else "-"
                dead_model = np.sum([c["dead_days_restored_model_train"] for c in u["seeds"].values()], axis=0).tolist()
                dead_val = np.sum(v.dead_days_valid, axis=0).tolist() if v.dead_days_valid else "-"
                rows.append([f"`{v.name}`", _pct_list(mean_usage), f"{eff:.2f}", str(dead_all), str(dead_upto), str(dead_model), str(dead_val)])
            lines.append(md_table(
                ["Variant", "Usage per agent, % (seed mean)", "Effective agents over time", "Dead agent-days, training (all epochs)",
                 "... up to the restored epoch", "Restored model, training days", "Restored model, validation days"],
                rows,
            ))  # fmt: skip
            lines += ["", "Usage counts an agent's dead days (constant output, score 0; DEVIATIONS 4) as 0. Dead agent-days are summed over seeds, per agent."]

    if rules is not None:
        lines += _rules_markdown(rules, ref)

    lines += ["", "## Per seed", ""]
    lines.append(md_table(
        ["Variant", "Seed", "IC", "ICIR", "Sharpe", "Max drawdown", "Annual return", "Best epoch", "Epochs"],
        [[f"`{v.name}`", p["seed"], f"{p['ic']['ic']:.4f}", f"{p['ic']['icir']:.3f}", f"{p['port']['sharpe']:.2f}", f"{p['port']['mdd']:.1%}",
          f"{p['port']['ar']:.1%}", p["best_epoch"], p["epochs_run"]] for v in variants for p in v.per_seed],
    ))  # fmt: skip
    if figs:
        lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    return "\n".join(lines) + "\n"


def _rules_markdown(rules: dict, ref: Variant) -> list[str]:
    target = rules["target"]
    lines = ["", "## Rules (docs/RULES.md, scheduler balance)", ""]
    lines += [f"Target usage ({rules['target_kind']} shares of the training days), %: {_pct_list(target)}. "
              "Rule M must hold in every seed, for the restored model and the last epoch trained.", ""]  # fmt: skip
    lines.append(md_table(
        ["Strength c_bal", "Variant", "M0 all seeds trained", "M1 usage >= half target", "M2 own agent leads its mood", "Lowest usage / target", "Rule M"],
        [[f"{c:g}", f"`{name}`", _yes(m["m0"]), _yes(m["m1"]), _yes(m["m2"]), f"{m['lowest_usage_over_target']:.2f}", "**pass**" if m["pass"] else "fail"]
         for c, (name, m) in zip(rules["strengths"], rules["rule_m"].items())],
    ))  # fmt: skip
    s = rules["rule_s"]
    chosen = f"c* = {s['c_star']:g} (`{rules['c_star_variant']}`)" if s["c_star"] is not None else s["outcome"]
    lines += ["", f"- **Rule S:** {chosen}."]
    k = rules["rule_k"]
    if k is not None:
        nxt = k["next_strength"]
        lines += [
            f"- **Rule K** (c* minus `{ref.name}`, IC): {k['ic_diff']:+.4f} [{k['ic_diff_low']:+.4f}, {k['ic_diff_high']:+.4f}].",
            f"  1. 95% range entirely above 0: {_yes(k['range_above_0'])}.",
            f"  2. seeds with a higher IC than the reference's same seed: {k['seed_wins']} of {k['seeds']} (need {k['seeds_needed']}): {_yes(k['seeds_ok'])}.",
            "  3. next strength up "
            + ("(none: c* is the top of the grid)" if nxt is None else f"c_bal = {nxt['strength']:g} passes Rule M: {_yes(nxt['passes_m'])}, mean IC {nxt['mean_ic']:.4f}")
            + f": {_yes(k['next_ok'])}.",
            f"  - Verdict: **{k['verdict']}**. Sharpe (second opinion): {k['sharpe_diff']:+.2f} [{k['sharpe_diff_low']:+.2f}, {k['sharpe_diff_high']:+.2f}], {k['sharpe_second_opinion']}.",
        ]
    lines += [f"- **Outcome:** {rules['outcome']}."]
    p = rules.get("rule_p")
    if p is not None:
        lines += [
            f"- **Rule P (placebo):** average placebo IC over the {len(p['groups'])} run groups {p['pooled_ic']:+.4f} "
            f"[{p['pooled_low']:+.4f}, {p['pooled_high']:+.4f}] (standard error {p['pooled_se']:.4f}).",
            f"  - P1 range not entirely above 0: {_yes(p['p1'])}. P2 chosen strength not significant and not \"better\": {_yes(p['p2'])}.",
            f"  - **Placebo {'passed' if p['pass'] else 'FAILED'}.**",
        ]
    return lines


def run_compare(
    runs: list[tuple[Config, str, str]], reference: str | None, out_dir: Path, n_boot: int = N_BOOT, block: int = BLOCK, rng_seed: int = RNG_SEED,
    rule: str | None = None, ledger_path: Path | None = None,
) -> Path:  # fmt: skip
    """runs: (config, config path, run-group tag) triples. Writes comparison.md / .json and figures to out_dir."""
    if rule is not None and rule not in RULES:
        raise CompareError(f"unknown rule {rule!r}; known: {RULES}")
    variants = [load_variant(cfg, path, tag) for cfg, path, tag in runs]
    check_comparable(variants)
    tags = [v.tag for v in variants]
    reference = reference or tags[0]
    if reference not in tags:
        raise CompareError(f"reference '{reference}' is not one of the compared run groups {tags}")
    ref_index = tags.index(reference)
    ref = variants[ref_index]
    draws = bootstrap_advantage(variants, n_boot, block, rng_seed)
    stats = summarize(variants, ref_index, n_boot, block, rng_seed, draws=draws)
    placebo = variants[0].cfg.is_placebo
    deflated = deflation(variants, stats, ledger_path) if (ledger_path is not None and not placebo and ledger_path.exists()) else None
    usage = {v.name: training_usage(v) for v in variants}
    rules = apply_scheduler_balance_rules(variants, ref, stats, draws) if rule == "scheduler_balance" else None
    out_dir.mkdir(parents=True, exist_ok=True)
    from detmaso.eval.compare_plots import comparison_figures, rule_figures

    figs = comparison_figures(variants, ref, stats, out_dir / "figures")
    if rules is not None:
        figs += rule_figures(variants, ref, stats, rules, usage, out_dir / "figures")
    meta = {"n_boot": n_boot, "block": block, "rng_seed": rng_seed}
    (out_dir / "comparison.md").write_text(write_markdown(variants, ref, stats, figs, meta, deflated, usage, rules))
    payload = {
        "split": "valid",
        "reference": {"variant": ref.name, "tag": ref.tag},
        "placebo": placebo,
        **meta,
        "runs": [
            {"variant": v.name, "tag": v.tag, "config": v.config_path, "config_hash": v.cfg.hash(), "data_hash": v.cfg.data_hash(),
             "model_kind": v.cfg.model.kind, "weights": v.n_weights, "seeds": v.seeds, "agents": v.agents, "per_seed": v.per_seed,
             "dead_days_valid": v.dead_days_valid, "training_usage": usage.get(v.name)}
            for v in variants
        ],
        "dates": {"first": str(ref.decision_dates[0].date()), "last": str(ref.decision_dates[-1].date()),
                  "trading_days": len(ref.decision_dates), "ic_days": len(ref.ic_dates)},
        **stats,
        "deflated": deflated,
        "rules": rules,
    }  # fmt: skip
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=_json_default))
    return out_dir / "comparison.md"


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return float(obj)
