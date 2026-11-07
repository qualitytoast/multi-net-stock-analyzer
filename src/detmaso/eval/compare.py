"""Side-by-side comparison of trained variants on the validation period (TODO.md Step 1; plan D3).

Each run group must already be evaluated with ``detmaso evaluate --split valid``, which saves
runs/<market>/<tag>/series_valid.parquet (per seed and decision date: daily IC, portfolio and
benchmark returns, scheduler weights) and results_valid.json. The test period is never used.

Statistics
- Per seed: IC, ICIR, Sharpe, max drawdown, annual return and excess-return metrics, recomputed from
  the saved series with the report's own functions and checked against the report's numbers.
- Per variant: mean +- std over seeds.
- Reference advantage = reference minus variant (for drawdown: variant minus reference, so positive
  always means the reference did better), with a two-level paired block bootstrap. Each draw:
  (1) redraws the evaluation days as circular blocks of ``block`` trading days (20 = the label
  horizon: neighbouring days' 20-day labels overlap, so single days are not independent), one draw
  shared by every variant, which pairs the comparison by day; (2) redraws each variant's seeds with
  replacement (training randomness; seeds are not paired across variants). The seed-mean metric is
  recomputed and the advantage recorded; the 95% range is the 2.5-97.5% percentiles of the draws.
- Rule fixed before seeing any results (plan D3): the reference is better if the 95% range of its
  advantage lies entirely above 0, the variant is better if it lies entirely below 0, otherwise there
  is no detectable difference. IC is the primary metric and Sharpe the second opinion. Max drawdown
  is not bootstrapped (block resampling breaks the day order it depends on).
- Early-stopping check: every model's epoch was picked by the same validation IC compared here,
  which flatters runs whose validation IC swings from epoch to epoch. Next to the picked epoch's IC,
  the report shows the validation IC averaged over every epoch trained (no picking).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config
from detmaso.data.features import FEATURE_NAMES
from detmaso.eval.metrics import ic_summary, portfolio_metrics
from detmaso.mdtable import md_table
from detmaso.model.factory import build_model

BOOT_METRICS = ("ic", "icir", "sharpe")
REFERENCE_BETTER, VARIANT_BETTER, NO_DIFFERENCE = "reference better", "variant better", "no detectable difference"


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


def _seed_frames(series: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(s): g.sort_values("decision_date").reset_index(drop=True) for s, g in series.groupby("seed", sort=True)}


def seed_metrics(ic: np.ndarray, port: np.ndarray, bench: np.ndarray, start, end, qlib_days: int) -> dict:
    """The report's per-seed metrics from one seed's daily series (eval.report.evaluate_seed)."""
    return {"ic": ic_summary(ic), "port": portfolio_metrics(port, start, end, qlib_days), "excess": portfolio_metrics(port - bench, start, end, qlib_days)}


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
            "mean_weights": np.mean([frames[s][w_cols].to_numpy(float).mean(axis=0) for s in seeds], axis=0).round(3).tolist(),
        }
    n_weights = sum(p.numel() for p in build_model(cfg, d_in=len(FEATURE_NAMES)).parameters())
    return Variant(cfg.variant, tag, config_path, cfg, seeds, ic, port, bench, decision_dates[in_ic], decision_dates, ret_dates, per_seed, agents, n_weights)


def check_comparable(variants: list[Variant]) -> None:
    ref = variants[0]
    for v in variants[1:]:
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


def bootstrap_advantage(
    variants: list[Variant], ref_index: int, n_boot: int, block: int, rng_seed: int, chunk: int = 1000
) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap draws of the seed-mean metric for every variant: {variant name: {metric: [n_boot]}}."""
    rng = np.random.default_rng(rng_seed)
    n_ic, n_ret = variants[0].ic.shape[1], variants[0].port.shape[1]
    years = (variants[0].ret_dates[-1] - variants[0].decision_dates[0]).days / 365.25
    days_per_year = n_ret / years
    ic_idx = block_indices(rng, n_ic, block, n_boot)
    ret_idx = block_indices(rng, n_ret, block, n_boot)
    seed_idx = {v.name: rng.integers(0, len(v.seeds), size=(n_boot, len(v.seeds))) for v in variants}
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
    """Plan D3, fixed before seeing results."""
    if ci_low > 0:
        return REFERENCE_BETTER
    if ci_high < 0:
        return VARIANT_BETTER
    return NO_DIFFERENCE


def _seed_values(v: Variant, metric: str) -> np.ndarray:
    section = {"ic": "ic", "icir": "ic", "sharpe": "port", "ar": "port", "mdd": "port"}[metric]
    return np.array([p[section][metric] for p in v.per_seed])


def summarize(variants: list[Variant], ref_index: int, n_boot: int, block: int, rng_seed: int) -> dict:
    ref = variants[ref_index]
    draws = bootstrap_advantage(variants, ref_index, n_boot, block, rng_seed)
    out = {"variants": {}, "advantage": {}}
    for v in variants:
        stats = {}
        for metric in ("ic", "icir", "sharpe", "mdd", "ar"):
            vals = _seed_values(v, metric)
            stats[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "min": float(vals.min()), "max": float(vals.max())}
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


def _fmt(metric: str, x: float) -> str:
    return f"{x:.1%}" if metric in ("ar", "mdd", "excess_ar", "excess_mdd") else (f"{x:.2f}" if "sharpe" in metric else f"{x:.3f}" if metric == "icir" else f"{x:.4f}")


def _signed(metric: str, x: float) -> str:
    return ("+" if x >= 0 else "") + _fmt(metric, x)


def write_markdown(variants: list[Variant], ref: Variant, stats: dict, figs: list[str], meta: dict) -> str:
    v0 = variants[0]
    lines = [f"# Variant comparison: {v0.cfg.market}, validation split", ""]
    lines += [
        f"- Reference: `{ref.name}` (run group `{ref.tag}`). Advantage = reference minus variant (drawdown: variant minus reference), so a positive number means the reference did better.",
        f"- Evaluated dates: {v0.decision_dates[0].date()} .. {v0.decision_dates[-1].date()} ({len(v0.decision_dates)} trading days; IC uses {len(v0.ic_dates)}). The test period is not used.",
        f"- Data tag `{v0.cfg.data.data_tag}`, data hash `{v0.cfg.data_hash()}` (shared by every run group).",
        f"- Bootstrap: {meta['n_boot']:,} draws; days redrawn in circular {meta['block']}-day blocks (one draw shared by all variants); seeds redrawn per variant; RNG seed {meta['rng_seed']}.",
        "- Rule fixed in advance (plan D3): the reference is better if the 95% range lies entirely above 0, the variant is better if entirely below 0, otherwise no detectable difference. IC is primary; Sharpe is the second opinion.",
        "",
        "## Run groups",
        "",
    ]
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
            ["Variant", "Mean abs agent rank corr", "Scheduler entropy (max ln n)", "Effective agents per day", "Mean agent weights"],
            [[f"`{v.name}`", f"{v.agents['mean_abs_corr']:.3f}", f"{v.agents['entropy']:.3f} (max {np.log(v.agents['n_agents']):.2f})",
              f"{v.agents['effective_agents_per_day']:.2f}", str(v.agents["mean_weights"])] for v in multi],
        ))  # fmt: skip
        lines += ["", "Effective agents per day = mean over days of exp(scheduler entropy): 1 = one agent takes all the weight, n = equal weights."]

    lines += ["", "## Per seed", ""]
    lines.append(md_table(
        ["Variant", "Seed", "IC", "ICIR", "Sharpe", "Max drawdown", "Annual return", "Best epoch", "Epochs"],
        [[f"`{v.name}`", p["seed"], f"{p['ic']['ic']:.4f}", f"{p['ic']['icir']:.3f}", f"{p['port']['sharpe']:.2f}", f"{p['port']['mdd']:.1%}",
          f"{p['port']['ar']:.1%}", p["best_epoch"], p["epochs_run"]] for v in variants for p in v.per_seed],
    ))  # fmt: skip
    if figs:
        lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    return "\n".join(lines) + "\n"


def run_compare(runs: list[tuple[Config, str, str]], reference: str | None, out_dir: Path, n_boot: int = 10_000, block: int = 20, rng_seed: int = 0) -> Path:
    """runs: (config, config path, run-group tag) triples. Writes comparison.md / .json and figures to out_dir."""
    variants = [load_variant(cfg, path, tag) for cfg, path, tag in runs]
    check_comparable(variants)
    tags = [v.tag for v in variants]
    reference = reference or tags[0]
    if reference not in tags:
        raise CompareError(f"reference '{reference}' is not one of the compared run groups {tags}")
    ref_index = tags.index(reference)
    ref = variants[ref_index]
    stats = summarize(variants, ref_index, n_boot, block, rng_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    from detmaso.eval.compare_plots import comparison_figures

    figs = comparison_figures(variants, ref, stats, out_dir / "figures")
    meta = {"n_boot": n_boot, "block": block, "rng_seed": rng_seed}
    (out_dir / "comparison.md").write_text(write_markdown(variants, ref, stats, figs, meta))
    payload = {
        "split": "valid",
        "reference": {"variant": ref.name, "tag": ref.tag},
        **meta,
        "runs": [
            {"variant": v.name, "tag": v.tag, "config": v.config_path, "config_hash": v.cfg.hash(), "data_hash": v.cfg.data_hash(),
             "model_kind": v.cfg.model.kind, "weights": v.n_weights, "seeds": v.seeds, "agents": v.agents, "per_seed": v.per_seed}
            for v in variants
        ],
        "dates": {"first": str(ref.decision_dates[0].date()), "last": str(ref.decision_dates[-1].date()),
                  "trading_days": len(ref.decision_dates), "ic_days": len(ref.ic_dates)},
        **stats,
    }  # fmt: skip
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=float))
    return out_dir / "comparison.md"
