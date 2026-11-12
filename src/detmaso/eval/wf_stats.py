"""Walk-forward statistics (docs/RULES.md, Rules WT and WK): a run group's daily series, metrics, resampling.

A walk-forward run group saves ``series_wf.parquet``: one row per seed and test date, with the fold (test year),
daily IC (on the purged IC dates), the paper backtest's gross return and benchmark, the long-short spread, the
day's buys and sells without and with blocked trades, and the scheduler weights.

Resampling (Rule WK): within each test window the days are redrawn as circular 20-day blocks, one draw shared
by every run group (paired by day); each run group's seeds are redrawn with replacement from its own random
stream (RNG seed and its name), once per draw for all folds. Draws are made chunk by chunk in a fixed order
(for each chunk of ``CHUNK`` draws, for each fold: IC blocks, then return blocks), so the same inputs always
give the same ranges, whichever other run groups are compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config
from detmaso.eval.compare import block_indices, variant_rng
from detmaso.eval.metrics import ic_summary, portfolio_metrics
from detmaso.eval.trading import CostSetting, cost_rates, cost_settings, net_returns

N_BOOT, BLOCK, RNG_SEED, CHUNK = 10_000, 20, 0, 500
SERIES_FILE, RESULTS_FILE = "series_wf.parquet", "results_wf.json"
SERIES_COLUMNS = ("ic", "port", "bench", "ls", "buy", "sell", "port_blk", "buy_blk", "sell_blk", "cash_blk", "blocked_buys", "blocked_sells")
REALISTIC = CostSetting("realistic", None)
AFTER_COSTS = "realistic_blocked"  # seed_metrics()["joined"] key of the headline after-cost metrics (Rule WT)


class WFError(RuntimeError):
    pass


@dataclass
class WFRun:
    """One walk-forward run group: per fold (test year) arrays of shape [seeds, days]."""

    name: str
    tag: str
    config_path: str
    cfg: Config
    seeds: list[int]
    years: list[int]
    dec: dict[int, pd.DatetimeIndex]  # decision dates per fold
    ret: dict[int, pd.DatetimeIndex]  # the day each holding-period return is realized
    in_ic: dict[int, np.ndarray]  # IC dates (answers end inside the window)
    cols: dict[int, dict[str, np.ndarray]]
    info: dict = field(default_factory=dict)  # results_wf.json (fold facts, moods), when loaded from disk

    def ic(self, year: int) -> np.ndarray:
        return self.cols[year]["ic"][:, self.in_ic[year]]

    def gross(self, year: int, blocked: bool) -> np.ndarray:
        return self.cols[year]["port_blk" if blocked else "port"]

    def net(self, year: int, setting: CostSetting, blocked: bool) -> np.ndarray:
        c = self.cols[year]
        sfx = "_blk" if blocked else ""
        buy_rate, sell_rate = cost_rates(self.cfg.trading, setting, self.dec[year])
        return net_returns(c[f"port{sfx}"], c[f"buy{sfx}"], c[f"sell{sfx}"], buy_rate, sell_rate)

    def realistic(self, year: int) -> np.ndarray:
        """The headline after-cost return: realistic costs, with blocked trades (Rule WT)."""
        return self.net(year, REALISTIC, True)

    def stitched(self, fn) -> np.ndarray:
        """[S, all test days]: fn(year) joined over the folds in order."""
        return np.concatenate([fn(y) for y in self.years], axis=1)

    @property
    def start(self) -> pd.Timestamp:
        return self.dec[self.years[0]][0]

    @property
    def end(self) -> pd.Timestamp:
        return self.ret[self.years[-1]][-1]

    @property
    def n_days(self) -> int:
        return sum(len(self.dec[y]) for y in self.years)

    @property
    def days_per_year(self) -> float:
        return self.n_days / ((self.end - self.start).days / 365.25)


def frame_to_run(frame: pd.DataFrame, cfg: Config, config_path: str, tag: str, info: dict | None = None) -> WFRun:
    """Group a saved daily series into per-fold [seed, day] arrays (every seed must cover the same dates)."""
    seeds = sorted(int(s) for s in frame["seed"].unique())
    years = sorted(int(y) for y in frame["year"].unique())
    dec, ret, in_ic, cols = {}, {}, {}, {}
    w_cols = sorted((c for c in frame.columns if c.startswith("w") and c[1:].isdigit()), key=lambda c: int(c[1:]))
    for y in years:
        fy = frame[frame["year"] == y]
        per_seed = {s: g.sort_values("decision_date").reset_index(drop=True) for s, g in fy.groupby("seed", sort=True)}
        if sorted(per_seed) != seeds:
            raise WFError(f"run group '{tag}': fold {y} lacks seeds {sorted(set(seeds) - set(per_seed))}")
        first = per_seed[seeds[0]]
        for s in seeds[1:]:
            if not (first["decision_date"].equals(per_seed[s]["decision_date"]) and first["in_ic"].equals(per_seed[s]["in_ic"])):
                raise WFError(f"run group '{tag}': seeds of fold {y} were evaluated on different dates")
        dec[y] = pd.DatetimeIndex(first["decision_date"])
        ret[y] = pd.DatetimeIndex(first["ret_date"])
        in_ic[y] = first["in_ic"].to_numpy(bool)
        cols[y] = {c: np.stack([per_seed[s][c].to_numpy(float) for s in seeds]) for c in (*SERIES_COLUMNS, "entropy", *w_cols)}
    return WFRun(cfg.variant, tag, config_path, cfg, seeds, years, dec, ret, in_ic, cols, info or {})


def load_wf_run(cfg: Config, config_path: str, tag: str) -> WFRun:
    """A fully evaluated walk-forward run group: every fold of the config and every configured seed."""
    from detmaso.walkforward.runner import group_hash

    run = cfg.run_path / tag
    if not cfg.is_walkforward:
        raise WFError(f"{config_path} is not a walk-forward config")
    if group_hash(cfg, tag) != cfg.hash():
        raise WFError(f"run group '{tag}' was not trained with {config_path} (config hash mismatch)")
    series, results = run / SERIES_FILE, run / RESULTS_FILE
    if not series.exists() or not results.exists():
        raise WFError(f"run group '{tag}' has no walk-forward evaluation; run `detmaso evaluate --split walkforward` first")
    info = json.loads(results.read_text())
    out = frame_to_run(pd.read_parquet(series), cfg, config_path, tag, info)
    if out.years != cfg.walkforward.years:
        raise WFError(f"run group '{tag}' was evaluated on folds {out.years}, not all of {cfg.walkforward.years}")
    if out.seeds != sorted(cfg.train.seeds):
        raise WFError(f"run group '{tag}' was evaluated on seeds {out.seeds}, not its configured seeds {sorted(cfg.train.seeds)}")
    return out


# Point estimates -----------------------------------------------------------------------------------------
def seed_metrics(run: WFRun) -> dict:
    """Per-seed metrics: yearly IC and ICIR [S, F], average IC [S], and joined-series portfolio metrics."""
    S = len(run.seeds)
    ic_year = np.array([[ic_summary(run.ic(y)[s])["ic"] for y in run.years] for s in range(S)])
    icir_year = np.array([[ic_summary(run.ic(y)[s])["icir"] for y in run.years] for s in range(S)])

    def pm(x: np.ndarray, start, end) -> list[dict]:
        return [portfolio_metrics(x[s], start, end, run.cfg.backtest.qlib_days) for s in range(S)]

    bench = run.stitched(lambda y: run.cols[y]["bench"])
    out = {
        "ic_year": ic_year,
        "icir_year": icir_year,
        "ic_avg": ic_year.mean(axis=1),
        "joined": {},
        "yearly": {},
    }
    settings = cost_settings(run.cfg.trading)
    for blocked in (False, True):
        mode = "blocked" if blocked else "unblocked"
        g = run.stitched(lambda y: run.gross(y, blocked))
        out["joined"][f"gross_{mode}"] = pm(g, run.start, run.end)
        for st in settings:
            x = run.stitched(lambda y: run.net(y, st, blocked))
            out["joined"][f"{st.name}_{mode}"] = pm(x, run.start, run.end)
            if f"{st.name}_{mode}" == AFTER_COSTS:
                out["joined"]["excess_realistic"] = pm(x - bench, run.start, run.end)
    for y in run.years:
        s0, s1 = run.dec[y][0], run.ret[y][-1]
        c = run.cols[y]
        out["yearly"][y] = {
            "gross": pm(run.gross(y, False), s0, s1),
            "realistic": pm(run.realistic(y), s0, s1),
            "excess_realistic": pm(run.realistic(y) - c["bench"], s0, s1),
            "ls_mean": np.nanmean(c["ls"], axis=1),
            "ls_sharpe": np.nanmean(c["ls"], axis=1) / np.nanstd(c["ls"], axis=1, ddof=1) * np.sqrt(len(c["ls"][0]) / ((s1 - s0).days / 365.25)),
            "turnover": np.maximum(c["buy"], c["sell"]).mean(axis=1),
            "turnover_blk": np.maximum(c["buy_blk"], c["sell_blk"]).mean(axis=1),
            "blocked_per_day": (c["blocked_buys"] + c["blocked_sells"]).mean(axis=1),
            "cash_blk": c["cash_blk"].mean(axis=1),
        }
    ls = run.stitched(lambda y: run.cols[y]["ls"])
    out["ls_mean"] = np.nanmean(ls, axis=1)
    out["ls_sharpe"] = out["ls_mean"] / np.nanstd(ls, axis=1, ddof=1) * np.sqrt(run.days_per_year)
    out["turnover"] = run.stitched(lambda y: np.maximum(run.cols[y]["buy"], run.cols[y]["sell"])).mean(axis=1)
    out["turnover_blk"] = run.stitched(lambda y: np.maximum(run.cols[y]["buy_blk"], run.cols[y]["sell_blk"])).mean(axis=1)
    out["blocked_per_day"] = run.stitched(lambda y: run.cols[y]["blocked_buys"] + run.cols[y]["blocked_sells"]).mean(axis=1)
    out["cash_blk"] = run.stitched(lambda y: run.cols[y]["cash_blk"]).mean(axis=1)
    return out


def mean_std(x) -> dict:
    a = np.asarray(x, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0}


def metric_values(per_seed: list[dict], key: str) -> np.ndarray:
    return np.array([m[key] for m in per_seed])


# Resampling ----------------------------------------------------------------------------------------------
def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, n: int, days_per_year: float) -> np.ndarray:
    mean = s1 / n
    var = np.maximum((s2 - n * mean**2) / (n - 1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return mean / np.sqrt(var) * np.sqrt(days_per_year)


def check_same_layout(runs: list[WFRun]) -> None:
    ref = runs[0]
    for r in runs[1:]:
        if r.years != ref.years:
            raise WFError(f"'{r.tag}' and '{ref.tag}' cover different folds")
        for y in ref.years:
            if not (r.dec[y].equals(ref.dec[y]) and np.array_equal(r.in_ic[y], ref.in_ic[y])):
                raise WFError(f"'{r.tag}' and '{ref.tag}' were evaluated on different dates in fold {y}")


def bootstrap(runs: list[WFRun], n_boot: int = N_BOOT, block: int = BLOCK, rng_seed: int = RNG_SEED, chunk: int = CHUNK) -> dict[str, dict]:
    """Draws per run group: yearly seed-mean IC [F, B], its average over the years [B], and the joined
    Sharpe of the realistic after-cost return and of the gross (paper backtest) return [B]."""
    check_same_layout(runs)
    ref = runs[0]
    years = ref.years
    F = len(years)
    rng = np.random.default_rng(rng_seed)
    seed_idx = {r.name: variant_rng(rng_seed, r.name).integers(0, len(r.seeds), size=(n_boot, len(r.seeds))) for r in runs}
    ics = {r.name: {y: r.ic(y) for y in years} for r in runs}
    rets = {r.name: {"sharpe_net": {y: r.realistic(y) for y in years}, "sharpe_gross": {y: r.gross(y, False) for y in years}} for r in runs}
    out = {r.name: {"ic_year": np.empty((F, n_boot)), "sharpe_net": np.empty(n_boot), "sharpe_gross": np.empty(n_boot)} for r in runs}
    n_total = ref.n_days
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        b = hi - lo
        sums = {r.name: {k: [np.zeros((len(r.seeds), b)), np.zeros((len(r.seeds), b))] for k in rets[r.name]} for r in runs}
        for fi, y in enumerate(years):
            ic_idx = block_indices(rng, int(ref.in_ic[y].sum()), block, b)
            ret_idx = block_indices(rng, len(ref.dec[y]), block, b)
            for r in runs:
                per_seed = np.nanmean(ics[r.name][y][:, ic_idx], axis=-1)  # [S, b]; a day without an IC is left out, as in seed_metrics
                out[r.name]["ic_year"][fi, lo:hi] = np.take_along_axis(per_seed.T, seed_idx[r.name][lo:hi], axis=1).mean(axis=1)
                for k, by_year in rets[r.name].items():
                    x = by_year[y][:, ret_idx]  # [S, b, n_f]
                    sums[r.name][k][0] += x.sum(axis=-1)
                    sums[r.name][k][1] += (x * x).sum(axis=-1)
        for r in runs:
            for k in rets[r.name]:
                sr = _sharpe_from_sums(*sums[r.name][k], n_total, r.days_per_year)  # [S, b]
                out[r.name][k][lo:hi] = np.take_along_axis(sr.T, seed_idx[r.name][lo:hi], axis=1).mean(axis=1)
    for r in runs:
        out[r.name]["ic_avg"] = out[r.name]["ic_year"].mean(axis=0)
    return out


def pct(x: float) -> str:
    """A probability as a percentage, or "n/a" when it could not be computed (too few tries for a spread)."""
    return f"{x:.0%}" if np.isfinite(x) else "n/a"


def pct_range(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")
