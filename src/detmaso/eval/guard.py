"""Guards against fooling ourselves (CLAUDE.md "Guards"; rules in docs/RULES.md).

Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014): the probability that a strategy's true Sharpe ratio
beats SR0, roughly the best Sharpe that N tries of pure luck would show:

    SR0 = sd(tries' Sharpe) * [(1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e))]
    DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurtosis - 1) / 4 * SR^2) )

with daily (not annualized) Sharpe ratios, T daily returns, and plain (not "excess") kurtosis. Here it is
computed on the portfolio's daily excess return over the equal-weighted eligible stocks (its stock-picking
edge): the market's own direction is not something the model chose.

Deflated IC uses the same logic, but the IC's standard error comes from the 20-day block resampling,
because neighbouring days' 20-day labels overlap (the formula's T independent days would overstate the data
about 20-fold): Phi((IC - IC0) / SE), IC0 = sd(tries' IC) * the same expected-maximum factor.

Rules G, M, S, K and P of the scheduler-balance change, and the walk-forward keep rule (WK) and placebo rule (WP),
are implemented below as plain functions that return their verdict together with every number behind it, so they
are applied by code, not by eye.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329
PILOT_STRENGTHS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GRID_STEPS = (10**-0.5, 1.0, 10**0.5, 10.0, 10**1.5, 100.0)  # g0/sqrt(10) ... 100 g0, half-decade steps
BETTER, WORSE, NO_DIFFERENCE = "better", "worse", "no detectable difference"
KEPT, DROPPED = "kept for re-check", "dropped"
IMPROVEMENT, NO_IMPROVEMENT = "improvement", "no improvement"
YEAR_WIN_SHARE = 8 / 11  # the general keep rule: wins in at least 8 of 11 walk-forward years
NOT_FIXED, NO_SPECIALISTS = "not fixed at any tested strength", "all agents used, but no specialists"


# Deflated metrics ----------------------------------------------------------------------------------
def expected_max_z(n_trials: int) -> float:
    """Approximate expected maximum of n independent standard normals (0 for fewer than 2 tries)."""
    if n_trials < 2:
        return 0.0
    return (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))


def return_moments(x: np.ndarray) -> tuple[float, float]:
    """(skewness, plain kurtosis) of a return series; plain kurtosis is 3 for a normal distribution."""
    x = np.asarray(x, dtype=np.float64)
    return float(skew(x)), float(kurtosis(x, fisher=False))


def deflated_sharpe(sr: float, n_obs: int, skewness: float, kurt: float, sr_sd_trials: float, n_trials: int) -> dict:
    """Bailey and Lopez de Prado (2014). sr and sr_sd_trials are per-period (not annualized)."""
    sr0 = sr_sd_trials * expected_max_z(n_trials)
    var = 1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr**2
    if not (np.isfinite(var) and var > 0 and n_obs > 1):
        return {"dsr": float("nan"), "sr0": float(sr0)}
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(var)
    return {"dsr": float(norm.cdf(z)), "sr0": float(sr0)}


def deflated_ic(ic: float, ic_se: float, ic_sd_trials: float, n_trials: int) -> dict:
    ic0 = ic_sd_trials * expected_max_z(n_trials)
    if not (np.isfinite(ic_se) and ic_se > 0):
        return {"dic": float("nan"), "ic0": float(ic0)}
    return {"dic": float(norm.cdf((ic - ic0) / ic_se)), "ic0": float(ic0)}


def excess_sharpe_inputs(port: np.ndarray, bench: np.ndarray) -> dict:
    """Seed-mean daily Sharpe of the excess return, and seed-mean skewness and kurtosis ([seeds, days] inputs)."""
    x = np.asarray(port, dtype=np.float64) - np.asarray(bench, dtype=np.float64)
    srs, skews, kurts = [], [], []
    for row in x:
        srs.append(row.mean() / row.std(ddof=1))
        s, k = return_moments(row)
        skews.append(s)
        kurts.append(k)
    return {"sr": float(np.mean(srs)), "n_obs": int(x.shape[1]), "skew": float(np.mean(skews)), "kurt": float(np.mean(kurts))}


# Scheduler-balance rules (docs/RULES.md) ------------------------------------------------------------
def round_1sig(x: float) -> float:
    return float(f"{x:.0e}")


def rule_g(pilot_m1: dict[float, bool], placebo: bool = False) -> dict:
    """Rule G. pilot_m1: pilot strength -> did M1 hold at the last epoch in both pilot seeds?"""
    passing = sorted(c for c, ok in pilot_m1.items() if ok)
    if passing:
        g0 = passing[0]
        return {"anchor": g0, "grid": [round_1sig(g0 * f) for f in GRID_STEPS], "stop": False}
    if placebo:  # the placebo process keeps going on the pilot strengths themselves
        return {"anchor": None, "grid": sorted(pilot_m1), "stop": False}
    return {"anchor": None, "grid": None, "stop": True}


def usage_and_routing(w: np.ndarray, live: np.ndarray, regime: np.ndarray, n_moods: int) -> dict:
    """Usage (dead agent-days count as 0) and the mood x agent routing table, from per-day weights."""
    wl = np.asarray(w, dtype=np.float64) * np.asarray(live, dtype=bool)
    routing = np.full((n_moods, wl.shape[1]), np.nan)
    days = np.zeros(n_moods, dtype=int)
    for k in range(n_moods):
        sel = regime == k
        days[k] = int(sel.sum())
        if days[k]:
            routing[k] = wl[sel].mean(axis=0)
    return {"usage": wl.mean(axis=0), "routing": routing, "mood_days": days}


def m1_holds(usage: np.ndarray, target: np.ndarray) -> bool:
    t = np.asarray(target)
    return bool(np.all(np.asarray(usage)[t > 0] >= 0.5 * t[t > 0]))


def m2_holds(routing: np.ndarray, mood_days: np.ndarray) -> bool:
    """Each mood's own agent has the largest average weight on that mood's days (moods with days only)."""
    return all(int(np.nanargmax(routing[k])) == k for k in range(len(mood_days)) if mood_days[k] > 0)


def rule_m(target: np.ndarray, checks: dict[int, dict[str, dict]], seeds_expected: list[int], routing_required: bool = True) -> dict:
    """Rule M. checks[seed][checkpoint] = usage_and_routing(...) for checkpoints "best" and "last"."""
    m0 = sorted(checks) == sorted(seeds_expected) and all({"best", "last"} <= set(checks[s]) for s in checks)
    per = {
        s: {ck: {"m1": m1_holds(c["usage"], target), "m2": m2_holds(c["routing"], c["mood_days"])} for ck, c in checks[s].items()}
        for s in checks
    }
    m1 = bool(per) and all(v["m1"] for s in per for v in per[s].values())
    m2 = bool(per) and all(v["m2"] for s in per for v in per[s].values())
    return {"m0": m0, "m1": m1, "m2": m2, "pass": bool(m0 and m1 and (m2 or not routing_required)), "per_seed": per}


def rule_s(m_by_strength: dict[float, dict]) -> dict:
    """Rule S: the smallest grid strength that satisfies Rule M, or the stop reason."""
    strengths = sorted(m_by_strength)
    passing = [c for c in strengths if m_by_strength[c]["pass"]]
    if passing:
        return {"c_star": passing[0], "outcome": "chosen"}
    if not any(m_by_strength[c]["m0"] and m_by_strength[c]["m1"] for c in strengths):
        return {"c_star": None, "outcome": NOT_FIXED}
    return {"c_star": None, "outcome": NO_SPECIALISTS}


def rule_k(diff_low: float, diff_high: float, cand_ic: dict[int, float], ref_ic: dict[int, float], next_strength: dict | None, ref_mean_ic: float) -> dict:
    """Rule K for c* vs the reference. diff_* = 95% range of (c* minus reference) mean IC.

    next_strength: {"strength", "passes_m", "mean_ic"} for the next grid strength above c*, or None.
    """
    seeds = sorted(set(cand_ic) & set(ref_ic))
    wins = sum(cand_ic[s] > ref_ic[s] for s in seeds)
    need = math.ceil(0.8 * len(seeds))  # at least 4 of 5 seeds
    part1 = diff_low > 0
    part2 = len(seeds) > 0 and wins >= need
    part3 = next_strength is not None and next_strength["passes_m"] and next_strength["mean_ic"] > ref_mean_ic
    verdict = BETTER if (part1 and part2 and part3) else WORSE if diff_high < 0 else NO_DIFFERENCE
    return {
        "verdict": verdict, "range_above_0": part1, "seed_wins": wins, "seeds": len(seeds), "seeds_needed": need,
        "seeds_ok": part2, "next_strength": next_strength, "next_ok": part3,
    }  # fmt: skip


def outcome(rule_m_passes: bool, verdict: str) -> str:
    return KEPT if (rule_m_passes and verdict != WORSE) else DROPPED


def rule_p(pooled_low: float, chosen: dict | None) -> dict:
    """Rule P. pooled_low: lower end of the 95% range of the average placebo IC over all placebo run groups.

    chosen: {"ic_low": lower end of the chosen strength's IC range, "k_verdict": Rule K verdict} or None.
    """
    p1 = not pooled_low > 0  # the range is not entirely above 0
    p2 = chosen is None or (not chosen["ic_low"] > 0 and chosen["k_verdict"] != BETTER)
    return {"p1": p1, "p2": p2, "pass": bool(p1 and p2)}


# Walk-forward keep rule (docs/RULES.md, Rule WK) --------------------------------------------------------------
def _refuse_non_finite(rule: str, numbers: dict[str, float]) -> None:
    """A rule gives no verdict on a number that could not be computed."""
    bad = [name for name, v in numbers.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"Rule {rule} cannot be applied: not finite: {', '.join(bad)}")


def keep_rule(
    yearly: dict[int, dict], average: dict, seed_gains: dict[int, float], sharpe: dict, has_setting: bool,
    neighbor_gain: float | None = None, target_years: list[int] | None = None,
) -> dict:  # fmt: skip
    """Rule WK for a candidate against a reference.

    yearly[year] = {"gain", "low", "high"}: seed-mean IC gain in that test year and its 95% range;
    average = {"gain", "low", "high"}: the mean of the yearly gains; seed_gains[seed] = that seed's mean yearly gain;
    sharpe = {"diff", "low", "high"}: after-cost Sharpe difference; neighbor_gain = the next setting's average gain
    (required when has_setting); target_years = the years a targeted change declared in advance (None = general form).
    Raises when any of these numbers is not finite.
    """
    years = sorted(yearly)
    _refuse_non_finite("WK", {
        **{f"{y} {k}": yearly[y][k] for y in years for k in ("gain", "low", "high")},
        **{f"average {k}": average[k] for k in ("gain", "low", "high")},
        **{f"seed {s} gain": g for s, g in seed_gains.items()},
        **{f"after-cost Sharpe {k}": sharpe[k] for k in ("diff", "low", "high")},
        **({} if neighbor_gain is None else {"next setting's gain": neighbor_gain}),
    })  # fmt: skip
    k1 = average["low"] > 0
    wins = [y for y in years if yearly[y]["gain"] > 0]
    if target_years:
        unknown = sorted(set(target_years) - set(years))
        if unknown:
            raise ValueError(f"target years {unknown} are not walk-forward test years {years}")
        targets = sorted(set(target_years))
        target_wins = [y for y in targets if yearly[y]["gain"] > 0]
        need = len(targets) // 2 + 1
        worse_elsewhere = [y for y in years if y not in targets and yearly[y]["high"] < 0]
        k2 = len(target_wins) >= need and not worse_elsewhere
        k2_detail = {"form": "targeted", "target_years": targets, "target_wins": target_wins, "target_wins_needed": need,
                     "other_years_worse": worse_elsewhere}  # fmt: skip
    else:
        need = math.ceil(YEAR_WIN_SHARE * len(years) - 1e-9)
        k2 = len(wins) >= need
        k2_detail = {"form": "general", "wins_needed": need}
    seeds = sorted(seed_gains)
    seed_wins = sum(seed_gains[s] > 0 for s in seeds)
    seeds_needed = math.ceil(0.8 * len(seeds))
    k3 = len(seeds) > 0 and seed_wins >= seeds_needed
    if has_setting and neighbor_gain is None:
        raise ValueError("the change has a tunable setting: name the next setting over (its run group) to apply K4")
    k4 = (neighbor_gain is not None and neighbor_gain > 0) if has_setting else True
    k5 = not sharpe["high"] < 0
    passed = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k5": k5}
    verdict = IMPROVEMENT if all(passed.values()) else WORSE if average["high"] < 0 else NO_IMPROVEMENT
    return {
        "verdict": verdict, **passed, "average": average, "year_wins": wins, "n_years": len(years), "k2_detail": k2_detail,
        "seed_wins": seed_wins, "seeds": len(seeds), "seeds_needed": seeds_needed, "has_setting": has_setting,
        "neighbor_gain": neighbor_gain, "sharpe": sharpe,
    }  # fmt: skip


def rule_wp(levels: dict[str, tuple[float, float]], keep_verdicts: dict[str, str]) -> dict:
    """Rule WP (walk-forward placebo). levels[group] = 95% range of its average IC; keep_verdicts[pair] = Rule WK
    verdicts between the placebo groups, in both directions. Raises when a range bound is not finite."""
    _refuse_non_finite("WP", {f"{group} {end}": v for group, (lo, hi) in levels.items() for end, v in (("low", lo), ("high", hi))})
    p1 = all(not lo > 0 for lo, _ in levels.values())
    p2 = all(v != IMPROVEMENT for v in keep_verdicts.values())
    return {"p1": p1, "p2": p2, "pass": bool(p1 and p2), "levels": levels, "keep_verdicts": keep_verdicts}
