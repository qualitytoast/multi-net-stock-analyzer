"""Guards (CLAUDE.md): deflated Sharpe / IC, the try ledger, and the scheduler-balance rules as code."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from detmaso.eval import guard, ledger
from detmaso.eval.compare import block_indices, deflation, summarize
from tests.conftest import ROOT, synthetic_config
from tests.test_compare import _market, _variant


# Deflated Sharpe and deflated IC ------------------------------------------------------------------
def test_deflated_sharpe_reproduces_the_published_worked_example():
    """Bailey and Lopez de Prado (2014): N = 100 tries, yearly Sharpe 2.5, yearly variance of the tries'
    Sharpe 0.5, T = 1250 days, skew -3, kurtosis 10 -> SR0 = 0.1132 per day, DSR = 0.9004."""
    r = guard.deflated_sharpe(2.5 / math.sqrt(250), 1250, -3.0, 10.0, math.sqrt(0.5 / 250), 100)
    assert r["sr0"] == pytest.approx(0.1132, abs=1e-4)
    assert r["dsr"] == pytest.approx(0.9004, abs=1e-4)
    wrong = guard.deflated_sharpe(2.5 / math.sqrt(250), 1250, -3.0, 10.0 - 3.0, math.sqrt(0.5 / 250), 100)
    assert wrong["dsr"] == pytest.approx(0.9018, abs=1e-4)  # what "excess" kurtosis would give by mistake


def test_moments_use_plain_kurtosis():
    x = np.random.default_rng(0).normal(size=200_000)
    s, k = guard.return_moments(x)
    assert abs(s) < 0.02 and k == pytest.approx(3.0, abs=0.05)


def test_expected_max_grows_with_the_number_of_tries():
    assert guard.expected_max_z(1) == 0.0
    vals = [guard.expected_max_z(n) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_deflated_sharpe_is_calibrated_under_pure_luck():
    """The best of 10 pure-luck strategies looks real (> 95%) about 40% of the time without deflation;
    deflated, it should do so at most about 5% of the time."""
    rng = np.random.default_rng(0)
    raw = deflated = 0
    for _ in range(400):
        x = rng.normal(0, 0.01, size=(10, 250))
        sr = x.mean(1) / x.std(1, ddof=1)
        b = int(np.argmax(sr))
        s, k = guard.return_moments(x[b])
        raw += guard.deflated_sharpe(sr[b], 250, s, k, sr.std(ddof=1), 1)["dsr"] > 0.95
        deflated += guard.deflated_sharpe(sr[b], 250, s, k, sr.std(ddof=1), 10)["dsr"] > 0.95
    assert raw / 400 > 0.25 and deflated / 400 <= 0.06


def test_deflated_ic_is_calibrated_with_overlapping_labels():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(200):
        e = rng.normal(size=(10, 279))
        ic = np.stack([np.convolve(row, np.ones(20) / 20, mode="valid") for row in e]) * 0.1  # 20-day overlap
        means = ic.mean(1)
        b = int(np.argmax(means))
        se = ic[b][block_indices(rng, 260, 20, 300)].mean(1).std(ddof=1)
        hits += guard.deflated_ic(means[b], se, means.std(ddof=1), 10)["dic"] > 0.95
    assert hits / 200 <= 0.06


# Try ledger ---------------------------------------------------------------------------------------
def test_ledger_counts_each_real_configuration_once_in_order(tmp_path):
    path = tmp_path / "trials.csv"
    a = synthetic_config(tmp_path, variant="a", loss={"c_bal": 0.1})
    b = synthetic_config(tmp_path, variant="b", loss={"c_bal": 0.2})
    ledger.register(path, a, "a.yaml", "ga", [0, 1])
    ledger.register(path, a, "a.yaml", "ga2", [0])  # an identical configuration under another run group
    ledger.register(path, b, "b.yaml", "gb", [0])
    ledger.register(path, synthetic_config(tmp_path, variant="p", placebo={"shuffle_seed": 0}), "p.yaml", "gp", [0])
    ledger.register(path, synthetic_config(tmp_path, variant="t", train={"validate": False}), "t.yaml", "gt", [0])
    ledger.set_seed_status(path, b, "gb", 0, "aborted: TrainingAborted")
    df = ledger.read(path)
    assert len(df) == 4
    assert df["kind"].tolist() == [ledger.REAL, ledger.REAL, ledger.PLACEBO, ledger.PILOT]
    assert df["try_no"].tolist()[:2] == [1, 2] and df["try_no"].isna().tolist()[2:] == [True, True]
    row_a = df[df["config_hash"] == a.hash()].iloc[0]
    assert row_a["run_groups"] == "ga;ga2" and json.loads(row_a["seed_status"]) == {"ga2:0": "started", "ga:0": "started", "ga:1": "started"}
    assert json.loads(df[df["config_hash"] == b.hash()].iloc[0]["seed_status"]) == {"gb:0": "aborted: TrainingAborted"}
    assert len(ledger.tries(path)) == 2 and ledger.try_number(path, b.hash()) == 2
    ledger.record_metrics(path, a, {"valid_days": 278, "ic": 0.05, "ic_sd_seeds": 0.01, "sharpe": 0.5, "excess_sharpe": 0.2, "excess_sr_daily": 0.013})
    assert ledger.tries(path).iloc[0]["ic"] == pytest.approx(0.05)


def test_ledger_survives_concurrent_writers(tmp_path):
    """Several training processes register at once; no update may be lost."""
    path = tmp_path / "trials.csv"
    script = (
        "import sys; from tests.conftest import synthetic_config; from detmaso.eval import ledger; from pathlib import Path\n"
        "k = int(sys.argv[1]); p = Path(sys.argv[2])\n"
        "for j in range(10):\n"
        "    cfg = synthetic_config(p.parent, variant=f'v{k}_{j}', loss={'c_bal': 1 + 100 * k + j})\n"
        "    ledger.register(p, cfg, 'x.yaml', f'g{k}', [j])\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", script, str(k), str(path)], cwd=ROOT) for k in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    df = ledger.read(path)
    assert len(df) == 40 and sorted(df["try_no"].tolist()) == list(range(1, 41))


# Rules (docs/RULES.md) ----------------------------------------------------------------------------
def test_rule_g_places_the_grid_around_the_pilot_anchor():
    g = guard.rule_g({1e-5: False, 1e-4: False, 1e-3: True, 1e-2: True, 1e-1: True, 1.0: True})
    assert g["anchor"] == 1e-3 and g["grid"] == [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1] and not g["stop"]
    assert guard.rule_g({c: False for c in guard.PILOT_STRENGTHS})["stop"]
    placebo = guard.rule_g({c: False for c in guard.PILOT_STRENGTHS}, placebo=True)
    assert not placebo["stop"] and placebo["grid"] == sorted(guard.PILOT_STRENGTHS)


def _days(n_moods: int = 3, per: int = 10) -> np.ndarray:
    return np.repeat(np.arange(n_moods), per)


def test_usage_excludes_dead_days_and_routing_rejects_a_constant_blend():
    regime = _days()
    target = np.array([1 / 3] * 3)
    specialists = np.eye(3)[regime] * 0.98 + 0.01
    live = np.ones_like(specialists, dtype=bool)
    good = guard.usage_and_routing(specialists, live, regime, 3)
    assert guard.m1_holds(good["usage"], target) and guard.m2_holds(good["routing"], good["mood_days"])
    blend = np.tile(target, (len(regime), 1)) + np.array([0.01, 0.0, -0.01])  # same mix every day
    b = guard.usage_and_routing(blend, live, regime, 3)
    assert guard.m1_holds(b["usage"], target) and not guard.m2_holds(b["routing"], b["mood_days"])
    dead = live.copy()
    dead[regime == 2, 2] = False  # agent 2 gets its weight only on days it is dead
    d = guard.usage_and_routing(specialists, dead, regime, 3)
    assert d["usage"][2] < 0.01 and not guard.m1_holds(d["usage"], target)


def test_rule_m_needs_every_seed_and_both_checkpoints():
    regime, target = _days(), np.array([1 / 3] * 3)
    ok = guard.usage_and_routing(np.eye(3)[regime], np.ones((30, 3), bool), regime, 3)
    full = {s: {"best": ok, "last": ok} for s in range(5)}
    assert guard.rule_m(target, full, list(range(5)))["pass"]
    missing_last = {**full, 4: {"best": ok}}
    assert not guard.rule_m(target, missing_last, list(range(5)))["pass"]
    assert not guard.rule_m(target, {s: full[s] for s in range(4)}, list(range(5)))["pass"]  # an aborted seed


def test_rule_s_picks_the_smallest_passing_strength_or_names_the_stop():
    ok = {"m0": True, "m1": True, "m2": True, "pass": True}
    no_routing = {"m0": True, "m1": True, "m2": False, "pass": False}
    unfixed = {"m0": True, "m1": False, "m2": False, "pass": False}
    assert guard.rule_s({0.1: ok, 0.01: no_routing, 1.0: ok})["c_star"] == 0.1
    assert guard.rule_s({0.1: unfixed, 1.0: unfixed})["outcome"] == guard.NOT_FIXED
    assert guard.rule_s({0.1: unfixed, 1.0: no_routing})["outcome"] == guard.NO_SPECIALISTS


def test_rule_k_needs_all_three_parts():
    ref = {s: 0.06 for s in range(5)}
    cand = {s: 0.08 for s in range(5)}
    nxt = {"strength": 0.3, "passes_m": True, "mean_ic": 0.07}
    assert guard.rule_k(0.01, 0.03, cand, ref, nxt, 0.06)["verdict"] == guard.BETTER
    assert guard.rule_k(-0.01, 0.03, cand, ref, nxt, 0.06)["verdict"] == guard.NO_DIFFERENCE  # range touches 0
    three = {**cand, 3: 0.05, 4: 0.05}
    assert guard.rule_k(0.01, 0.03, three, ref, nxt, 0.06)["verdict"] == guard.NO_DIFFERENCE  # 3 of 5 seeds
    four = {**cand, 4: 0.05}
    assert guard.rule_k(0.01, 0.03, four, ref, nxt, 0.06)["verdict"] == guard.BETTER  # 4 of 5 is enough
    assert guard.rule_k(0.01, 0.03, cand, ref, None, 0.06)["verdict"] == guard.NO_DIFFERENCE  # top of the grid
    assert guard.rule_k(0.01, 0.03, cand, ref, {**nxt, "passes_m": False}, 0.06)["verdict"] == guard.NO_DIFFERENCE
    assert guard.rule_k(-0.05, -0.01, cand, ref, nxt, 0.06)["verdict"] == guard.WORSE
    assert guard.outcome(True, guard.NO_DIFFERENCE) == guard.KEPT and guard.outcome(True, guard.WORSE) == guard.DROPPED
    assert guard.outcome(False, guard.BETTER) == guard.DROPPED


def test_rule_p_checks_are_one_sided():
    assert guard.rule_p(-0.01, None)["pass"] and guard.rule_p(0.0, None)["pass"]
    assert not guard.rule_p(0.001, None)["p1"]
    assert not guard.rule_p(-0.01, {"ic_low": 0.002, "k_verdict": guard.NO_DIFFERENCE})["p2"]
    assert not guard.rule_p(-0.01, {"ic_low": -0.01, "k_verdict": guard.BETTER})["p2"]
    assert guard.rule_p(-0.01, {"ic_low": -0.03, "k_verdict": guard.WORSE})["pass"]  # below 0 is not "finding" anything


# Compare: order-independent resampling, level ranges, deflation ------------------------------------
def test_a_variants_draws_do_not_depend_on_the_other_variants(tmp_path):
    cfg = synthetic_config(tmp_path)
    ref, a, b = (_variant(cfg, n, *_market(5, seed=i)) for i, n in enumerate(("ref", "a", "b")))
    one = summarize([ref, a, b], 0, n_boot=400, block=20, rng_seed=0)["advantage"]["a"]
    two = summarize([ref, b, a], 0, n_boot=400, block=20, rng_seed=0)["advantage"]["a"]
    three = summarize([ref, a], 0, n_boot=400, block=20, rng_seed=0)["advantage"]["a"]
    assert one == two == three


def test_level_ranges_and_deflation(tmp_path):
    path = tmp_path / "trials.csv"
    cfgs = [synthetic_config(tmp_path, variant=f"v{i}", loss={"c_bal": 0.1 * (i + 1)}) for i in range(3)]
    variants = [_variant(c, c.variant, *_market(5, seed=i)) for i, c in enumerate(cfgs)]
    variants = [dataclasses.replace(v, cfg=c) for v, c in zip(variants, cfgs)]
    for c in cfgs:
        ledger.register(path, c, "x.yaml", c.variant, [0])
        ledger.record_metrics(path, c, {"valid_days": 278, "ic": 0.05 + 0.01 * cfgs.index(c), "ic_sd_seeds": 0.01, "sharpe": 0.5,
                                        "excess_sharpe": 0.2, "excess_sr_daily": 0.01 + 0.005 * cfgs.index(c)})  # fmt: skip
    stats = summarize(variants, 0, n_boot=400, block=20, rng_seed=0)
    for v in variants:
        s = stats["variants"][v.name]["ic"]
        assert s["ci_low"] <= s["mean"] <= s["ci_high"] and s["se"] > 0
    d = deflation(variants, stats, path)
    assert d["n_tries"] == 3 and [d["variants"][v.name]["try_no"] for v in variants] == [1, 2, 3]
    assert all(0.0 <= d["variants"][v.name]["dsr"] <= 1.0 and 0.0 <= d["variants"][v.name]["dic"] <= 1.0 for v in variants)
