"""Step 1 comparison: bootstrap statistics, the fixed decision rule, and the safety checks."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from detmaso.eval.compare import (
    NO_DIFFERENCE,
    REFERENCE_BETTER,
    VARIANT_BETTER,
    CompareError,
    Variant,
    block_indices,
    check_comparable,
    load_variant,
    rule,
    run_compare,
    seed_metrics,
    summarize,
)
from tests.conftest import synthetic_config


def _variant(cfg, name: str, ic: np.ndarray, port: np.ndarray, bench: np.ndarray | None = None) -> Variant:
    n_seeds, n_days = port.shape
    dates = pd.bdate_range("2023-01-02", periods=n_days + 1)
    decision, ret = dates[:-1], dates[1:]
    bench = np.zeros_like(port) if bench is None else bench
    per_seed = [
        {"seed": s, **seed_metrics(ic[s], port[s], bench[s], decision[0], ret[-1], 238), "best_epoch": 1, "epochs_run": 1, "val_ic_all_epochs": 0.0}
        for s in range(n_seeds)
    ]
    cfg = dataclasses.replace(cfg, variant=name)
    return Variant(name, name, f"{name}.yaml", cfg, list(range(n_seeds)), ic, port, bench, decision[: ic.shape[1]], decision, ret, per_seed, None, 0)


def _market(n_seeds: int, seed: int = 0, n_ic: int = 258, n_days: int = 278):
    rng = np.random.default_rng(seed)
    ic = 0.05 + 0.10 * rng.standard_normal((n_seeds, n_ic))
    port = 0.0005 + 0.01 * rng.standard_normal((n_seeds, n_days))
    return ic, port


def test_block_indices_are_circular_blocks():
    idx = block_indices(np.random.default_rng(0), n_days=50, block=20, n_boot=7)
    assert idx.shape == (7, 50) and idx.min() >= 0 and idx.max() < 50
    for row in idx:
        for start in range(0, 50, 20):
            block = row[start : start + 20]
            assert np.all(np.diff(block) % 50 == 1)


def test_rule_is_the_fixed_d3_rule():
    assert rule(0.01, 0.02) == REFERENCE_BETTER
    assert rule(-0.02, -0.01) == VARIANT_BETTER
    assert rule(-0.01, 0.01) == NO_DIFFERENCE
    assert rule(0.0, 0.01) == NO_DIFFERENCE  # a range touching 0 is not "entirely above 0"


def test_identical_single_seed_variants_have_exactly_zero_advantage(tmp_path):
    cfg = synthetic_config(tmp_path)
    ic, port = _market(1)
    stats = summarize([_variant(cfg, "ref", ic, port), _variant(cfg, "copy", ic.copy(), port.copy())], 0, n_boot=500, block=20, rng_seed=0)
    for metric in ("ic", "icir", "sharpe"):
        a = stats["advantage"]["copy"][metric]
        assert a["observed"] == a["ci_low"] == a["ci_high"] == 0.0 and a["verdict"] == NO_DIFFERENCE


def test_identical_multi_seed_variants_straddle_zero(tmp_path):
    """Seeds are redrawn independently per variant, so two copies differ draw by draw, around 0."""
    cfg = synthetic_config(tmp_path)
    ic, port = _market(5)
    stats = summarize([_variant(cfg, "ref", ic, port), _variant(cfg, "copy", ic.copy(), port.copy())], 0, n_boot=2000, block=20, rng_seed=0)
    for metric in ("ic", "icir", "sharpe"):
        a = stats["advantage"]["copy"][metric]
        assert a["observed"] == 0.0 and a["ci_low"] < 0.0 < a["ci_high"] and a["verdict"] == NO_DIFFERENCE


def test_a_clearly_shifted_variant_is_detected_in_both_directions(tmp_path):
    cfg = synthetic_config(tmp_path)
    ic, port = _market(5)
    variants = [
        _variant(cfg, "ref", ic, port),
        _variant(cfg, "worse", ic - 0.08, port - 0.003),
        _variant(cfg, "better", ic + 0.08, port + 0.003),
    ]
    stats = summarize(variants, 0, n_boot=2000, block=20, rng_seed=0)
    worse, better = stats["advantage"]["worse"], stats["advantage"]["better"]
    assert worse["ic"]["observed"] == pytest.approx(0.08) and worse["ic"]["ci_low"] > 0
    assert worse["ic"]["verdict"] == worse["sharpe"]["verdict"] == REFERENCE_BETTER
    assert better["ic"]["verdict"] == better["sharpe"]["verdict"] == VARIANT_BETTER
    assert worse["ic"]["share_positive"] == 1.0 and better["ic"]["share_positive"] == 0.0


def test_bootstrap_is_reproducible_with_a_fixed_seed(tmp_path):
    cfg = synthetic_config(tmp_path)
    ic, port = _market(5)
    other_ic, other_port = _market(5, seed=1)
    variants = [_variant(cfg, "ref", ic, port), _variant(cfg, "other", other_ic, other_port)]
    a = summarize(variants, 0, n_boot=500, block=20, rng_seed=7)
    b = summarize(variants, 0, n_boot=500, block=20, rng_seed=7)
    c = summarize(variants, 0, n_boot=500, block=20, rng_seed=8)
    assert a == b
    assert a["advantage"]["other"]["ic"]["ci_low"] != c["advantage"]["other"]["ic"]["ci_low"]


def test_variants_with_different_data_dates_or_names_are_refused(tmp_path):
    cfg = synthetic_config(tmp_path)
    ic, port = _market(2)
    ref = _variant(cfg, "ref", ic, port)
    other_splits = _variant(dataclasses.replace(cfg, splits=dataclasses.replace(cfg.splits, purge_days=10)), "b", ic, port)
    with pytest.raises(CompareError, match="data hash"):
        check_comparable([ref, other_splits])
    shorter = _variant(cfg, "c", ic[:, :-1], port[:, :-1])
    with pytest.raises(CompareError, match="different dates"):
        check_comparable([ref, shorter])
    with pytest.raises(CompareError, match="unique"):
        check_comparable([ref, _variant(cfg, "ref", ic, port)])


def test_load_variant_refuses_foreign_or_unevaluated_run_groups(tmp_path):
    cfg = synthetic_config(tmp_path)
    (tmp_path / "foreign").mkdir()
    (tmp_path / "foreign" / "config.json").write_text(json.dumps({"hash": "0000000000000000"}))
    with pytest.raises(CompareError, match="config hash mismatch"):
        load_variant(cfg, "x.yaml", "foreign")
    (tmp_path / "fresh").mkdir()
    (tmp_path / "fresh" / "config.json").write_text(json.dumps({"hash": cfg.hash()}))
    with pytest.raises(CompareError, match="no saved valid evaluation"):
        load_variant(cfg, "x.yaml", "fresh")
    with pytest.raises(CompareError, match="validation period only"):
        load_variant(cfg, "x.yaml", "fresh", split="test")
    dates = pd.bdate_range("2021-01-04", periods=3)
    pd.DataFrame({"seed": 0, "decision_date": dates}).to_parquet(tmp_path / "fresh" / "series_valid.parquet", index=False)
    (tmp_path / "fresh" / "results_valid.json").write_text(json.dumps({"per_seed": [{"seed": 0}]}))
    with pytest.raises(CompareError, match=r"evaluated on seeds \[0\], not its configured seeds \[0, 1, 2, 3, 4\]"):
        load_variant(cfg, "x.yaml", "fresh")


def test_compare_command_refuses_the_test_split(tmp_path):
    from detmaso.cli import main

    with pytest.raises(SystemExit, match="validation period only"):
        main(["compare", "--runs", "a.yaml:x", "b.yaml:y", "--out", str(tmp_path), "--split", "test"])


def test_run_compare_end_to_end_on_synthetic_runs(tmp_path):
    """Train two tiny variants, save their validation evaluation as `evaluate` does, then compare."""
    from detmaso.eval.report import _json_default, evaluate_seed
    from detmaso.train.trainer import train_seed
    from tests.test_end_to_end import synthetic_panel_and_splits

    panel, splits = synthetic_panel_and_splits()
    runs = tmp_path / "runs"
    paper = synthetic_config(runs, train={"max_epochs": 3, "seeds": (0, 1)}, model={"dtype": "float64"})
    single = synthetic_config(
        runs, variant="single_test", train={"max_epochs": 3, "seeds": (0, 1)},
        model={"dtype": "float64", "kind": "single_mlp"}, loss={"c_div": 0.0, "c_ent": 0.0},
    )  # fmt: skip
    for cfg, tag in ((paper, "ref"), (single, "single")):
        group = runs / tag
        group.mkdir(parents=True)
        (group / "config.json").write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, default=str))
        results = []
        for seed in (0, 1):
            train_seed(cfg, panel, splits, seed, group / f"seed{seed}")
            results.append(evaluate_seed(cfg, panel, group / f"seed{seed}", splits.valid_fit, splits.valid, panel.dates, None))
        pd.concat([r.daily for r in results], ignore_index=True).to_parquet(group / "series_valid.parquet", index=False)
        per_seed = [{"seed": r.seed, "ic": r.ic, "port": r.port, "excess": r.excess} for r in results]
        (group / "results_valid.json").write_text(json.dumps({"per_seed": per_seed}, default=_json_default))

    out = tmp_path / "cmp"
    path = run_compare([(paper, "paper.yaml", "ref"), (single, "single.yaml", "single")], reference="ref", out_dir=out, n_boot=300)
    text = path.read_text()
    payload = json.loads((out / "comparison.json").read_text())
    assert "## Advantage of `paper` over each variant" in text and "## Agents and scheduler" in text
    assert set(payload["advantage"]) == {"single_test"}
    assert payload["advantage"]["single_test"]["ic"]["verdict"] in (REFERENCE_BETTER, VARIANT_BETTER, NO_DIFFERENCE)
    assert [r["weights"] for r in payload["runs"]][1] < [r["weights"] for r in payload["runs"]][0]
    assert sorted(p.name for p in (out / "figures").iterdir()) == ["advantage.png", "cumulative_return.png", "metrics.png"]


def test_scheduler_balance_rules_end_to_end_on_synthetic_runs(tmp_path):
    """Train a reference and two penalty strengths, save their evaluation and training-day usage, then
    apply docs/RULES.md Rules M, S and K with `compare --rule scheduler_balance`."""
    from detmaso.eval.report import _json_default, evaluate_seed
    from detmaso.eval.scheduler_diag import write_usage
    from detmaso.train.trainer import train_seed
    from tests.test_end_to_end import synthetic_panel_and_splits

    panel, splits = synthetic_panel_and_splits()
    runs = tmp_path / "runs"
    common = {"train": {"max_epochs": 3, "seeds": (0, 1)}, "model": {"dtype": "float64"}}
    ref = synthetic_config(runs, **common)
    strengths = [synthetic_config(runs, variant=f"bal_{c}", loss={"c_bal": c, "balance_target": "regime"}, **common) for c in (0.1, 1.0)]
    for cfg, tag in [(ref, "ref"), (strengths[0], "b1"), (strengths[1], "b2")]:
        group = runs / tag
        group.mkdir(parents=True)
        (group / "config.json").write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, default=str))
        results = []
        for seed in (0, 1):
            train_seed(cfg, panel, splits, seed, group / f"seed{seed}")
            results.append(evaluate_seed(cfg, panel, group / f"seed{seed}", splits.valid_fit, splits.valid, panel.dates, None))
        pd.concat([r.daily for r in results], ignore_index=True).to_parquet(group / "series_valid.parquet", index=False)
        per_seed = [{"seed": r.seed, "ic": r.ic, "port": r.port, "excess": r.excess, "dead_days": r.dead_days} for r in results]
        (group / "results_valid.json").write_text(json.dumps({"per_seed": per_seed}, default=_json_default))
        write_usage(cfg, tag, panel, splits)

    out = tmp_path / "cmp"
    runs_spec = [(ref, "paper.yaml", "ref"), (strengths[0], "b1.yaml", "b1"), (strengths[1], "b2.yaml", "b2")]
    text = run_compare(runs_spec, reference="ref", out_dir=out, n_boot=300, rule="scheduler_balance").read_text()
    payload = json.loads((out / "comparison.json").read_text())
    rules = payload["rules"]
    assert "## Rules (docs/RULES.md, scheduler balance)" in text and "## Training days" in text
    assert rules["strengths"] == [0.1, 1.0] and set(rules["rule_m"]) == {"bal_0.1", "bal_1.0"}
    assert rules["target"] == pytest.approx([0.5, 0.0, 0.5, 0.0, 0.0], abs=0.02)  # synthetic regimes: agents 0 and 2
    assert rules["outcome"] in ("kept for re-check", "dropped", "not fixed at any tested strength", "all agents used, but no specialists")
    assert "rule_p" not in rules  # real (not placebo) runs
    figs = sorted(p.name for p in (out / "figures").iterdir())
    assert {"strength_sweep.png", "usage_vs_target.png", "routing.png"} <= set(figs)
