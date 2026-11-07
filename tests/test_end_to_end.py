"""Plan tests 28-30: synthetic-market training, determinism, and real-data pipeline sanity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from detmaso.data.panel import Panel
from detmaso.data.splits import make_splits
from tests.conftest import synthetic_config


def synthetic_panel_and_splits(n_days: int = 360, n_stocks: int = 80, seed: int = 0):
    """A market where the label depends on feature 3 in regime 0 and on feature 7 otherwise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    X = rng.normal(size=(n_days, n_stocks, 34)).astype(np.float32)
    regime = (np.arange(n_days) // 30) % 2  # alternating 30-day regimes
    signal = np.where(regime[:, None] == 0, X[..., 3], X[..., 7])
    y = (0.02 * signal + 0.05 * rng.normal(size=(n_days, n_stocks))).astype(np.float32)
    s = rng.normal(size=(n_days, 12)).astype(np.float32)
    s[:, 0] += 3.0 * (regime * 2 - 1)  # the scheduler can see the regime
    ok = np.ones((n_days, n_stocks), dtype=bool)
    panel = Panel(
        dates=dates,
        code_list=[f"S{i}" for i in range(n_stocks)],
        feature_names=[f"f{i}" for i in range(34)],
        code_idx=torch.arange(n_stocks).repeat(n_days, 1),
        X=torch.from_numpy(X),
        y=torch.from_numpy(y),
        r_next=torch.from_numpy((y / 20).astype(np.float32)),
        universe=torch.from_numpy(ok),
        tradable=torch.from_numpy(ok),
        label_ok=torch.from_numpy(ok),
        s_raw=torch.from_numpy(s),
        s_sched=torch.from_numpy(s),
        regime=torch.from_numpy(regime.astype(np.int64) * 2),  # agents 0 and 2
        index_close=torch.ones(n_days),
    )
    splits = make_splits(dates, ("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-03-31"), ("2021-04-01", "2021-12-31"), 20)
    return panel, splits


# 28 ------------------------------------------------------------------------------------------
def test_synthetic_market_is_learned(tmp_path):
    from detmaso.train.trainer import train_seed

    cfg = synthetic_config(tmp_path, train={"max_epochs": 6}, model={"dtype": "float64"})
    panel, splits = synthetic_panel_and_splits()
    summary = train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "seed0")
    assert summary["best_val_ic_smoothed"] > 0.1
    history = [json.loads(line) for line in (tmp_path / "seed0" / "history.jsonl").read_text().splitlines()]
    assert history[-1]["mean_abs_corr"] < 0.99


# 29 ------------------------------------------------------------------------------------------
def test_training_is_deterministic(tmp_path):
    from detmaso.train.trainer import train_seed

    cfg = synthetic_config(tmp_path, train={"max_epochs": 3}, model={"dtype": "float64"})
    panel, splits = synthetic_panel_and_splits()
    a = train_seed(cfg, panel, splits, seed=3, out_dir=tmp_path / "a")
    b = train_seed(cfg, panel, splits, seed=3, out_dir=tmp_path / "b")
    assert a["val_ic_curve"] == b["val_ic_curve"]


# 30 ------------------------------------------------------------------------------------------
@pytest.mark.data
def test_pipeline_sanity_single_feature_ic_on_training_period(real):
    """China A-shares: short-term reversal and low-volatility effects make these ICs negative."""
    from detmaso.losses.stats import daily_spearman

    cfg, panel, regimes, splits = real
    pos = panel.date_pos(splits.train_fit)
    y, m = panel.y[pos].numpy(), panel.label_ok[pos].numpy()
    for name in ("mom_20", "vol_20"):
        j = panel.feature_names.index(name)
        ic = np.nanmean(daily_spearman(panel.X[pos, :, j].numpy(), y, m))
        assert ic < 0, (name, ic)


# Step 1: paper path unchanged, single-network baseline ----------------------------------------
GOLDEN_VAL_IC = [0.30065209052550823, 0.3278283893790223, 0.328831031837361]
GOLDEN_LOSS = [-0.5917139715520398, -1.9705881877762073, -2.1203809532437607]
GOLDEN_PARAM_ABS_SUM = 724.0504740523643


def test_paper_training_path_matches_the_golden_run(tmp_path):
    """Recorded with the code as of the CSI300 MVP (before Step 1): the paper path must not change."""
    from detmaso.train.trainer import train_seed

    cfg = synthetic_config(tmp_path, train={"max_epochs": 3}, model={"dtype": "float64"})
    panel, splits = synthetic_panel_and_splits()
    summary = train_seed(cfg, panel, splits, seed=3, out_dir=tmp_path / "seed3")
    history = [json.loads(line) for line in (tmp_path / "seed3" / "history.jsonl").read_text().splitlines()]
    state = torch.load(tmp_path / "seed3" / "model.pt", weights_only=True)
    assert summary["val_ic_curve"] == pytest.approx(GOLDEN_VAL_IC, rel=1e-9, abs=0)
    assert [h["loss"] for h in history] == pytest.approx(GOLDEN_LOSS, rel=1e-9, abs=0)
    assert float(sum(v.double().abs().sum() for v in state.values())) == pytest.approx(GOLDEN_PARAM_ABS_SUM, rel=1e-9, abs=0)


def _single_config(tmp_path, epochs: int, state: bool = False):
    return synthetic_config(
        tmp_path, variant="single_test", train={"max_epochs": epochs},
        model={"dtype": "float64", "kind": "single_mlp", "market_state_input": state}, loss={"c_div": 0.0, "c_ent": 0.0},
    )  # fmt: skip


def test_single_network_learns_and_market_state_helps_on_a_regime_switching_market(tmp_path):
    """The synthetic label depends on feature 3 in one regime and feature 7 in the other; only the
    market state reveals the regime, so a network that sees it can do better than one that cannot."""
    from detmaso.train.trainer import train_seed

    panel, splits = synthetic_panel_and_splits()
    plain = train_seed(_single_config(tmp_path / "a", 6), panel, splits, seed=0, out_dir=tmp_path / "a" / "seed0")
    state = train_seed(_single_config(tmp_path / "b", 6, state=True), panel, splits, seed=0, out_dir=tmp_path / "b" / "seed0")
    assert plain["best_val_ic_smoothed"] > 0.1
    assert state["best_val_ic_smoothed"] > plain["best_val_ic_smoothed"] + 0.03


def test_single_network_training_is_deterministic(tmp_path):
    from detmaso.train.trainer import train_seed

    panel, splits = synthetic_panel_and_splits()
    a = train_seed(_single_config(tmp_path / "a", 3), panel, splits, seed=1, out_dir=tmp_path / "a" / "s")
    b = train_seed(_single_config(tmp_path / "b", 3), panel, splits, seed=1, out_dir=tmp_path / "b" / "s")
    assert a["val_ic_curve"] == b["val_ic_curve"]


@pytest.mark.parametrize("kind", ["detmaso", "single_mlp"])
def test_evaluate_seed_saves_daily_series_for_both_model_kinds(tmp_path, kind):
    from detmaso.eval.report import evaluate_seed
    from detmaso.train.trainer import train_seed

    if kind == "single_mlp":
        cfg = _single_config(tmp_path, 3)
    else:
        cfg = synthetic_config(tmp_path, train={"max_epochs": 3}, model={"dtype": "float64"})
    panel, splits = synthetic_panel_and_splits()
    train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "seed0")
    r = evaluate_seed(cfg, panel, tmp_path / "seed0", splits.valid_fit, splits.valid, panel.dates, None)
    n_agents = 1 if kind == "single_mlp" else 5
    assert (r.shapley is None) == (kind == "single_mlp")
    assert len(r.daily) == len(splits.valid) and int(r.daily["in_ic"].sum()) == len(splits.valid_fit)
    assert np.isfinite(r.daily.loc[r.daily["in_ic"], "ic"]).all() and r.daily.loc[~r.daily["in_ic"], "ic"].isna().all()
    w = r.daily[[f"w{i}" for i in range(n_agents)]].to_numpy()
    assert np.allclose(w.sum(axis=1), 1.0) and f"w{n_agents}" not in r.daily
    assert r.ic["ic"] == pytest.approx(np.nanmean(r.daily["ic"]), abs=1e-15)
