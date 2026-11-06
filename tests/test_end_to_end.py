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
