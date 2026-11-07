"""Shared fixtures: synthetic markets (run anywhere) and the real prepared data (marked `data`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from detmaso.config import Config, DataConfig, SplitConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def synthetic_fields(n_days: int = 320, n_stocks: int = 8, seed: int = 0, suspend: bool = True) -> dict[str, pd.DataFrame]:
    """Random-walk OHLCV for a few stocks, with a couple of suspension gaps."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"S{i:02d}" for i in range(n_stocks)]
    ret = rng.normal(0.0005, 0.02, size=(n_days, n_stocks))
    close = 10.0 * np.exp(np.cumsum(ret, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, size=close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, size=close.shape)))
    volume = rng.lognormal(10, 0.5, size=close.shape)
    factor = np.ones_like(close)
    frames = {
        "open": open_, "high": high, "low": low, "close": close, "volume": volume, "factor": factor,
    }  # fmt: skip
    out = {k: pd.DataFrame(v, index=dates, columns=codes) for k, v in frames.items()}
    if suspend:
        for k in ("open", "high", "low", "close", "volume", "factor"):
            out[k].iloc[150:153, 1] = np.nan  # stock 1 suspended 3 days
            out[k].iloc[200:201, 3] = np.nan  # stock 3 suspended 1 day
    return out


def synthetic_config(run_dir: Path, variant: str = "paper", **overrides) -> Config:
    """A small config for synthetic panels; overrides are {section: {field: value}}, applied together."""
    import dataclasses

    from detmaso.config import LossConfig, ModelConfig, TrainConfig

    cfg = Config(
        market="synthetic",
        run_dir=str(run_dir),
        data=DataConfig(
            provider_uri="unused", region="cn", data_tag="synthetic", data_sha256="-", universe="all",
            index_symbol="-", load_start="2020-01-01", load_end="2021-12-31",
        ),
        splits=SplitConfig(train=("2020-01-01", "2020-12-31"), valid=("2021-01-01", "2021-03-31"), test=("2021-04-01", "2021-12-31"), purge_days=20),
        model=ModelConfig(hidden=(16, 8, 4), activation="gelu"),
        loss=LossConfig(),
        train=TrainConfig(max_epochs=4, patience=8, weight_decay_mode="decoupled"),
    )  # fmt: skip
    sections = {section: dataclasses.replace(getattr(cfg, section), **values) for section, values in overrides.items()}
    return dataclasses.replace(cfg, variant=variant, **sections)  # validated once, as a whole


@pytest.fixture
def fields():
    return synthetic_fields()


@pytest.fixture(scope="session")
def real():
    """The prepared CSI300 data (skips if `detmaso prepare` has not been run)."""
    cfg = load_config(ROOT / "configs/csi300.yaml")
    prepared = (ROOT / cfg.run_dir / "prepared" / "panel.pt")
    if not prepared.exists():
        pytest.skip("prepared data missing; run `detmaso prepare --config configs/csi300.yaml`")
    import os

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        from detmaso.data.panel import load_prepared

        panel, regimes, splits = load_prepared(cfg)
    finally:
        os.chdir(cwd)
    return cfg, panel, regimes, splits
