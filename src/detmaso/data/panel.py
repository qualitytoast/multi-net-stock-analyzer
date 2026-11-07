"""Assemble per-date model inputs: features, labels, masks, market state and regimes.

Masks (plan A3):
  universe_t = index member on t AND traded on t-1 (features computable; known before t)
               -> drives agent-output z-scoring, the scores, and the portfolio pool
  tradable_t = universe_t AND traded on t (can be bought at t's close)
  label_ok_t = universe_t AND the 20-day label exists -> used ONLY inside loss / IC / Shapley
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detmaso.config import Config
from detmaso.data.features import FEATURE_NAMES, compute_features
from detmaso.data.labels import forward_return, next_day_return
from detmaso.data.market_state import market_state
from detmaso.data.qlib_io import RawPanel, load_or_build_raw_panel
from detmaso.data.regimes import RegimeModel, fit_regimes
from detmaso.data.splits import Splits, make_splits

log = logging.getLogger(__name__)


@dataclass
class Panel:
    dates: pd.DatetimeIndex  # T model dates (train start .. test end)
    code_list: list[str]
    feature_names: list[str]
    code_idx: torch.Tensor  # [T, N] long, -1 = padding
    X: torch.Tensor  # [T, N, D] float32 (processed features; 0 outside the universe)
    y: torch.Tensor  # [T, N] float32 20-day forward return (NaN if missing)
    r_next: torch.Tensor  # [T, N] float32 next-day return (NaN if not traded on t+1)
    universe: torch.Tensor  # [T, N] bool
    tradable: torch.Tensor  # [T, N] bool
    label_ok: torch.Tensor  # [T, N] bool
    s_raw: torch.Tensor  # [T, 12] raw market state
    s_sched: torch.Tensor  # [T, 12] standardized + clipped (scheduler input)
    regime: torch.Tensor  # [T] long, agent index of the day's regime
    index_close: torch.Tensor  # [T] index level on t (plots only)

    def date_pos(self, dates: pd.DatetimeIndex) -> torch.Tensor:
        pos = self.dates.get_indexer(dates)
        if (pos < 0).any():
            raise KeyError(f"{int((pos < 0).sum())} dates are not in the panel")
        return torch.as_tensor(pos, dtype=torch.long)

    def save(self, path: Path) -> None:
        payload = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        payload["dates"] = np.asarray(self.dates.values, dtype="datetime64[ns]").astype(np.int64)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "Panel":
        p = torch.load(path, weights_only=False)
        p["dates"] = pd.DatetimeIndex(p["dates"].astype("datetime64[ns]"))
        return cls(**p)


def cross_sectional_process(F: pd.DataFrame, universe: pd.DataFrame, clip: float) -> pd.DataFrame:
    """Per date over the universe: z-score (population std), clip, NaN -> 0 (plan A8-A10)."""
    G = F.where(universe)
    mu = G.mean(axis=1)
    sd = G.std(axis=1, ddof=0).replace(0.0, np.nan)
    Z = G.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)
    Z = Z.mask(universe & Z.isna(), 0.0)
    return Z.where(universe)


def make_masks(close: pd.DataFrame, member: pd.DataFrame, horizon: int):
    """The three stock masks (plan A3) and the labels, from closes and point-in-time membership.

    universe_t: member on t AND traded on t-1 (known before t; drives scores and the portfolio pool)
    tradable_t: universe_t AND traded on t (can be bought at t's close)
    label_ok_t: universe_t AND the horizon-day label exists (used only by the loss, IC and Shapley)
    """
    traded = close.notna()
    universe = member & traded.shift(1, fill_value=False)
    tradable = universe & traded
    labels = forward_return(close, horizon)
    label_ok = universe & labels.notna()
    return universe, tradable, label_ok, labels


@dataclass
class PreparedData:
    raw: RawPanel
    features_raw: dict[str, pd.DataFrame]
    state: pd.DataFrame
    splits: Splits
    regimes: RegimeModel
    panel: Panel
    masks: dict[str, pd.DataFrame]
    labels: pd.DataFrame


def build(cfg: Config, rebuild_raw: bool = False) -> PreparedData:
    raw = load_or_build_raw_panel(cfg, rebuild=rebuild_raw)
    close = raw.fields["close"]
    universe, tradable, label_ok, labels = make_masks(close, raw.member, cfg.features.horizon)
    r_next = next_day_return(close)

    log.info("computing features")
    feats = compute_features(raw.fields, cfg.features.min_window_coverage)
    state = market_state(close, raw.member, cfg.regimes.windows, cfg.features.min_window_coverage)

    cal = raw.calendar
    model_dates = cal[(cal >= pd.Timestamp(cfg.splits.train[0])) & (cal <= pd.Timestamp(cfg.splits.test[1]))]
    splits = make_splits(model_dates, cfg.splits.train, cfg.splits.valid, cfg.splits.test, cfg.splits.purge_days)

    log.info("fitting regimes on %d training dates", len(splits.train))
    regimes = fit_regimes(
        state,
        splits.train,
        n_agents=cfg.regimes.n_agents,
        k_range=(cfg.regimes.k_min, cfg.regimes.k_max),
        n_init=cfg.regimes.n_init,
        seed=cfg.regimes.seed,
    )

    log.info("processing features cross-sectionally")
    processed = [cross_sectional_process(feats[n], universe, cfg.features.zscore_clip) for n in FEATURE_NAMES]
    rows = cal.get_indexer(model_dates)
    feat_cube = np.stack([p.to_numpy(dtype=np.float32)[rows] for p in processed], axis=-1)  # [T, codes, D]

    uni = universe.to_numpy()[rows]
    n_max = int(uni.sum(axis=1).max())
    T, n_codes = uni.shape
    D = len(FEATURE_NAMES)
    code_idx = np.full((T, n_max), -1, dtype=np.int64)
    X = np.zeros((T, n_max, D), dtype=np.float32)
    y = np.full((T, n_max), np.nan, dtype=np.float32)
    rn = np.full((T, n_max), np.nan, dtype=np.float32)
    m_uni = np.zeros((T, n_max), dtype=bool)
    m_trd = np.zeros((T, n_max), dtype=bool)
    m_lab = np.zeros((T, n_max), dtype=bool)
    lab = labels.to_numpy(dtype=np.float32)[rows]
    nxt = r_next.to_numpy(dtype=np.float32)[rows]
    trd = tradable.to_numpy()[rows]
    lok = label_ok.to_numpy()[rows]
    for i in range(T):
        cols = np.flatnonzero(uni[i])
        k = len(cols)
        code_idx[i, :k] = cols
        X[i, :k] = feat_cube[i, cols]
        y[i, :k] = lab[i, cols]
        rn[i, :k] = nxt[i, cols]
        m_uni[i, :k] = True
        m_trd[i, :k] = trd[i, cols]
        m_lab[i, :k] = lok[i, cols]

    s_raw = state.loc[model_dates].to_numpy(dtype=np.float64)
    if np.isnan(s_raw).any():
        raise ValueError("market state has NaN inside the model period; load more warm-up history")
    clip = cfg.regimes.scheduler_input_clip
    s_sched = np.clip(regimes.standardize(s_raw), -clip, clip)
    regime = regimes.assign(s_raw)

    panel = Panel(
        dates=model_dates,
        code_list=raw.codes,
        feature_names=list(FEATURE_NAMES),
        code_idx=torch.from_numpy(code_idx),
        X=torch.from_numpy(X),
        y=torch.from_numpy(y),
        r_next=torch.from_numpy(rn),
        universe=torch.from_numpy(m_uni),
        tradable=torch.from_numpy(m_trd),
        label_ok=torch.from_numpy(m_lab),
        s_raw=torch.from_numpy(s_raw.astype(np.float32)),
        s_sched=torch.from_numpy(s_sched.astype(np.float32)),
        regime=torch.from_numpy(regime.astype(np.int64)),
        index_close=torch.from_numpy(raw.index_close.loc[model_dates].to_numpy(dtype=np.float32)),
    )
    masks = {"universe": universe, "tradable": tradable, "label_ok": label_ok}
    return PreparedData(raw, feats, state, splits, regimes, panel, masks, labels)


def artifacts_dir(cfg: Config) -> Path:
    return cfg.run_path / "prepared"


def save_prepared(data: PreparedData, cfg: Config) -> None:
    d = artifacts_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    data.panel.save(d / "panel.pt")
    data.regimes.to_json(d / "regimes.json")
    data.state.to_parquet(d / "market_state.parquet")
    meta = {
        "data_tag": cfg.data.data_tag,
        "data_hash": cfg.data_hash(),
        "config_hash": cfg.hash(),
        "splits": {k: [str(v[0].date()), str(v[-1].date()), len(v)] for k, v in vars(data.splits).items()},
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_prepared(cfg: Config) -> tuple[Panel, RegimeModel, Splits]:
    d = artifacts_dir(cfg)
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("data_hash") != cfg.data_hash():
        raise ValueError(
            f"prepared data was built with data config {meta.get('data_hash')}, current is {cfg.data_hash()}; "
            "re-run `detmaso prepare`"
        )
    panel = Panel.load(d / "panel.pt")
    regimes = RegimeModel.from_json(d / "regimes.json")
    splits = make_splits(panel.dates, cfg.splits.train, cfg.splits.valid, cfg.splits.test, cfg.splits.purge_days)
    if cfg.is_placebo:  # applied here so that train, evaluate, usage and compare all see the same answers
        from detmaso.data.placebo import shuffle_answers

        panel = shuffle_answers(panel, cfg.placebo.block_days, cfg.placebo.shuffle_seed)
    return panel, regimes, splits
