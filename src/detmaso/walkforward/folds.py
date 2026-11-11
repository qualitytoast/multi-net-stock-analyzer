"""Walk-forward folds and their market moods (docs/RULES.md, Rule WL).

Fold Y trains on every model date from the history start to the end of year Y - E - 1, early-stops on the
E years before Y (E = walkforward.early_stop_years) and is tested on year Y; the last fold's test window runs
on to walkforward.end. Each fold is an ordinary ``Splits`` (train = training, valid = early stopping, test =
test window), so the paper trainer runs on it unchanged. Purges: ``train_fit`` and ``valid_fit`` drop the last
``purge_days`` dates of their window, and so does ``Fold.test_ic`` (IC days whose 20-day answers end inside
the test window). The portfolio uses every test date.

Moods (``walkforward.moods: per_fold``): the K-means moods and the scheduler's input scaling are refit on every
model date before the fold's test window (market state only, never answers), named by the fixed rule A16, and
cached per fold next to the prepared data. ``fold_panel`` swaps them into the panel.

Nothing here may touch the locked test period: ``check_fold`` refuses a fold whose test window, answers or
next-day returns reach ``splits.test[0]``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detmaso.config import Config
from detmaso.data.panel import Panel, artifacts_dir
from detmaso.data.regimes import RegimeModel, fit_regimes
from detmaso.data.splits import Splits, make_splits


class FoldError(ValueError):
    pass


@dataclass(frozen=True)
class Fold:
    year: int  # the test year (the fold's name)
    splits: Splits  # train / train_fit: training; valid / valid_fit: early stopping; test: test window
    test_ic: pd.DatetimeIndex  # test-window dates whose answers end inside the window

    @property
    def name(self) -> str:
        return f"fold{self.year}"

    @property
    def pre_test(self) -> pd.DatetimeIndex:
        """Every model date before the test window (what the fold's moods are fit on)."""
        return self.splits.train.append(self.splits.valid)

    def describe(self) -> dict:
        s = self.splits

        def span(idx: pd.DatetimeIndex) -> list:
            return [str(idx[0].date()), str(idx[-1].date()), len(idx)]

        return {"year": self.year, "train": span(s.train), "train_fit": span(s.train_fit), "early_stop": span(s.valid),
                "early_stop_ic": span(s.valid_fit), "test": span(s.test), "test_ic": span(self.test_ic)}  # fmt: skip


def fold_layout(cfg: Config, dates: pd.DatetimeIndex) -> list[Fold]:
    """The folds of a walk-forward config over the panel's model dates, each checked by ``check_fold``."""
    wf = cfg.walkforward
    if wf is None:
        raise FoldError("this config has no walkforward section")
    purge = cfg.splits.purge_days
    folds = []
    for year in wf.years:
        es_first = year - wf.early_stop_years
        end = wf.end if year == wf.last_test_year else f"{year}-12-31"
        s = make_splits(dates, (cfg.splits.train[0], f"{es_first - 1}-12-31"), (f"{es_first}-01-01", f"{year - 1}-12-31"), (f"{year}-01-01", end), purge)
        fold = Fold(year, s, s.test[:-purge] if purge else s.test)
        check_fold(fold, dates, pd.Timestamp(cfg.splits.test[0]), cfg.features.horizon)
        folds.append(fold)
    return folds


def check_fold(fold: Fold, dates: pd.DatetimeIndex, test_start: pd.Timestamp, horizon: int) -> None:
    """Order, purges and the test-period guard: no answer or return of the fold may reach ``test_start``."""
    s = fold.splits
    for name, idx in (("training", s.train_fit), ("early-stopping", s.valid_fit), ("test", s.test), ("test IC", fold.test_ic)):
        if len(idx) == 0:
            raise FoldError(f"fold {fold.year}: no {name} dates")
    if not (s.train[-1] < s.valid[0] and s.valid[-1] < s.test[0]):
        raise FoldError(f"fold {fold.year}: windows overlap or are out of order")

    def label_end(day: pd.Timestamp) -> pd.Timestamp:
        pos = dates.get_loc(day) + horizon
        if pos >= len(dates):
            raise FoldError(f"fold {fold.year}: the answer of {day.date()} lies beyond the loaded dates")
        return dates[pos]

    # Each window's answers end inside it (so nothing trained or picked on reaches the next window).
    for name, used, window_end in (("training", s.train_fit, s.train[-1]), ("early-stopping", s.valid_fit, s.valid[-1]), ("test", fold.test_ic, s.test[-1])):
        if label_end(used[-1]) > window_end:
            raise FoldError(f"fold {fold.year}: {name} answers reach past the window's end {window_end.date()}")
    pos = dates.get_loc(s.test[-1])
    if pos + 1 >= len(dates) or dates[pos + 1] >= test_start or s.test[-1] >= test_start:
        raise FoldError(f"fold {fold.year}: its test window or its last next-day return reaches the locked test period ({test_start.date()})")


# Moods per fold ------------------------------------------------------------------------------------------
def fold_dir(cfg: Config, fold: Fold) -> Path:
    return artifacts_dir(cfg) / "folds" / fold.name


def load_market_state(cfg: Config) -> pd.DataFrame:
    """The 12 raw market-state columns saved by `detmaso prepare` (float64; answers are never involved)."""
    return pd.read_parquet(artifacts_dir(cfg) / "market_state.parquet")


def fit_fold_regimes(cfg: Config, state: pd.DataFrame, fold: Fold) -> RegimeModel:
    r = cfg.regimes
    return fit_regimes(state, fold.pre_test, n_agents=r.n_agents, k_range=(r.k_min, r.k_max), n_init=r.n_init, seed=r.seed)


def fold_regimes(cfg: Config, state: pd.DataFrame, fold: Fold, fit: bool = True) -> RegimeModel:
    """The fold's moods, from the cache when it matches (data hash and fit dates), else fitted and cached."""
    d = fold_dir(cfg, fold)
    want = {"data_hash": cfg.data_hash(), "fit_first": str(fold.pre_test[0].date()), "fit_last": str(fold.pre_test[-1].date()),
            "n_fit_dates": len(fold.pre_test), "moods": cfg.walkforward.moods}  # fmt: skip
    meta_path = d / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) == want:
        return RegimeModel.from_json(d / "regimes.json")
    if not fit:
        raise FoldError(f"{fold.name}: moods are not cached in {d}; train the run group first")
    reg = fit_fold_regimes(cfg, state, fold)
    if pd.Timestamp(reg.fit_end) >= fold.splits.test[0]:
        raise FoldError(f"{fold.name}: moods were fit on dates up to {reg.fit_end}, inside the test window")
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".regimes.{os.getpid()}.tmp"  # atomic writes: another process may be reading or fitting the same fold
    reg.to_json(tmp)
    os.replace(tmp, d / "regimes.json")
    tmp.write_text(json.dumps(want, indent=2))
    os.replace(tmp, meta_path)
    return reg


def fold_panel(panel: Panel, state: pd.DataFrame, regimes: RegimeModel, clip: float) -> Panel:
    """The panel with the fold's mood labels and scheduler inputs (computed as in `prepare`, from float64 state)."""
    s = state.loc[panel.dates].to_numpy(dtype=np.float64)
    if np.isnan(s).any():
        raise FoldError("market state has NaN inside the model period")
    s_sched = np.clip(regimes.standardize(s), -clip, clip)
    return dataclasses.replace(
        panel,
        s_sched=torch.from_numpy(s_sched.astype(np.float32)),
        regime=torch.from_numpy(regimes.assign(s).astype(np.int64)),
    )


def mood_summary(regimes: RegimeModel, state: pd.DataFrame, fold: Fold, previous: tuple[RegimeModel, Fold] | None) -> dict:
    """Per fold: silhouette-best K, mood shares of the training and test days, and the drift from the previous fold.

    Drift: each mood's centre shift (in the current fold's standardized units) and the share of the previous
    fold's fit days whose mood changes between the two fits.
    """
    n = len(regimes.names)

    def shares(idx: pd.DatetimeIndex) -> list[float]:
        lab = regimes.assign(state.loc[idx].to_numpy(dtype=np.float64))
        return (np.bincount(lab, minlength=n) / len(lab)).round(4).tolist()

    out = {
        "fit": [str(fold.pre_test[0].date()), str(fold.pre_test[-1].date())],
        "best_k": int(max(regimes.silhouette, key=regimes.silhouette.get)),
        "train_shares": shares(fold.splits.train_fit),
        "test_shares": shares(fold.splits.test),
        "centre_shift": None,
        "relabelled": None,
    }
    if previous is not None:
        prev, prev_fold = previous
        prev_raw = prev.centroids * prev.std + prev.mean
        shift = np.linalg.norm(regimes.centroids - regimes.standardize(prev_raw), axis=1)
        s = state.loc[prev_fold.pre_test].to_numpy(dtype=np.float64)
        out["centre_shift"] = shift.round(3).tolist()
        out["relabelled"] = float(np.mean(prev.assign(s) != regimes.assign(s)))
    return out
