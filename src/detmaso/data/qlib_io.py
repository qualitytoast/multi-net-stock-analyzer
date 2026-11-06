"""Qlib access: calendar, point-in-time index membership and raw daily fields.

Everything is pulled once into wide (date x stock) DataFrames and cached as parquet, so the
rest of the pipeline is plain pandas/numpy that can be unit-tested for look-ahead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from detmaso.config import Config

log = logging.getLogger(__name__)

RAW_FIELDS = ("open", "high", "low", "close", "volume", "factor")


def init_qlib(cfg: Config) -> None:
    import qlib

    qlib.init(
        provider_uri=str(Path(cfg.data.provider_uri).expanduser()),
        region=cfg.data.region,
        kernels=cfg.data.qlib_kernels,
    )


@dataclass
class RawPanel:
    """Raw daily data for every stock that was ever an index member in the load window."""

    fields: dict[str, pd.DataFrame]  # field -> (date x code), adjusted prices, NaN when not traded
    member: pd.DataFrame  # (date x code) bool, point-in-time index membership
    index_close: pd.Series  # the index level (price index) per date
    calendar: pd.DatetimeIndex

    @property
    def codes(self) -> list[str]:
        return list(self.member.columns)


def _membership_frame(spans: dict, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    member = pd.DataFrame(False, index=calendar, columns=sorted(spans))
    for code, periods in spans.items():
        for start, end in periods:
            member.loc[pd.Timestamp(start) : pd.Timestamp(end), code] = True
    return member


def load_raw_panel(cfg: Config) -> RawPanel:
    """Load from Qlib (no membership filter on prices, so each stock keeps its full history)."""
    from qlib.data import D

    init_qlib(cfg)
    start, end = cfg.data.load_start, cfg.data.load_end
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))
    spans = D.list_instruments(
        D.instruments(cfg.data.universe), start_time=start, end_time=end, as_list=False
    )
    member = _membership_frame(spans, calendar)
    codes = list(member.columns)
    log.info("loading %d fields for %d ever-members over %d dates", len(RAW_FIELDS), len(codes), len(calendar))
    raw = D.features(codes, [f"${f}" for f in RAW_FIELDS], start_time=start, end_time=end, freq="day")
    fields = {}
    for f in RAW_FIELDS:
        wide = raw[f"${f}"].unstack(level="instrument")
        fields[f] = wide.reindex(index=calendar, columns=codes).astype("float64")
    idx = D.features([cfg.data.index_symbol], ["$close"], start_time=start, end_time=end, freq="day")
    index_close = idx["$close"].droplevel("instrument").reindex(calendar).astype("float64")
    return RawPanel(fields=fields, member=member, index_close=index_close, calendar=calendar)


def cache_dir(cfg: Config) -> Path:
    return cfg.run_path / "cache" / f"raw_{cfg.data.data_tag}"


def save_raw_panel(panel: RawPanel, cfg: Config) -> None:
    d = cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for name, frame in panel.fields.items():
        frame.to_parquet(d / f"{name}.parquet")
    panel.member.to_parquet(d / "member.parquet")
    panel.index_close.to_frame("index_close").to_parquet(d / "index_close.parquet")
    meta = {
        "data_tag": cfg.data.data_tag,
        "data_sha256": cfg.data.data_sha256,
        "load_start": cfg.data.load_start,
        "load_end": cfg.data.load_end,
        "universe": cfg.data.universe,
        "index_symbol": cfg.data.index_symbol,
        "n_codes": len(panel.codes),
        "n_dates": len(panel.calendar),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_or_build_raw_panel(cfg: Config, rebuild: bool = False) -> RawPanel:
    d = cache_dir(cfg)
    wanted = {
        "data_tag": cfg.data.data_tag,
        "load_start": cfg.data.load_start,
        "load_end": cfg.data.load_end,
        "universe": cfg.data.universe,
        "index_symbol": cfg.data.index_symbol,
    }
    if (d / "meta.json").exists():
        meta = json.loads((d / "meta.json").read_text())
        if any(meta.get(k) != v for k, v in wanted.items()):
            log.info("raw cache does not match the config %s; rebuilding", wanted)
            rebuild = True
    if not rebuild and (d / "meta.json").exists():
        fields = {f: pd.read_parquet(d / f"{f}.parquet") for f in RAW_FIELDS}
        member = pd.read_parquet(d / "member.parquet")
        index_close = pd.read_parquet(d / "index_close.parquet")["index_close"]
        return RawPanel(fields=fields, member=member, index_close=index_close, calendar=member.index)
    panel = load_raw_panel(cfg)
    save_raw_panel(panel, cfg)
    return panel
