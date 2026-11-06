"""Typed configuration. Every plan ASSUMPTION that is a value lives here (see configs/*.yaml)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    provider_uri: str
    region: str
    data_tag: str
    data_sha256: str
    universe: str
    index_symbol: str
    load_start: str
    load_end: str
    qlib_kernels: int = 1  # Qlib's joblib multiprocessing is very slow on macOS; 1 = in-process


@dataclass(frozen=True)
class SplitConfig:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]
    purge_days: int = 20  # A6
    sub_window: tuple[str, str] = ("2024-03-01", "2025-04-18")  # A45


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 20  # label horizon tau
    min_window_coverage: float = 0.8  # A8 rule (2)
    zscore_clip: float = 5.0  # A9
    min_label_ok: int = 50  # A8 rule (6)


@dataclass(frozen=True)
class RegimeConfig:
    windows: tuple[int, ...] = (5, 20, 60, 120)
    n_agents: int = 5  # D2 = a
    k_min: int = 2
    k_max: int = 10  # A14
    n_init: int = 50
    seed: int = 0
    scheduler_input_clip: float = 5.0  # A13


@dataclass(frozen=True)
class ModelConfig:
    hidden: tuple[int, ...] = (64, 32, 16)  # A17
    activation: str = "relu"  # A17: "relu" or "gelu"
    dtype: str = "float32"  # A35: "float32" or "float64"
    encoder_hidden: int = 16  # A19
    head_hidden: int = 32  # paper: W_1 in R^{32x16}


@dataclass(frozen=True)
class LossConfig:
    c_ic: float = 1.0
    c_sr: float = 0.5
    c_div: float = 0.1
    c_ent: float = 0.01
    lambda_mask: float = 0.1  # D5 = a
    kappa_soft: float = 1.0  # D4 = a
    shrink: float = 1e-4  # A23
    sharpe_min_points: int = 20  # A24
    sharpe_eps: float = 1e-8
    sharpe_window: int | None = None  # A24: None = expanding within the epoch (T_train = #train dates)


@dataclass(frozen=True)
class TrainConfig:
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    weight_decay_mode: str = "coupled"  # A29: "coupled" (Adam L2) or "decoupled" (AdamW)
    clip_norm: float = 1.0
    max_epochs: int = 200
    patience: int = 8
    smooth: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "auto"  # A35 / D9: CUDA if present, else CPU (Apple GPU not used)
    dead_std: float = 1e-6  # A27: raw output spread below this counts as a constant agent
    max_skip_frac: float = 0.01
    max_consecutive_skips: int = 20
    max_nonfinite: int = 10
    grad_norm_every: int = 50  # log per-term gradient norms every N steps
    diversity_last_epochs: int = 40  # A44


@dataclass(frozen=True)
class BacktestConfig:
    top_frac: float = 0.10  # A37
    qlib_days: int = 238  # Qlib risk_analysis convention, reported alongside


@dataclass(frozen=True)
class Config:
    market: str
    run_dir: str
    data: DataConfig
    splits: SplitConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir).expanduser()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Stable hash of everything that affects results (used for the test-period lock)."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _tuplify(value):
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _build(cls, raw: dict | None):
    raw = raw or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _tuplify(v) for k, v in raw.items()})


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        market=raw["market"],
        run_dir=raw["run_dir"],
        data=_build(DataConfig, raw["data"]),
        splits=_build(SplitConfig, raw["splits"]),
        features=_build(FeatureConfig, raw.get("features")),
        regimes=_build(RegimeConfig, raw.get("regimes")),
        model=_build(ModelConfig, raw.get("model")),
        loss=_build(LossConfig, raw.get("loss")),
        train=_build(TrainConfig, raw.get("train")),
        backtest=_build(BacktestConfig, raw.get("backtest")),
    )
