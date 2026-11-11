"""Typed configuration. Every plan ASSUMPTION that is a value lives here (see configs/*.yaml).

Settings added after the paper reproduction are declared with ``post_paper(default)``. Their default
reproduces the paper method exactly, and while a flag keeps its default it is left out of the config
hashes, so the paper config keeps the hash recorded in locks/<market>_test.json. A ``label(...)``
field is descriptive only and never hashed; an ``eval_only(...)`` section (trading costs) only changes how
trained models are scored, so it is never hashed either (changing it never retrains or creates a new try).

A YAML config may start from another one with ``base: other.yaml`` (a path relative to the file) and
override only what differs; nested sections are merged key by key. ``base`` may also be a list, merged in
order (later files win), e.g. ``base: [../csi300_single_mlp.yaml, _layer.yaml]`` for a walk-forward variant.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

MODEL_KINDS = ("detmaso", "single_mlp")
BALANCE_TARGETS = ("uniform", "regime")
WF_MOODS = ("per_fold",)
_POST_PAPER, _LABEL, _EVAL = "post_paper", "label", "eval"


def post_paper(default):
    """A switch added after the paper reproduction; its default reproduces the paper exactly."""
    return field(default=default, metadata={"hash": _POST_PAPER})


def label(default):
    """Descriptive metadata that does not affect results; never part of a config hash."""
    return field(default=default, metadata={"hash": _LABEL})


def eval_only(factory):
    """Settings that only change how trained models are scored (never part of a config hash)."""
    return field(default_factory=factory, metadata={"hash": _EVAL})


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
    activation: str = "gelu"  # A17, revised to GELU (DEVIATIONS 1); "relu" also supported
    dtype: str = "float64"  # A35, revised to float64 (DEVIATIONS 3); "float32" also supported
    encoder_hidden: int = 16  # A19: hidden width of each per-window MLP (its output is fixed at 16)
    head_hidden: int = 32  # paper: W_1 in R^{32x16}
    # Step 1 (ROADMAP.md): "single_mlp" = one agent-shaped network, no scheduler, no regime mask.
    kind: str = post_paper("detmaso")
    # Step 1, single_mlp only: append the 12 market-state inputs (the scheduler's s_t) to every stock.
    market_state_input: bool = post_paper(False)

    def __post_init__(self):
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"model.kind must be one of {MODEL_KINDS}, got {self.kind!r}")


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
    # Scheduler load balancing (ROADMAP "Fix the scheduler collapse"; docs/RULES.md): the loss adds
    # c_bal * KL(target || usage), usage = the scheduler's mean weight per agent over all training days.
    c_bal: float = post_paper(0.0)
    # "uniform" = equal shares; "regime" = each regime's share of the training days.
    balance_target: str = post_paper("uniform")

    def __post_init__(self):
        object.__setattr__(self, "c_bal", float(self.c_bal))  # 1 and 1.0 must give the same fingerprint
        if self.c_bal < 0:
            raise ValueError(f"loss.c_bal must be >= 0, got {self.c_bal}")
        if self.balance_target not in BALANCE_TARGETS:
            raise ValueError(f"loss.balance_target must be one of {BALANCE_TARGETS}, got {self.balance_target!r}")


@dataclass(frozen=True)
class TrainConfig:
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    weight_decay_mode: str = "decoupled"  # A29, revised to AdamW (DEVIATIONS 2); "coupled" = Adam L2
    clip_norm: float = 1.0
    max_epochs: int = 200
    patience: int = 8
    smooth: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "auto"  # A35 / D9: CUDA if present, else CPU (Apple GPU not used)
    dead_std: float = 1e-12  # A27: raw spread below this = constant output (float64 scale; DEVIATIONS 3-4)
    max_skip_frac: float = 0.01
    max_consecutive_skips: int = 20
    max_nonfinite: int = 10
    grad_norm_every: int = 50  # log per-term gradient norms every N steps
    diversity_last_epochs: int = 40  # A44
    # False = a training-only run (the strength pilot, docs/RULES.md Rule G): exactly max_epochs, the
    # validation period is never evaluated, the last epoch is the model, and `evaluate` refuses the run.
    validate: bool = post_paper(True)


@dataclass(frozen=True)
class PlaceboConfig:
    """Placebo test (docs/RULES.md, Rule P): shuffle the answers so there is nothing real to find.

    shuffle_seed None = real answers. Otherwise, within each block of ``block_days`` model dates, the
    stocks labelled on every day of the block get another such stock's 20-day labels and next-day returns.
    """

    shuffle_seed: int | None = None
    block_days: int = 20

    def __post_init__(self):
        if self.block_days < 1:
            raise ValueError(f"placebo.block_days must be >= 1, got {self.block_days}")


@dataclass(frozen=True)
class BacktestConfig:
    top_frac: float = 0.10  # A37
    qlib_days: int = 238  # Qlib risk_analysis convention, reported alongside


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward evaluation (docs/RULES.md, Rule WL); the section is absent (None) in every other config.

    Fold Y (first_test_year <= Y <= last_test_year): train from splits.train[0] to the end of year
    Y - early_stop_years - 1, early-stop on the early_stop_years before Y, test on year Y. The last fold's
    test window runs on to ``end``, which must lie before the locked test period.
    """

    first_test_year: int = 2013
    last_test_year: int = 2023
    end: str = "2024-02-28"
    early_stop_years: int = 1
    moods: str = "per_fold"  # Rule WL: K-means moods and scheduler scaling refit on each fold's pre-test days

    def __post_init__(self):
        if self.moods not in WF_MOODS:
            raise ValueError(f"walkforward.moods must be one of {WF_MOODS}, got {self.moods!r}")
        if self.early_stop_years < 1:
            raise ValueError("walkforward.early_stop_years must be >= 1")
        if self.last_test_year < self.first_test_year:
            raise ValueError("walkforward.last_test_year must be >= first_test_year")
        end = pd.Timestamp(self.end)
        if not (pd.Timestamp(f"{self.last_test_year}-01-01") <= end < pd.Timestamp(f"{self.last_test_year + 2}-01-01")):
            raise ValueError(f"walkforward.end {self.end} must fall in {self.last_test_year} or the year after")

    @property
    def years(self) -> list[int]:
        return list(range(self.first_test_year, self.last_test_year + 1))


@dataclass(frozen=True)
class TradingConfig:
    """How trained models are traded in the walk-forward reports (docs/RULES.md, Rule WT); never hashed.

    Rates are in basis points (1 bp = 0.01%) of the traded value, charged on every buy and every sell.
    Realistic setting: buys pay fee_bps + slippage_bps; sells pay the same plus the stamp duty in force on
    the trade date (stamp_duty_bps: (first date, bps) steps; a date before the first step is an error).
    """

    flat_bps: tuple[float, ...] = (0.0, 10.0, 20.0)
    fee_bps: float = 3.0  # broker commission + exchange, management and transfer fees, each side
    slippage_bps: float = 5.0  # each side
    stamp_duty_bps: tuple[tuple[str, float], ...] = (("2008-09-19", 10.0), ("2023-08-28", 5.0))  # sells only

    def __post_init__(self):
        object.__setattr__(self, "flat_bps", tuple(float(x) for x in self.flat_bps))
        object.__setattr__(self, "fee_bps", float(self.fee_bps))
        object.__setattr__(self, "slippage_bps", float(self.slippage_bps))
        object.__setattr__(self, "stamp_duty_bps", tuple((str(d), float(b)) for d, b in self.stamp_duty_bps))
        dates = [pd.Timestamp(d) for d, _ in self.stamp_duty_bps]
        if not dates or dates != sorted(dates):
            raise ValueError("trading.stamp_duty_bps needs at least one (date, bps) step, in date order")
        if min(self.flat_bps, default=0.0) < 0 or self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("trading costs must be >= 0")


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
    # Whole section left out of both hashes while off; never part of data_hash (the prepared data is reused).
    placebo: PlaceboConfig = post_paper(PlaceboConfig())
    # Walk-forward evaluation (docs/RULES.md, Rule WL); None = the single train / validation / test split.
    walkforward: WalkForwardConfig | None = post_paper(None)
    trading: TradingConfig = eval_only(TradingConfig)
    variant: str = label("paper")  # name shown in reports and comparisons

    def __post_init__(self):
        m, loss = self.model, self.loss
        if self.walkforward is not None:
            self._check_walkforward()
        if self.variant == "paper" and _post_paper_changed(self):
            raise ValueError("this config changes post-paper flags, so it is not the paper method: give it a variant name")
        if m.kind == "single_mlp" and (loss.c_div != 0 or loss.c_ent != 0):
            raise ValueError("model.kind single_mlp trains with the IC and Sharpe terms only: set loss.c_div and loss.c_ent to 0")
        if m.market_state_input and m.kind != "single_mlp":
            raise ValueError("model.market_state_input is only supported with model.kind single_mlp")
        if loss.c_bal > 0 and m.kind != "detmaso":
            raise ValueError("loss.c_bal balances a scheduler, so it needs model.kind detmaso")
        if loss.balance_target != "uniform" and loss.c_bal == 0:
            raise ValueError("loss.balance_target only matters with loss.c_bal > 0")

    def _check_walkforward(self) -> None:
        wf, sp = self.walkforward, self.splits
        if not self.train.validate:
            raise ValueError("walk-forward folds early-stop on their own year, so train.validate must stay true")
        if pd.Timestamp(wf.end) >= pd.Timestamp(sp.test[0]):
            raise ValueError(f"walkforward.end {wf.end} must lie before the locked test period ({sp.test[0]})")
        first_train_year = wf.first_test_year - wf.early_stop_years - 1
        if pd.Timestamp(sp.train[0]).year > first_train_year:
            raise ValueError(f"the first fold needs training data in {first_train_year}; splits.train starts {sp.train[0]}")
        if sp.purge_days < self.features.horizon:
            raise ValueError("walk-forward purges must cover the label horizon (splits.purge_days >= features.horizon)")

    @property
    def is_placebo(self) -> bool:
        return self.placebo.shuffle_seed is not None

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir).expanduser()

    @property
    def is_walkforward(self) -> bool:
        return self.walkforward is not None

    def to_dict(self) -> dict:
        """Every field, including labels and post-paper flags at their defaults (written to config.json)."""
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Stable hash of everything that affects results (used for the test-period lock)."""
        payload = json.dumps(_hash_view(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def data_hash(self) -> str:
        """Hash of the sections that determine the prepared data (features, labels, regimes)."""
        d = _hash_view(self)
        payload = json.dumps({k: d[k] for k in ("data", "splits", "features", "regimes")}, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _field_default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    raise ValueError(f"post-paper field {f.name!r} needs a default")


def _post_paper_changed(obj) -> bool:
    """True if any post-paper flag (in any section) differs from its default."""
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if f.metadata.get("hash") == _POST_PAPER and value != _field_default(f):
            return True
        if dataclasses.is_dataclass(value) and _post_paper_changed(value):
            return True
    return False


def _hash_view(obj):
    """What the config hashes see: labels and evaluation-only settings dropped, post-paper flags dropped while
    at their default."""
    if dataclasses.is_dataclass(obj):
        out = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            rule = f.metadata.get("hash")
            if rule in (_LABEL, _EVAL) or (rule == _POST_PAPER and value == _field_default(f)):
                continue
            out[f.name] = _hash_view(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_hash_view(v) for v in obj]
    return obj


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


def _merge(base: dict, override: dict) -> dict:
    """Override ``base`` key by key; nested mappings merge, everything else is replaced."""
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _read_yaml(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    """A config file with its ``base:`` chain resolved (base paths are relative to the file; a list of bases
    is merged in order, later ones winning, and the file itself wins over all of them)."""
    path = path.expanduser().resolve()
    if path in chain:
        raise ValueError("config base cycle: " + " -> ".join(str(p) for p in (*chain, path)))
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    merged: dict = {}
    for b in [base] if isinstance(base, str) else list(base):
        merged = _merge(merged, _read_yaml(path.parent / b, (*chain, path)))
    return _merge(merged, raw)


_TOP_LEVEL = {f.name for f in dataclasses.fields(Config)}


def load_config(path: str | Path) -> Config:
    raw = _read_yaml(Path(path))
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
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
        placebo=_build(PlaceboConfig, raw.get("placebo")),
        walkforward=None if raw.get("walkforward") is None else _build(WalkForwardConfig, raw["walkforward"]),
        trading=_build(TradingConfig, raw.get("trading")),
        variant=raw.get("variant", "paper"),
    )
