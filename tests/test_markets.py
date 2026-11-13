"""The extra China universes (docs/RULES.md, "More data": Rules ND and CU): CSI500, CSI1000 and the whole A-share market.

Config layout: the validation-period config is configs/<market>.yaml; the walk-forward variants are
configs/walkforward/<market>/paper.yaml and single_mlp.yaml. Every market has its own name (its own try ledger
and test lock) and its own run folders. The paper config and CSI300's walk-forward configs must not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from detmaso.config import TradingConfig, WalkForwardConfig, load_config
from detmaso.walkforward.folds import fold_layout
from tests.conftest import ROOT
from tests.test_config import _flat

CONFIGS = ROOT / "configs"
MARKETS = ("csi500", "csi1000", "ashare")
TEST_PERIOD = ("2024-03-01", "2026-03-01")
INDEX_SERIES = ("SH000300", "SH000852", "SH000905", "SH000906", "SH000985", "SZ399300")  # in Qlib's `all` list, not stocks
SINGLE_NETWORK = {"model.kind": "single_mlp", "loss.c_div": 0.0, "loss.c_ent": 0.0}
QA_ON = {"qa.membership_checks": True}

# What each validation-period config changes from configs/csi300.yaml, and nothing else.
VALIDATION = {
    "csi500": {"market": "csi500", "run_dir": "runs/csi500", "data.universe": "csi500", "data.index_symbol": "SH000905", **QA_ON},
    "csi1000": {
        "market": "csi1000", "run_dir": "runs/csi1000", "data.universe": "csi1000", "data.index_symbol": "SH000852",
        "data.load_start": "2015-01-01", "splits.train": ("2016-01-01", "2022-12-31"), **QA_ON,
    },
    "ashare": {
        "market": "ashare", "run_dir": "runs/ashare", "variant": "paper_listed120", "data.universe": "all", "data.index_symbol": "SH000985",
        "data.min_traded_days": 120, "data.exclude_codes": INDEX_SERIES,
        "trading.price_limits": (*TradingConfig().price_limits, (("BJ",), "1990-01-01", 0.30)), **QA_ON,
    },
}  # fmt: skip
# What each walk-forward paper variant changes from its market's validation-period config.
WALKFORWARD = {
    "csi500": {"run_dir": "runs/csi500_wf", "variant": "wf_paper", "data.load_start": "2007-01-01", "splits.train": ("2008-01-01", "2022-12-31")},
    "csi1000": {"run_dir": "runs/csi1000_wf", "variant": "wf_paper"},
    "ashare": {"run_dir": "runs/ashare_wf", "variant": "wf_paper_listed120", "data.load_start": "2007-01-01", "splits.train": ("2008-01-01", "2022-12-31")},
}
WF_YEARS = {"csi500": list(range(2013, 2024)), "csi1000": list(range(2018, 2024)), "ashare": list(range(2013, 2024))}


def _changed(base, other) -> dict:
    a, b = _flat(base.to_dict()), _flat(other.to_dict())
    assert set(a) == set(b)
    return {k: b[k] for k in a if a[k] != b[k]}


def _wf(cfg) -> dict:
    """A config's flat view without its walkforward section (None in a validation-period config)."""
    return {k: v for k, v in _flat(cfg.to_dict()).items() if not k.startswith("walkforward")}


def _load(market: str):
    return (
        load_config(CONFIGS / f"{market}.yaml"),
        load_config(CONFIGS / "walkforward" / market / "paper.yaml"),
        load_config(CONFIGS / "walkforward" / market / "single_mlp.yaml"),
    )


def _calendar(cfg) -> pd.DatetimeIndex:
    """Weekdays standing in for the trading calendar: the model dates a fold layout is built on."""
    days = pd.bdate_range(cfg.data.load_start, cfg.data.load_end)
    return days[(days >= pd.Timestamp(cfg.splits.train[0])) & (days <= pd.Timestamp(cfg.splits.test[1]))]


# The paper config and CSI300's walk-forward configs ---------------------------------------------------------------
def test_paper_and_csi300_walkforward_hashes_are_unchanged():
    paper = load_config(CONFIGS / "csi300.yaml")
    entry = json.loads((ROOT / "locks/csi300_test.json").read_text())["entries"][0]
    assert paper.hash() == entry["config_hash"] == "d2a65e9a6a237d2c" and paper.data_hash() == entry["data_hash"] == "8e81084fb70adad6"
    # every setting added for the new markets is off in the paper config
    assert paper.data.min_traded_days == 0 and paper.data.exclude_codes == () and not paper.qa.membership_checks
    assert paper.trading == TradingConfig() and paper.variant == "paper"
    frozen = {
        "csi300_single_mlp.yaml": ("ef0560a807c0b155", "8e81084fb70adad6"),
        "walkforward/paper.yaml": ("ea869cb9c0c6e5d3", "7470fa641726f353"),
        "walkforward/single_mlp.yaml": ("f17dc1e5694a60b3", "7470fa641726f353"),
    }
    for name, hashes in frozen.items():
        cfg = load_config(CONFIGS / name)
        assert (cfg.hash(), cfg.data_hash()) == hashes, name
        assert cfg.market == "csi300" and not cfg.qa.membership_checks and cfg.data.min_traded_days == 0


# The new configs ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("market", MARKETS)
def test_validation_config_changes_only_what_its_market_needs(market):
    paper, (cfg, _, _) = load_config(CONFIGS / "csi300.yaml"), _load(market)
    assert _changed(paper, cfg) == VALIDATION[market]
    assert cfg.market == market and cfg.run_dir == f"runs/{market}" and cfg.walkforward is None
    assert cfg.splits.test == TEST_PERIOD and cfg.splits.valid == paper.splits.valid  # the same locked test period everywhere
    assert cfg.qa.membership_checks  # Rule ND
    assert cfg.hash() != paper.hash() and cfg.data_hash() != paper.data_hash()
    # the model, the loss and the training recipe are the paper's
    assert (cfg.model, cfg.loss, cfg.train, cfg.features, cfg.regimes, cfg.backtest) == (paper.model, paper.loss, paper.train, paper.features, paper.regimes, paper.backtest)


@pytest.mark.parametrize("market", MARKETS)
def test_walkforward_variants(market):
    cfg, wf, single = _load(market)
    a, b = _wf(cfg), _wf(wf)
    assert {k: b[k] for k in a if a[k] != b[k]} == WALKFORWARD[market]
    assert wf.market == market and wf.run_dir == f"runs/{market}_wf" and wf.splits.test == TEST_PERIOD
    assert wf.walkforward.years == WF_YEARS[market] and wf.walkforward.end == "2024-02-28"
    assert pd.Timestamp(wf.walkforward.end) < pd.Timestamp(wf.splits.test[0])
    # the single network differs from the walk-forward paper variant by the single-network settings only, as for CSI300
    assert _changed(wf, single) == {"variant": wf.variant.replace("wf_paper", "wf_single_mlp"), **SINGLE_NETWORK}
    csi300 = _changed(load_config(CONFIGS / "walkforward/paper.yaml"), load_config(CONFIGS / "walkforward/single_mlp.yaml"))
    assert csi300 == {"variant": "wf_single_mlp", **SINGLE_NETWORK}
    assert single.data_hash() == wf.data_hash() and single.run_dir == wf.run_dir  # one prepared dataset per run folder
    assert len({cfg.hash(), wf.hash(), single.hash()}) == 3


def test_every_run_folder_and_market_name_is_its_own():
    csi300 = [load_config(CONFIGS / n) for n in ("csi300.yaml", "walkforward/paper.yaml")]
    new = [c for m in MARKETS for c in _load(m)[:2]]
    folders = [c.run_dir for c in csi300 + new]
    assert len(set(folders)) == len(folders) == 8
    assert {c.market for c in new} == set(MARKETS) and "csi300" not in {c.market for c in new}
    assert len({c.data_hash() for c in csi300 + new}) == 7  # CSI1000's two folders hold the same prepared data: see below


def test_csi500_folds_are_the_csi300_folds():
    """Rule CU: the same splits and the same 11 walk-forward folds as CSI300."""
    ref = load_config(CONFIGS / "walkforward/paper.yaml")
    for market in ("csi500", "ashare"):
        _, wf, _ = _load(market)
        assert wf.walkforward == ref.walkforward == WalkForwardConfig() and wf.splits == ref.splits
        assert (wf.data.load_start, wf.data.load_end) == (ref.data.load_start, ref.data.load_end)
        ours, theirs = fold_layout(wf, _calendar(wf)), fold_layout(ref, _calendar(ref))
        assert [f.describe() for f in ours] == [f.describe() for f in theirs] and len(ours) == 11


def test_csi1000_history_starts_in_2016_with_six_folds():
    """Rule CU: history from 2016-01-01; test years 2018-2023; the first fold trains on 2016 and early-stops on 2017."""
    cfg, wf, _ = _load("csi1000")
    assert cfg.splits.train == wf.splits.train == ("2016-01-01", "2022-12-31") and cfg.data.load_start == wf.data.load_start == "2015-01-01"
    assert wf.data_hash() == cfg.data_hash()  # same load window, history and splits: the same prepared data in both folders
    folds = fold_layout(wf, _calendar(wf))
    assert [f.year for f in folds] == [2018, 2019, 2020, 2021, 2022, 2023]
    first = folds[0].splits
    assert set(first.train.year) == {2016} and set(first.valid.year) == {2017} and set(first.test.year) == {2018}
    assert first.train[0] == pd.Timestamp("2016-01-01") and first.train_fit[-1] < first.valid[0] and first.valid_fit[-1] < first.test[0]
    for f in folds:
        assert f.splits.train[0].year == 2016 and f.splits.train[-1].year == f.year - 2 and set(f.splits.valid.year) == {f.year - 1}
    last = folds[-1].splits
    assert last.test[0].year == 2023 and last.test[-1] == pd.Timestamp("2024-02-28") < pd.Timestamp(TEST_PERIOD[0])


def test_ashare_filters_and_price_limits():
    """Rule CU: new listings enter on their 121st traded day; China's price limits plus 30% for Beijing codes."""
    from detmaso.eval.trading import price_limit

    cfg, wf, single = _load("ashare")
    for c in (cfg, wf, single):
        assert c.data.universe == "all" and c.data.min_traded_days == 120 and c.data.exclude_codes == INDEX_SERIES
        assert c.variant != "paper" and c.trading.price_limits == VALIDATION["ashare"]["trading.price_limits"]
        assert c.trading.max_price_limit == 0.30
    codes = ["SH600000", "SZ000001", "SH688981", "SZ300750", "SZ301000", "BJ430017", "BJ830799", "BJ871396", "BJ920002"]
    dates = pd.DatetimeIndex(["2012-01-04", "2020-08-21", "2020-08-24", "2021-11-15", "2024-02-29"])
    lim = price_limit(codes, dates, cfg.trading)
    assert np.array_equal(lim[:, :5], price_limit(codes[:5], dates, TradingConfig()))  # the existing China rules, untouched
    assert (lim[:, 5:] == 0.30).all()
    # the trading settings other than the limits are CSI300's
    assert (cfg.trading.flat_bps, cfg.trading.fee_bps, cfg.trading.slippage_bps, cfg.trading.stamp_duty_bps) == (
        TradingConfig().flat_bps, TradingConfig().fee_bps, TradingConfig().slippage_bps, TradingConfig().stamp_duty_bps,
    )  # fmt: skip


# The pinned archive (skipped where it is not on disk) -------------------------------------------------------------
STOCK_PREFIXES = ("SH60", "SH68", "SZ00", "SZ30", "BJ")


def _archive() -> Path:
    root = Path(load_config(CONFIGS / "csi300.yaml").data.provider_uri).expanduser()
    if not (root / "instruments" / "all.txt").exists():
        pytest.skip(f"the pinned archive is not on disk ({root})")
    return root


def _instruments(name: str) -> pd.DataFrame:
    return pd.read_csv(_archive() / "instruments" / f"{name}.txt", sep="\t", header=None, names=["code", "start", "end"], parse_dates=["start", "end"])


@pytest.mark.data
def test_every_non_stock_series_in_the_all_list_is_excluded():
    """Qlib keeps index series in its `all` list. Judged from the codes alone: anything that is not a Shanghai, Shenzhen
    or Beijing stock code must be in the A-share config's exclude list, and nothing else may be."""
    codes = sorted(set(_instruments("all")["code"]))
    not_stocks = tuple(c for c in codes if not c.startswith(STOCK_PREFIXES))
    assert not_stocks == load_config(CONFIGS / "ashare.yaml").data.exclude_codes == INDEX_SERIES


@pytest.mark.data
def test_csi1000_member_list_is_full_on_every_day_the_first_market_state_reads():
    """Rule CU: the list has 3 members before 2015-05-29. The first model date's market state reads the member returns
    of the 120 trading days before it; all of them must lie after that day, with a full list. Dates before the test
    period only."""
    cfg = load_config(CONFIGS / "csi1000.yaml")
    cal = pd.DatetimeIndex(pd.read_csv(_archive() / "calendars/day.txt", header=None)[0])
    cal = cal[(cal >= pd.Timestamp(cfg.data.load_start)) & (cal < pd.Timestamp(cfg.splits.test[0]))]
    spans = _instruments("csi1000")
    per_day = pd.Series([int(((spans["start"] <= d) & (spans["end"] >= d)).sum()) for d in cal], index=cal)
    assert per_day[:"2015-05-28"].max() == 3 and per_day["2015-05-29":].min() >= 1000 and per_day.max() <= 1002
    first = int(np.argmax(cal >= pd.Timestamp(cfg.splits.train[0])))
    window = cal[first - max(cfg.regimes.windows) - 1 : first]  # 120 returns need 121 closes
    assert len(window) == 121 and window[0] > pd.Timestamp("2015-05-29") and (per_day[window] >= 1000).all()


@pytest.mark.data
def test_csi1000_model_dates_do_not_depend_on_the_nearly_empty_member_list():
    """Wiping the member list before 2015-05-29 leaves the market state of every model date before the test period
    unchanged (needs `detmaso prepare --config configs/csi1000.yaml`). "Unchanged" is up to the rounding residue of
    pandas' running sums (measured: at most 5e-12 of the value, 1.6e-15 in absolute terms; the float32 values the
    panel stores are all identical): no window of a model date contains a day before 2015-07-08."""
    from detmaso.data.market_state import market_state
    from detmaso.data.panel import artifacts_dir
    from detmaso.data.qlib_io import cache_dir

    cfg = load_config(CONFIGS / "csi1000.yaml")
    raw, prepared = ROOT / cache_dir(cfg), ROOT / artifacts_dir(cfg)
    if not (raw / "member.parquet").exists() or not (prepared / "market_state.parquet").exists():
        pytest.skip("csi1000 is not prepared")
    member, close = pd.read_parquet(raw / "member.parquet"), pd.read_parquet(raw / "close.parquet")
    before_test = member.index < pd.Timestamp(cfg.splits.test[0])
    member, close = member.loc[before_test], close.loc[before_test]
    assert member.loc[:"2015-05-28"].sum(axis=1).max() == 3
    wiped = member.copy()
    wiped.loc[:"2015-05-28"] = False
    model = member.index[member.index >= pd.Timestamp(cfg.splits.train[0])]
    saved = pd.read_parquet(prepared / "market_state.parquet").loc[model]
    full = market_state(close, member, cfg.regimes.windows, cfg.features.min_window_coverage).loc[model]
    cut = market_state(close, wiped, cfg.regimes.windows, cfg.features.min_window_coverage).loc[model]
    assert not full.isna().any().any()
    pd.testing.assert_frame_equal(full, saved, check_exact=True)  # the state the prepared data carries, bit for bit
    np.testing.assert_allclose(cut.to_numpy(), full.to_numpy(), rtol=1e-9, atol=0)
    assert (cut.to_numpy(np.float32) == full.to_numpy(np.float32)).mean() > 0.999
