"""Walk-forward evaluation (docs/RULES.md, Rules WL, WT, WK, WP): config, folds, moods, training, evaluation, compare."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from detmaso.config import RegimeConfig, SplitConfig, TradingConfig, WalkForwardConfig, load_config
from detmaso.data.labels import forward_return, next_day_return
from detmaso.data.market_state import state_names
from detmaso.data.panel import Panel
from detmaso.data.regimes import fit_regimes
from detmaso.data.splits import make_splits
from detmaso.eval import ledger
from detmaso.walkforward.folds import FoldError, check_fold, fold_layout
from tests.conftest import ROOT, synthetic_config

PAPER = ROOT / "configs/csi300.yaml"


# Config ------------------------------------------------------------------------------------------------------
def test_walkforward_configs_keep_the_paper_hashes_and_change_only_the_layer():
    paper = load_config(PAPER)
    assert paper.hash() == "d2a65e9a6a237d2c" and paper.data_hash() == "8e81084fb70adad6" and paper.walkforward is None
    wf = load_config(ROOT / "configs/walkforward/paper.yaml")
    single = load_config(ROOT / "configs/walkforward/single_mlp.yaml")
    assert wf.walkforward == WalkForwardConfig() and wf.walkforward.years == list(range(2013, 2024))
    assert wf.run_dir == "runs/csi300_wf" and wf.data.load_start == "2007-01-01" and wf.splits.train[0] == "2008-01-01"
    assert dataclasses.replace(wf, walkforward=None, run_dir=paper.run_dir, data=paper.data, splits=paper.splits, variant="paper") == paper
    # list bases merge in order: the single network keeps its own loss settings under the walk-forward layer
    assert single.model.kind == "single_mlp" and single.loss.c_div == 0.0 and single.walkforward == wf.walkforward
    assert wf.data_hash() == single.data_hash() != paper.data_hash()
    assert len({wf.hash(), single.hash(), paper.hash()}) == 3


def test_trading_settings_are_never_hashed(tmp_path):
    cfg = synthetic_config(tmp_path)
    other = dataclasses.replace(cfg, trading=TradingConfig(slippage_bps=9.0, flat_bps=(0, 5)))
    assert other.hash() == cfg.hash() and other.data_hash() == cfg.data_hash()
    assert other.trading.flat_bps == (0.0, 5.0)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"walkforward": WalkForwardConfig(2021, 2021, "2021-04-30")}, "before the locked test period"),
        ({"walkforward": WalkForwardConfig(2020, 2020, "2020-12-31", early_stop_years=1)}, "needs training data in 2018"),
    ],
)
def test_invalid_walkforward_layouts_raise(tmp_path, change, message):
    cfg = synthetic_config(tmp_path)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(cfg, variant="wf", **change)


def test_early_stopping_is_fixed_at_the_one_year_before_the_test_year():
    """Decision D4 (docs/RULES.md, Rule WL): each fold early-stops on calendar year Y-1, so no other length is a layout."""
    assert WalkForwardConfig().early_stop_years == 1
    for years in (0, 2):
        with pytest.raises(ValueError, match="early_stop_years must be 1.*decision D4"):
            WalkForwardConfig(early_stop_years=years)


# Fold layout ---------------------------------------------------------------------------------------------------
def _wf_config(run_dir: Path, variant: str = "wf_test", **overrides):
    cfg = synthetic_config(run_dir, variant=variant, train={"max_epochs": 3, "seeds": (0, 1)}, model={"dtype": "float64"}, **overrides)
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, load_start="2016-01-01", load_end="2021-04-30"),
        splits=SplitConfig(train=("2016-01-01", "2019-12-31"), valid=("2020-01-01", "2020-02-28"), test=("2020-03-03", "2021-03-31"), purge_days=20),
        regimes=RegimeConfig(k_min=5, k_max=5, n_init=2),
        walkforward=WalkForwardConfig(first_test_year=2018, last_test_year=2019, end="2020-02-28"),
    )


DATES = pd.bdate_range("2016-01-01", "2021-03-31")


def test_fold_layout_purges_and_never_reaches_the_test_period(tmp_path):
    cfg = _wf_config(tmp_path)
    folds = fold_layout(cfg, DATES)
    assert [f.year for f in folds] == [2018, 2019]
    for f in folds:
        s = f.splits
        assert s.train[0] == DATES[0] and s.train[-1].year == f.year - 2 and set(s.valid.year) == {f.year - 1}
        assert len(s.train) - len(s.train_fit) == 20 and len(s.valid) - len(s.valid_fit) == 20 and len(s.test) - len(f.test_ic) == 20
        assert s.train_fit[-1] < s.valid[0] and s.valid_fit[-1] < s.test[0]
    assert folds[-1].splits.test[-1] == pd.Timestamp("2020-02-28")
    assert set(folds[-1].pre_test) == set(folds[-1].splits.train) | set(folds[-1].splits.valid)


def test_check_fold_refuses_a_window_touching_the_test_period(tmp_path):
    cfg = _wf_config(tmp_path)
    fold = fold_layout(cfg, DATES)[-1]
    with pytest.raises(FoldError, match="locked test period"):
        check_fold(fold, DATES, pd.Timestamp("2020-02-28"), 20)  # the last next-day return would reach it
    unpurged = dataclasses.replace(fold, test_ic=fold.splits.test)
    with pytest.raises(FoldError, match="answers reach past"):
        check_fold(unpurged, DATES, pd.Timestamp("2020-03-03"), 20)


# A synthetic prepared market on disk ----------------------------------------------------------------------------
def make_market(cfg, n_stocks: int = 60, seed: int = 0, stuck: tuple[int, list[int]] | None = None) -> None:
    """Writes the prepared panel, market state, regimes and raw price cache that `prepare` would.

    stuck = (calendar row, stock columns): those stocks fall 15% to their lower price limit on that row (they
    cannot be sold at its close) and are suspended for one day 25 rows later.
    """
    from detmaso.data.panel import artifacts_dir
    from detmaso.data.qlib_io import cache_dir

    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2016-01-01", "2021-04-30")
    codes = [f"SH60{i:04d}" for i in range(n_stocks)]
    T = len(DATES)
    X = rng.normal(size=(len(cal), n_stocks, 34)).astype(np.float32)
    drift = 0.004 * X[..., 3] + 0.015 * rng.normal(size=X.shape[:2])  # tomorrow's move depends on feature 3
    close = pd.DataFrame(10 * np.exp(np.cumsum(drift, axis=0)), index=cal, columns=codes)
    close.iloc[300:303, 2] = np.nan  # a suspension
    if stuck is not None:
        close.iloc[stuck[0] :, stuck[1]] *= 0.85
        close.iloc[stuck[0] + 25, stuck[1]] = np.nan
    low = close * 0.99
    if stuck is not None:
        low.iloc[stuck[0], stuck[1]] = close.iloc[stuck[0], stuck[1]]  # closed at the day's low
    regime_wave = np.sin(np.arange(len(cal)) / 40.0)
    state = pd.DataFrame(rng.normal(size=(len(cal), 12)) + regime_wave[:, None] * np.linspace(0.5, 3, 12), index=cal, columns=state_names((5, 20, 60, 120)))
    rows = cal.get_indexer(DATES)
    y = forward_return(close, 20).to_numpy(np.float32)[rows]
    rn = next_day_return(close).to_numpy(np.float32)[rows]
    traded = close.notna().to_numpy()[rows]
    universe = np.ones((T, n_stocks), dtype=bool)
    tradable = traded
    label_ok = np.isfinite(y)
    splits = make_splits(DATES, cfg.splits.train, cfg.splits.valid, cfg.splits.test, 20)
    reg = fit_regimes(state, splits.train, n_agents=5, k_range=(5, 5), n_init=2, seed=0)
    s64 = state.loc[DATES].to_numpy(np.float64)
    panel = Panel(
        dates=DATES, code_list=codes, feature_names=[f"f{i}" for i in range(34)],
        code_idx=torch.arange(n_stocks).repeat(T, 1), X=torch.from_numpy(X[rows]), y=torch.from_numpy(y), r_next=torch.from_numpy(rn),
        universe=torch.from_numpy(universe), tradable=torch.from_numpy(tradable), label_ok=torch.from_numpy(label_ok),
        s_raw=torch.from_numpy(s64.astype(np.float32)), s_sched=torch.from_numpy(np.clip(reg.standardize(s64), -5, 5).astype(np.float32)),
        regime=torch.from_numpy(reg.assign(s64)), index_close=torch.ones(T),
    )  # fmt: skip
    d = artifacts_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    panel.save(d / "panel.pt")
    reg.to_json(d / "regimes.json")
    state.to_parquet(d / "market_state.parquet")
    (d / "meta.json").write_text(json.dumps({"data_hash": cfg.data_hash()}))
    raw = cache_dir(cfg)
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(True, index=cal, columns=codes).to_parquet(raw / "member.parquet")
    close.to_parquet(raw / "close.parquet")
    (close * 1.01).to_parquet(raw / "high.parquet")
    low.to_parquet(raw / "low.parquet")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Two walk-forward run groups (Det-MASO and a single network) trained and evaluated on the synthetic market."""
    from detmaso.eval.walkforward import evaluate_walkforward
    from detmaso.walkforward.runner import train_walkforward

    root = tmp_path_factory.mktemp("wf")
    cfg = _wf_config(root / "runs")
    single = dataclasses.replace(cfg, variant="wf_single", model=dataclasses.replace(cfg.model, kind="single_mlp"),
                                 loss=dataclasses.replace(cfg.loss, c_div=0.0, c_ent=0.0))  # fmt: skip
    make_market(cfg)
    trials = root / "trials.csv"
    status = train_walkforward(cfg, "wf.yaml", "wf", [0, 1], jobs=1, ledger_path=trials)
    train_walkforward(single, "single.yaml", "single", [0, 1], jobs=1, ledger_path=trials)
    evaluate_walkforward(cfg, "wf.yaml", "wf", [0, 1], ledger_path=trials)
    evaluate_walkforward(single, "single.yaml", "single", [0, 1], ledger_path=trials)
    return {"cfg": cfg, "single": single, "trials": trials, "status": status, "root": root}


def test_training_runs_every_fold_and_seed_and_is_counted_once(trained):
    cfg, trials = trained["cfg"], trained["trials"]
    assert trained["status"] == {f"Y{y}:s{s}": "trained" for y in (2018, 2019) for s in (0, 1)}
    row = ledger.read(trials).set_index("config_hash").loc[cfg.hash()]
    assert int(row["try_no"]) == 1 and row["kind"] == "real"
    assert json.loads(row["seed_status"]) == {f"wf:Y{y}:s{s}": "trained" for y in (2018, 2019) for s in (0, 1)}
    assert row["wf_years"] == 2 and np.isfinite(row["wf_ic"]) and np.isfinite(row["wf_sharpe_net"])
    # each fold's moods were fit only on days before its test window
    from detmaso.walkforward.folds import fold_dir

    for f in fold_layout(cfg, DATES):
        reg = json.loads((fold_dir(cfg, f) / "regimes.json").read_text())
        assert pd.Timestamp(reg["fit_end"]) < f.splits.test[0]


def test_training_resumes_and_parallel_jobs_match_serial(trained, tmp_path):
    from detmaso.walkforward.runner import job_dir, train_walkforward

    cfg = trained["cfg"]
    assert train_walkforward(cfg, "wf.yaml", "wf", [0, 1], jobs=1) == {}  # everything is done already
    copy = dataclasses.replace(cfg, run_dir=str(tmp_path / "runs"))
    make_market(copy)
    train_walkforward(copy, "wf.yaml", "par", [0, 1], years=[2018], jobs=2)
    for s in (0, 1):
        a = torch.load(job_dir(cfg, "wf", 2018, s) / "model.pt", weights_only=True)
        b = torch.load(job_dir(copy, "par", 2018, s) / "model.pt", weights_only=True)
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_interrupted_jobs_are_marked_aborted_and_resume(trained, tmp_path):
    """CLAUDE.md, "Count tries": aborted runs stay visible in the try ledger, as `aborted: <reason>`."""
    import detmaso.train.trainer as trainer
    from detmaso.walkforward.runner import train_walkforward

    cfg, trials = trained["cfg"], tmp_path / "trials.csv"
    real_train_seed, calls = trainer.train_seed, []

    def interrupted_on_the_second_job(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_train_seed(*args, **kwargs)

    def statuses() -> dict:
        return json.loads(ledger.read(trials).set_index("config_hash").loc[cfg.hash()]["seed_status"])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer, "train_seed", interrupted_on_the_second_job)
        with pytest.raises(KeyboardInterrupt):
            train_walkforward(cfg, "wf.yaml", "stop", [0, 1], jobs=1, ledger_path=trials)
    stopped = "aborted: KeyboardInterrupt"
    assert statuses() == {"stop:Y2018:s0": "trained", "stop:Y2018:s1": stopped, "stop:Y2019:s0": stopped, "stop:Y2019:s1": stopped}
    assert train_walkforward(cfg, "wf.yaml", "stop", [0, 1], years=[2018], jobs=1, ledger_path=trials) == {"Y2018:s1": "trained"}
    assert statuses() == {"stop:Y2018:s0": "trained", "stop:Y2018:s1": "trained", "stop:Y2019:s0": stopped, "stop:Y2019:s1": stopped}


def test_a_run_deflates_with_its_own_walkforward_metrics_in_the_ledger(trained, tmp_path):
    """docs/RULES.md, "Deflated metrics for walk-forward results": the spread comes from the walk-forward tries once
    3 have metrics. The third try's own report already counts itself, as a later `compare` does."""
    from detmaso.eval.walkforward import evaluate_walkforward
    from detmaso.eval.wf_stats import RESULTS_FILE
    from detmaso.walkforward.runner import train_walkforward

    cfg, trials = trained["cfg"], tmp_path / "trials.csv"
    earlier = {"a": (0.010, 0.02), "b": (0.030, 0.05)}  # two earlier tries: walk-forward IC, daily after-cost excess Sharpe
    for epochs, (name, (ic, sr)) in zip((10, 11), earlier.items()):
        other = dataclasses.replace(cfg, variant=name, train=dataclasses.replace(cfg.train, max_epochs=epochs))
        ledger.register(trials, other, f"{name}.yaml", name, [])
        ledger.record_wf_metrics(trials, other, {"wf_years": 2, "wf_ic": ic, "wf_ic_sd_seeds": 0.0, "wf_sharpe_net": 1.0, "wf_excess_sr_daily_net": sr})
    third = dataclasses.replace(cfg, variant="wf_third", train=dataclasses.replace(cfg.train, max_epochs=4))
    train_walkforward(third, "third.yaml", "third", [0, 1], jobs=1, ledger_path=trials)
    evaluate_walkforward(third, "third.yaml", "third", [0, 1], ledger_path=trials)
    own = ledger.read(trials).set_index("config_hash").loc[third.hash()]
    deflated = json.loads((third.run_path / "third" / RESULTS_FILE).read_text())["deflated"]
    assert deflated["try_no"] == 3 and deflated["n_tries"] == 3 and deflated["source"] == "the 3 walk-forward tries"
    assert deflated["ic_sd"] == pytest.approx(np.std([0.010, 0.030, own["wf_ic"]], ddof=1))
    assert deflated["sr_sd"] == pytest.approx(np.std([0.02, 0.05, own["wf_excess_sr_daily_net"]], ddof=1))
    assert "Try #3 of 3 counted tries" in (third.run_path / "third" / "report_wf.md").read_text()


def test_no_real_return_enters_a_placebo_evaluation(tmp_path):
    """docs/RULES.md, Rule WT: placebo holdings earn shuffled next-day returns only. Two thirds of the stocks close at
    their lower limit on the first day of a placebo block that they are not shuffled in (a later suspension leaves them
    without a 20-day answer there), so the ones picked the day before stay held with their real return in the panel.
    Changing that real return must change nothing in the placebo's daily series."""
    from detmaso.data.panel import artifacts_dir
    from detmaso.eval.walkforward import evaluate_walkforward
    from detmaso.eval.wf_stats import SERIES_FILE
    from detmaso.walkforward.runner import train_walkforward

    cfg = _wf_config(tmp_path / "runs", variant="wf_placebo")
    cfg = dataclasses.replace(cfg, placebo=dataclasses.replace(cfg.placebo, shuffle_seed=0))
    row, cols = 560, list(range(40))  # placebo blocks are 20 model dates: row 560 opens the block after rows 540-559
    assert cfg.placebo.block_days == 20 and DATES[row - 20].year == DATES[row + 39].year == 2018
    make_market(cfg, stuck=(row, cols))
    train_walkforward(cfg, "placebo.yaml", "placebo", [0], years=[2018], jobs=1)

    def series() -> pd.DataFrame:
        evaluate_walkforward(cfg, "placebo.yaml", "placebo", [0], years=[2018])
        return pd.read_parquet(cfg.run_path / "placebo" / SERIES_FILE)

    before = series()
    assert before.set_index("decision_date").loc[DATES[row], "blocked_sells"] >= 1  # a stuck stock is held over that day
    panel = Panel.load(artifacts_dir(cfg) / "panel.pt")
    panel.r_next[row, cols] = 0.5  # the real next-day return of the stuck stocks: +50%
    panel.save(artifacts_dir(cfg) / "panel.pt")
    pd.testing.assert_frame_equal(series(), before)


def test_blocked_trades_follow_the_price_limits_in_the_config(tmp_path):
    """The walk-forward evaluation takes the market's daily price limits from the config (trading.price_limits: an
    evaluation-only setting, so nothing is retrained). 40 stocks fall 15% to the day's low on one day: a lower-limit
    close under China's 10% limit, where they cannot be sold, and an ordinary close under a 30% limit."""
    from detmaso.eval.walkforward import evaluate_walkforward
    from detmaso.eval.wf_stats import SERIES_FILE
    from detmaso.walkforward.runner import train_walkforward

    cfg = _wf_config(tmp_path / "runs")
    row, cols = 560, list(range(40))
    make_market(cfg, stuck=(row, cols))
    train_walkforward(cfg, "wf.yaml", "limits", [0], years=[2018], jobs=1)

    def blocked_sells(c) -> pd.Series:
        evaluate_walkforward(c, "wf.yaml", "limits", [0], years=[2018])
        return pd.read_parquet(c.run_path / "limits" / SERIES_FILE).set_index("decision_date")["blocked_sells"]

    assert cfg.trading.price_limits == TradingConfig().price_limits  # China's limits are the default
    china = blocked_sells(cfg)
    wide = dataclasses.replace(cfg, trading=TradingConfig(price_limits=[[[], "1990-01-01", 0.30]]))
    assert wide.hash() == cfg.hash()  # same run group, same models
    loose = blocked_sells(wide)
    assert china.loc[DATES[row]] >= 1 and loose.loc[DATES[row]] == 0
    none = dataclasses.replace(cfg, trading=TradingConfig(price_limits=[[["SZ"], "1990-01-01", 0.10]]))  # no limit for the SH codes
    with pytest.raises(ValueError, match="no price limit is defined"):
        blocked_sells(none)


def test_evaluation_series_reproduce_the_paper_backtest_and_purge_ic(trained):
    from detmaso.eval.wf_stats import SERIES_FILE, load_wf_run

    cfg = trained["cfg"]
    run = load_wf_run(cfg, "wf.yaml", "wf")
    assert run.years == [2018, 2019] and run.seeds == [0, 1]
    for f in fold_layout(cfg, DATES):
        assert run.dec[f.year].equals(f.splits.test) and int(run.in_ic[f.year].sum()) == len(f.test_ic)
        assert np.isnan(run.cols[f.year]["ic"][:, ~run.in_ic[f.year]]).all()
    daily = pd.read_parquet(cfg.run_path / "wf" / SERIES_FILE)
    assert (daily["decision_date"] < pd.Timestamp(cfg.splits.test[0])).all() and (daily["ret_date"] < pd.Timestamp(cfg.splits.test[0])).all()
    assert daily["port_blk"].notna().all() and (daily["cash_blk"] >= 0).all()
    report = (cfg.run_path / "wf" / "report_wf.md").read_text()
    assert "Try #1 of" in report and "Deflated Sharpe" in report and "## By test year" in report


def test_compare_reads_evaluated_groups_and_refuses_a_group_of_another_config(trained):
    """The keep and placebo verdicts themselves, and the refusal to mix placebo and real run groups, are checked on
    planted run groups in tests/test_keep_rule.py."""
    from detmaso.eval.wf_compare import run_wf_compare
    from detmaso.eval.wf_stats import WFError

    cfg, single, root = trained["cfg"], trained["single"], trained["root"]
    out = root / "cmp"
    runs = [(cfg, "wf.yaml", "wf"), (single, "single.yaml", "single")]
    run_wf_compare(runs, "wf", out, rule="keep", candidate="single", no_setting=True, ledger_path=trained["trials"], n_boot=200)
    payload = json.loads((out / "comparison.json").read_text())
    assert (payload["rules"]["candidate"], payload["rules"]["reference"]) == ("wf_single", "wf_test")
    assert {name: d["try_no"] for name, d in payload["deflated"].items()} == {"wf_test": 1, "wf_single": 2}  # in training order
    assert set(payload["advantage"]["wf_single"]["yearly"]) == {"2018", "2019"}
    with pytest.raises(WFError, match="needs --neighbor"):
        run_wf_compare(runs, "wf", out, rule="keep", candidate="single", ledger_path=None, n_boot=50)
    other = dataclasses.replace(single, placebo=dataclasses.replace(single.placebo, shuffle_seed=0), variant="placebo_single")
    with pytest.raises(WFError, match="run group 'single' was not trained with single.yaml \\(config hash mismatch\\)"):
        run_wf_compare([(cfg, "wf.yaml", "wf"), (other, "single.yaml", "single")], "wf", out, n_boot=50)


def test_identical_run_groups_never_count_as_an_improvement(trained):
    from detmaso.eval.wf_compare import advantage
    from detmaso.eval.wf_stats import bootstrap, load_wf_run, seed_metrics
    from detmaso.eval.guard import keep_rule

    cfg = trained["cfg"]
    a = load_wf_run(cfg, "wf.yaml", "wf")
    b = load_wf_run(dataclasses.replace(cfg, variant="wf_twin"), "wf.yaml", "wf")
    boot = bootstrap([a, b], n_boot=200)
    x = advantage(b, a, {r.name: seed_metrics(r) for r in (a, b)}, boot)
    assert all(v["gain"] == 0 for v in x["yearly"].values()) and x["sharpe"]["diff"] == 0
    assert keep_rule(x["yearly"], x["average"], x["seed_gains"], x["sharpe"], has_setting=False)["verdict"] == "no improvement"


def test_cli_takes_a_next_setting_or_no_tunable_setting_never_both(tmp_path, capsys):
    from detmaso.cli import main

    with pytest.raises(SystemExit):
        main(["compare", "--split", "walkforward", "--runs", "a.yaml:x", "b.yaml:y", "c.yaml:z", "--out", str(tmp_path),
              "--rule", "keep", "--candidate", "y", "--neighbor", "z", "--no-tunable-setting"])  # fmt: skip
    assert "--no-tunable-setting: not allowed with argument --neighbor" in capsys.readouterr().err


def test_cli_routes_walkforward_configs(tmp_path):
    from detmaso.cli import main

    with pytest.raises(SystemExit, match="--split walkforward"):
        main(["evaluate", "--config", str(ROOT / "configs/walkforward/paper.yaml"), "--split", "valid"])
    with pytest.raises(SystemExit, match="--split walkforward"):
        main(["evaluate", "--config", str(PAPER), "--split", "walkforward"])
    with pytest.raises(SystemExit, match="walk-forward configs only"):
        main(["train", "--config", str(PAPER), "--jobs", "2", "--tag", "never"])


def test_test_lock_refuses_walkforward_configs(tmp_path):
    from detmaso.eval.lock import TestLockError, open_test_period

    with pytest.raises(TestLockError, match="walk-forward"):
        open_test_period(_wf_config(tmp_path), "wf", [0, 1])
