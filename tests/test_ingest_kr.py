"""The South Korea converter (docs/RULES.md, Rules ND, KG and KU): factor, suspensions, universe, index, Qlib files.

Synthetic markets run anywhere. Tests marked ``data`` need a download (the pinned source, Qlib's dump_bin.py or
the converted dataset; see scripts/get_kr_data.sh) and skip when it is missing. No test reads a real price dated
in the locked test period (from 2024-03-01).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from detmaso.config import load_config
from detmaso.data.ingest import kr
from tests.conftest import ROOT, synthetic_config

DAYS = pd.bdate_range("2019-11-01", "2021-01-15")  # selections: 2020-05-29 and 2020-11-30
MAY, NOV = pd.Timestamp("2020-05-29"), pd.Timestamp("2020-11-30")
JUN, DEC = pd.Timestamp("2020-06-01"), pd.Timestamp("2020-12-01")


def stock(code, name, close, *, days=DAYS, shares=10.0, market="KOSPI", volume=100.0) -> pd.DataFrame:
    """Source-layout rows for one stock: change = close - previous close, so the base price is the previous close."""
    close = np.broadcast_to(np.asarray(close, dtype="float64"), len(days)).copy()
    return pd.DataFrame({
        "Code": code, "Name": name, "Market": market, "Date": days, "Open": close, "High": close, "Low": close,
        "Close": close, "Volume": volume, "Changes": np.diff(close, prepend=close[0]), "Marcap": close * shares,
    })  # fmt: skip


def prepared(*stocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """(usable rows of codes ending in 0, trading calendar), as the converter builds them from the source."""
    raw = kr.standardize(pd.concat(stocks, ignore_index=True))
    return raw[kr.common_rows(raw)].reset_index(drop=True), pd.DatetimeIndex(sorted(raw["Date"].unique()))


def market() -> list[pd.DataFrame]:
    """Five ordinary KOSPI stocks (A largest .. E smallest) and four that must never be picked."""
    return [
        stock("000010", "A", 50.0),
        stock("000020", "B", 40.0),
        stock("000030", "C", 30.0),
        stock("000040", "D", 20.0),
        stock("000050", "E", 10.0),
        stock("000065", "A우", 900.0),  # preferred share: the code does not end in 0
        stock("000070", "K", 900.0, market="KOSDAQ"),
        stock("000080", "테스트리츠", 900.0),  # a REIT by name
        stock("000090", "테스트스팩1호", 900.0),  # a SPAC by name
    ]


def picks(universe: kr.Universe, day: pd.Timestamp) -> list[str]:
    return list(universe.ranked[day]["Code"])


# Factor (Rule KG2) -------------------------------------------------------------------------------------------
def test_split_gives_the_right_factor():
    days = DAYS[:6]
    # 2-for-1 on day 2 (raw 102 -> 52, up 1 from the base price 51), 1-for-10 reverse split on day 4 (53 -> 540).
    rows = stock("000010", "A", [100, 102, 52, 53, 540, 540], days=days)
    rows["Changes"] = [0, 2, 1, 1, 10, 0]
    rows["Volume"] = [1000, 1000, 2000, 2000, 200, 200]
    out = kr.adjust(kr.standardize(rows))
    np.testing.assert_allclose(out["factor"], [1, 1, 2, 2, 0.2, 0.2])
    np.testing.assert_allclose(out["close"], [100, 102, 104, 106, 108, 108])  # the adjusted close has no jump
    np.testing.assert_allclose(out["volume"], [1000, 1000, 1000, 1000, 1000, 1000])  # raw volume / factor
    np.testing.assert_allclose(out["volume"] * out["factor"], rows["Volume"])  # volume x factor = raw volume
    np.testing.assert_allclose(out["open"] / out["factor"], rows["Open"])  # adjusted price / factor = raw price


def test_factor_is_one_on_the_first_row_and_chains_per_stock():
    a = stock("000010", "A", [100, 2, 2], days=DAYS[:3])  # 50-for-1 on day 1
    a["Changes"] = [0, 0, 0]
    b = stock("000020", "B", [70, 70, 70], days=DAYS[:3])
    out = kr.adjust(kr.standardize(pd.concat([a, b], ignore_index=True)))
    np.testing.assert_allclose(out.loc[out["Code"] == "000010", "factor"], [1, 50, 50])
    np.testing.assert_allclose(out.loc[out["Code"] == "000020", "factor"], [1, 1, 1])  # A's factor does not leak into B


def test_row_without_a_base_price_carries_the_factor():
    rows = stock("000010", "A", [100, 50, 50, 55], days=DAYS[:4])
    rows["Changes"] = [0, 0, 50, 5]  # day 1: 2-for-1; day 2: change = close, so the source gives no base price
    out = kr.adjust(kr.standardize(rows))
    np.testing.assert_allclose(out["factor"], [1, 2, 2, 2])
    assert out["no_base"].tolist() == [False, False, True, False]


def test_suspended_day_has_no_price():
    days = DAYS[:8]
    rows = stock("000010", "A", [100, 100, 100, 100, 2, 2, 2, 2], days=days)
    rows["Changes"] = 0.0
    halted = rows["Date"].isin(days[1:4])  # a 3-day halt, then a 50-for-1 split (the Samsung 2018 pattern)
    rows.loc[halted, ["Open", "High", "Low", "Volume"]] = 0.0
    rows = rows[rows["Date"] != days[6]]  # and one day on which the stock is missing from the snapshot
    adjusted = kr.adjust(kr.standardize(rows))
    assert adjusted["reopened"].tolist() == [False, False, False, False, True, False, False]  # the step right after the halt
    frame = kr.stock_frame(adjusted, days)
    assert list(frame.index) == list(days)
    no_price = frame[["open", "high", "low", "close", "volume"]].isna().all(axis=1)
    assert no_price.tolist() == [False, True, True, True, False, False, True, False]
    assert frame.loc[~no_price, ["open", "high", "low", "close", "volume"]].notna().all().all()
    np.testing.assert_allclose(frame["factor"], [1, 1, 1, 1, 50, 50, 50, 50])  # carried over days without trading
    np.testing.assert_allclose(frame["close"].dropna(), 100.0)  # no jump across the halt and the split


def test_traded_row_with_missing_open_high_low_keeps_its_close():
    rows = stock("000010", "A", [100, 101, 102], days=DAYS[:3])
    rows.loc[1, ["Open", "High", "Low"]] = 0.0
    out = kr.adjust(kr.standardize(rows))
    assert out.loc[1, ["open", "high", "low"]].isna().all() and out.loc[1, "close"] == 101 and out.loc[1, "volume"] == 100


def test_rows_with_a_zero_close_and_short_codes_are_handled():
    rows = stock("10", "A", [100, 0, 100], days=DAYS[:3])  # the early years store 000010 as "10"
    usable, _ = prepared(rows)
    assert usable["Code"].tolist() == ["000010", "000010"] and usable["Date"].tolist() == [DAYS[0], DAYS[2]]


# Universe (Rule KU) ------------------------------------------------------------------------------------------
def test_selection_schedule():
    schedule = kr.selection_schedule(DAYS)
    assert [(s.date, s.effective, s.window_start) for s in schedule] == [
        (MAY, JUN, pd.Timestamp("2019-12-01")),
        (NOV, DEC, pd.Timestamp("2020-06-01")),
    ]
    # No selection in a month that is not over yet, and none whose window starts before the data.
    assert [s.date for s in kr.selection_schedule(DAYS[DAYS <= "2020-11-27"])] == [MAY]
    assert [s.date for s in kr.selection_schedule(DAYS[DAYS >= "2019-12-02"])] == [NOV]


def test_universe_picks_the_largest_eligible_stocks():
    caps, calendar = prepared(*market())
    universe = kr.build_universe(caps, calendar, top_n=3)
    assert picks(universe, MAY) == ["000010", "000020", "000030"] == picks(universe, NOV)
    assert universe.spans == {code: [(JUN, calendar[-1])] for code in ("000010", "000020", "000030")}  # spans joined
    ranked = universe.ranked[MAY]
    np.testing.assert_allclose(ranked["avg_cap"], [500, 400, 300])
    assert (ranked["cap_days"] == ranked["window_days"]).all()


@pytest.mark.parametrize(
    ("name", "excluded"),
    [
        ("삼성전자", False), ("메리츠화재", False), ("메리츠금융지주", False), ("두산인프라코어", False), ("삼호", False),
        ("한국토지신탁", False), ("아모레퍼시픽", False), ("SNG21", False), ("NAVER", False),
        ("롯데리츠", True), ("이리츠코크렙", True), ("교보메리츠", True), ("맥쿼리인프라", True), ("우리스팩1호", True),
        ("동북아12호", True), ("하이골드 12호", True), ("한국ANKOR유전", True), ("박현주3호", True), ("KODEX 200", True),
        ("맵스리얼티1", True), ("sk리츠", True),
    ],
)  # fmt: skip
def test_name_patterns(name, excluded):
    assert (kr.exclusion(name) is not None) == excluded


def test_name_patterns_are_the_ones_fixed_in_the_rules():
    """docs/RULES.md (Rule KU, Details) lists every pattern word for word; change it there first."""
    rules = (ROOT / "docs/RULES.md").read_text()
    for label, pattern in kr.EXCLUSION_PATTERNS:
        assert f"`{pattern}`" in rules, label


def test_picks_do_not_change_when_later_data_is_altered():
    caps, calendar = prepared(*market())
    base = kr.build_universe(caps, calendar, top_n=3)
    rng = np.random.default_rng(0)
    later = caps["Date"] > MAY
    altered = caps.copy()
    altered.loc[later, "Marcap"] = rng.uniform(1, 1e6, int(later.sum()))  # any future size ordering
    altered.loc[later & (altered["Code"] == "000020"), "Name"] = "테스트스팩2호"  # becomes a SPAC by name
    altered.loc[later & (altered["Code"] == "000030"), "Market"] = "KOSDAQ"  # leaves KOSPI
    altered = altered[~(later & (altered["Code"] == "000010") & (altered["Date"] > "2020-07-31"))]  # delisted
    new = kr.build_universe(altered, calendar, top_n=3)
    pd.testing.assert_frame_equal(new.ranked[MAY], base.ranked[MAY])
    assert picks(new, NOV) != picks(base, NOV)  # the alteration is real: the next re-pick sees it
    # The same on the selection date itself does change the picks (the test above is not vacuous).
    on_day = caps.copy()
    on_day.loc[(on_day["Date"] == MAY) & (on_day["Code"] == "000030"), "Market"] = "KOSDAQ"
    assert picks(kr.build_universe(on_day, calendar, top_n=3), MAY) == ["000010", "000020", "000040"]
    # And picking from data cut off at the selection date gives the same answer.
    cut = kr.pick(caps[caps["Date"] <= MAY], calendar[calendar <= JUN], base.schedule[0], top_n=3)
    pd.testing.assert_frame_equal(cut, base.ranked[MAY])


def test_the_80_percent_window_rule():
    window = DAYS[(DAYS >= "2019-12-01") & (DAYS <= MAY)]
    need = -(-4 * len(window) // 5)  # the smallest day count with 5 x days >= 4 x window days
    assert 5 * need >= 4 * len(window) > 5 * (need - 1)
    enough = stock("000110", "X", 900.0, days=DAYS[DAYS >= window[-need]])  # listed just long enough
    short = stock("000120", "Y", 800.0, days=DAYS[DAYS >= window[-(need - 1)]])  # one day short
    no_cap = stock("000130", "Z", 700.0)  # rows on every day, but no market cap on too many of them
    no_cap.loc[no_cap["Date"].isin(window[: len(window) - need + 1]), "Marcap"] = 0.0
    gone = stock("000140", "W", 600.0, days=DAYS[DAYS < MAY])  # every day but the selection date: not listed on it
    caps, calendar = prepared(*market(), enough, short, no_cap, gone)
    universe = kr.build_universe(caps, calendar, top_n=3)
    assert picks(universe, MAY) == ["000110", "000010", "000020"]
    assert universe.ranked[MAY].loc[0, ["cap_days", "window_days"]].tolist() == [need, len(window)]
    assert picks(universe, NOV) == ["000110", "000120", "000130"]  # by November all three have a full window


def test_delisted_stock_keeps_its_history_and_leaves_the_universe():
    last = pd.Timestamp("2020-08-14")
    stocks = market()
    stocks[1] = stock("000020", "B", 40.0, days=DAYS[DAYS <= last])  # B stops for good in August
    caps, calendar = prepared(*stocks)
    universe = kr.build_universe(caps, calendar, top_n=3)
    conv = kr.convert(caps, calendar, universe)
    assert universe.spans["000020"] == [(JUN, last)]
    assert picks(universe, MAY) == ["000010", "000020", "000030"] and picks(universe, NOV) == ["000010", "000030", "000040"]
    members = conv.member.sum(axis=1)
    assert (members[JUN:last] == 3).all() and (members[calendar[calendar > last][0] : NOV] == 2).all()  # not replaced
    assert (members[DEC:] == 3).all() and not conv.member.loc[:MAY].any().any()
    # Full history up to the delisting, also before the stock was a member; then it simply ends.
    frame = conv.frames["000020"]
    assert frame.index[0] == DAYS[0] and frame.index[-1] == last and frame["close"].notna().all()
    # D is picked in November only, and still has its whole history.
    assert universe.spans["000040"] == [(DEC, calendar[-1])] and conv.frames["000040"].index[0] == DAYS[0]


def test_stock_that_changes_board():
    stocks = market()
    stocks[2] = stock("000030", "C", 30.0, market=np.where(DAYS < "2020-07-01", "KOSPI", "KOSDAQ"))  # leaves KOSPI
    stocks.append(stock("000150", "F", 45.0, market=np.where(DAYS < "2020-04-01", "KOSDAQ", "KOSPI")))  # joins KOSPI
    caps, calendar = prepared(*stocks)
    universe = kr.build_universe(caps, calendar, top_n=3)
    conv = kr.convert(caps, calendar, universe)
    # F is eligible in May (KOSPI on the selection date), ranked on all its rows in the window, whichever board.
    assert picks(universe, MAY) == ["000010", "000150", "000020"]
    assert universe.ranked[MAY].loc[1, "cap_days"] == universe.ranked[MAY].loc[1, "window_days"]
    # C was picked in neither round here; make it a member first, then move it.
    stocks[0] = stock("000010", "A", 5.0)  # A shrinks, so the May picks are F, B, C
    caps, calendar = prepared(*stocks)
    universe = kr.build_universe(caps, calendar, top_n=3)
    conv = kr.convert(caps, calendar, universe)
    assert picks(universe, MAY) == ["000150", "000020", "000030"]
    assert universe.spans["000030"] == [(JUN, NOV)]  # a member until the next effective date, whatever its board
    assert "000030" not in picks(universe, NOV)  # labelled KOSDAQ on the selection date
    frame = conv.frames["000030"]
    assert frame.index[0] == DAYS[0] and frame.index[-1] == DAYS[-1] and frame["close"].notna().all()  # history on both boards


# Index series ------------------------------------------------------------------------------------------------
def index_market(split: bool) -> list[pd.DataFrame]:
    rng = np.random.default_rng(1)
    stocks = [stock(code, code, (50.0 - 10 * i) * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS))))) for i, code in enumerate(("000010", "000020", "000030", "000040"))]
    a, b = stocks[0], stocks[1]
    if split:  # A: 5-for-1 on 2020-07-01 (price / 5, market cap unchanged, the change measured from the new base price)
        after = a["Date"] >= "2020-07-01"
        a.loc[after, ["Open", "High", "Low", "Close"]] /= 5.0
        a["Changes"] = np.diff(a["Close"], prepend=a["Close"].iloc[0])
        day = a.index[a["Date"] == "2020-07-01"][0]
        a.loc[day, "Changes"] = a.loc[day, "Close"] - a.loc[day - 1, "Close"] / 5.0
    # B: halted for a week in September (rows stay, zero volume).
    b.loc[b["Date"].between("2020-09-07", "2020-09-11"), ["Open", "High", "Low", "Volume"]] = 0.0
    return stocks


def test_index_series_matches_a_plain_loop():
    caps, calendar = prepared(*index_market(split=True))
    universe = kr.build_universe(caps, calendar, top_n=3)
    conv = kr.convert(caps, calendar, universe)

    level, expected = kr.INDEX_BASE, {MAY: kr.INDEX_BASE}
    by_code = {code: part.set_index("Date") for code, part in conv.adjusted.groupby("Code")}
    for t in calendar[calendar > MAY]:
        num = den = 0.0
        for code in conv.member.columns[conv.member.loc[t]]:
            rows = by_code[code]
            earlier = rows[rows.index < t]
            weight = earlier["marcap"].dropna().iloc[-1]
            traded_before = earlier["close"].dropna()
            today = rows["close"].get(t, np.nan)
            ret = 0.0 if np.isnan(today) or traded_before.empty else today / traded_before.iloc[-1] - 1.0
            num, den = num + weight * ret, den + weight
        level *= 1.0 + (num / den if den else 0.0)
        expected[t] = level
    expected = pd.Series(expected)
    assert conv.index.loc[: calendar[calendar < MAY][-1]].isna().all() and conv.index.loc[MAY] == kr.INDEX_BASE
    np.testing.assert_allclose(conv.index.loc[MAY:], expected, rtol=1e-12)
    assert conv.index.loc[JUN:].nunique() > 100  # it moves
    # A split moves nothing: the same market without the split gives the same index.
    plain_caps, _ = prepared(*index_market(split=False))
    plain = kr.convert(plain_caps, calendar, kr.build_universe(plain_caps, calendar, top_n=3))
    assert plain.adjusted["factor"].eq(1.0).all() and conv.adjusted["factor"].max() == pytest.approx(5.0)
    np.testing.assert_allclose(conv.index.loc[MAY:], plain.index.loc[MAY:], rtol=1e-12)


# Go/no-go checks can fail ------------------------------------------------------------------------------------
def kg_rows() -> pd.DataFrame:
    days = pd.bdate_range("2015-01-01", "2021-06-30")

    def split_stock(code, price, ratio, day):
        s = stock(code, code, np.where(days < day, price, price / ratio), days=days)
        s.loc[s["Date"] == day, "Changes"] = 0.0  # the exchange's base price is the old close / ratio
        return s

    sam = split_stock(kr.SAMSUNG, 2_650_000.0, 50, pd.Timestamp("2018-05-04"))
    sam.loc[sam["Date"].isin(pd.to_datetime([*kr.HALT_DAYS, "2018-05-01"])), ["Open", "High", "Low", "Volume"]] = 0.0
    others = [split_stock(code, 1_000_000.0, ratio, pd.Timestamp(f"{month}-15")) for code, _, ratio, month in kr.OTHER_SPLITS]
    return kr.standardize(pd.concat([sam, *others], ignore_index=True))


def test_kg2_and_kg3_pass_and_fail():
    rows = kg_rows()
    calendar = pd.DatetimeIndex(sorted(rows["Date"].unique()))
    assert kr.check_kg2(rows).passed and kr.check_kg3(rows, calendar).passed
    unadjusted = rows.copy()  # a source whose change is measured from the old close: the split is not adjusted
    hit = (unadjusted["Code"] == kr.SAMSUNG) & (unadjusted["Date"] == kr.SAMSUNG_AFTER_DAY)
    unadjusted.loc[hit, "Changes"] = 53_000.0 - 2_650_000.0
    assert not kr.check_kg2(unadjusted).passed
    two_wrong = rows.copy()
    for code, _, ratio, month in kr.OTHER_SPLITS[:2]:
        hit = (two_wrong["Code"] == code) & (two_wrong["Date"] == f"{month}-15")
        two_wrong.loc[hit, "Changes"] = 1_000_000.0 / ratio - 1_000_000.0
    assert not kr.check_kg2(two_wrong).passed
    traded_halt = rows.copy()  # a source that shows trading on a halt day
    hit = (traded_halt["Code"] == kr.SAMSUNG) & (traded_halt["Date"] == kr.HALT_DAYS[1])
    traded_halt.loc[hit, ["Open", "High", "Low", "Volume"]] = [2_650_000.0, 2_650_000.0, 2_650_000.0, 10.0]
    assert not kr.check_kg3(traded_halt, calendar).passed


def test_kg1_pass_and_fail():
    def code_rows(code, last, market="KOSPI"):
        return pd.DataFrame({"Code": code, "Market": market, "Date": pd.to_datetime(["2010-01-04", last]), "Volume": 5.0})

    known = [code_rows(code, f"{year}-06-30") for code, _, year in kr.KNOWN_DELISTINGS]
    ended = [code_rows(f"{i:05d}0", "2023-02-28") for i in range(20001, 20198)]  # 197 + the 3 known ones = 200
    alive = [code_rows("999990", "2024-02-29"), code_rows("999980", "2020-01-02", market="KOSDAQ")]
    meta = pd.concat([*known, *ended, *alive], ignore_index=True)
    assert kr.check_kg1(meta).passed
    assert not kr.check_kg1(meta[meta["Code"] != "200010"]).passed  # 199 ended KOSPI codes
    late = meta.copy()  # a stock that ends on 2023-03-01 is not "more than one year before 2024-03-01"
    late.loc[(late["Code"] == "200010") & (late["Date"] == "2023-02-28"), "Date"] = pd.Timestamp("2023-03-01")
    assert not kr.check_kg1(late).passed
    wrong_year = pd.concat([code_rows(code, "2019-06-28") for code, _, _ in kr.KNOWN_DELISTINGS[:2]] + [known[2], *ended, *alive])
    assert not kr.check_kg1(wrong_year).passed  # only 1 of the 3 known delistings ends in its stated year
    after_lock = pd.concat([meta, code_rows("200010", "2024-03-04")])  # rows from the test period are never read
    assert kr.check_kg1(after_lock).passed


# Pins and configs --------------------------------------------------------------------------------------------
def test_the_download_script_and_the_converter_use_the_same_pins(tmp_path):
    """``--print-pins`` is the script's no-network mode: it prints the pins it sourced, which must be the ones this
    module read from scripts/kr_data.pins. Nothing is fetched or written."""
    env = {**os.environ, "DL_DIR": str(tmp_path / "dl"), "TOOLS_DIR": str(tmp_path / "tools")}
    done = subprocess.run([str(ROOT / "scripts/get_kr_data.sh"), "--print-pins"], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert dict(line.split("=", 1) for line in done.stdout.splitlines()) == kr.PINS and done.stderr == ""
    assert not any(tmp_path.iterdir())
    assert re.fullmatch(r"[0-9a-f]{40}", kr.PINNED_COMMIT) and re.fullmatch(r"[0-9a-f]{40}", kr.DUMP_BIN_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{64}", kr.DUMP_BIN_SHA256)


def test_the_download_script_rejects_an_unknown_or_extra_argument_before_fetching_anything(tmp_path):
    env = {**os.environ, "DL_DIR": str(tmp_path / "dl"), "TOOLS_DIR": str(tmp_path / "tools")}
    for args in (["--print-pin"], ["--write-manifest"], ["--print-pins", "junk"]):
        done = subprocess.run([str(ROOT / "scripts/get_kr_data.sh"), *args], capture_output=True, text=True, env=env)
        assert done.returncode == 2 and done.stdout == ""
        assert done.stderr.strip() == f"unknown argument(s): {' '.join(args)} (the only option is --print-pins)"
    assert not any(tmp_path.iterdir())


def test_a_malformed_manifest_line_is_a_reported_error_not_a_traceback(tmp_path, capsys):
    manifest = tmp_path / "kr_data.sha256"
    manifest.write_text("0123abcd  data/marcap-1995.parquet\n0123abcd\n")
    with pytest.raises(kr.KrError, match=r"kr_data\.sha256:2: expected"):
        kr.read_manifest(manifest)
    out = tmp_path / "go_no_go.md"
    assert kr.main(["go-no-go", "--source", str(tmp_path), "--manifest", str(manifest), "--out", str(out)]) == 2
    assert capsys.readouterr().err.startswith("error: ") and not out.exists()


def test_kr200_config_names_the_pinned_data():
    manifest = ROOT / "scripts/kr_data.sha256"
    assert sorted(kr.read_manifest(manifest)) == [f"data/marcap-{year}.parquet" for year in range(1995, 2027)]
    cfg = load_config(ROOT / "configs/kr200.yaml")
    assert cfg.data.data_sha256 == kr.sha256_of(manifest)  # what data_sha256 means for this dataset
    assert cfg.data.provider_uri.endswith(f"kr_data_{kr.PINNED_COMMIT[:7]}") and cfg.data.data_tag.endswith(kr.PINNED_COMMIT[:7])
    assert (cfg.data.universe, cfg.data.index_symbol, cfg.market) == (kr.UNIVERSE_NAME, kr.INDEX_SYMBOL, "kr200")


def test_korea_configs_keep_the_locked_test_period_and_the_paper_recipe():
    paper = load_config(ROOT / "configs/csi300.yaml")
    main = load_config(ROOT / "configs/kr200.yaml")
    wf = load_config(ROOT / "configs/walkforward/kr200/paper.yaml")
    single = load_config(ROOT / "configs/walkforward/kr200/single_mlp.yaml")
    assert main.splits == paper.splits and main.splits.test == ("2024-03-01", "2026-03-01")
    assert (main.model, main.loss, main.train, main.features, main.regimes) == (paper.model, paper.loss, paper.train, paper.features, paper.regimes)
    china_wf = load_config(ROOT / "configs/walkforward/paper.yaml")
    assert wf.walkforward == china_wf.walkforward and wf.splits == china_wf.splits and wf.data.load_start == "2007-01-01"
    assert (main.run_dir, wf.run_dir, single.run_dir) == ("runs/kr200", "runs/kr200_wf", "runs/kr200_wf")
    assert single.model.kind == "single_mlp" and wf.data_hash() == single.data_hash()
    for cfg in (main, wf, single):
        assert cfg.qa.membership_checks
        assert cfg.trading.price_limits == (((), "1998-12-07", 0.15), ((), "2015-06-15", 0.30))
        steps = cfg.trading.stamp_duty_bps
        assert steps[0] == ("2007-01-01", 30.0) and pd.Timestamp(steps[0][0]) <= pd.Timestamp(wf.data.load_start)
        assert [bps for _, bps in steps] == [30.0, 25.0, 23.0, 20.0, 18.0, 15.0, 20.0]


# Round trip through Qlib (needs Qlib's dump_bin.py) -----------------------------------------------------------
@pytest.fixture(scope="module")
def dump_bin() -> Path:
    path = Path(f"~/data/tools/qlib_{kr.DUMP_BIN_COMMIT[:7]}/dump_bin.py").expanduser()
    if not path.is_file() or kr.sha256_of(path) != kr.DUMP_BIN_SHA256:
        pytest.skip("Qlib's dump_bin.py is missing; run scripts/get_kr_data.sh")
    return path


@pytest.mark.data
def test_round_trip_through_qlib(tmp_path, dump_bin):
    from qlib.data import D

    from detmaso.data.qlib_io import RAW_FIELDS, load_raw_panel

    stocks = market()
    b = stocks[1]
    b.loc[b["Date"].between("2020-09-07", "2020-09-09"), ["Open", "High", "Low", "Volume"]] = 0.0  # a halt
    stocks[1] = b[b["Date"] <= "2020-10-16"]  # then B stops for good
    c = stocks[2].copy()
    after = c["Date"] >= "2020-07-01"  # C: 2-for-1 split
    c.loc[after, ["Open", "High", "Low", "Close"]] = 15.0
    c["Changes"] = 0.0
    stocks[2] = c[c["Date"] != "2020-12-10"]  # and one day missing from the snapshot
    stocks.append(stock("0030R0", "R", 35.0, days=DAYS[DAYS >= "2019-11-20"]))  # a code with a letter, listed later
    caps, calendar = prepared(*stocks)
    universe = kr.build_universe(caps, calendar, top_n=3)
    conv = kr.convert(caps, calendar, universe)
    qlib_dir = tmp_path / "kr_qlib"
    kr.write_qlib(conv, qlib_dir, dump_bin, workers=2)
    assert sorted(conv.frames) == ["000010", "000020", "000030", "0030R0"]
    assert (qlib_dir / "calendars/day.txt").read_text().split() == [d.strftime("%Y-%m-%d") for d in calendar]
    assert (qlib_dir / "features/kr0030r0/close.day.bin").is_file()  # Qlib's lower-case folder names

    # 1. The raw files: start index, then float32 values, NaN where there is no price.
    for code, frame in conv.frames.items():
        for name in kr.FIELDS:
            start, values = kr.read_bin(qlib_dir, kr.symbol(code), name)
            assert start == calendar.get_loc(frame.index[0])
            np.testing.assert_array_equal(values, frame[name].to_numpy(dtype="float32"))

    # 2. Through the project's loader, which goes through Qlib's own reader.
    cfg = synthetic_config(tmp_path / "run", data={
        "provider_uri": str(qlib_dir), "region": "us", "universe": kr.UNIVERSE_NAME, "index_symbol": kr.INDEX_SYMBOL,
        "load_start": str(calendar[0].date()), "load_end": str(calendar[-1].date()),
    })  # fmt: skip
    raw = load_raw_panel(cfg)
    assert list(raw.calendar) == list(calendar)
    assert raw.codes == [kr.symbol(code) for code in sorted(universe.spans)]
    expected_member = conv.member[sorted(universe.spans)].rename(columns=kr.symbol)
    pd.testing.assert_frame_equal(raw.member, expected_member, check_names=False, check_freq=False)
    for code in universe.spans:
        frame = conv.frames[code].reindex(calendar)
        for name in RAW_FIELDS:
            got = raw.fields[name][kr.symbol(code)].to_numpy()
            np.testing.assert_array_equal(got, frame[name].to_numpy(dtype="float32").astype("float64"), err_msg=f"{code} {name}")
    np.testing.assert_array_equal(raw.index_close.to_numpy(), conv.index.to_numpy(dtype="float32").astype("float64"))
    # The cases the loader must see: the halt and the missing day have no price, the delisted stock ends, and the
    # split leaves the adjusted close flat while raw volume = volume x factor still holds.
    close, factor = raw.fields["close"], raw.fields["factor"]
    assert close.loc["2020-09-07":"2020-09-09", "KR000020"].isna().all() and close.loc["2020-09-10", "KR000020"] == 40.0
    assert close.loc["2020-10-19":, "KR000020"].isna().all() and not raw.member.loc["2020-10-19":, "KR000020"].any()
    assert np.isnan(close.loc["2020-12-10", "KR000030"]) and factor.loc["2020-12-10", "KR000030"] == 2.0
    assert (close["KR000030"].dropna() == 30.0).all()
    np.testing.assert_allclose((raw.fields["volume"] * factor)["KR000030"].dropna(), 100.0)

    # 3. Qlib's own reader directly, for the index and a stock outside the universe file.
    index = D.features([kr.INDEX_SYMBOL], ["$close", "$factor"], start_time=str(calendar[0].date()), end_time=str(calendar[-1].date()))
    assert index["$close"].notna().sum() == conv.index.notna().sum() and index["$close"].dropna().iloc[0] == kr.INDEX_BASE
    listed = D.list_instruments(D.instruments("all"), as_list=True)
    assert sorted(listed) == sorted([kr.symbol(code) for code in conv.frames] + [kr.INDEX_SYMBOL])

    # 4. Deterministic: writing the same conversion again gives the same bytes.
    kr.write_qlib(conv, tmp_path / "again", dump_bin, workers=1)
    assert kr.tree_digest(tmp_path / "again") == kr.tree_digest(qlib_dir)


# The real download and the converted dataset -----------------------------------------------------------------
@pytest.fixture(scope="module")
def kr_cfg():
    """The walk-forward Korea config, if the converted dataset is on disk at the pinned commit."""
    cfg = load_config(ROOT / "configs/walkforward/kr200/paper.yaml")
    info = Path(cfg.data.provider_uri).expanduser() / kr.BUILD_INFO
    if not info.is_file():
        pytest.skip("converted Korea dataset missing; run scripts/get_kr_data.sh and the converter")
    if json.loads(info.read_text())["source_commit"] != kr.PINNED_COMMIT:
        pytest.skip("the converted Korea dataset is from another source commit")
    return cfg


@pytest.fixture(scope="module")
def kr_source() -> Path:
    source = Path(f"~/data/marcap/{kr.PINNED_COMMIT}").expanduser()
    if not (source / "data").is_dir():
        pytest.skip("pinned Korea source missing; run scripts/get_kr_data.sh")
    return source


@pytest.mark.data
def test_real_samsung_2018_split(kr_cfg):
    """Rule KG2 and KG3 on the files as written: read with Qlib's own reader."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(Path(kr_cfg.data.provider_uri).expanduser()), region=kr_cfg.data.region, kernels=1)
    got = D.features([kr.symbol(kr.SAMSUNG)], ["$close", "$factor", "$volume"], start_time="2018-04-20", end_time="2018-05-11")
    got = got.droplevel("instrument")
    before, after = pd.Timestamp("2018-04-27"), pd.Timestamp(kr.SAMSUNG_AFTER_DAY)
    assert abs(got.loc[after, "$factor"] / got.loc[before, "$factor"] / kr.SAMSUNG_SPLIT_RATIO - 1.0) <= kr.SPLIT_TOLERANCE
    assert got.loc[pd.to_datetime(list(kr.HALT_DAYS)), ["$close", "$volume"]].isna().all().all()
    assert got.loc[pd.to_datetime(list(kr.HALT_DAYS)), "$factor"].notna().all()  # the factor is carried
    raw_close = got["$close"] / got["$factor"]
    np.testing.assert_allclose(raw_close.loc[[before, after]], [2_650_000, 51_900], rtol=1e-6)
    np.testing.assert_allclose(got.loc[after, "$volume"] * got.loc[after, "$factor"], 39_565_391, rtol=1e-6)
    assert abs(got.loc[after, "$close"] / got.loc[before, "$close"] - (51_900 / 53_000)) < 1e-6  # -2.08% across the split


@pytest.mark.data
def test_real_dataset_through_the_loader(kr_cfg):
    from detmaso.data.qlib_io import load_raw_panel

    info = json.loads((Path(kr_cfg.data.provider_uri).expanduser() / kr.BUILD_INFO).read_text())
    assert info["no_base_rows_from_2007"] == 0  # docs/RULES.md, Rule KU details: no row without a base price from 2007
    raw = load_raw_panel(kr_cfg)
    days = raw.calendar[raw.calendar < kr.CHECK_END]  # Rule ND: nothing from the locked test period is checked
    assert days[0] == pd.Timestamp("2007-01-02") and days[-1] == pd.Timestamp("2024-02-29")
    member, close = raw.member.loc[days], raw.fields["close"].loc[days]
    per_day = member.sum(axis=1)
    assert per_day.max() == kr.TOP_N and per_day.min() >= 190
    assert (member & close.notna()).to_numpy().sum() / member.to_numpy().sum() > 0.99  # priced share of member-days
    assert all(code.startswith(kr.CODE_PREFIX) and code.endswith("0") and len(code) == 8 for code in raw.codes)
    assert raw.index_close.loc[days].notna().all() and (raw.index_close.loc[days] > 0).all()
    assert (raw.fields["factor"].loc[days] > 0).to_numpy()[raw.fields["factor"].loc[days].notna().to_numpy()].all()


@pytest.mark.data
def test_real_members_are_the_picks_of_the_selection_date(kr_cfg, kr_source):
    """One re-pick recomputed from the source with data up to its selection date only, against the written spans."""
    year = kr.standardize(pd.read_parquet(kr_source / "data/marcap-2018.parquet", columns=list(kr.ROW_COLUMNS)))
    calendar = pd.DatetimeIndex(sorted(year["Date"].unique()))
    selection = next(s for s in kr.selection_schedule(calendar) if s.date == pd.Timestamp("2018-11-30"))
    caps = year[kr.common_rows(year) & (year["Date"] <= selection.date)]
    ranked = kr.pick(caps, calendar, selection)
    assert len(ranked) == kr.TOP_N and ranked.loc[0, "Code"] == kr.SAMSUNG
    lines = (Path(kr_cfg.data.provider_uri).expanduser() / "instruments" / f"{kr.UNIVERSE_NAME}.txt").read_text().splitlines()
    spans = [line.split("\t") for line in lines]
    on_day = {code for code, start, end in spans if start <= str(selection.effective.date()) <= end}
    assert on_day == {kr.symbol(code) for code in ranked["Code"]}
