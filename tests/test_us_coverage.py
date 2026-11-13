"""The free US coverage check (Rule UC): ticker normalization, the weekday expansion of membership, the date-range
match and the 95.0% bar, on small synthetic inputs. The pinned files are never read here."""

from __future__ import annotations

import os
import re
import subprocess
import zipfile
from datetime import date

import pytest

from detmaso.data.ingest import us_coverage as uc
from detmaso.data.ingest.us_coverage import CoverageError, Listing
from tests.conftest import ROOT

D = date.fromisoformat


def stock(ticker, start, end, exchange="NYSE", asset_type="Stock", currency="USD"):
    return Listing(uc.normalize_ticker(ticker), exchange, asset_type, currency, D(start) if start else None, D(end) if end else None)


def membership(*rows):
    return uc.read_membership(["date,tickers\n", *(f'{day},"{tickers}"\n' for day, tickers in rows)])


# Normalization ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("brk.b", "BRK-B"), ("BF/B", "BF-B"), ("BRK-B", "BRK-B"), (" aapl ", "AAPL"), ("RDS.A", "RDS-A"), ("AAL-199702", "AAL-199702"), ("04I:BE", "04I:BE"), ("A B", "A B")],
)
def test_normalize_ticker_changes_only_case_and_class_separators(raw, expected):
    assert uc.normalize_ticker(raw) == expected


# Days ---------------------------------------------------------------------------------------------------------
def test_weekdays_keep_holidays_and_both_ends_and_drop_weekends():
    days = uc.weekdays(D("2008-01-01"), D("2008-01-07"))
    assert days == [D("2008-01-01"), D("2008-01-02"), D("2008-01-03"), D("2008-01-04"), D("2008-01-07")]  # New Year's Day stays
    assert uc.weekdays(D("2008-01-05"), D("2008-01-06")) == []


def test_the_rule_window_has_4218_weekdays():
    days = uc.weekdays(uc.WINDOW_START, uc.WINDOW_END)
    assert (days[0], days[-1], len(days)) == (D("2008-01-01"), D("2024-02-29"), 4218)
    assert uc.BAR_PER_MILLE == 950


# Membership ---------------------------------------------------------------------------------------------------
def test_membership_is_the_last_row_on_or_before_each_weekday():
    rows = membership(("2007-12-31", "A,B"), ("2008-01-03", "A,C"), ("2008-01-05", "A,C,D"), ("2008-01-09", "A"))  # 01-05 is a Saturday
    got = uc.member_days(rows, uc.weekdays(D("2008-01-01"), D("2008-01-08")))
    assert got == {
        "A": [D("2008-01-01"), D("2008-01-02"), D("2008-01-03"), D("2008-01-04"), D("2008-01-07"), D("2008-01-08")],
        "B": [D("2008-01-01"), D("2008-01-02")],  # gone from the row's own date on
        "C": [D("2008-01-03"), D("2008-01-04"), D("2008-01-07"), D("2008-01-08")],
        "D": [D("2008-01-07"), D("2008-01-08")],  # a Saturday row first counts on the Monday
    }
    assert uc.members_on(rows, D("2008-01-06")) == {"A", "C", "D"}


def test_membership_tickers_are_normalized_and_counted_once_per_day():
    rows = membership(("2008-01-01", "brk.b,BRK/B, x ,,BF.B"))
    assert rows == [(D("2008-01-01"), frozenset({"BRK-B", "X", "BF-B"}))]
    assert uc.member_days(rows, [D("2008-01-01")]) == {"BRK-B": [D("2008-01-01")], "X": [D("2008-01-01")], "BF-B": [D("2008-01-01")]}


def test_membership_rows_after_the_window_are_not_read():
    lines = ["date,tickers\n", '2008-01-01,"A"\n', '2024-02-29,"A,B"\n', '2024-03-01,"A,B,C"\n', "not a date at all\n"]
    assert [day for day, _ in uc.read_membership(lines)] == [D("2008-01-01"), D("2024-02-29")]


@pytest.mark.parametrize(
    "lines",
    [
        ["day,tickers\n", '2008-01-01,"A"\n'],  # another header
        ["date,tickers\n", '2008-01-02,"A"\n', '2008-01-01,"A"\n'],  # not sorted
        ["date,tickers\n", '2008-01-01,"A"\n', '2008-01-01,"B"\n'],  # a date twice
        ["date,tickers\n", '01/02/2008,"A"\n'],  # another date format
        ["date,tickers\n", "2008-01-01,A,B\n"],  # an unquoted list
    ],
)
def test_a_membership_file_in_another_format_gives_no_verdict(lines):
    with pytest.raises(CoverageError):
        uc.read_membership(lines)


def test_a_window_that_starts_before_the_membership_gives_no_verdict():
    with pytest.raises(CoverageError, match="no row dated on or before 2008-01-01"):
        uc.member_days(membership(("2008-01-02", "A")), uc.weekdays(D("2008-01-01"), D("2008-01-04")))


# The match ----------------------------------------------------------------------------------------------------
def coverage_of(ticker, days, listings):
    (tc,) = uc.match({ticker: days}, uc.by_ticker(listings))
    return tc


def test_the_date_range_must_cover_the_day_with_both_ends_included():
    days = uc.weekdays(D("2008-01-01"), D("2008-01-10"))
    tc = coverage_of("XYZ", days, [stock("XYZ", "2008-01-03", "2008-01-08")])
    assert [d.isoformat() for d, m in zip(tc.days, tc.matched) if m] == ["2008-01-03", "2008-01-04", "2008-01-07", "2008-01-08"]
    assert (tc.n_days, tc.n_matched, tc.n_unmatched, tc.has_stock_row, tc.has_any_row) == (8, 4, 4, True, True)
    assert tc.unmatched_days == [D("2008-01-01"), D("2008-01-02"), D("2008-01-09"), D("2008-01-10")]


def test_a_reused_ticker_matches_only_inside_a_row_and_never_in_the_gap_between_rows():
    days = uc.weekdays(D("2008-01-01"), D("2008-01-31"))
    old_company = stock("XYZ", "1999-05-03", "2008-01-04", exchange="PINK")
    new_company = stock("XYZ", "2008-01-21", "2020-06-30", exchange="NASDAQ")
    tc = coverage_of("XYZ", days, [new_company, old_company])
    matched = [d for d, m in zip(tc.days, tc.matched) if m]
    assert matched == uc.weekdays(D("2008-01-01"), D("2008-01-04")) + uc.weekdays(D("2008-01-21"), D("2008-01-31"))
    assert tc.n_unmatched == len(uc.weekdays(D("2008-01-07"), D("2008-01-18")))
    # A later company under the same ticker gives no match for the earlier member, but still counts ticker-only.
    later_only = coverage_of("XYZ", uc.weekdays(D("2008-01-01"), D("2008-01-04")), [stock("XYZ", "2015-01-02", "2020-01-02")])
    assert (later_only.n_matched, later_only.has_stock_row) == (0, True)


def test_a_listing_without_dates_never_matches():
    days = uc.weekdays(D("2008-01-01"), D("2008-01-04"))
    for row in (stock("XYZ", None, None), stock("XYZ", "2000-01-03", None), stock("XYZ", None, "2030-01-01")):
        assert not row.covers(days[0])
        tc = coverage_of("XYZ", days, [row])
        assert (tc.n_matched, tc.has_stock_row) == (0, True)  # it still counts in the ticker-only extra
    both = coverage_of("XYZ", days, [stock("XYZ", None, None), stock("XYZ", "2008-01-02", "2008-01-02")])
    assert both.matched == (False, True, False, False)


@pytest.mark.parametrize(
    "row",
    [
        stock("XYZ", "2000-01-03", "2030-01-01", asset_type="ETF"),
        stock("XYZ", "2000-01-03", "2030-01-01", asset_type="Mutual Fund"),
        stock("XYZ", "2000-01-03", "2030-01-01", asset_type="stock"),
        stock("XYZ", "2000-01-03", "2030-01-01", currency="CNY"),
        stock("XYZ", "2000-01-03", "2030-01-01", exchange="LSE"),
        stock("XYZ", "2000-01-03", "2030-01-01", exchange="SHG"),
        stock("XYZW", "2000-01-03", "2030-01-01"),
    ],
)
def test_only_a_us_stock_row_of_the_same_ticker_counts(row):
    tc = coverage_of("XYZ", [D("2008-01-01")], [row])
    assert (tc.n_matched, tc.has_stock_row) == (0, False)
    assert tc.has_any_row is (row.ticker == "XYZ")  # the upper-bound extra counts a row of any kind


@pytest.mark.parametrize("exchange", ["NYSE", "NASDAQ", "AMEX", "PINK", "OTCGREY", "EXPM", ""])
def test_us_venues_count_including_over_the_counter_and_a_blank_exchange(exchange):
    assert coverage_of("XYZ", [D("2008-01-01")], [stock("XYZ", "2000-01-03", "2030-01-01", exchange=exchange)]).matched == (True,)


def test_class_shares_match_across_separators():
    for theirs in ("BRK-B", "BRK.B", "brk/b"):
        assert coverage_of("BRK-B", [D("2008-01-01")], [stock(theirs, "2000-01-03", "2030-01-01")]).matched == (True,)
    assert coverage_of("BRK-B", [D("2008-01-01")], [stock("BRKB", "2000-01-03", "2030-01-01")]).matched == (False,)


# Tiingo's file ------------------------------------------------------------------------------------------------
HEADER = "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"


def test_read_tiingo_normalizes_tickers_and_keeps_missing_dates_as_missing():
    rows = uc.read_tiingo([HEADER, "brk.b,NYSE,Stock,USD,1996-05-09,2026-11-11\n", "OLD,,Stock,USD,,\n", "000001,SHE,Stock,CNY,2007-01-04,2026-11-12\n"])
    assert rows[0] == Listing("BRK-B", "NYSE", "Stock", "USD", D("1996-05-09"), D("2026-11-11")) and rows[0].is_stock
    assert rows[1] == Listing("OLD", "", "Stock", "USD", None, None) and rows[1].is_stock and not rows[1].dated
    assert not rows[2].is_stock


@pytest.mark.parametrize(
    "lines",
    [
        ["ticker,exchange,assetType,startDate,endDate\n", "A,NYSE,Stock,2000-01-03,2030-01-01\n"],  # a column is missing
        [HEADER, "A,NYSE,Stock,USD,2000-01-03\n"],  # a short row
        [HEADER, "A,NYSE,Stock,USD,01/03/2000,2030-01-01\n"],  # another date format
        [HEADER, "A,NYSE,Stock,USD,2000-02-30,2030-01-01\n"],  # not a real date
        [HEADER, "A,TSX,Stock,USD,2000-01-03,2030-01-01\n"],  # an exchange the rule's details do not name
        [HEADER, "A,TSX,ETF,CAD,2000-01-03,2030-01-01\n"],  # the same, even on a row that is not a stock
    ],
)
def test_a_tiingo_file_the_rule_cannot_read_gives_no_verdict(lines):
    with pytest.raises(CoverageError):
        uc.read_tiingo(lines)


def test_every_exchange_is_in_exactly_one_list():
    assert not uc.US_EXCHANGES & uc.NON_US_EXCHANGES
    assert uc.NON_US_EXCHANGES == {"LSE", "SHE", "SHEB", "SHG", "SHGB"} and len(uc.US_EXCHANGES) == 18


# The bar ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("matched", "total", "expected"),
    [(950, 1000, True), (949, 1000, False), (19, 20, True), (1000, 1000, True), (0, 5, False), (1999109, 2104326, False), (1999110, 2104326, True)],
)
def test_the_bar_is_95_percent_on_the_exact_counts(matched, total, expected):
    assert uc.passes(matched, total) is expected  # 2104326 x 0.95 = 1999109.7


def test_no_member_days_gives_no_verdict():
    with pytest.raises(CoverageError):
        uc.passes(0, 0)


def test_printed_shares_are_rounded_down():
    assert uc.pct(94999, 100000) == "94.99%" and uc.pct(949999, 1000000) == "94.99%" and uc.pct(19, 20) == "95.00%"


# End to end ---------------------------------------------------------------------------------------------------
def small_case():
    """Window 2008-01-01 .. 2008-01-14 (10 weekdays). KEEP stays; GONE leaves on 01-08 when NEWCO joins; XYZ is a
    reused ticker; BF.B is a class share Tiingo writes another way; NODATE is listed without dates."""
    rows = membership(
        ("2007-12-31", "KEEP,GONE,XYZ,BF.B,NODATE"),
        ("2008-01-08", "KEEP,NEWCO,XYZ,BF.B,NODATE"),
        ("2008-01-11", "KEEP,NEWCO,BF.B,NODATE"),
    )
    listings = [
        stock("KEEP", "1990-01-02", "2026-11-11"),
        stock("NEWCO", "2008-01-09", "2026-11-11"),  # listed one day after it joined
        stock("XYZ", "2015-06-01", "2026-11-11"),  # another company, later
        stock("BFB", "1990-01-02", "2026-11-11"),
        stock("NODATE", None, None),
        stock("GONE", "1990-01-02", "2026-11-11", asset_type="ETF"),
    ]
    return uc.check(rows, listings, start=D("2008-01-01"), end=D("2008-01-14"))


def test_check_counts_member_days_leavers_and_the_extra():
    r = small_case()
    by = {tc.ticker: tc for tc in r.coverage}
    assert r.n_days == 10
    assert {t: (tc.n_days, tc.n_matched) for t, tc in by.items()} == {
        "KEEP": (10, 10), "GONE": (5, 0), "NEWCO": (5, 4), "XYZ": (8, 0), "BF-B": (10, 0), "NODATE": (10, 0),
    }  # fmt: skip
    assert uc.totals(r.coverage) == (14, 33, 38, 48)  # matched, ticker only (KEEP, NEWCO, XYZ, NODATE), any row (+ GONE), all
    assert r.leavers == {"GONE", "XYZ"} and not r.verdict
    assert uc.yearly(r.coverage) == {2008: (14, 33, 38, 48)}
    assert [tc.ticker for tc in r.unmatched] == ["BF-B", "NODATE", "XYZ", "GONE", "NEWCO"]
    assert r.reasons == {
        "XYZ": (uc.REUSED, "Tiingo lists it for 2015-06-01..2026-11-11 only, outside the membership dates; left the index on 2008-01-11"),
        "GONE": (uc.REUSED, "Tiingo lists it only as ETF (USD, NYSE); left the index on 2008-01-08, when NEWCO joined (possible rename)"),
        "BF-B": (uc.CLASS_SHARE, "no row under this ticker (Tiingo does list BFB); still a member at the end of the window"),
        "NODATE": (uc.MISSING_DATES, "Tiingo lists it as a stock without dates; still a member at the end of the window"),
        "NEWCO": (uc.MISSING_DATES, "Tiingo lists it for 2008-01-09..2026-11-11 only; still a member at the end of the window"),
    }
    assert all(reason.kind in uc.REASON_KINDS for reason in r.reasons.values())


def test_a_ticker_with_no_row_at_all_is_not_listed():
    r = uc.check(membership(("2007-12-31", "OLD,NEW"), ("2008-01-03", "NEW")), [stock("NEW", "2000-01-03", "2030-01-01")], start=D("2008-01-01"), end=D("2008-01-04"))
    assert r.reasons == {"OLD": (uc.NOT_LISTED, "no row under this ticker; left the index on 2008-01-03")}
    assert uc.totals(r.coverage) == (4, 4, 4, 6) and not r.verdict


def test_a_fully_covered_index_passes():
    rows = membership(("2007-12-31", "A,B"))
    r = uc.check(rows, [stock("A", "2000-01-03", "2030-01-01"), stock("B", "2008-01-01", "2008-01-04")], start=D("2008-01-01"), end=D("2008-01-04"))
    assert r.verdict and uc.totals(r.coverage) == (8, 8, 8, 8) and not r.unmatched and not r.leavers
    assert uc.verdict_line(r) == "PASS: 100.00% of member-days are matched (8 of 8); the bar is 95.0%."


def test_outputs_are_written_and_are_the_same_on_a_second_run(tmp_path):
    r = small_case()
    report, table = uc.write_outputs(r, tmp_path / "a")
    again = uc.write_outputs(small_case(), tmp_path / "b")
    assert report.read_bytes() == again[0].read_bytes() and table.read_bytes() == again[1].read_bytes()
    text = report.read_text()
    assert "**FAIL: 29.16% of member-days are matched (14 of 48); the bar is 95.0%.**" in text
    assert "| 2008 | 10 | 48 | 14 | 29.16% | 68.75% |" in text
    assert "| Left before 2008-01-15 | 2 | 2 | 13 | 0 | 0.00% | 61.53% |" in text
    assert uc.PINNED_SP500_COMMIT in text and uc.PINNED_TIINGO_ZIP_SHA256 in text and uc.PINNED_MEMBERSHIP_SHA256 in text
    assert chr(0x2014) not in text  # no em dash in generated files
    lines = table.read_text().splitlines()
    assert lines[0].startswith("ticker,unmatched_member_days,member_days,first_member_date,last_member_date,")
    assert lines[0].endswith(",tiingo_stock_rows,likely_reason,detail")
    assert lines[1].startswith("BF-B,10,10,2008-01-01,2008-01-14,2008-01-01,2008-01-14,,class share,no row under this ticker")
    assert lines[3].startswith('XYZ,8,8,2008-01-01,2008-01-10,2008-01-01,2008-01-10,2015-06-01..2026-11-11 NYSE,reused ticker,"Tiingo lists it for')
    assert lines[5].startswith("NEWCO,1,5,2008-01-08,2008-01-14,2008-01-08,2008-01-08,2008-01-09..2026-11-11 NYSE,missing dates,Tiingo lists it for")
    assert len(lines) == 6
    assert "| reused ticker | 2 | 13 | 27.08% |" in text and "| not listed | 0 | 0 | 0.00% |" in text
    assert "(any row exists under the ticker, whatever its asset type, currency, exchange and dates): 38 = 79.16%. This is below the bar too" in text


def test_the_command_line_refuses_files_that_are_not_the_pinned_ones(tmp_path, capsys):
    members = tmp_path / "members.csv"
    members.write_text('date,tickers\n2007-12-31,"A"\n')
    listed = tmp_path / "supported_tickers.zip"
    with zipfile.ZipFile(listed, "w") as z:
        z.writestr(uc.TIINGO_CSV_NAME, HEADER + "A,NYSE,Stock,USD,2000-01-03,2030-01-01\n")
    out = tmp_path / "out"
    assert uc.main(["--membership", str(members), "--tiingo", str(listed), "--out-dir", str(out)]) == 2
    assert "not the pinned one" in capsys.readouterr().err and not out.exists()
    # The same files run through the check itself (no pin check): the whole real window, fully covered.
    r = uc.run_check(members, listed, out)
    assert r.verdict and uc.totals(r.coverage) == (4218, 4218, 4218, 4218) and (out / uc.REPORT_FILE).exists()


def test_an_unreadable_tiingo_zip_is_a_reported_error_not_a_traceback(tmp_path):
    members = tmp_path / "members.csv"
    members.write_text('date,tickers\n2007-12-31,"A"\n')
    wrong_name = tmp_path / "wrong_name.zip"
    with zipfile.ZipFile(wrong_name, "w") as z:
        z.writestr("supported-tickers.csv", HEADER)
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip file")
    for path in (wrong_name, corrupt):
        with pytest.raises(CoverageError, match=re.escape(path.name)):
            uc.run_check(members, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_fetch_script_rejects_an_unknown_or_extra_argument_before_fetching_anything(tmp_path):
    env = {**os.environ, "SP500_DIR": str(tmp_path / "sp500"), "TIINGO_ROOT": str(tmp_path / "tiingo")}
    for args in (["--new-pin"], ["--print-pins", "junk"]):
        done = subprocess.run([str(ROOT / "scripts/get_us_coverage_data.sh"), *args], capture_output=True, text=True, env=env)
        assert done.returncode == 2 and done.stdout == ""
        assert done.stderr.strip() == f"unknown argument(s): {' '.join(args)} (the only option is --print-pins)"
    assert not any(tmp_path.iterdir())


def test_the_fetch_script_and_the_check_use_the_same_pins(tmp_path):
    """``--print-pins`` is the script's no-network mode: it prints the pins it sourced, which must be the ones this
    module read from scripts/us_coverage_data.pins. Nothing is fetched or written."""
    env = {**os.environ, "SP500_DIR": str(tmp_path / "sp500"), "TIINGO_ROOT": str(tmp_path / "tiingo")}
    done = subprocess.run([str(ROOT / "scripts/get_us_coverage_data.sh"), "--print-pins"], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert dict(line.split("=", 1) for line in done.stdout.splitlines()) == uc.PINS and done.stderr == ""
    assert not any(tmp_path.iterdir())
    assert re.fullmatch(r"[0-9a-f]{40}", uc.PINNED_SP500_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{64}", uc.PINNED_MEMBERSHIP_SHA256) and re.fullmatch(r"[0-9a-f]{64}", uc.PINNED_TIINGO_ZIP_SHA256)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", uc.PINNED_TIINGO_DOWNLOADED_AT)
    assert uc.PINNED_MEMBERSHIP_FILE.endswith(".csv") and uc.TIINGO_URL.startswith("https://")
