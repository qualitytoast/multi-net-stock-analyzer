"""The membership timing audit (Rule MA): its pure logic on small made-up inputs. Nothing here opens a connection."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("audit_membership", ROOT / "scripts" / "audit_membership.py")
am = importlib.util.module_from_spec(_spec)
sys.modules["audit_membership"] = am
_spec.loader.exec_module(am)

TINY = am.Index("tiny", "TINY", "query_tiny", 4)


def weekdays(first: str, last: str) -> list[str]:
    import datetime as dt

    day, end, out = dt.date.fromisoformat(first), dt.date.fromisoformat(last), []
    while day <= end:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return out


def codes(*numbers: int) -> frozenset[str]:
    return frozenset(f"SH{600000 + n}" for n in numbers)


# Month ends ------------------------------------------------------------------------------------------------------
def test_month_ends_are_the_last_trading_day_of_each_month_in_range():
    days = weekdays("2020-12-01", "2021-04-30")
    days.remove("2021-03-31")  # a holiday on the month's last weekday
    # January 2021 ends on a Sunday, February on a Sunday, March's last weekday is a holiday.
    assert am.month_ends(days, "2021-01", "2021-03") == ["2021-01-29", "2021-02-26", "2021-03-30"]
    assert am.month_ends(days, "2020-12", "2020-12") == ["2020-12-31"]


def test_month_ends_refuse_a_month_without_trading_days():
    days = [d for d in weekdays("2021-01-01", "2021-04-30") if not d.startswith("2021-02")]
    with pytest.raises(ValueError, match="2021-02"):
        am.month_ends(days, "2021-01", "2021-03")


# Lists -----------------------------------------------------------------------------------------------------------
def test_mismatch_share_is_stocks_in_one_list_over_stocks_in_either():
    assert am.mismatch_share({"a", "b", "c"}, {"b", "c", "d"}) == 2 / 4
    assert am.mismatch_share({"a", "b"}, {"a", "b"}) == 0
    assert am.mismatch_share({"a"}, {"b"}) == 1
    assert am.mismatch_share({"a", "b", "c"}, {"a"}) == 2 / 3
    with pytest.raises(ValueError):
        am.mismatch_share(set(), set())


def test_codes_are_rewritten_to_the_archive_form():
    assert am.norm_code("sh.600000") == "SH600000"
    assert am.norm_code("sz.000001") == "SZ000001"
    with pytest.raises(ValueError):
        am.norm_code("600000.SH")


# The archive's spans ---------------------------------------------------------------------------------------------
def test_spans_become_runs_joined_across_weekends_and_cut_at_the_calendar_end():
    cal = am.Calendar(weekdays("2021-01-04", "2021-01-29"))
    spans = {
        "A": [("2021-01-04", "2021-01-08"), ("2021-01-09", "2021-01-15")],  # touch: one run
        "B": [("2021-01-04", "2021-01-08"), ("2021-01-11", "2021-01-15")],  # a weekend apart: still one run
        "C": [("2021-01-04", "2021-01-06"), ("2021-01-13", "2021-03-31")],  # a real gap; the end is cut off
        "D": [("2021-02-01", "2021-03-31")],  # entirely after the calendar: not there at all
        "E": [("2021-01-09", "2021-01-10")],  # a weekend only: no trading day
    }
    runs = am.to_runs(spans, cal)
    assert runs == {"A": [(0, 9)], "B": [(0, 9)], "C": [(0, 2), (7, 19)]}
    assert am.members_on(runs, 2) == {"A", "B", "C"}
    assert am.members_on(runs, 5) == {"A", "B"}
    assert am.members_on(runs, 12) == {"C"}


def test_run_events_skip_the_first_day_and_runs_that_are_still_open():
    runs = {"A": [(0, 9)], "C": [(0, 2), (7, 19)], "F": [(4, 19)]}
    events = am.run_events(runs, first_day=0, last_day=19)
    assert events == {("A", am.DROP): [10], ("C", am.DROP): [3], ("C", am.ADD): [7], ("F", am.ADD): [4]}


# Finding BaoStock's change days ----------------------------------------------------------------------------------
def test_find_changes_locates_every_change_day_and_skips_equal_halves():
    lists = {day: codes(1, 2) for day in range(0, 3)} | {day: codes(1, 3) for day in range(3, 6)} | {day: codes(4, 3) for day in range(6, 9)}
    asked = []

    def members(day):
        asked.append(day)
        return lists[day]

    steps = am.find_changes(0, 8, members)
    assert [(s.prev_day, s.day, s.added, s.dropped) for s in steps] == [(2, 3, codes(3), codes(2)), (5, 6, codes(4), codes(1))]
    assert 1 not in asked and 7 not in asked  # halves with equal ends are not searched
    assert am.find_changes(0, 2, members) == []


def test_find_changes_steps_over_unusable_days():
    lists = {0: codes(1), 1: None, 2: None, 3: codes(2), 4: codes(2)}
    (step,) = am.find_changes(0, 4, lists.get)
    assert (step.prev_day, step.day, step.added, step.dropped) == (0, 3, codes(2), codes(1))
    with pytest.raises(ValueError):
        am.find_changes(1, 4, lists.get)


def test_net_changes_keep_what_lasts_and_count_what_was_undone():
    steps = [
        am.Step(0, 1, codes(5, 6), codes(1)),  # 6 comes in and goes out again below
        am.Step(3, 4, frozenset(), codes(6)),
        am.Step(6, 7, codes(7), frozenset()),
    ]
    kept, undone = am.net_changes(steps, old=codes(1, 2), new=codes(2, 5, 7))
    assert [(code, kind, step.day) for code, kind, step in kept] == [("SH600005", am.ADD, 1), ("SH600007", am.ADD, 7), ("SH600001", am.DROP, 1)]
    assert undone == 2


# Pairing a BaoStock change with the archive's ----------------------------------------------------------------------
def test_match_changes_pairs_the_nearest_within_the_window():
    assert am.match_changes([100], [90, 300], 250) == [90]
    assert am.match_changes([100], [112], 250) == [112]
    assert am.match_changes([100], [351], 250) == [None]  # 251 days apart: not the same change
    assert am.match_changes([100], [350], 250) == [350]
    assert am.match_changes([100], [], 250) == [None]
    assert am.match_changes([], [5], 250) == []


def test_match_changes_uses_each_archive_day_once_and_keeps_the_order():
    assert am.match_changes([100, 343], [340], 250) == [None, 340]
    assert am.match_changes([100, 343], [110], 250) == [110, None]
    assert am.match_changes([5, 300, 600], [2, 310, 590], 250) == [2, 310, 590]
    assert am.match_changes([100, 200], [150, 160], 250) == [150, 160]  # two pairs beat one closer pair


# Late or early ---------------------------------------------------------------------------------------------------
def test_timing_tells_late_from_early_and_clear_from_unclear():
    # BaoStock: old list updated 2021-01-04 (Monday), new list first served on 2021-01-11.
    assert am.timing(10, "2021-01-25", "2021-01-04") == am.LATE
    assert am.timing(0, "2021-01-11", "2021-01-04") == am.SAME_DAY
    # The archive shows it on the 6th: after BaoStock's last old list, so BaoStock could not have shown it yet.
    assert am.timing(-3, "2021-01-06", "2021-01-04") == am.EARLY_UNCLEAR
    # The archive shows it on or before the day of a BaoStock list that still lacked it: clearly earlier.
    assert am.timing(-5, "2021-01-04", "2021-01-04") == am.EARLY_CLEAR
    assert am.timing(-20, "2020-12-14", "2021-01-04") == am.EARLY_CLEAR


def test_lag_stats_and_nearest_rank_percentile():
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 40]
    assert am.percentile(lags, 90) == 9
    assert am.percentile(lags, 91) == 40
    assert am.percentile([7], 90) == 7
    assert am.lag_stats(lags) == {"n": 10, "min": 1, "median": 5.5, "p90": 9, "max": 40}
    assert am.lag_stats([-3, 0, 12]) == {"n": 3, "min": -3, "median": 0, "p90": 12, "max": 12}
    assert am.lag_stats([]) is None


# One index from end to end ---------------------------------------------------------------------------------------
class FakeBaoStock:
    """A list per trading day that changes only on Mondays, stamped with that Monday, like the real service."""

    def __init__(self, cal, changes, start, frozen=None):
        self.cal, self.changes, self.start = cal, changes, start  # changes: {monday: (added, dropped)}
        self.frozen = frozen  # (first, end): no new list is stored from `first` until `end`
        self.broken: dict[str, list[list[str]]] = {}
        self.asked: list[str] = []

    def rows(self, date):
        if date in self.broken:
            return self.broken[date]
        import datetime as dt

        if self.frozen and self.frozen[0] <= date < self.frozen[1]:
            date = self.cal.days[self.cal.first_on_or_after(self.frozen[0]) - 1]

        asked = dt.date.fromisoformat(date)
        stamp = (asked - dt.timedelta(days=asked.weekday())).isoformat()  # the Monday of that week
        members = set(self.start)
        for day, (added, dropped) in sorted(self.changes.items()):
            if day <= date:
                members = (members | set(added)) - set(dropped)
        return [[stamp, f"{c[:2].lower()}.{c[2:]}", "name"] for c in sorted(members)]

    def get(self, index, date):
        self.asked.append(date)
        return am.to_answer(date, self.rows(date))


@pytest.fixture
def market():
    """Three years of weekdays. The archive and BaoStock start with the same four stocks; then:

    - SH600005 replaces SH600001: BaoStock on Monday 2021-03-15, the archive 10 trading days later (late);
    - SH600006 replaces SH600002: BaoStock on Monday 2021-06-14, the archive on 2021-05-31 (clearly early);
    - SH600007 replaces SH600003: BaoStock on Monday 2021-09-13, the archive on Thursday 2021-09-09 (early,
      but inside BaoStock's weekly step);
    - SH600008 replaces SH600004 in BaoStock on 2021-11-15 and the archive never follows;
    - SH600009 replaces SH600005 in BaoStock on 2023-11-13, too close to the end for "never" to be said.
    """
    cal = am.Calendar(weekdays("2020-01-01", "2023-12-29"))
    end = cal.days[-1]
    spans = {
        "SH600001": [("2020-01-01", "2021-03-28")],
        "SH600005": [("2021-03-29", end)],
        "SH600002": [("2020-01-01", "2021-05-30")],
        "SH600006": [("2021-05-31", end)],
        "SH600003": [("2020-01-01", "2021-09-08")],
        "SH600007": [("2021-09-09", end)],
        "SH600004": [("2020-01-01", end)],
    }
    bao = FakeBaoStock(cal, {
        "2021-03-15": (codes(5), codes(1)), "2021-06-14": (codes(6), codes(2)), "2021-09-13": (codes(7), codes(3)),
        "2021-11-15": (codes(8), codes(4)), "2023-11-13": (codes(9), codes(5)),
    }, start=codes(1, 2, 3, 4))  # fmt: skip
    ends = am.month_ends(cal.days, "2020-01", "2023-12")
    return cal, spans, bao, ends


def test_audit_index_dates_every_change_and_sorts_late_early_never(market):
    cal, spans, bao, ends = market
    result = am.audit_index(TINY, cal, ends, spans, bao.get, window=250)
    got = {(c.code, c.kind): c for c in result["changes"]}
    assert len(got) == len(result["changes"]) == 10

    late = got["SH600005", am.ADD]
    assert (late.date, late.archive_date, late.lag, late.timing, late.status) == ("2021-03-15", "2021-03-29", 10, am.LATE, am.MATCHED)
    assert (late.prev_date, late.last_old_update, late.update, late.exact) == ("2021-03-12", "2021-03-08", "2021-03-15", True)
    assert (late.gap, late.week) == (5, True)
    assert got["SH600001", am.DROP].lag == 10

    early = got["SH600006", am.ADD]
    assert (early.date, early.archive_date, early.lag, early.timing) == ("2021-06-14", "2021-05-31", -10, am.EARLY_CLEAR)
    assert got["SH600002", am.DROP].timing == am.EARLY_CLEAR

    unclear = got["SH600007", am.ADD]
    assert (unclear.date, unclear.archive_date, unclear.lag, unclear.timing) == ("2021-09-13", "2021-09-09", -2, am.EARLY_UNCLEAR)

    for key in (("SH600008", am.ADD), ("SH600004", am.DROP)):
        assert (got[key].status, got[key].lag, got[key].archive_date) == (am.NEVER, None, None)
    # The 2023-11 pair: the add has no archive counterpart yet; the drop of SH600005 neither.
    assert got["SH600009", am.ADD].status == am.TOO_RECENT
    assert got["SH600005", am.DROP].status == am.TOO_RECENT
    assert result["undone"] == 0 and result["archive_only"] == []


def test_audit_index_month_end_mismatch(market):
    cal, spans, bao, ends = market
    result = am.audit_index(TINY, cal, ends, spans, bao.get, window=250)
    months = {m.date: m for m in result["months"]}
    assert len(months) == 48 and all(m.status == "ok" for m in months.values())
    assert months["2021-02-26"].share == 0
    # End of March 2021: both show the swap again (the archive caught up on the 29th).
    assert months["2021-03-31"].share == 0
    # End of May 2021: the archive already swapped 2 for 6, BaoStock has not: 2 stocks in one list, 5 in either.
    may = months["2021-05-31"]
    assert (may.only_archive, may.only_baostock, may.either, may.share) == (("SH600006",), ("SH600002",), 5, 2 / 5)
    # From 2021-11-15 BaoStock holds 8 instead of 4 and the archive never follows.
    assert months["2021-11-30"].share == 2 / 5


def test_audit_index_sets_aside_month_ends_with_no_or_a_wrong_sized_list(market):
    cal, spans, bao, ends = market
    bao.broken["2020-06-30"] = []
    bao.broken["2020-07-31"] = [["2020-07-27", "sh.600001", "name"]]  # 1 stock where 4 are expected
    result = am.audit_index(TINY, cal, ends, spans, bao.get, window=250)
    months = {m.date: m for m in result["months"]}
    assert months["2020-06-30"].status == "empty" and months["2020-07-31"].status == "size"
    assert sum(m.status == "ok" for m in result["months"]) == 46
    assert len(result["changes"]) == 10  # the changes are still found across the gap


def test_audit_index_never_asks_for_a_locked_date(market):
    cal, spans, bao, ends = market
    am.audit_index(TINY, cal, ends, spans, bao.get, window=250)
    assert max(bao.asked) <= cal.days[-1]
    assert set(bao.asked) <= set(cal.days)


def test_an_add_the_archive_has_held_since_its_first_day_is_not_called_never_shown():
    cal = am.Calendar(weekdays("2020-01-01", "2022-12-30"))
    end = cal.days[-1]
    spans = {f"SH60000{n}": [("2020-01-01", end)] for n in (1, 2, 3, 5)}
    bao = FakeBaoStock(cal, {"2020-03-02": (codes(5), codes(4))}, start=codes(1, 2, 3, 4))
    result = am.audit_index(TINY, cal, am.month_ends(cal.days, "2020-01", "2022-12"), spans, bao.get, window=250)
    got = {(c.code, c.kind): c.status for c in result["changes"]}
    assert got == {("SH600005", am.ADD): am.FROM_START, ("SH600004", am.DROP): am.NEVER}


def test_a_stretch_where_baostock_skips_updates_is_told_apart_from_an_early_archive():
    """The index swaps SH600004 for SH600010 on Monday 2022-04-11. The archive shows it 10 trading days late, on
    2022-04-25. BaoStock stores no list from 2022-03-07 to 2022-06-05, so it shows the swap only on 2022-06-06."""
    cal = am.Calendar(weekdays("2020-01-01", "2023-12-29"))
    end = cal.days[-1]
    spans = {f"SH60000{n}": [("2020-01-01", end)] for n in (1, 2, 3)}
    spans |= {"SH600004": [("2020-01-01", "2022-04-24")], "SH600010": [("2022-04-25", end)]}
    bao = FakeBaoStock(cal, {"2022-04-11": (codes(10), codes(4))}, start=codes(1, 2, 3, 4), frozen=("2022-03-07", "2022-06-06"))
    result = am.audit_index(TINY, cal, am.month_ends(cal.days, "2020-01", "2023-12"), spans, bao.get, window=250)

    (add, drop) = result["changes"]
    assert (add.code, add.prev_date, add.date, add.last_old_update, add.update) == ("SH600010", "2022-06-03", "2022-06-06", "2022-02-28", "2022-06-06")
    # The archive is 30 trading days ahead of BaoStock's day, but BaoStock had no list in between: not a clear case.
    assert (add.archive_date, add.lag, add.timing, add.gap, add.week) == ("2022-04-25", -30, am.EARLY_UNCLEAR, 70, False)
    assert (drop.code, drop.lag, drop.timing, drop.week) == ("SH600004", -30, am.EARLY_UNCLEAR, False)

    months = {m.date: m for m in result["months"]}
    assert [(months[d].update, months[d].age, months[d].one_list_only) for d in ("2022-02-28", "2022-03-31", "2022-04-29", "2022-05-31", "2022-06-30")] == [
        ("2022-02-28", 0, 0), ("2022-02-28", 31, 0), ("2022-02-28", 60, 2), ("2022-02-28", 92, 2), ("2022-06-27", 3, 0),
    ]  # fmt: skip
    fresh = am.mismatch_totals([m for m in result["months"] if m.age <= am.FRESH_DAYS])
    assert (am.mismatch_totals(result["months"])["one"], fresh["one"], fresh["months"]) == (4, 0, 45)
    facts = am.update_date_facts(result, cal)
    assert (facts["oldest"].date, facts["oldest"].age) == ("2022-06-03", 95)
    assert facts["fresh"] < facts["answers"] and facts["on_stamp"] == facts["exact_steps"] == 1
    assert set(facts["weekdays"]) == {"Monday"} and facts["one_list_per_stamp"] == facts["stamps"]


def test_a_change_is_dated_to_the_week_only_if_baostock_lists_are_a_trading_week_apart():
    def change(gap, exact=True):
        return am.Change("SH600001", am.ADD, "2021-01-08", "2021-01-11", "2021-01-04", "2021-01-11", exact, gap, am.MATCHED, "2021-01-25", 10, am.LATE)

    assert change(5).week and change(3).week
    assert not change(10).week  # BaoStock skipped a Monday
    assert not change(5, exact=False).week
    assert not change(None).week
    assert am.to_answer("2021-01-08", [["2021-01-04", "sh.600001", "x"]]).age == 4
    assert am.to_answer("2021-01-08", []).age is None


# The cache -------------------------------------------------------------------------------------------------------
class FakeSource:
    version = "test"

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def fetch(self, index, date):
        self.calls.append((index.key, date))
        return self.answers[date], 1


def test_cache_saves_answers_and_reruns_offline_without_a_source(tmp_path):
    rows = [["2021-01-04", "sh.600001", "一"], ["2021-01-04", "sz.000002", "二"]]
    source = FakeSource({"2021-01-06": rows, "2021-01-07": []})
    online = am.Cache(tmp_path, source)
    answer = online.get(TINY, "2021-01-06")
    assert (answer.rows, answer.codes, answer.update_date) == (2, {"SH600001", "SZ000002"}, "2021-01-04")
    assert online.get(TINY, "2021-01-07").rows == 0 and online.get(TINY, "2021-01-07").update_date is None
    online.get(TINY, "2021-01-06")
    # One request for the full answer; an empty answer is asked for twice before it is believed.
    assert source.calls == [("tiny", "2021-01-06"), ("tiny", "2021-01-07"), ("tiny", "2021-01-07")]
    online.write_sha_list()
    with (tmp_path / "tiny" / "2021-01-06.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [am.FIELDS, *rows]
    assert [row["date"] for row in online.log_rows(TINY)] == ["2021-01-06", "2021-01-07"]
    assert sorted(am.Cache(tmp_path, None).read_sha_list()) == ["tiny/2021-01-06.csv", "tiny/2021-01-07.csv"]

    offline = am.Cache(tmp_path, None)
    assert offline.get(TINY, "2021-01-06") == answer
    with pytest.raises(SystemExit, match="not in the cache"):
        offline.get(TINY, "2021-01-08")


def test_cache_refuses_locked_dates_and_files_that_changed(tmp_path):
    source = FakeSource({"2021-01-06": [["2021-01-04", "sh.600001", "x"]], "2021-01-07": [["2021-01-04", "sh.600001", "x"]]})
    cache = am.Cache(tmp_path, source)
    with pytest.raises(ValueError, match="locked"):
        cache.get(TINY, am.LOCK_START)
    assert source.calls == []
    cache.get(TINY, "2021-01-06")
    cache.write_sha_list()
    cache.get(TINY, "2021-01-07")  # downloaded after the list was written
    with pytest.raises(SystemExit, match="does not list"):
        am.Cache(tmp_path, None)
    cache.write_sha_list()
    am.Cache(tmp_path, None)
    (tmp_path / "tiny" / "2021-01-06.csv").write_text("updateDate,code,code_name\n2021-01-04,sh.600002,x\n")
    with pytest.raises(SystemExit, match="does not match"):
        am.Cache(tmp_path, None)


def test_offline_needs_a_sha_list(tmp_path):
    with pytest.raises(SystemExit, match="no SHA256SUMS"):
        am.Cache(tmp_path, None)


# The whole script, offline ---------------------------------------------------------------------------------------
def test_main_rebuilds_the_same_report_offline_and_stays_out_of_the_locked_period(tmp_path):
    """A made-up archive and a made-up BaoStock over the real audit window: one index change in June 2015 that
    BaoStock shows on Monday 2015-06-15 and the archive 11 trading days later, on 2015-06-30."""
    days = weekdays("2006-12-01", "2024-06-28")  # runs into the locked period on purpose
    cal = am.Calendar([d for d in days if d < am.LOCK_START])
    archive = tmp_path / "archive"
    (archive / "calendars").mkdir(parents=True)
    (archive / "instruments").mkdir()
    (archive / "calendars" / "day.txt").write_text("\n".join(days) + "\n")
    cache_dir = tmp_path / "cache" / "2026-01-01"
    cache_dir.mkdir(parents=True)

    class Source(FakeSource):
        def __init__(self):
            self.calls, self.by_index = [], {}

        def fetch(self, index, date):
            self.calls.append((index.key, date))
            return self.by_index[index.key].rows(date), 1

    source = Source()
    cache = am.Cache(cache_dir, source)
    ends = am.month_ends(days, am.FIRST_MONTH, am.LAST_MONTH)
    for index in am.INDICES:
        start = frozenset(f"SH{600000 + n}" for n in range(index.size))
        lines = [f"{code}\t2006-12-01\t2024-06-28" for code in sorted(start - {"SH600000"})]
        lines += ["SH600000\t2006-12-01\t2015-06-29", "SZ000001\t2015-06-30\t2024-06-28"]
        (archive / "instruments" / f"{index.key}.txt").write_text("\n".join(lines) + "\n")
        source.by_index[index.key] = FakeBaoStock(cal, {"2015-06-15": ({"SZ000001"}, {"SH600000"})}, start)
        am.audit_index(index, cal, ends, am.read_spans(archive / "instruments" / f"{index.key}.txt"), cache.get)
    cache.write_sha_list()
    assert max(date for _, date in source.calls) < am.LOCK_START

    out, events = tmp_path / "report.md", tmp_path / "events.csv"
    args = ["--offline", "--archive-dir", str(archive), "--cache-dir", str(cache_dir), "--out", str(out)]
    am.main([*args, "--events-csv", str(events)])
    first = out.read_text(encoding="utf-8")
    am.main(args)
    assert out.read_text(encoding="utf-8") == first  # same cache, same report
    assert chr(0x2014) not in first and chr(0x2013) not in first  # no em or en dash in the report
    # Once in the table of changes dated to the week, once in the table of all changes.
    assert first.count("| CSI300 | all | both | 2 | 2 | 11 | 11 | 11 | 11 | 2 | 0 | 0 | 0 | 0 (0.0%) |") == 2
    assert "in none of the 4 paired changes does the archive show the change clearly EARLIER" in first
    # 206 month ends; the lists differ only at the end of June 2015... which the archive has caught up with by
    # the 30th, so every month end agrees.
    assert "| CSI500 | all | 206 | 0 | 103000 | **0.00%** | 0.0 | 206 |" in first
    with events.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["index"], row["code"], row["kind"], row["baostock_date"], row["archive_date"], row["lag"]) for row in rows} == {
        (key, code, kind, "2015-06-15", "2015-06-30", "11")
        for key in ("csi300", "csi500") for code, kind in (("SZ000001", "add"), ("SH600000", "drop"))
    }  # fmt: skip
    assert not [path for path in cache_dir.glob("*/*.csv") if path.stem >= am.LOCK_START]


def test_unpaired_facts_tell_an_archive_ahead_change_from_one_baostock_made_earlier():
    """By the report's own definition: the archive is ahead when every BaoStock list dated on or after its day still
    shows the old state (an add it lacks, a drop it still holds). Lists dated before that day and unusable ones
    say nothing."""
    def answer(date, update, *numbers):
        return am.Answer(date, len(numbers), codes(*numbers), (update,))

    answers = [
        answer("2021-01-29", "2021-01-25", 1, 2, 3, 4),
        answer("2021-02-26", "2021-02-22", 1, 2, 3, 4),
        answer("2021-03-31", "2021-03-29", 1, 2, 3, 4),
        answer("2021-04-30", "2021-04-26", 1, 2),  # obviously wrong size: unusable, never counted
    ]
    result = {"index": TINY, "answers": answers, "archive_only": [
        ("2021-02-01", am.ADD, "SH600005"),  # no later list holds it: the archive is ahead
        ("2021-02-01", am.DROP, "SH600001"),  # every later list still holds it: the archive is ahead
        ("2021-02-01", am.ADD, "SH600002"),  # every later list holds it: BaoStock had it already
        ("2021-03-30", am.DROP, "SH600003"),  # the 2021-03-29 list is dated before that day: no later list
    ]}  # fmt: skip
    facts = am.unpaired_facts(result)
    assert [(f["code"], f["lists"], f["holds"], f["ahead"]) for f in facts] == [
        ("SH600005", 2, 0, True), ("SH600001", 2, 2, True), ("SH600002", 2, 2, False), ("SH600003", 0, 0, False)]
    assert facts[0]["dated"] == ("2021-02-22", "2021-03-29") and facts[3]["dated"] is None
    assert am.later_lists(facts[0]) == "the 2 BaoStock lists dated 2021-02-22 to 2021-03-29"
    assert am.later_lists(facts[3]) == "no later BaoStock list"
    note = am.unpaired_summary([{"index": TINY}], {"tiny": facts})
    assert note.startswith("Archive changes with no BaoStock change to pair with are outside this count")
    assert "2 of 4 show the archive ahead of every later BaoStock list: TINY SH600005, added 2021-02-01, missing from the 2 BaoStock lists dated 2021-02-22 to 2021-03-29; TINY SH600001, dropped 2021-02-01, still in the 2 BaoStock lists dated 2021-02-22 to 2021-03-29." in note
    assert am.unpaired_summary([{"index": TINY}], {"tiny": []}) == "Every archive change in the window has a BaoStock change to pair with."


def test_held_from_start_needs_an_unbroken_run_from_the_first_day_within_the_window():
    assert am.held_from_start([(0, 500)], first_day=0, day=40, window=250)
    assert not am.held_from_start([(0, 30), (60, 500)], first_day=0, day=40, window=250)  # the run ended before
    assert not am.held_from_start([(5, 500)], first_day=0, day=40, window=250)  # not there on the first day
    assert not am.held_from_start([(0, 500)], first_day=0, day=300, window=250)  # too long ago to be the same change
    assert not am.held_from_start([], first_day=0, day=40, window=250)


# Totals for the report -------------------------------------------------------------------------------------------
def test_mismatch_totals_pool_the_usable_month_ends():
    months = [
        am.Month("2021-01-29", 4, 4, "ok", ("A",), ("B",), 5),
        am.Month("2021-02-26", 4, 4, "ok", (), (), 4),
        am.Month("2021-03-31", 4, 0, "empty"),
        am.Month("2021-04-30", 4, 1, "size"),
    ]
    assert am.mismatch_totals(months) == {"months": 2, "one": 2, "either": 9, "share": 2 / 9, "identical": 1, "per_month": 1.0}
    assert am.mismatch_totals(months[2:]) is None


def test_lag_totals_count_each_outcome_and_leave_unpinned_days_out_of_the_lags():
    def change(status, lag=None, timing=None, exact=True):
        return am.Change("SH600001", am.ADD, "2021-01-08", "2021-01-11", "2021-01-04", "2021-01-11", exact, 5, status, None, lag, timing)

    changes = [
        change(am.MATCHED, 10, am.LATE), change(am.MATCHED, 12, am.LATE), change(am.MATCHED, 0, am.SAME_DAY),
        change(am.MATCHED, -2, am.EARLY_UNCLEAR), change(am.MATCHED, -30, am.EARLY_CLEAR),
        change(am.MATCHED, 99, am.LATE, exact=False),  # BaoStock's day is only known as a range
        change(am.NEVER), change(am.NEVER), change(am.TOO_RECENT), change(am.FROM_START),
    ]  # fmt: skip
    totals = am.lag_totals(changes)
    assert (totals["changes"], totals["paired"], totals["timed"]) == (10, 6, 5)
    assert totals["stats"] == {"n": 5, "min": -30, "median": 0, "p90": 12, "max": 12}
    assert totals["timing"] == {am.LATE: 2, am.SAME_DAY: 1, am.EARLY_UNCLEAR: 1, am.EARLY_CLEAR: 1}
    assert (totals["never"], totals["never_share"], totals["too_recent"], totals["from_start"]) == (2, 2 / 8, 1, 1)
    empty = am.lag_totals([])
    assert empty["stats"] is None and empty["never_share"] is None


def test_calendar_patterns_of_review_days():
    cal = am.Calendar([d for d in weekdays("2019-01-01", "2021-12-31") if d not in ("2019-01-01", "2021-06-14")])
    # June 2019: the second Friday is the 14th, so the review day is Monday the 17th.
    assert am.after_second_friday("2019-06-17", cal) and not am.after_second_friday("2019-06-18", cal)
    assert not am.after_second_friday("2019-06-10", cal) and not am.after_second_friday("2019-07-15", cal)
    # June 2021: the Monday after the second Friday (the 14th) is a holiday, so it is Tuesday the 15th.
    assert am.after_second_friday("2021-06-15", cal)
    # December 2020: the 1st is a Tuesday, the second Friday is the 11th.
    assert am.after_second_friday("2020-12-14", cal)
    assert am.first_of_half_year("2019-01-02", cal) and am.first_of_half_year("2019-07-01", cal)
    assert not am.first_of_half_year("2019-01-03", cal) and not am.first_of_half_year("2019-06-03", cal)


def test_member_counts_and_overdue_stock_days(market):
    cal, spans, bao, ends = market
    runs = am.to_runs({"A": [("2020-01-01", "2020-01-03")], "B": [("2020-01-02", "2020-01-08")]}, cal)
    assert am.member_counts(runs, 6) == [1, 2, 2, 1, 1, 1]  # weekdays from Wednesday 2020-01-01
    result = am.audit_index(TINY, cal, ends, spans, bao.get, window=250)
    assert set(result["archive_sizes"]) == {4}
    # Only one drop is late: SH600001, held 10 trading days too long. The early and never-shown ones add nothing.
    got = am.overdue(result, cal, "2020-01-01")
    assert got == {"held": 10, "member_days": 4 * len(cal.days), "share": 10 / (4 * len(cal.days))}
    assert am.overdue(result, cal, "2021-04-01")["held"] == 0


def test_a_week_in_which_the_market_is_closed_is_not_a_skipped_update():
    """BaoStock stores no list in a week without trading. A change right after such a week is still dated to the
    week: 5 trading days lie between its last old list and its first new one."""
    closed = set(weekdays("2021-02-15", "2021-02-19"))
    cal = am.Calendar([d for d in weekdays("2020-01-01", "2022-12-30") if d not in closed])
    end = cal.days[-1]
    spans = {f"SH60000{n}": [("2020-01-01", end)] for n in (1, 2, 3)}
    spans |= {"SH600004": [("2020-01-01", "2021-02-28")], "SH600005": [("2021-03-01", end)]}
    bao = FakeBaoStock(cal, {"2021-02-22": (codes(5), codes(4))}, start=codes(1, 2, 3, 4), frozen=("2021-02-15", "2021-02-22"))
    result = am.audit_index(TINY, cal, am.month_ends(cal.days, "2020-01", "2022-12"), spans, bao.get, window=250)
    add = next(c for c in result["changes"] if c.kind == am.ADD)
    assert (add.prev_date, add.date, add.last_old_update, add.update) == ("2021-02-12", "2021-02-22", "2021-02-08", "2021-02-22")
    assert (add.gap, add.week, add.lag) == (5, True, 5)


def test_review_days_are_told_apart_by_where_they_fall_in_the_calendar():
    cal = am.Calendar(weekdays("2019-01-01", "2020-12-31"))

    def change(number, date, last_old, archive_date, lag):
        kind = am.ADD if number % 2 else am.DROP
        return am.Change(f"SH6{number:05d}", kind, "-", date, last_old, date, True, 5, am.MATCHED, archive_date, lag, am.timing(lag, archive_date, last_old))

    # A review that BaoStock dates to the Monday after the second Friday of June; the archive is 9 days late.
    june = [change(n, "2019-06-17", "2019-06-10", "2019-06-28", 9) for n in range(12)]
    # A review that the archive dates to the first trading day of January; BaoStock's Monday list is 3 days behind.
    january = [change(100 + n, "2020-01-06", "2019-12-30", "2020-01-01", -3) for n in range(10)]
    # One stock swapped between reviews, shown by the archive 86 trading days later.
    single = [change(200 + n, "2019-09-02", "2019-08-26", "2019-12-31", 86) for n in range(2)]
    result = {"index": TINY, "changes": june + january + single}

    found = am.pattern_stats(result, cal)
    assert list(found) == list(am.PATTERNS[:2])
    assert (found[am.PATTERNS[0]]["days"], found[am.PATTERNS[0]]["stats"]["n"], found[am.PATTERNS[0]]["stats"]["max"]) == (["2019-06-17"], 12, 9)
    assert (found[am.PATTERNS[1]]["days"], found[am.PATTERNS[1]]["stats"]["min"]) == (["2020-01-06"], -3)
    assert am.other_day_lags(result) == {"n": 2, "min": 86, "median": 86, "p90": 86, "max": 86}
    summary = am.delay_summary([result], cal)
    assert "at the 1 twice-yearly review since 2019-06-17 that BaoStock dates to the week, the archive is 9 to 9 trading days late" in summary
    assert "86 (TINY)" in summary
    assert am.delay_summary([{"index": TINY, "changes": single}], cal) is None
    assert (am.count(1, "add"), am.count(0, "add"), am.count(3, "drop")) == ("1 add", "0 adds", "3 drops")
