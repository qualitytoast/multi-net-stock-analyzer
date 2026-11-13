#!/usr/bin/env python3
"""Membership timing audit (docs/RULES.md, Rule MA).

Compares the pinned archive's CSI300 and CSI500 member lists with BaoStock's, and measures how late (or early)
the archive shows index changes. A report only: it changes no data and no decision depends on it.

Download what is not cached yet (anonymous BaoStock login), then write the report:

    python3 -m uv run --with baostock python scripts/audit_membership.py

Rebuild the same report from the cache alone, with no network:

    python3 -m uv run python scripts/audit_membership.py --offline --cache-dir ~/data/baostock_membership/<date>

Every date that is queried or counted is before 2024-03-01, the first day of the locked test period.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import csv
import datetime as dt
import hashlib
import math
import os
import re
import signal
import socket
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

FIRST_MONTH = "2007-01"
LAST_MONTH = "2024-02"
LOCK_START = "2024-03-01"  # first day of the locked test period: no date from here on is queried or counted
ARCHIVE_DIR = "~/.qlib/qlib_data/cn_data_2026-11-04"
ARCHIVE_TAG = "2026-11-04"
CACHE_ROOT = "~/data/baostock_membership"
REPORT = Path(__file__).resolve().parents[1] / "results" / "csi300" / "membership_audit.md"

SIZE_TOLERANCE = 0.02  # a BaoStock list more than 2% away from the index's size is "obviously wrong"
FRESH_DAYS = 6  # a list stored every Monday is at most 6 days old; an older one means BaoStock skipped updates
WEEK = 5  # trading days from one Monday list to the next when no update is skipped
MATCH_WINDOW = 250  # trading days (about one year): changes further apart are not "the same change"
FIELDS = ["updateDate", "code", "code_name"]
SHA_LIST = "SHA256SUMS"
REVIEW_MIN = 10  # a BaoStock change day that moves at least this many stocks is called a review day
WORST_MONTHS = 10
EXAMPLES = 40

ADD, DROP = "add", "drop"
LATE, SAME_DAY, EARLY_UNCLEAR, EARLY_CLEAR = "late", "same day", "early, between two BaoStock lists", "early, clear"
MATCHED, NEVER, TOO_RECENT, FROM_START = "matched", "never shown", "too close to the cut-off", "in the archive from its first day"


@dataclass(frozen=True)
class Index:
    key: str  # the archive's name for it, and the cache folder
    name: str
    query: str  # the BaoStock function
    size: int


INDICES = (
    Index("csi300", "CSI300", "query_hs300_stocks", 300),
    Index("csi500", "CSI500", "query_zz500_stocks", 500),
)


# Dates ---------------------------------------------------------------------------------------------------------
def read_calendar(path: Path) -> list[str]:
    days = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if days != sorted(set(days)):
        raise ValueError(f"{path}: trading days are not sorted and unique")
    return days


def month_ends(days: Iterable[str], first_month: str, last_month: str) -> list[str]:
    """The last trading day of every month from `first_month` to `last_month` (both "YYYY-MM", inclusive)."""
    last: dict[str, str] = {}
    for day in days:
        month = day[:7]
        if first_month <= month <= last_month:
            last[month] = max(day, last.get(month, day))
    year, month = (int(part) for part in first_month.split("-"))
    wanted = []
    while f"{year:04d}-{month:02d}" <= last_month:
        wanted.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    missing = [m for m in wanted if m not in last]
    if missing:
        raise ValueError(f"no trading day in the calendar for: {', '.join(missing)}")
    return [last[m] for m in wanted]


def days_between(first: str, last: str) -> int:
    return (dt.date.fromisoformat(last) - dt.date.fromisoformat(first)).days


class Calendar:
    """Trading days as positions, so that a lag is a difference of two positions."""

    def __init__(self, days: list[str]):
        self.days = days
        self.pos = {day: i for i, day in enumerate(days)}

    def first_on_or_after(self, date: str) -> int:
        return bisect.bisect_left(self.days, date)  # len(days) if there is none

    def last_on_or_before(self, date: str) -> int:
        return bisect.bisect_right(self.days, date) - 1  # -1 if there is none


# The archive's membership ----------------------------------------------------------------------------------------
def read_spans(path: Path) -> dict[str, list[tuple[str, str]]]:
    """The archive's membership file: one line per stock and span (code, first date, last date)."""
    spans: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        code, start, end = line.split("\t")
        if start > end:
            raise ValueError(f"{path}: span ends before it starts: {line!r}")
        spans[code].append((start, end))
    return dict(spans)


def to_runs(spans: dict[str, list[tuple[str, str]]], cal: Calendar) -> dict[str, list[tuple[int, int]]]:
    """Per stock, its unbroken runs of membership as (first, last) trading-day positions in `cal`.

    Spans that touch or overlap are joined, and whatever lies outside the calendar is cut off, so a calendar
    that stops before the locked test period keeps every later date out of the audit."""
    runs: dict[str, list[tuple[int, int]]] = {}
    for code, stock_spans in spans.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(stock_spans):
            first, last = cal.first_on_or_after(start), cal.last_on_or_before(end)
            if first > last:
                continue  # no trading day inside the span
            if merged and first <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], last))
            else:
                merged.append((first, last))
        if merged:
            runs[code] = merged
    return runs


def member_counts(runs: dict[str, list[tuple[int, int]]], days: int) -> list[int]:
    """How many stocks the archive lists on each trading day."""
    change = [0] * (days + 1)
    for stock_runs in runs.values():
        for first, last in stock_runs:
            change[first] += 1
            change[last + 1] -= 1
    counts, total = [], 0
    for step in change[:days]:
        total += step
        counts.append(total)
    return counts


def members_on(runs: dict[str, list[tuple[int, int]]], day: int) -> frozenset[str]:
    return frozenset(code for code, stock_runs in runs.items() if any(a <= day <= b for a, b in stock_runs))


def run_events(runs: dict[str, list[tuple[int, int]]], first_day: int, last_day: int) -> dict[tuple[str, str], list[int]]:
    """(stock, add or drop) -> the days on which the archive first shows that change.

    An add is a run's first day, a drop is the day after a run's last day. A run that starts on the archive's
    first day or is still open on the last day shows no change there."""
    events: dict[tuple[str, str], list[int]] = defaultdict(list)
    for code, stock_runs in runs.items():
        for first, last in stock_runs:
            if first > first_day:
                events[code, ADD].append(first)
            if last < last_day:
                events[code, DROP].append(last + 1)
    return dict(events)


# Pure comparisons ------------------------------------------------------------------------------------------------
def norm_code(code: str) -> str:
    """BaoStock's "sh.600000" -> the archive's "SH600000"."""
    match = re.fullmatch(r"(sh|sz|bj)\.(\d{6})", code)
    if not match:
        raise ValueError(f"unexpected BaoStock stock code: {code!r}")
    return match.group(1).upper() + match.group(2)


def mismatch_share(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Stocks in exactly one of the two lists / stocks in either list (Rule MA)."""
    either = len(a | b)
    if either == 0:
        raise ValueError("both lists are empty")
    return len(a ^ b) / either


@dataclass(frozen=True)
class Step:
    """BaoStock's list differs between two trading days with no usable answer in between."""

    prev_day: int  # last day with the old list
    day: int  # first day with the new list
    added: frozenset[str]
    dropped: frozenset[str]


def find_changes(lo: int, hi: int, members: Callable[[int], frozenset[str] | None]) -> list[Step]:
    """Where BaoStock's list changes between trading days `lo` and `hi`, found by halving the interval.

    `members(day)` is the list on a day, or None when that day's answer is unusable; `lo` and `hi` must be
    usable. Halves whose two ends carry the same list are not searched further."""
    steps: list[Step] = []
    todo = [(lo, hi)]
    while todo:
        a, b = todo.pop()
        old, new = members(a), members(b)
        if old is None or new is None:
            raise ValueError("find_changes needs a usable list at both ends")
        if old == new:
            continue
        middle = (a + b) // 2
        inside = sorted(range(a + 1, b), key=lambda day: (abs(day - middle), day))
        split = next((day for day in inside if members(day) is not None), None)
        if split is None:
            steps.append(Step(a, b, frozenset(new - old), frozenset(old - new)))
        else:
            todo += [(a, split), (split, b)]
    return sorted(steps, key=lambda step: step.day)


def net_changes(steps: list[Step], old: frozenset[str], new: frozenset[str]) -> tuple[list[tuple[str, str, Step]], int]:
    """The changes that are still there at the interval's end: (stock, add or drop, step), plus the number of
    stock moves that were undone again inside the interval. A stock that moves more than once is dated by its
    last move."""
    last: dict[tuple[str, str], Step] = {}
    moves = 0
    for step in steps:  # in date order, so a later move replaces an earlier one
        for kind, codes in ((ADD, step.added), (DROP, step.dropped)):
            for code in codes:
                last[code, kind] = step
                moves += 1
    kept = [(code, ADD, last[code, ADD]) for code in sorted(new - old)]
    kept += [(code, DROP, last[code, DROP]) for code in sorted(old - new)]
    return kept, moves - len(kept)


def match_changes(ours: list[int], theirs: list[int], window: int) -> list[int | None]:
    """Pair each day in `ours` (BaoStock) with a day in `theirs` (the archive), both sorted.

    The order is kept, each archive day is used at most once and a pair is at most `window` trading days
    apart. Among all such pairings: as many pairs as possible, then the smallest total distance. Returns the
    archive day, or None, for each of ours."""
    n, m = len(ours), len(theirs)
    # best[i][j] = (unpaired of ours, total distance) for ours[i:] against theirs[j:]
    best: list[list[tuple[int, int]]] = [[(n - i, 0)] * (m + 1) for i in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            options = [(best[i + 1][j][0] + 1, best[i + 1][j][1]), best[i][j + 1]]
            distance = abs(theirs[j] - ours[i])
            if distance <= window:
                options.append((best[i + 1][j + 1][0], best[i + 1][j + 1][1] + distance))
            best[i][j] = min(options)
    out: list[int | None] = []
    i = j = 0
    while i < n:
        distance = abs(theirs[j] - ours[i]) if j < m else None
        if distance is not None and distance <= window and best[i][j] == (best[i + 1][j + 1][0], best[i + 1][j + 1][1] + distance):
            out.append(theirs[j])
            i, j = i + 1, j + 1
        elif j < m and best[i][j] == best[i][j + 1]:
            j += 1
        else:
            out.append(None)
            i += 1
    return out


def held_from_start(stock_runs: list[tuple[int, int]], first_day: int, day: int, window: int) -> bool:
    """The archive lists the stock without a break from its own first day up to `day`, at most `window` days on.
    Its add day is then unknown (on or before the archive's first day), so no lag can be given."""
    return bool(stock_runs) and stock_runs[0][0] == first_day and stock_runs[0][1] >= day and day - first_day <= window


def timing(lag: int, archive_date: str, last_old_update: str | None) -> str:
    """Late, same day or early. An early change is "clear" only if the archive showed it on or before the date
    of a BaoStock list that still lacked it; otherwise BaoStock simply had no newer list to show it in."""
    if lag > 0:
        return LATE
    if lag == 0:
        return SAME_DAY
    return EARLY_CLEAR if last_old_update is not None and archive_date <= last_old_update else EARLY_UNCLEAR


def percentile(values: list[int], q: float) -> int:
    """Nearest rank: the smallest value with at least q% of the values at or below it."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q / 100 * len(ordered))) - 1]


def lag_stats(lags: list[int]) -> dict[str, float | int] | None:
    if not lags:
        return None
    return {"n": len(lags), "min": min(lags), "median": statistics.median(lags), "p90": percentile(lags, 90), "max": max(lags)}


# BaoStock answers and their cache --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Answer:
    date: str
    rows: int
    codes: frozenset[str]
    update_dates: tuple[str, ...]  # distinct, sorted; BaoStock stamps every row with one

    @property
    def update_date(self) -> str | None:
        return self.update_dates[-1] if self.update_dates else None

    @property
    def age(self) -> int | None:
        """Calendar days from the list's update date to the day it was asked for."""
        return days_between(self.update_date, self.date) if self.update_dates else None


def usable(answer: Answer, index: Index) -> bool:
    return answer.rows > 0 and abs(len(answer.codes) - index.size) <= SIZE_TOLERANCE * index.size


def to_answer(date: str, rows: list[list[str]]) -> Answer:
    return Answer(date, len(rows), frozenset(norm_code(row[1]) for row in rows), tuple(sorted({row[0] for row in rows})))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Unreachable(RuntimeError):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    """Stop a request that hangs (the client library loops forever on a closed connection)."""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expired(signum, frame):
        raise TimeoutError(f"no answer within {seconds} s")

    before = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)


class BaoStock:
    """One anonymous BaoStock session: a pause before every request, retries with growing waits."""

    def __init__(self, pause: float = 0.3, tries: int = 6, timeout: int = 60):
        import baostock  # only needed online; `--with baostock` provides it without a project dependency

        self.bs = baostock
        self.version = getattr(baostock, "__version__", "unknown")
        self.pause, self.tries, self.timeout = pause, tries, timeout
        self.logged_in = False
        self.requests = 0
        socket.setdefaulttimeout(timeout)

    def _once(self, index: Index, date: str) -> list[list[str]]:
        with time_limit(self.timeout):
            if not self.logged_in:
                login = self.bs.login()  # the library's default: user "anonymous"
                if login.error_code != "0":
                    raise RuntimeError(f"login failed: {login.error_code} {login.error_msg}")
                self.logged_in = True
            time.sleep(self.pause)
            self.requests += 1
            result = getattr(self.bs, index.query)(date=date)
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0":
                raise RuntimeError(f"{index.query}({date}) failed: {result.error_code} {result.error_msg}")
            if rows and result.fields != FIELDS:
                raise RuntimeError(f"{index.query}({date}) returned unexpected columns: {result.fields}")
            return rows

    def fetch(self, index: Index, date: str) -> tuple[list[list[str]], int]:
        """The rows for one index and date, and how many tries it took."""
        problem: Exception | None = None
        for attempt in range(1, self.tries + 1):
            try:
                return self._once(index, date), attempt
            except Exception as error:  # noqa: BLE001 - any failure is retried, then reported
                problem = error
                self.close()
                if attempt < self.tries:
                    wait = min(60, 2**attempt)
                    print(f"  try {attempt} of {self.tries} failed ({error}); waiting {wait} s", file=sys.stderr)
                    time.sleep(wait)
        raise Unreachable(f"BaoStock gave no answer for {index.query}({date}) in {self.tries} tries; last error: {problem}")

    def close(self) -> None:
        if self.logged_in:
            with contextlib.suppress(Exception), time_limit(self.timeout):
                self.bs.logout()
        self.logged_in = False


class Cache:
    """Every BaoStock answer as a CSV file: <dir>/<index>/<date>.csv. With no source it is read-only (offline)."""

    def __init__(self, directory: Path, source: BaoStock | None):
        self.dir, self.source = directory, source
        self.seen: dict[tuple[str, str], Answer] = {}
        self.downloaded = 0
        listed = self.read_sha_list()
        on_disk = self.files()
        for name, digest in listed.items():
            if not (self.dir / name).is_file() or sha256(self.dir / name) != digest:
                raise SystemExit(f"cache check failed: {self.dir / name} is missing or does not match {SHA_LIST}")
        if source is None:
            unlisted = sorted(set(on_disk) - set(listed))
            if not listed or unlisted:
                raise SystemExit(f"cache check failed: {self.dir} has no {SHA_LIST} or holds files it does not list ({unlisted[:3]})")

    def files(self) -> list[str]:
        return sorted(str(path.relative_to(self.dir)) for path in self.dir.glob("*/*.csv"))

    def read_sha_list(self) -> dict[str, str]:
        path = self.dir / SHA_LIST
        if not path.is_file():
            return {}
        return {name: digest for digest, name in (line.split("  ", 1) for line in path.read_text().splitlines() if line)}

    def write_sha_list(self) -> None:
        text = "".join(f"{sha256(self.dir / name)}  {name}\n" for name in self.files())
        (self.dir / SHA_LIST).write_text(text)

    def get(self, index: Index, date: str) -> Answer:
        if date >= LOCK_START:
            raise ValueError(f"{date} is in the locked test period")
        if (index.key, date) in self.seen:
            return self.seen[index.key, date]
        path = self.dir / index.key / f"{date}.csv"
        if not path.is_file():
            if self.source is None:
                raise SystemExit(f"{path} is not in the cache; run again without --offline to download it")
            self.download(index, date, path)
        with path.open(newline="", encoding="utf-8") as handle:
            header, *rows = list(csv.reader(handle))
        if header != FIELDS:
            raise ValueError(f"{path}: unexpected header {header}")
        self.seen[index.key, date] = to_answer(date, rows)
        return self.seen[index.key, date]

    def download(self, index: Index, date: str, path: Path) -> None:
        rows, tries = self.source.fetch(index, date)
        if not rows:  # an empty answer is checked once more before it is believed
            rows, more = self.source.fetch(index, date)
            tries += more
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with part.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(FIELDS)
            writer.writerows(rows)
        os.replace(part, path)
        log = self.dir / f"{index.key}_download_log.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if new:
                writer.writerow(["date", "downloaded_at_utc", "rows", "tries", "client"])
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            writer.writerow([date, now, len(rows), tries, f"baostock {self.source.version}"])
        self.downloaded += 1
        if self.downloaded % 25 == 0:
            print(f"  {self.downloaded} answers downloaded (latest: {index.key} {date})", file=sys.stderr)

    def log_rows(self, index: Index) -> list[dict[str, str]]:
        log = self.dir / f"{index.key}_download_log.csv"
        if not log.is_file():
            return []
        with log.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


# The audit of one index ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Month:
    date: str
    archive: int  # stocks in the archive's list
    baostock: int  # stocks in BaoStock's list
    status: str  # "ok", "empty" (BaoStock returned nothing) or "size" (obviously wrong size)
    only_archive: tuple[str, ...] = ()
    only_baostock: tuple[str, ...] = ()
    either: int = 0
    update: str | None = None  # the update date of BaoStock's list
    age: int | None = None  # calendar days from that update date to the month end

    @property
    def one_list_only(self) -> int:
        return len(self.only_archive) + len(self.only_baostock)

    @property
    def share(self) -> float:
        return self.one_list_only / self.either


@dataclass(frozen=True)
class Change:
    """One stock that BaoStock adds or drops, and when the archive shows the same change."""

    code: str
    kind: str
    prev_date: str  # last trading day on which BaoStock still served the old list
    date: str  # first trading day on which BaoStock serves the new list
    last_old_update: str | None  # update date of that old list
    update: str | None  # update date of the new list
    exact: bool  # the two days are neighbours (nothing unusable in between)
    gap: int | None  # trading days from the date of the last old list to the date of the first new list
    status: str  # MATCHED, NEVER, TOO_RECENT or FROM_START
    archive_date: str | None = None
    lag: int | None = None  # trading days; above 0 the archive is late, below 0 it is early
    timing: str | None = None

    @property
    def week(self) -> bool:
        """BaoStock dates the change to the week: its last old list and its first new list are at most one
        trading week apart (a week in which the market is closed does not count as a skipped update)."""
        return self.exact and self.gap is not None and self.gap <= WEEK


def audit_index(index: Index, cal: Calendar, ends: list[str], spans: dict[str, list[tuple[str, str]]],
                get: Callable[[Index, str], Answer], window: int = MATCH_WINDOW) -> dict:  # fmt: skip
    """Everything the report says about one index. `cal` must stop before the locked test period."""
    runs = to_runs(spans, cal)
    first_day = min(first for stock_runs in runs.values() for first, _ in stock_runs)
    last_day = len(cal.days) - 1
    asked: dict[str, Answer] = {}

    def ask(date: str) -> Answer:
        if date not in asked:
            asked[date] = get(index, date)
        return asked[date]

    months = []
    for date in ends:
        ours, theirs = members_on(runs, cal.pos[date]), ask(date)
        if not usable(theirs, index):
            months.append(Month(date, len(ours), len(theirs.codes), "empty" if theirs.rows == 0 else "size"))
        elif not ours:
            months.append(Month(date, 0, len(theirs.codes), "no archive list"))
        else:
            only_ours, only_theirs = sorted(ours - theirs.codes), sorted(theirs.codes - ours)
            months.append(Month(date, len(ours), len(theirs.codes), "ok", tuple(only_ours), tuple(only_theirs),
                                len(ours | theirs.codes), theirs.update_date, theirs.age))  # fmt: skip

    def members(day: int) -> frozenset[str] | None:
        answer = ask(cal.days[day])
        return answer.codes if usable(answer, index) else None

    found: list[tuple[str, str, Step]] = []
    undone = 0
    good_ends = [cal.pos[m.date] for m in months if m.status in ("ok", "no archive list")]
    steps: list[Step] = []
    for lo, hi in zip(good_ends, good_ends[1:]):
        between = find_changes(lo, hi, members)
        kept, extra = net_changes(between, members(lo), members(hi))
        steps += between
        found += kept
        undone += extra

    events = run_events(runs, first_day, last_day)
    by_stock: dict[tuple[str, str], list[Step]] = defaultdict(list)
    for code, kind, step in found:
        by_stock[code, kind].append(step)
    changes: list[Change] = []
    used: set[tuple[str, str, int]] = set()
    for (code, kind), stock_steps in by_stock.items():
        stock_steps.sort(key=lambda step: step.day)
        shown = match_changes([step.day for step in stock_steps], events.get((code, kind), []), window)
        for step, archive_day in zip(stock_steps, shown):
            old_stamp, new_stamp = ask(cal.days[step.prev_day]).update_date, ask(cal.days[step.day]).update_date
            base = dict(
                code=code, kind=kind, prev_date=cal.days[step.prev_day], date=cal.days[step.day],
                last_old_update=old_stamp, update=new_stamp, exact=step.day == step.prev_day + 1,
                gap=cal.first_on_or_after(new_stamp) - cal.first_on_or_after(old_stamp) if old_stamp and new_stamp else None,
            )  # fmt: skip
            if archive_day is not None:
                used.add((code, kind, archive_day))
                lag = archive_day - step.day
                changes.append(Change(**base, status=MATCHED, archive_date=cal.days[archive_day], lag=lag,
                                      timing=timing(lag, cal.days[archive_day], base["last_old_update"])))  # fmt: skip
            elif kind == ADD and held_from_start(runs.get(code, []), first_day, step.day, window):
                changes.append(Change(**base, status=FROM_START, archive_date=cal.days[first_day]))
            elif last_day - step.day < window:
                changes.append(Change(**base, status=TOO_RECENT))
            else:
                changes.append(Change(**base, status=NEVER))
    changes.sort(key=lambda c: (c.date, c.kind, c.code))

    # The other direction: changes the archive shows after the first month end that BaoStock never shows.
    start = cal.pos[ends[0]]
    archive_only = sorted(
        (cal.days[day], kind, code)
        for (code, kind), days in events.items() for day in days
        if day > start and (code, kind, day) not in used
    )  # fmt: skip

    return {
        "index": index, "months": months, "changes": changes, "undone": undone, "archive_only": archive_only,
        "steps": steps, "archive_first": cal.days[first_day], "archive_dates": snapshot_dates(spans, ends[0], cal.days[-1]),
        "answers": [asked[date] for date in sorted(asked)], "window": window, "last_day": cal.days[last_day],
        "archive_sizes": member_counts(runs, len(cal.days)),
    }  # fmt: skip


def snapshot_dates(spans: dict[str, list[tuple[str, str]]], first: str, last: str) -> list[str]:
    """The distinct days from `first` to `last` on which any span of the archive starts: the only days on which
    its list can change."""
    return sorted({start for stock_spans in spans.values() for start, _ in stock_spans if first <= start <= last})


# Numbers for the report ------------------------------------------------------------------------------------------
def mismatch_totals(months: list[Month]) -> dict | None:
    """Pooled over month ends: all stocks in exactly one list / all stocks in either list."""
    ok = [m for m in months if m.status == "ok"]
    if not ok:
        return None
    one, either = sum(m.one_list_only for m in ok), sum(m.either for m in ok)
    return {
        "months": len(ok), "one": one, "either": either, "share": one / either,
        "identical": sum(m.one_list_only == 0 for m in ok), "per_month": one / len(ok),
    }  # fmt: skip


def lag_totals(changes: list[Change]) -> dict:
    """Counts and lag statistics for a set of BaoStock changes. Lags count only changes whose BaoStock day is
    pinned to a single trading day."""
    paired = [c for c in changes if c.status == MATCHED]
    timed = [c for c in paired if c.exact]
    never = sum(c.status == NEVER for c in changes)
    return {
        "changes": len(changes), "paired": len(paired), "timed": len(timed), "stats": lag_stats([c.lag for c in timed]),
        "timing": Counter(c.timing for c in timed), "never": never,
        "never_share": never / (len(paired) + never) if paired or never else None,
        "too_recent": sum(c.status == TOO_RECENT for c in changes), "from_start": sum(c.status == FROM_START for c in changes),
    }  # fmt: skip


def overdue(result: dict, cal: Calendar, first: str) -> dict:
    """From `first` on: the stock-days on which the archive still lists a stock that BaoStock had already
    dropped (the positive lags of the paired drops), against all of the archive's member-days."""
    held = sum(c.lag for c in result["changes"] if c.kind == DROP and c.status == MATCHED and c.exact and c.lag > 0 and c.date >= first)
    member_days = sum(result["archive_sizes"][cal.first_on_or_after(first):])
    return {"held": held, "member_days": member_days, "share": held / member_days}


def update_date_facts(result: dict, cal: Calendar) -> dict:
    """How BaoStock stamps its lists, measured on every answer the audit used."""
    answers = [a for a in result["answers"] if a.rows]
    lists_by_stamp: dict[str, set[frozenset[str]]] = defaultdict(set)
    for answer in answers:
        if len(answer.update_dates) == 1:
            lists_by_stamp[answer.update_date].add(answer.codes)
    by_date = {a.date: a for a in result["answers"]}
    exact = [step for step in result["steps"] if step.day == step.prev_day + 1]
    # A new list "starts on its update date" if that date is after the last day with the old list and not
    # after the first day with the new one (it can be a non-trading day in between).
    on_stamp = sum(cal.days[step.prev_day] < (by_date[cal.days[step.day]].update_date or "") <= cal.days[step.day] for step in exact)
    stamps = sorted(lists_by_stamp)
    oldest = max(answers, key=lambda a: (a.age, a.date))
    return {
        "answers": len(answers), "one_stamp": sum(len(a.update_dates) == 1 for a in answers),
        "stamp_not_after": sum(a.update_date <= a.date for a in answers),
        "fresh": sum(a.age <= FRESH_DAYS for a in answers), "oldest": oldest,
        "stamps": len(stamps), "weekdays": Counter(dt.date.fromisoformat(stamp).strftime("%A") for stamp in stamps),
        "one_list_per_stamp": sum(len(lists) == 1 for lists in lists_by_stamp.values()),
        "steps": len(result["steps"]), "exact_steps": len(exact), "on_stamp": on_stamp,
    }  # fmt: skip


def archive_gap(result: dict, cal: Calendar) -> dict:
    """The archive's list can only change on the days its spans start: how far apart those days are."""
    days = [cal.pos[d] for d in result["archive_dates"] if d in cal.pos]
    gaps = [b - a for a, b in zip(days, days[1:])]
    return {"dates": len(days), "median": statistics.median(gaps) if gaps else None, "max": max(gaps, default=None)}


# The report ------------------------------------------------------------------------------------------------------
def unpaired_facts(result: dict) -> list[dict]:
    """For each archive change that BaoStock never pairs: what BaoStock's lists dated on or after the archive's
    day (among those this audit asked for) say about that stock. A list is one update date, however many answers
    carry it. "ahead" means every such list still shows the old state, so the archive shows the change before
    BaoStock does, the pattern of a clearly early change."""
    index = result["index"]
    facts = []
    for day, kind, code in result["archive_only"]:
        later: dict[str, bool] = {}
        for a in result["answers"]:
            if usable(a, index) and a.update_date is not None and a.update_date >= day:
                later[a.update_date] = later.get(a.update_date, False) or code in a.codes
        holds = sum(later.values())
        dated = sorted(later)
        facts.append({
            "date": day, "kind": kind, "code": code, "lists": len(later), "holds": holds,
            "dated": (dated[0], dated[-1]) if dated else None,
            "ahead": bool(later) and holds == (0 if kind == ADD else len(later)),
        })  # fmt: skip
    return facts


def later_lists(fact: dict) -> str:
    if fact["dated"] is None:
        return "no later BaoStock list"
    first, last = fact["dated"]
    return f"the BaoStock list dated {first}" if first == last else f"the {fact['lists']} BaoStock lists dated {first} to {last}"


def unpaired_summary(results: list[dict], unpaired: dict[str, list[dict]]) -> str:
    """One sentence on the archive changes BaoStock never pairs, for "In short" and section 4."""
    total = sum(len(facts) for facts in unpaired.values())
    if not total:
        return "Every archive change in the window has a BaoStock change to pair with."
    ahead = [(r["index"].name, f) for r in results for f in unpaired[r["index"].key] if f["ahead"]]
    named = "; ".join(
        f"{name} {f['code']}, {'added' if f['kind'] == ADD else 'dropped'} {f['date']}, "
        f"{'missing from' if f['kind'] == ADD else 'still in'} {later_lists(f)}" for name, f in ahead)  # fmt: skip
    return (f"Archive changes with no BaoStock change to pair with are outside this count (section 3, \"The other direction\"): "
            f"{len(ahead)} of {total} {'shows' if len(ahead) == 1 else 'show'} the archive ahead of every later BaoStock list"
            + (f": {named}." if ahead else "."))  # fmt: skip


MODEL_START = "2012-01-01"  # the first date the model trains on (configs/csi300.yaml)


def pct(share: float | None, digits: int = 2) -> str:
    return "n/a" if share is None else f"{100 * share:.{digits}f}%"


def num(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def group_changes(changes: list[Change]) -> list[list]:
    """Changes that share BaoStock day, archive day and outcome, as one table row with example stocks."""
    groups: dict[tuple, list[Change]] = defaultdict(list)
    for change in changes:
        key = (change.date, change.last_old_update or "n/a", change.archive_date or "never", change.lag, change.timing or change.status)
        groups[key].append(change)
    rows = []
    for (date, last_old, archive_date, lag, outcome), members in sorted(groups.items(), key=lambda item: [str(part) for part in item[0]]):
        adds = sorted(c.code for c in members if c.kind == ADD)
        drops = sorted(c.code for c in members if c.kind == DROP)
        examples = ", ".join((adds + drops)[:4]) + (" ..." if len(members) > 4 else "")
        rows.append([date, last_old, archive_date, num(lag), outcome, len(adds), len(drops), examples])
    return rows


CHANGE_HEADERS = [
    "BaoStock's new list starts on", "BaoStock's last old list is dated", "Archive shows it on", "Lag (trading days)",
    "Outcome", "Adds", "Drops", "Stocks",
]  # fmt: skip
LAG_HEADERS = [
    "BaoStock changes", "With a lag", "Median lag", "90th percentile", "Longest", "Shortest", "Late", "Same day",
    "Early, between two BaoStock lists", "Early, clear", "Never shown (share)",
]  # fmt: skip


def lag_row(label: list[str], totals: dict) -> list:
    stats, timing_counts = totals["stats"] or {}, totals["timing"]
    return [
        *label, totals["changes"], totals["timed"], num(stats.get("median")), num(stats.get("p90")), num(stats.get("max")),
        num(stats.get("min")), timing_counts[LATE], timing_counts[SAME_DAY], timing_counts[EARLY_UNCLEAR], timing_counts[EARLY_CLEAR],
        f"{totals['never']} ({pct(totals['never_share'], 1)})",
    ]  # fmt: skip


def lag_table(results: list[dict], keep: Callable[[Change], bool]) -> list[list]:
    rows = []
    for r in results:
        for label, first in (("all", ""), (f"{MODEL_START[:7]} on", MODEL_START)):
            for kind_label, kinds in (("adds", (ADD,)), ("drops", (DROP,)), ("both", (ADD, DROP))):
                picked = [c for c in r["changes"] if c.date >= first and c.kind in kinds and keep(c)]
                rows.append(lag_row([r["index"].name, label, kind_label], lag_totals(picked)))
    return rows


def review_table(results: list[dict]) -> list[list]:
    """Lags of week-dated changes, split by how many stocks BaoStock moves on the same day."""
    rows = []
    for r in results:
        week = [c for c in r["changes"] if c.week]
        per_day = Counter(c.date for c in week)
        for label, keep in ((f"review days ({REVIEW_MIN} or more stocks move)", lambda n: n >= REVIEW_MIN), (f"other days (fewer than {REVIEW_MIN})", lambda n: n < REVIEW_MIN)):
            picked = [c for c in week if keep(per_day[c.date])]
            totals = lag_totals(picked)
            stats = totals["stats"] or {}
            rows.append([
                r["index"].name, label, len({c.date for c in picked}), totals["changes"], totals["timed"], num(stats.get("median")),
                num(stats.get("p90")), num(stats.get("max")), num(stats.get("min")), f"{totals['never']} ({pct(totals['never_share'], 1)})",
            ])  # fmt: skip
    return rows


def after_second_friday(day: str, cal: Calendar) -> bool:
    """The first trading day after the second Friday of June or December."""
    date = dt.date.fromisoformat(day)
    if date.month not in (6, 12):
        return False
    first = date.replace(day=1)
    second_friday = first + dt.timedelta(days=(4 - first.weekday()) % 7 + 7)
    return cal.days[cal.first_on_or_after((second_friday + dt.timedelta(days=1)).isoformat())] == day


def first_of_half_year(day: str, cal: Calendar) -> bool:
    """The first trading day of January or July."""
    return day[5:7] in ("01", "07") and (cal.pos[day] == 0 or cal.days[cal.pos[day] - 1][:7] != day[:7])


PATTERNS = (
    "BaoStock's day is the first trading day after the second Friday of June or December",
    "the archive's day is the first trading day of January or July",
    "neither of the two",
)
PATTERN_PHRASES = dict(zip(PATTERNS, (
    "that BaoStock dates to the first trading day after the second Friday of June or December",
    "that the archive dates to the first trading day of January or July",
    "that fit neither pattern",
)))  # fmt: skip


def pattern_stats(result: dict, cal: Calendar) -> dict[str, dict]:
    """Where one index's week-dated review days fall in the calendar, with the lags of their changes."""
    week = [c for c in result["changes"] if c.week and c.status == MATCHED]
    per_day = Counter(c.date for c in week)
    found: dict[str, dict] = {pattern: {"days": [], "lags": []} for pattern in PATTERNS}
    for day in sorted(day for day, moves in per_day.items() if moves >= REVIEW_MIN):
        on_day = [c for c in week if c.date == day]
        archive_day = Counter(c.archive_date for c in on_day).most_common(1)[0][0]
        pattern = PATTERNS[0] if after_second_friday(day, cal) else PATTERNS[1] if first_of_half_year(archive_day, cal) else PATTERNS[2]
        found[pattern]["days"].append(day)
        found[pattern]["lags"] += [c.lag for c in on_day]
    return {pattern: {**entry, "stats": lag_stats(entry["lags"])} for pattern, entry in found.items() if entry["days"]}


def pattern_table(results: list[dict], cal: Calendar) -> list[list]:
    rows = []
    for r in results:
        for pattern, entry in pattern_stats(r, cal).items():
            stats = entry["stats"]
            rows.append([r["index"].name, pattern, len(entry["days"]), f"{entry['days'][0]} to {entry['days'][-1]}", stats["n"], num(stats["median"]), num(stats["min"]), num(stats["max"])])
    return rows


def other_day_lags(result: dict) -> dict | None:
    """Lag statistics of the week-dated changes on days when fewer than REVIEW_MIN stocks move."""
    week = [c for c in result["changes"] if c.week and c.status == MATCHED]
    per_day = Counter(c.date for c in week)
    return lag_stats([c.lag for c in week if per_day[c.date] < REVIEW_MIN])


def count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def capped(rows: list[list]) -> list[str]:
    """A table of grouped changes, cut at EXAMPLES rows with a note."""
    from detmaso.mdtable import md_table

    out = [md_table(CHANGE_HEADERS, rows[:EXAMPLES])]
    if len(rows) > EXAMPLES:
        out += ["", f"({len(rows) - EXAMPLES} more rows; the full list is in \"Every day on which BaoStock's list changes\" below.)"]
    return out


def delay_sources(results: list[dict], cal: Calendar) -> list[str]:
    """One bullet per index: the delay at the twice-yearly reviews and at the single-stock changes in between."""
    lines = []
    for r in results:
        name, parts = r["index"].name, []
        for pattern, entry in pattern_stats(r, cal).items():
            stats, days = entry["stats"], entry["days"]
            span = f"at the {count(len(days), 'review')} from {days[0]} to {days[-1]} {PATTERN_PHRASES[pattern]}"
            parts.append(f"{span}, the lag is {num(stats['min'])} to {num(stats['max'])} trading days (median {num(stats['median'])})")
        other = other_day_lags(r)
        if other:
            parts.append(f"the {other['n']} changes on other days (single stocks moving between reviews) wait for the archive's next "
                         f"change day: median {num(other['median'])}, 90th percentile {num(other['p90'])}, longest {num(other['max'])} trading days")  # fmt: skip
        if parts:
            lines.append(f"- **{name}, where the delay comes from** (changes that BaoStock dates to the week): " + "; ".join(parts) + ".")
    return lines


def delay_summary(results: list[dict], cal: Calendar) -> str | None:
    """The same in one bullet for the summary; None when no review day fits the June and December pattern."""
    mid_year = [entry for r in results if (entry := pattern_stats(r, cal).get(PATTERNS[0]))]
    if not mid_year:
        return None
    first = min(entry["days"][0] for entry in mid_year)
    reviews = max(len(entry["days"]) for entry in mid_year)
    low, high = min(entry["stats"]["min"] for entry in mid_year), max(entry["stats"]["max"] for entry in mid_year)
    longest = ", ".join(f"{num(other['max'])} ({r['index'].name})" for r in results if (other := other_day_lags(r)))
    return (f"- **Where the delay comes from** (section 4): at the {count(reviews, 'twice-yearly review')} since {first} that BaoStock "
            f"dates to the week, the archive is {num(low)} to {num(high)} trading days late. Single stocks that move between reviews "
            f"wait for the archive's next change day; the longest waits, in trading days: {longest}.")  # fmt: skip


def render(results: list[dict], cal: Calendar, ends: list[str], cache: dict) -> str:
    """The whole report as markdown. Depends only on the archive, the calendar and the cached answers."""
    from detmaso.mdtable import md_table

    out: list[str] = []
    say = out.append
    window, last_day = results[0]["window"], results[0]["last_day"]
    facts = {r["index"].key: update_date_facts(r, cal) for r in results}
    all_changes = [c for r in results for c in r["changes"]]
    timed = [c for c in all_changes if c.status == MATCHED and c.exact]
    clear = [c for c in timed if c.timing == EARLY_CLEAR]
    unclear = [c for c in timed if c.timing == EARLY_UNCLEAR]
    stale_months = [(r["index"].name, m) for r in results for m in r["months"] if m.status == "ok" and m.age > FRESH_DAYS]
    unpaired = {r["index"].key: unpaired_facts(r) for r in results}
    unpaired_note = unpaired_summary(results, unpaired)

    say("# Membership timing audit: CSI300 and CSI500 (Rule MA)")
    say("")
    say("Written by `scripts/audit_membership.py` (docs/RULES.md, Rule MA). A report only: it changes no data and no")
    say(f"decision depends on it. Every date that was queried or counted is before {LOCK_START}, the first day of the")
    say("locked test period. No model was trained, so this adds no tries.")
    say("")

    say("## In short")
    say("")
    for r in results:
        name = r["index"].name
        everything = mismatch_totals(r["months"])
        fresh = mismatch_totals([m for m in r["months"] if m.status == "ok" and m.age <= FRESH_DAYS])
        say(f"- **{name}, month-end lists:** {pct(everything['share'])} of names differ between the archive and BaoStock "
            f"({everything['one']} stocks in exactly one list out of {everything['either']} in either, {everything['months']} month ends; "
            f"identical lists at {everything['identical']} of them). Leaving out the {everything['months'] - fresh['months']} month ends where "
            f"BaoStock's own list was out of date: {pct(fresh['share'])}.")  # fmt: skip
    for r in results:
        name = r["index"].name
        week, everything = lag_totals([c for c in r["changes"] if c.week]), lag_totals(r["changes"])
        stats, every = week["stats"] or {}, everything["stats"] or {}
        model = lag_totals([c for c in r["changes"] if c.week and c.date >= MODEL_START])
        recent = model["stats"] or {}
        say(f"- **{name}, delay:** the archive shows an index change a median of {num(stats.get('median'))} trading days after "
            f"BaoStock (90th percentile {num(stats.get('p90'))}, longest {num(stats.get('max'))}), over the {week['timed']} changes that "
            f"BaoStock dates to the week. From {MODEL_START[:7]} on, the dates the model uses: median {num(recent.get('median'))}, "
            f"90th percentile {num(recent.get('p90'))}, longest {num(recent.get('max'))} ({model['timed']} changes). Over all "
            f"{everything['timed']} paired changes, including those where BaoStock skipped updates: median {num(every.get('median'))}, "
            f"90th percentile {num(every.get('p90'))}, longest {num(every.get('max'))}. The archive never shows "
            f"{everything['never']} of BaoStock's {everything['changes']} changes ({pct(everything['never_share'], 1)}).")  # fmt: skip
    if delay_summary(results, cal):
        say(delay_summary(results, cal))
    if clear:
        say(f"- **Look-ahead check: {len(clear)} of {len(timed)} paired changes ({pct(len(clear) / len(timed), 1)}) show up in the archive "
            "clearly EARLIER than in BaoStock**, that is on or before the date of a BaoStock list that still lacked the "
            "change. They are listed in section 3. Each is either look-ahead in the archive or a late entry in "
            f"BaoStock; this audit cannot tell which. {unpaired_note}")  # fmt: skip
    else:
        say(f"- **Look-ahead check: in none of the {len(timed)} paired changes does the archive show the change clearly EARLIER "
            f"than BaoStock** (on or before the date of a BaoStock list that still lacked it). {unpaired_note}")  # fmt: skip
    if unclear:
        in_week = [c for c in unclear if c.week]
        say(f"- Another {len(unclear)} changes ({pct(len(unclear) / len(timed), 1)}) show up in the archive before BaoStock's new list "
            "but after BaoStock's last old one. BaoStock had no list in between, so these say nothing either way about "
            f"look-ahead. For {len(in_week)} of them the two BaoStock lists are a week apart and the archive is at most "
            f"{-min((c.lag for c in in_week), default=0)} trading days ahead, which is what an archive that is exactly on time looks like "
            f"next to a weekly list; the other {len(unclear) - len(in_week)} fall where BaoStock skipped updates (section 3).")  # fmt: skip
    say(f"- BaoStock is an unofficial source, not ground truth, and it was itself out of date at {len(stale_months)} of the "
        f"{sum(m.status == 'ok' for r in results for m in r['months'])} month-end lists compared. See \"Limits\".")  # fmt: skip
    say("")

    say("## Why this audit exists")
    say("")
    say("On every day the model ranks the stocks that the pinned archive lists as index members on that day")
    say(f"(`{ARCHIVE_DIR}`, files `instruments/csi300.txt` and `instruments/csi500.txt`). The project's notes say")
    say("the archive's maintainer rebuilt these lists from snapshots of the index's weights, not from the index")
    say("provider's change notices. A list rebuilt that way can show an index change on the wrong day:")
    say("")
    say("- **late** is stale but honest: for a while the model ranks a slightly outdated list;")
    say("- **early** is look-ahead: the model would know that a stock joins or leaves the index before it happened.")
    say("")
    say("The audit measures the direction and the size of the error against a second source.")
    say("")

    say("## Method")
    say("")
    say("**The archive.** Each line of a membership file is a stock and a span of dates; a stock is a member on a day")
    say("if one of its spans covers that day. Spans that touch are joined. The list can only change on the days on")
    say("which spans start, and there are few of them:")
    say("")
    rows = []
    for r in results:
        gap = archive_gap(r, cal)
        rows.append([r["index"].name, r["archive_first"], gap["dates"], num(gap["median"]), num(gap["max"])])
    say(md_table(["Index", "Archive's first day", f"Days on which its list can change, {ends[0]} to {last_day}", "Typical distance (trading days, median)", "Largest distance"], rows))
    say("")
    say("**BaoStock.** A free data service (Python package `baostock`, anonymous login, no account). It was asked for")
    say("the member list on one date at a time with `query_hs300_stocks(date=...)` (CSI300) and")
    say("`query_zz500_stocks(date=...)` (CSI500). Each answer carries the stock codes and an \"update date\".")
    say("")
    say(f"**Step 1, month ends.** The last trading day of every month from {FIRST_MONTH} to {LAST_MONTH} in the archive's")
    say(f"own trading calendar: {len(ends)} month ends, {ends[0]} to {ends[-1]}. On each one the two lists are compared.")
    say("Mismatch share = stocks in exactly one of the two lists / stocks in either list. The overall and yearly")
    say("figures pool the month ends (sum of the first count / sum of the second). A BaoStock answer that is empty or")
    say(f"more than {SIZE_TOLERANCE:.0%} away from the index's size (300 or 500 stocks) is counted as unusable and left out.")
    say("")
    say("**Step 2, the day of each change in BaoStock.** Whenever BaoStock's lists at two consecutive month ends")
    say("differ, the days in between are searched by halving: ask for the list in the middle, keep the half (or halves)")
    say("whose two ends differ, repeat until two neighbouring trading days carry different lists. The later of the two")
    say("is the day the change takes effect in BaoStock. The update date was not trusted in advance; it was checked")
    say("against these directly found days afterwards (section 2).")
    say("")
    say("**Step 3, the day of the same change in the archive.** For an add, the first day of a membership run of that")
    say("stock; for a drop, the first trading day after a run ends. A BaoStock change is paired with the archive change")
    say(f"of the same stock and the same direction that is nearest in time, at most {window} trading days (about one")
    say("year) away, each archive change used once. The window was fixed before any result was seen.")
    say("")
    say("**Lag** = the archive's day minus BaoStock's day, in trading days. Above 0 the archive is late, below 0 it is")
    say("early. A BaoStock change without a partner is:")
    say("")
    say(f"- *never shown*, if at least {window} trading days of archive history follow it;")
    say(f"- *too close to the cut-off*, if fewer follow (the archive may still show it after {last_day}, which this audit does not look at);")
    say("- *in the archive from its first day*, if the archive already lists the stock on its very first day, so the")
    say("  archive's own add day cannot be known.")
    say("")
    say("**Early changes come in two kinds**, because BaoStock only has a list for certain dates (section 2):")
    say("")
    say(f"- *{EARLY_UNCLEAR}*: the archive shows the change after the date of BaoStock's last old list and")
    say("  before BaoStock's first new list. BaoStock had no list in between, so it cannot tell who is right.")
    say(f"- *{EARLY_CLEAR}*: the archive shows the change on or before the date of a BaoStock list that still lacked it.")
    say("  The two sources contradict each other on that day. This is the look-ahead signal.")
    say("")
    say("**\"Dated to the week\".** BaoStock normally stores a new list every Monday, so its last old list and its first")
    say(f"new list are at most {WEEK} trading days apart and the change day is known to the week. Where BaoStock skipped")
    say("updates the two lists are further apart, BaoStock's change day is too late by an unknown amount, and the lag")
    say("says more about BaoStock than about the archive. Section 3 gives the lags both ways.")
    say("")
    say(f"**What was fixed when.** The {window}-day pairing window, the {SIZE_TOLERANCE:.0%} size rule, the percentile method and the two")
    say("kinds of \"early\" were fixed before any BaoStock list was compared with the archive (a first probe of seven")
    say("dates had only shown that answers carry a Monday update date). The splits by the age of BaoStock's list")
    say(f"(\"at most {FRESH_DAYS} days old\", \"dated to the week\") and by review days were added after the first comparisons")
    say("showed that BaoStock skips updates. The all-inclusive numbers are always shown next to them.")
    say("")

    say("## 1. Month-end lists")
    say("")
    rows = []
    for r in results:
        ok = [m for m in r["months"] if m.status == "ok"]
        subsets = (
            ("all", ok),
            (f"BaoStock's list at most {FRESH_DAYS} days old", [m for m in ok if m.age <= FRESH_DAYS]),
            ("BaoStock's list older (out of date)", [m for m in ok if m.age > FRESH_DAYS]),
            (f"{MODEL_START[:7]} on", [m for m in ok if m.date >= MODEL_START]),
            (f"{MODEL_START[:7]} on, BaoStock's list at most {FRESH_DAYS} days old", [m for m in ok if m.date >= MODEL_START and m.age <= FRESH_DAYS]),
        )  # fmt: skip
        for label, months in subsets:
            t = mismatch_totals(months)
            if t is None:
                rows.append([r["index"].name, label, 0, 0, 0, "n/a", "n/a", 0, "n/a"])
                continue
            worst = max(months, key=lambda m: (m.share, m.date))
            rows.append([r["index"].name, label, t["months"], t["one"], t["either"], f"**{pct(t['share'])}**", f"{t['per_month']:.1f}", t["identical"], f"{pct(worst.share)} ({worst.date})"])
    say(md_table(["Index", "Month ends", "Compared", "Stocks in exactly one list", "Stocks in either list", "Mismatch share", "Stocks in one list only, per month end", "Month ends with identical lists", "Worst month end"], rows))
    say("")
    say(f"\"{MODEL_START[:7]} on\" is the part the model uses (training starts {MODEL_START}). A BaoStock list is \"out of date\"")
    say(f"when its update date is more than {FRESH_DAYS} days before the month end (section 2); a mismatch there is BaoStock's")
    say("delay, not the archive's.")
    say("")
    on_month_end = "; ".join(
        f"{r['index'].name}: {len(set(r['archive_dates']) & set(ends))} of its {len(r['archive_dates'])} change days are month ends" for r in results)
    say(f"**Month ends flatter the archive.** Its list mostly changes on a month end ({on_month_end}), so a")
    say("month-end comparison lands just after the archive has caught up. The days in between are measured in")
    say("section 3, and section 4 turns them into a share of the universe.")
    say("")
    say("**Unusable BaoStock answers at month ends.** " + "; ".join(
        f"{r['index'].name}: {sum(m.status == 'empty' for m in r['months'])} empty, {sum(m.status == 'size' for m in r['months'])} with an obviously wrong size, "
        f"{sum(m.status == 'no archive list' for m in r['months'])} where the archive has no list" for r in results) + ".")  # fmt: skip
    bad = [(r["index"].name, m) for r in results for m in r["months"] if m.status != "ok"]
    if bad:
        say("")
        say(md_table(["Index", "Month end", "Problem", "Archive stocks", "BaoStock stocks"], [[name, m.date, m.status, m.archive, m.baostock] for name, m in bad]))
    say("")
    for label, field in (("BaoStock", "baostock"), ("Archive", "archive")):
        sizes = "; ".join(
            f"{r['index'].name}: " + ", ".join(f"{size} stocks at {count(times, 'month end')}" for size, times in Counter(getattr(m, field) for m in r["months"]).most_common())
            for r in results)  # fmt: skip
        say(f"{label} list sizes at the month ends: {sizes}.")
    say("")
    say("### Per year")
    say("")
    rows = []
    for year in sorted({m.date[:4] for r in results for m in r["months"]}):
        row = [year]
        for r in results:
            months = [m for m in r["months"] if m.date[:4] == year]
            t = mismatch_totals(months)
            stale = sum(m.status == "ok" and m.age > FRESH_DAYS for m in months)
            row += [pct(t["share"]), f"{t['one']} / {t['either']}", t["months"], stale] if t else ["n/a", "n/a", 0, 0]
        rows.append(row)
    headers = ["Year"]
    for r in results:
        headers += [f"{r['index'].name} mismatch share", "In one list / in either", "Month ends", "Of which BaoStock out of date"]
    say(md_table(headers, rows))
    say("")
    say("### Worst month ends")
    say("")
    for r in results:
        worst = sorted((m for m in r["months"] if m.status == "ok"), key=lambda m: (-m.share, m.date))[:WORST_MONTHS]
        say(f"**{r['index'].name}** (the {len(worst)} largest):")
        say("")
        say(md_table(["Month end", "In the archive only", "In BaoStock only", "In either", "Mismatch share", "BaoStock's list is dated", "Days old"],
                     [[m.date, len(m.only_archive), len(m.only_baostock), m.either, pct(m.share), m.update, m.age] for m in worst]))  # fmt: skip
        say("")

    say("## 2. How BaoStock dates its lists")
    say("")
    say("Measured on every answer the audit used (month ends and the extra days of step 2):")
    say("")
    rows = []
    for r in results:
        f = facts[r["index"].key]
        weekdays = ", ".join(f"{day} {count}" for day, count in f["weekdays"].most_common())
        rows.append([
            r["index"].name, f["answers"], f"{f['one_stamp']} of {f['answers']}", f"{f['stamp_not_after']} of {f['answers']}",
            f["stamps"], weekdays, f"{f['one_list_per_stamp']} of {f['stamps']}", f"{f['on_stamp']} of {f['exact_steps']}",
            f"{f['fresh']} of {f['answers']}", f"{f['oldest'].age} days (asked {f['oldest'].date}, dated {f['oldest'].update_date})",
        ])  # fmt: skip
    say(md_table([
        "Index", "Answers with stocks", "One update date for all rows", "Update date on or before the asked date",
        "Distinct update dates seen", "Their weekdays", "Update dates that always come with the same list",
        "Change days where the new list starts on its update date", f"Answers at most {FRESH_DAYS} days old", "Oldest answer",
    ], rows))  # fmt: skip
    say("")
    say("Reading: an answer is the latest list BaoStock stored on or before the asked date, and the update date is the")
    say("day that list was stored. \"Starts on its update date\" means the update date lies after the last trading day")
    say("with the old list and on or before the first trading day with the new one (it can be a holiday in between).")
    say("")
    if all(set(f["weekdays"]) == {"Monday"} for f in facts.values()):
        say("- **Every update date seen is a Monday: BaoStock stores at most one list per week.** A change that takes")
        say("  effect on a Wednesday appears in BaoStock on the following Monday at the earliest, up to 4 trading days")
        say("  later. So a negative lag of a few days does not prove that the archive is early.")
    if stale_months:
        say(f"- **BaoStock skipped updates in places.** At {len(stale_months)} month ends its list was more than {FRESH_DAYS} days old")
        say("  (table below). In such a stretch BaoStock shows an index change weeks or months after it happened.")
    reliable = all(f["on_stamp"] == f["exact_steps"] and f["one_list_per_stamp"] == f["stamps"] for f in facts.values())
    if reliable:
        say("- **Which day was used, and why:** the day found by direct queries. The update date agrees with it in every")
        say("  case, so it proved reliable, but the change day does not depend on it. The update date is used for two")
        say("  things: the date of BaoStock's last old list, which separates the two kinds of \"early\", and the age of")
        say("  a list, which shows where BaoStock itself was out of date.")
    else:
        say("- **Which day was used, and why:** the day found by direct queries, because the update date does not agree")
        say("  with it in every case (see the table).")
    undone = ", ".join(f"{r['index'].name} {r['undone']}" for r in results)
    say(f"- Stock moves that BaoStock undid again before the next usable month end, as far as the search saw them (not")
    say(f"  counted as changes; a lower bound, see \"Limits\"): {undone}. Changes whose BaoStock day could not be pinned")
    say(f"  to one trading day because of unusable answers: {sum(not c.exact for c in all_changes)}.")
    say("")
    if stale_months:
        say(f"Month ends at which BaoStock's list was more than {FRESH_DAYS} days old:")
        say("")
        say(md_table(["Index", "Month end", "BaoStock's list is dated", "Days old", "Stocks in one list only", "Mismatch share"],
                     [[name, m.date, m.update, m.age, m.one_list_only, pct(m.share)] for name, m in stale_months]))  # fmt: skip
        say("")

    say("## 3. How late the archive shows index changes")
    say("")
    say("Lags are in trading days (positive = the archive is late). The 90th percentile is the smallest lag that at")
    say("least 90% of the changes do not exceed. \"Never shown (share)\" = never shown / (paired + never shown).")
    say("")
    say(f"**Changes that BaoStock dates to the week** (its last old list and its first new list are at most {WEEK} trading")
    say("days apart). This is the cleaner measure of the archive's delay:")
    say("")
    say(md_table(["Index", "Period", "Change", *LAG_HEADERS], lag_table(results, lambda c: c.week)))
    say("")
    say("The same week-dated changes, split by what kind of day it was in BaoStock. The index is reviewed twice a")
    say("year, when many stocks move at once; in between, single stocks move now and then (a merger, a delisting, a")
    say("very large new listing):")
    say("")
    say(md_table(["Index", "BaoStock change day", "Days", "Changes", "With a lag", "Median lag", "90th percentile", "Longest", "Shortest", "Never shown (share)"], review_table(results)))
    say("")
    say("Where those review days fall in the calendar:")
    say("")
    say(md_table(["Index", "Review day", "Days", "BaoStock's days", "Changes", "Median lag", "Shortest", "Longest"], pattern_table(results, cal)))
    say("")
    say("A reading aid, not a finding: the index provider's twice-yearly reviews are generally described as taking")
    say("effect at the start of January and July until mid-2013, and on the trading day after the second Friday of")
    say("June and December since then. This audit did not check that against official notices. If it is right, a")
    say("row of the first kind is BaoStock on the true day with the archive behind it, a row of the second kind is")
    say("the archive on the true day with BaoStock's weekly list a few days behind, and a row of the third kind is a")
    say("review that the archive shows late by more than BaoStock's weekly step.")
    say("")
    say("**All changes**, including those where BaoStock skipped updates and its change day is too late:")
    say("")
    say(md_table(["Index", "Period", "Change", *LAG_HEADERS], lag_table(results, lambda c: True)))
    say("")
    extra = "; ".join(
        f"{r['index'].name}: {lag_totals(r['changes'])['too_recent']} too close to the cut-off, {lag_totals(r['changes'])['from_start']} in the archive from its first day"
        for r in results)  # fmt: skip
    say(f"Not paired and left out of the never-shown share: {extra}.")
    say("")

    say("### Changes the archive shows EARLIER than BaoStock")
    say("")
    for r in results:
        name = r["index"].name
        paired = [c for c in r["changes"] if c.status == MATCHED and c.exact]
        for kind in (EARLY_CLEAR, EARLY_UNCLEAR):
            cases = [c for c in paired if c.timing == kind]
            adds, drops = sum(c.kind == ADD for c in cases), sum(c.kind == DROP for c in cases)
            share = pct(len(cases) / len(paired), 1) if paired else "n/a"
            note = ""
            if kind == EARLY_UNCLEAR and cases:
                note = f" Of these, {sum(c.week for c in cases)} are dated to the week by BaoStock (the archive is at most a few days ahead of the next Monday list) and {sum(not c.week for c in cases)} fall where BaoStock skipped updates."
            say(f"**{name}, {kind}: {len(cases)} of {len(paired)} paired changes ({share}); {count(adds, 'add')}, {count(drops, 'drop')}.**{note}")
            say("")
            if cases:
                out.extend(capped(group_changes(cases)))
                say("")
    say("### Per year")
    say("")
    say("By the year of BaoStock's change day. The three lag columns use only the changes that BaoStock dates to the")
    say("week; \"Early\" counts both kinds over all changes, with the clear ones in brackets.")
    say("")
    rows = []
    for year in sorted({c.date[:4] for c in all_changes}):
        row = [year]
        for r in results:
            in_year = [c for c in r["changes"] if c.date[:4] == year]
            t, week = lag_totals(in_year), lag_totals([c for c in in_year if c.week])
            stats = week["stats"] or {}
            early = t["timing"][EARLY_UNCLEAR] + t["timing"][EARLY_CLEAR]
            row += [t["changes"], week["changes"], num(stats.get("median")), num(stats.get("min")), num(stats.get("max")), f"{early} ({t['timing'][EARLY_CLEAR]})", t["never"]]
        rows.append(row)
    headers = ["Year"]
    for r in results:
        headers += [f"{r['index'].name} changes", "Dated to the week", "Median lag", "Shortest", "Longest", "Early (clear)", "Never shown"]
    say(md_table(headers, rows))
    say("")

    say("### Changes the archive never shows")
    say("")
    for r in results:
        cases = [c for c in r["changes"] if c.status == NEVER]
        say(f"**{r['index'].name}: {len(cases)}** ({count(sum(c.kind == ADD for c in cases), 'add')}, {count(sum(c.kind == DROP for c in cases), 'drop')}).")
        say("")
        if cases:
            out.extend(capped(group_changes(cases)))
            say("")
    say("### The other direction: archive changes that BaoStock never shows")
    say("")
    for r in results:
        only = r["archive_only"]
        by_day = Counter(day for day, _, _ in only)
        top = ", ".join(f"{day} ({moves})" for day, moves in sorted(by_day.items(), key=lambda item: (-item[1], item[0]))[:8])
        say(f"- {r['index'].name}: {len(only)} ({count(sum(kind == ADD for _, kind, _ in only), 'add')}, {count(sum(kind == DROP for _, kind, _ in only), 'drop')}) after {ends[0]}"
            + (f". Days with the most: {top}." if only else "."))  # fmt: skip
    say("")
    say(f"These are stock moves in the archive with no BaoStock change of the same stock and direction within {window}")
    say("trading days. Some are the other half of a \"never shown\" case above (the two sources swapped different")
    say("stocks). One can also be the archive's side of a BaoStock move that was undone again before the next usable")
    say(f"month end (section 2), which this audit does not count as a change. One shortly after {ends[0]} can also")
    say("belong to a BaoStock change from before the audit window.")
    say("")
    if any(unpaired.values()):
        say("What BaoStock's lists dated on or after the archive's day (among those this audit asked for; a list is one")
        say("update date) say about each:")
        say("")
        say(md_table(
            ["Index", "Archive day", "Change", "Stock", "Later BaoStock lists (distinct update dates)", "Of which hold the stock", "Archive ahead of all of them"],
            [[r["index"].name, f["date"], f["kind"], f["code"], f"{f['lists']} ({f['dated'][0]} to {f['dated'][1]})" if f["dated"] else "0",
              f["holds"], "yes" if f["ahead"] else "no"] for r in results for f in unpaired[r["index"].key]],
        ))  # fmt: skip
        say("")
        say("Reading: an add that none of the later lists holds, or a drop that all of them still hold, is a change the")
        say("archive shows before BaoStock does, the pattern of a clearly early change above. These have no BaoStock")
        say("change to pair with, so they are not in the look-ahead count; \"In short\" and section 4 name them instead.")
        say("Later lists that agree with the archive mean BaoStock shows the same state and made the move before the")
        say(f"audit window or more than {window} trading days away.")
        say("")

    say("### Every day on which BaoStock's list changes")
    say("")
    say("One row per BaoStock change day, archive day and outcome; \"Stocks\" names up to four of them.")
    say("")
    for r in results:
        say(f"**{r['index'].name}:**")
        say("")
        say(md_table(CHANGE_HEADERS, group_changes(r["changes"])))
        say("")

    say("## 4. What this means for the project")
    say("")
    for r in results:
        name = r["index"].name
        recent = lag_totals([c for c in r["changes"] if c.date >= MODEL_START and c.week])
        stats = recent["stats"] or {}
        t = mismatch_totals([m for m in r["months"] if m.status == "ok" and m.date >= MODEL_START and m.age <= FRESH_DAYS])
        say(f"- **{name}, {MODEL_START[:7]} on (the dates the model uses):** an index change reaches the model's universe a median of "
            f"{num(stats.get('median'))} trading days after BaoStock shows it (90th percentile {num(stats.get('p90'))}, longest {num(stats.get('max'))}; "
            f"{recent['timed']} changes dated to the week). At a month end where BaoStock is up to date, {t['per_month']:.1f} of the "
            f"{r['index'].size} names differ on average ({pct(t['share'])}).")  # fmt: skip
    for line in delay_sources(results, cal):
        say(line)
    say("- **What a delay does.** For that many days after an index change, the universe still holds the stocks that")
    say("  left the index and lacks the ones that joined. That makes the universe slightly out of date. It uses no")
    say("  information from the future, so it cannot inflate a backtest the way look-ahead does.")
    for r in results:
        o = overdue(r, cal, MODEL_START)
        say(f"- **How much of the {r['index'].name} universe is out of date, {MODEL_START[:7]} on.** Stocks that BaoStock had already dropped "
            f"but the archive still listed add up to {o['held']} stock-days, {pct(o['share'])} of the archive's {o['member_days']} member-days "
            "(one stock on one trading day). About as many stock-days are missing on the other side (stocks that had "
            "joined). Where BaoStock skipped updates the delay cannot be measured and counts as zero here, so the true "
            "share is somewhat higher.")  # fmt: skip
    if clear:
        say(f"- **Look-ahead.** {len(clear)} paired changes are clearly earlier in the archive than in BaoStock (section 3). They need")
        say("  a look by the owner before the archive's membership is called free of look-ahead.")
    else:
        say("- **Look-ahead.** No paired change is clearly earlier in the archive than in BaoStock.")
    say(f"  {unpaired_note}")
    if unclear:
        say(f"- **Not settled by this audit.** {len(unclear)} changes appear in the archive between two BaoStock lists. Where the")
        say("  two lists are a week apart, the archive is at most a few days ahead of BaoStock's next Monday list, which")
        say("  is what an archive that is exactly on time looks like. Where BaoStock skipped updates, nothing can be")
        say("  said. Settling these needs the index provider's own effective dates, which this audit did not use.")
    say("- No data and no config was changed. Rule MA makes this a report, not a decision.")
    say("")

    say("## 5. Cache and hashes")
    say("")
    say(f"- Cache: `{cache['dir']}/<index>/<asked date>.csv`, one file per BaoStock answer, exactly as returned (columns")
    say(f"  `{', '.join(FIELDS)}`; an empty answer would be a file with only the header). Outside the repo, not committed.")
    for r in results:
        info = cache["index"][r["index"].key]
        say(f"- {r['index'].name}: {info['files']} answers ({len(ends)} month ends, {info['files'] - len(ends)} extra days), downloaded "
            f"{info['first']} to {info['last']} (UTC) with {info['client']}; {info['retried']} needed more than one try.")  # fmt: skip
    say(f"- Hash list: `{cache['dir']}/{SHA_LIST}` holds the sha256 of every answer file ({cache['listed']} lines, checkable with")
    say(f"  `shasum -a 256 -c {SHA_LIST}` inside the cache folder); the two download logs, which feed only the download")
    say("  times, client and retry counts above, are not in it. The sha256 of that list itself is")
    say(f"  `{cache['list_sha']}`.")
    say(f"- Archive: tag `{ARCHIVE_TAG}`; `instruments/csi300.txt` sha256 `{cache['archive']['csi300']}`,")
    say(f"  `instruments/csi500.txt` sha256 `{cache['archive']['csi500']}`,")
    say(f"  `calendars/day.txt` sha256 `{cache['archive']['calendar']}`.")
    say("- Rebuild this report without any network (it checks every answer file against the hash list first):")
    say(f"  `python3 -m uv run python scripts/audit_membership.py --offline --cache-dir {cache['dir']}`")
    say("- Requests went out one at a time per index, with a pause before each and up to 6 tries with growing waits.")
    say("  An empty answer would have been asked for a second time before it was believed.")
    say("")

    say("## 6. Limits of this audit")
    say("")
    say("- **BaoStock is not ground truth.** It is a free third-party service with no stated link to the index provider.")
    say("  Where the two sources differ, this audit cannot say which one is wrong. The index provider's own change")
    say("  notices and effective dates were not used.")
    say("- **BaoStock's lists are weekly at best.** Its change days are known only to the week (section 2). If its")
    say("  Monday list is right for that Monday, BaoStock shows a change up to 4 trading days after it took effect, so")
    say("  a lag here can understate the archive's true delay by that much, and an archive that is exactly on time can")
    say("  look a few days early.")
    if stale_months:
        say("- **BaoStock was out of date for whole stretches** (section 2). There its change days are too late, the")
        say("  month-end mismatch is its own delay, and a clearly early archive change could hide among the \"between two")
        say("  BaoStock lists\" cases without this audit being able to see it.")
    say("- **The two sources may share an origin.** If BaoStock's history was rebuilt from the same kind of weight")
    say("  snapshots as the archive, both can be late in the same way and agree with each other while both are wrong.")
    say("  Agreement here is weaker evidence than disagreement.")
    say("- **Only month ends were compared in step 1**, and step 2 searched between consecutive usable month ends by")
    say("  halving, never searching a half whose two ends carry the same list. A stock move that BaoStock undid again")
    say("  before the next usable month end is therefore seen only if a halving step happened to fall between the move")
    say("  and its reversal, even when that stretch had other changes. The \"undone\" count in section 2 is a lower")
    say("  bound, and a stock that moved more than once is dated by the last move the search saw, so its lag can come")
    say("  out too large by the unseen moves. Where a month end was unusable the search spans two months, so such a")
    say("  move can also be undone in the following month; its archive side then appears under \"The other direction\".")
    say(f"- **\"In the archive from its first day\" is given only within {window} trading days of that day.** A BaoStock add")
    say("  of a stock that the archive has listed without a break since its own first day, found later than that, is")
    say("  labelled never shown, although the archive's own add day is simply unknown.")
    say(f"- **The pairing window is a choice.** {window} trading days; a change the archive shows later than that counts as never shown.")
    say(f"- **Nothing from {LOCK_START} on was looked at**, so changes in the last months before it may still be shown")
    say("  later by the archive (\"too close to the cut-off\").")
    say("- **Only CSI300 and CSI500 membership was audited.** Prices, adjustment factors and the other universes")
    say("  (CSI1000, all A-shares) are not covered; the BaoStock package has list queries for CSI300, CSI500 and SSE50 only.")
    return "\n".join(out) + "\n"


def display(path: Path) -> str:
    """A path with the home folder written as "~", so the report does not depend on the user name."""
    home = str(Path.home())
    return "~" + str(path)[len(home):] if str(path).startswith(home) else str(path)


def cache_facts(cache: Cache, archive: Path) -> dict:
    facts: dict = {"dir": display(cache.dir), "index": {}}
    for index in INDICES:
        log = cache.log_rows(index)
        times = sorted(row["downloaded_at_utc"] for row in log)
        facts["index"][index.key] = {
            "files": sum(name.startswith(f"{index.key}/") for name in cache.files()),
            "first": times[0] if times else "n/a", "last": times[-1] if times else "n/a",
            "client": ", ".join(sorted({row["client"] for row in log})) or "n/a",
            "retried": sum(int(row["tries"]) > 1 for row in log),
        }  # fmt: skip
    facts["listed"] = len(cache.read_sha_list())
    facts["list_sha"] = sha256(cache.dir / SHA_LIST)
    facts["archive"] = {
        "csi300": sha256(archive / "instruments" / "csi300.txt"), "csi500": sha256(archive / "instruments" / "csi500.txt"),
        "calendar": sha256(archive / "calendars" / "day.txt"),
    }  # fmt: skip
    return facts


def write_events(path: Path, results: list[dict]) -> None:
    """Every BaoStock change with its archive day and lag, for looking at single cases."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["index", "code", "kind", "baostock_prev_date", "baostock_date", "baostock_last_old_update", "baostock_update", "exact", "update_gap_trading_days", "dated_to_week", "status", "archive_date", "lag", "timing"])
        for r in results:
            for c in r["changes"]:
                writer.writerow([r["index"].key, c.code, c.kind, c.prev_date, c.date, c.last_old_update, c.update, c.exact, c.gap, c.week, c.status, c.archive_date, c.lag, c.timing])


# Command line ----------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive-dir", default=ARCHIVE_DIR, help="the pinned archive (default: %(default)s)")
    parser.add_argument("--cache-dir", help=f"the cache to fill or reuse (default: {CACHE_ROOT}/<today>; offline: the newest one)")
    parser.add_argument("--offline", action="store_true", help="use the cache only; never open a connection")
    parser.add_argument("--only", choices=[index.key for index in INDICES], help="fill the cache for one index and write no report")
    parser.add_argument("--out", default=str(REPORT), help="the report to write (default: %(default)s)")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds to wait before every request (default: %(default)s)")
    parser.add_argument("--events-csv", help="also write every change with its lag to this file (keep it outside the repo)")
    return parser.parse_args(argv)


def pick_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir).expanduser()
    root = Path(CACHE_ROOT).expanduser()
    if not args.offline:
        return root / dt.date.today().isoformat()
    dated = sorted(path for path in root.glob("*") if (path / SHA_LIST).is_file())
    if not dated:
        raise SystemExit(f"no cache with a {SHA_LIST} under {root}; run once without --offline")
    return dated[-1]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive = Path(args.archive_dir).expanduser()
    days = read_calendar(archive / "calendars" / "day.txt")
    ends = month_ends(days, FIRST_MONTH, LAST_MONTH)
    cal = Calendar([day for day in days if day < LOCK_START])
    if ends[-1] >= LOCK_START:
        raise SystemExit("a month end falls in the locked test period")

    cache_dir = pick_cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = None if args.offline else BaoStock(pause=args.pause)
    cache = Cache(cache_dir, source)
    print(f"cache: {cache_dir} ({'offline' if args.offline else 'online'})", file=sys.stderr)
    results = []
    try:
        for index in INDICES:
            if args.only in (None, index.key):
                spans = read_spans(archive / "instruments" / f"{index.key}.txt")
                results.append(audit_index(index, cal, ends, spans, cache.get))
    except Unreachable as error:
        raise SystemExit(f"{error}\nNo other source was substituted. What is cached so far stays in {cache_dir}.") from error
    finally:
        if source is not None:
            source.close()
            cache.write_sha_list()
    for result in results:
        lags = [c.lag for c in result["changes"] if c.status == MATCHED]
        print(f"{result['index'].name}: {len(result['months'])} month ends, {len(result['changes'])} changes, lags {lag_stats(lags)}", file=sys.stderr)
    if args.only:
        return
    if args.events_csv:
        write_events(Path(args.events_csv).expanduser(), results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(results, cal, ends, cache_facts(cache, archive)), encoding="utf-8")
    print(f"report: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
