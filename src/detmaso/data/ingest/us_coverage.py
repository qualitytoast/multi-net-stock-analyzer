"""The free US coverage check (docs/RULES.md, Rule UC): how much of the S&P 500's history could be priced from free data.

Membership comes from fja05680/sp500 (unofficial members by date) and the price listing from Tiingo's public list of
supported tickers. Only the two lists are compared; no price is fetched and no API key is used. Run:

    python -m detmaso.data.ingest.us_coverage --membership <csv> --tiingo <zip> [--out-dir results/sp500]

`scripts/get_us_coverage_data.sh` fetches and verifies both inputs and prints their paths. The rule, and the details
fixed from the file formats before the first match, are in docs/RULES.md; this module applies them as written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

from detmaso.data.ingest.pins import SCRIPTS_DIR, read_pins
from detmaso.mdtable import md_table

# The rule -----------------------------------------------------------------------------------------------------
WINDOW_START = date(2008, 1, 1)
WINDOW_END = date(2024, 2, 29)  # the day before the locked test period; nothing after it is read from membership
BAR_PER_MILLE = 950  # pass if matched x 1000 >= all x 950, i.e. at least 95.0% on the exact counts

STOCK_ASSET_TYPE = "Stock"
STOCK_CURRENCY = "USD"
NON_US_EXCHANGES = frozenset({"LSE", "SHE", "SHEB", "SHG", "SHGB"})
US_EXCHANGES = frozenset({
    "NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "NYSE NAT", "BATS", "PINK", "OTCMKTS", "OTCGREY", "OTCBB",
    "OTCQB", "OTCQX", "OTCCE", "OTCD", "EXPM", "NMFQS", "",
})  # fmt: skip

# The pinned inputs (scripts/us_coverage_data.pins, the file scripts/get_us_coverage_data.sh sources) -----------
PINS = read_pins(SCRIPTS_DIR / "us_coverage_data.pins")
PINNED_SP500_COMMIT = PINS["SP500_COMMIT"]
PINNED_MEMBERSHIP_FILE = PINS["SP500_FILE"]
PINNED_MEMBERSHIP_SHA256 = PINS["SP500_FILE_SHA256"]
PINNED_TIINGO_DOWNLOADED_AT = PINS["TIINGO_DOWNLOADED_AT"]
PINNED_TIINGO_ZIP_SHA256 = PINS["TIINGO_ZIP_SHA256"]
TIINGO_URL = PINS["TIINGO_URL"]
TIINGO_CSV_NAME = "supported_tickers.csv"
TIINGO_COLUMNS = ("ticker", "exchange", "assetType", "priceCurrency", "startDate", "endDate")

REPORT_FILE = "coverage_check.md"
UNMATCHED_FILE = "coverage_unmatched.csv"
TOP_UNMATCHED = 30

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class CoverageError(ValueError):
    """An input cannot be read the way Rule UC needs, so no verdict is given."""


# Tickers and days ---------------------------------------------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    """Upper case, with "." and "/" written as "-" (class shares). Nothing else is changed."""
    return ticker.strip().upper().replace(".", "-").replace("/", "-")


def weekdays(start: date, end: date) -> list[date]:
    """Every Monday to Friday from `start` to `end`, both included. Market holidays are not removed."""
    return [d for d in (start + timedelta(n) for n in range((end - start).days + 1)) if d.weekday() < 5]


def _parse_date(text: str, where: str) -> date:
    if not _ISO_DATE.fullmatch(text):
        raise CoverageError(f"{where}: {text!r} is not a YYYY-MM-DD date")
    try:
        return date.fromisoformat(text)
    except ValueError as e:
        raise CoverageError(f"{where}: {text!r} is not a date") from e


# Membership (fja05680/sp500) ----------------------------------------------------------------------------------
Membership = list[tuple[date, frozenset[str]]]


def read_membership(lines: Iterable[str], last: date = WINDOW_END) -> Membership:
    """The rows (date, full member list from that date on) dated on or before `last`, tickers normalized."""
    reader = csv.reader(lines)
    if next(reader, None) != ["date", "tickers"]:
        raise CoverageError("membership file: the header is not 'date,tickers'")
    rows: Membership = []
    for n, rec in enumerate(reader, start=2):
        if not rec:
            continue
        if len(rec) != 2:
            raise CoverageError(f"membership file, line {n}: expected 2 columns, found {len(rec)}")
        day = _parse_date(rec[0], f"membership file, line {n}")
        if rows and day <= rows[-1][0]:
            raise CoverageError(f"membership file, line {n}: dates are not strictly increasing")
        if day > last:
            break
        rows.append((day, frozenset(normalize_ticker(t) for t in rec[1].split(",") if t.strip())))
    return rows


def members_on(rows: Membership, day: date) -> frozenset[str]:
    """The member list in force on `day`: the list of the last row dated on or before it."""
    current = None
    for row_day, members in rows:
        if row_day > day:
            break
        current = members
    if current is None:
        raise CoverageError(f"membership file: no row dated on or before {day}")
    return current


def member_days(rows: Membership, days: list[date]) -> dict[str, list[date]]:
    """Ticker -> the days (in order) on which it is in the member list in force."""
    if days and (not rows or rows[0][0] > days[0]):
        raise CoverageError(f"membership file: no row dated on or before {days[0]}")
    out: dict[str, list[date]] = defaultdict(list)
    r = 0
    for day in days:
        while r + 1 < len(rows) and rows[r + 1][0] <= day:
            r += 1
        for ticker in rows[r][1]:
            out[ticker].append(day)
    return dict(out)


def departures(rows: Membership) -> dict[str, tuple[date, tuple[str, ...]]]:
    """Ticker -> (date of the row from which it is last missing, tickers that joined in that same row)."""
    out = {}
    for (_, before), (day, after) in zip(rows, rows[1:]):
        joined = tuple(sorted(after - before))
        for ticker in before - after:
            out[ticker] = (day, joined)
    final = rows[-1][1] if rows else frozenset()
    return {t: v for t, v in out.items() if t not in final}


# Tiingo's list ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Listing:
    """One row of Tiingo's supported-tickers file; `ticker` is normalized, a missing date is None."""

    ticker: str
    exchange: str
    asset_type: str
    currency: str
    start: date | None
    end: date | None

    @property
    def is_stock(self) -> bool:
        return self.asset_type == STOCK_ASSET_TYPE and self.currency == STOCK_CURRENCY and self.exchange in US_EXCHANGES

    @property
    def dated(self) -> bool:
        return self.start is not None and self.end is not None

    def covers(self, day: date) -> bool:
        """A listing without dates never matches."""
        return self.dated and self.start <= day <= self.end

    def describe(self) -> str:
        where = self.exchange or "no exchange"
        return f"{self.start}..{self.end} {where}" if self.dated else f"no dates {where}"


def read_tiingo(lines: Iterable[str]) -> list[Listing]:
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or any(c not in reader.fieldnames for c in TIINGO_COLUMNS):
        raise CoverageError(f"Tiingo file: expected the columns {', '.join(TIINGO_COLUMNS)}; found {reader.fieldnames}")
    out = []
    for n, rec in enumerate(reader, start=2):
        if any(rec[c] is None for c in TIINGO_COLUMNS):
            raise CoverageError(f"Tiingo file, line {n}: fewer columns than the header")
        if rec["exchange"] not in US_EXCHANGES and rec["exchange"] not in NON_US_EXCHANGES:
            raise CoverageError(
                f"Tiingo file, line {n}: exchange {rec['exchange']!r} is in neither list of Rule UC's details; "
                "the rule has to say whether it counts before the check can run"
            )
        start, end = ((_parse_date(rec[c].strip(), f"Tiingo file, line {n}") if rec[c].strip() else None) for c in ("startDate", "endDate"))
        out.append(Listing(normalize_ticker(rec["ticker"]), rec["exchange"], rec["assetType"], rec["priceCurrency"], start, end))
    return out


def by_ticker(listings: Iterable[Listing]) -> dict[str, list[Listing]]:
    out: dict[str, list[Listing]] = defaultdict(list)
    for row in listings:
        out[row.ticker].append(row)
    return dict(out)


# The match ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TickerCoverage:
    ticker: str
    days: tuple[date, ...]  # its member-days in the window, in order
    matched: tuple[bool, ...]  # per member-day: a stock row of this ticker covers the day (the rule)
    has_stock_row: bool  # a stock row of this ticker exists, whatever its dates (an extra, not the rule)
    has_any_row: bool  # any row of this ticker exists, whatever its kind and dates (an extra, not the rule)

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def n_matched(self) -> int:
        return sum(self.matched)

    @property
    def n_unmatched(self) -> int:
        return self.n_days - self.n_matched

    @property
    def unmatched_days(self) -> list[date]:
        return [d for d, m in zip(self.days, self.matched) if not m]


def match(members: dict[str, list[date]], listings: dict[str, list[Listing]]) -> list[TickerCoverage]:
    """A member-day is matched if at least one stock row of the same ticker has start <= day <= end."""
    out = []
    for ticker in sorted(members):
        stock = [row for row in listings.get(ticker, []) if row.is_stock]
        spans = [(row.start, row.end) for row in stock if row.dated]
        matched = tuple(any(start <= day <= end for start, end in spans) for day in members[ticker])
        out.append(TickerCoverage(ticker, tuple(members[ticker]), matched, bool(stock), ticker in listings))
    return out


def passes(matched: int, total: int) -> bool:
    if total <= 0:
        raise CoverageError("no member-days in the window, so the bar cannot be applied")
    return matched * 1000 >= total * BAR_PER_MILLE


class Counts(NamedTuple):
    """Member-days. Only `matched` and `total` enter the rule; the two in between are reported as extras."""

    matched: int  # a stock row of the ticker covers the day
    ticker_only: int  # a stock row of the ticker exists, whatever its dates
    any_row: int  # any row of the ticker exists, whatever its kind, currency, exchange and dates
    total: int


def totals(coverage: Iterable[TickerCoverage]) -> Counts:
    matched = ticker_only = any_row = total = 0
    for tc in coverage:
        matched += tc.n_matched
        ticker_only += tc.n_days if tc.has_stock_row else 0
        any_row += tc.n_days if tc.has_any_row else 0
        total += tc.n_days
    return Counts(matched, ticker_only, any_row, total)


def yearly(coverage: Iterable[TickerCoverage]) -> dict[int, Counts]:
    """Member-days by the calendar year of the day."""
    out: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for tc in coverage:
        for day, m in zip(tc.days, tc.matched):
            row = out[day.year]
            row[0] += m
            row[1] += tc.has_stock_row
            row[2] += tc.has_any_row
            row[3] += 1
    return {year: Counts(*out[year]) for year in sorted(out)}


# Why a ticker is unmatched: a description worked out from the two files, not part of the rule ------------------
NOT_LISTED, REUSED, MISSING_DATES, CLASS_SHARE = "not listed", "reused ticker", "missing dates", "class share"
REASON_KINDS = {
    NOT_LISTED: "Tiingo's list has no row under this ticker. The company may trade under another ticker now (Tiingo files "
    "a company under one ticker, so a rename hides the old one), or Tiingo may not cover it.",
    REUSED: "Tiingo has rows under this ticker, but none is a US stock whose dates touch the membership dates: a fund, or "
    "a stock listed only before or after. The letters belong to something else in Tiingo's list.",
    MISSING_DATES: "Tiingo lists the ticker as a US stock, but its first and last date cover only part of the membership "
    "dates (or it has no dates).",
    CLASS_SHARE: "The ticker has a share-class suffix and Tiingo's list has no ticker written that way.",
}


class Reason(NamedTuple):
    kind: str  # one of REASON_KINDS
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def likely_reason(tc: TickerCoverage, listings: dict[str, list[Listing]], left: tuple[date, tuple[str, ...]] | None) -> Reason:
    """`left` is the ticker's entry in `departures`, or None if it is still a member at the end of the window."""
    rows = listings.get(tc.ticker, [])
    stock = [row for row in rows if row.is_stock]
    dated = sorted((row for row in stock if row.dated), key=lambda row: (row.start, row.end))
    if dated:
        spans = ", ".join(f"{row.start}..{row.end}" for row in dated)
        if any(row.start <= tc.days[-1] and row.end >= tc.days[0] for row in dated):
            kind, detail = MISSING_DATES, f"Tiingo lists it for {spans} only"
        else:
            kind, detail = REUSED, f"Tiingo lists it for {spans} only, outside the membership dates"
    elif stock:
        kind, detail = MISSING_DATES, "Tiingo lists it as a stock without dates"
    elif rows:
        kinds = ", ".join(sorted({f"{row.asset_type} ({row.currency}, {row.exchange or 'no exchange'})" for row in rows}))
        kind, detail = REUSED, f"Tiingo lists it only as {kinds}"
    elif "-" in tc.ticker:
        variants = [v for v in (tc.ticker.replace("-", ""), tc.ticker.split("-")[0]) if any(r.is_stock for r in listings.get(v, []))]
        kind, detail = CLASS_SHARE, "no row under this ticker" + (f" (Tiingo does list {' and '.join(variants)})" if variants else "")
    else:
        kind, detail = NOT_LISTED, "no row under this ticker"
    if left is None:
        return Reason(kind, detail + "; still a member at the end of the window")
    day, joined = left
    if not joined:
        return Reason(kind, detail + f"; left the index on {day}")
    names = ", ".join(joined[:4]) + (" and others" if len(joined) > 4 else "")
    return Reason(kind, detail + f"; left the index on {day}, when {names} joined (possible rename)")


# The report ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Result:
    start: date
    end: date
    n_days: int
    coverage: list[TickerCoverage]  # one entry per ticker with a member-day in the window, sorted by ticker
    leavers: frozenset[str]  # tickers with a member-day in the window that are not members on `end`
    reasons: dict[str, Reason]  # ticker with an unmatched member-day -> likely reason
    stock_rows: dict[str, str]  # the same tickers -> their Tiingo stock rows, described

    @property
    def unmatched(self) -> list[TickerCoverage]:
        """Tickers with at least one unmatched member-day, most unmatched member-days first."""
        return sorted((tc for tc in self.coverage if tc.n_unmatched), key=lambda tc: (-tc.n_unmatched, tc.ticker))

    @property
    def verdict(self) -> bool:
        counts = totals(self.coverage)
        return passes(counts.matched, counts.total)


def check(rows: Membership, listings: list[Listing], start: date = WINDOW_START, end: date = WINDOW_END) -> Result:
    days = weekdays(start, end)
    index = by_ticker(listings)
    coverage = match(member_days(rows, days), index)
    final = members_on(rows, end)
    left = departures(rows)
    unmatched = [tc for tc in coverage if tc.n_unmatched]
    return Result(
        start=start,
        end=end,
        n_days=len(days),
        coverage=coverage,
        leavers=frozenset(tc.ticker for tc in coverage if tc.ticker not in final),
        reasons={tc.ticker: likely_reason(tc, index, left.get(tc.ticker)) for tc in unmatched},
        stock_rows={tc.ticker: "; ".join(row.describe() for row in index.get(tc.ticker, []) if row.is_stock) for tc in unmatched},
    )


def pct(part: int, whole: int) -> str:
    """A share as a percentage, rounded down to 2 decimals so that a printed share never overstates the count."""
    return f"{part * 10000 // whole / 100:.2f}%" if whole else "n/a"


def verdict_line(result: Result) -> str:
    counts = totals(result.coverage)
    call = "PASS" if result.verdict else "FAIL"
    return (
        f"{call}: {pct(counts.matched, counts.total)} of member-days are matched "
        f"({counts.matched:,} of {counts.total:,}); the bar is {BAR_PER_MILLE / 10:.1f}%."
    )


def render_report(result: Result) -> str:
    cov = result.coverage
    matched, ticker_only, any_row, total = totals(cov)
    gone = [tc for tc in cov if tc.ticker in result.leavers]
    kept = [tc for tc in cov if tc.ticker not in result.leavers]
    g, k = totals(gone), totals(kept)
    unmatched = result.unmatched
    by_kind = {kind: [tc for tc in unmatched if result.reasons[tc.ticker].kind == kind] for kind in REASON_KINDS}
    after = result.end + timedelta(days=1)
    days_in_year: dict[int, int] = defaultdict(int)
    for day in weekdays(result.start, result.end):
        days_in_year[day.year] += 1

    lines = [
        "# Free US coverage check (Rule UC)",
        "",
        "Question: if the S&P 500 were built from free data (members from fja05680/sp500, prices from Tiingo), how much",
        "of its history could be priced? This check compares the member list with Tiingo's public list of supported",
        "tickers. It fetches no prices. The rule and its details were fixed in `docs/RULES.md` (Rule UC) before the",
        "match was first computed.",
        "",
        "## Verdict",
        "",
        f"**{verdict_line(result)}**",
        "",
    ]
    if result.verdict:
        lines += ["By Rule UC a pass makes the Tiingo price fetch a TODO item. The fetch is not started in this build."]
    else:
        lines += ["By Rule UC the free S&P 500 route is recorded as not good enough as it stands, and the owner is told."]
    lines += [
        "",
        "What a matched member-day shows: Tiingo's list has a US stock row under that ticker whose first and last date",
        "cover the day. What it does not show:",
        "",
        "- that Tiingo's free tier serves those prices. That stays unverified until a fetch;",
        "- that the row is the same company. A reused ticker whose dates happen to cover the day counts as matched;",
        "- that a price exists on every day in between. The list gives a first and a last date only;",
        "- that the member list is right. fja05680/sp500 is an unofficial list.",
        "",
        "## Numbers",
        "",
        f"- Window: {result.start} to {result.end}, every Monday to Friday ({result.n_days:,} days; weekday holidays count).",
        f"- Member-days: {total:,} ({total / result.n_days:.1f} members per day on average).",
        f"- Matched member-days (the rule): {matched:,} = {pct(matched, total)}. Unmatched: {total - matched:,}.",
        f"- Distinct tickers that were members in the window: {len(cov):,}. With at least one unmatched member-day: "
        f"{len(unmatched):,}, of which {sum(1 for tc in unmatched if not tc.n_matched):,} have no matched member-day at all.",
        f"- Not part of the rule, ticker only (a stock row exists under the ticker, whatever its dates): "
        f"{ticker_only:,} = {pct(ticker_only, total)}. The gap to the rule's number, {ticker_only - matched:,} member-days, "
        "is what the date ranges remove (reused tickers and missing dates).",
        f"- Not part of the rule, an upper bound for any reading of \"Tiingo lists that ticker as a stock\" (any row exists "
        f"under the ticker, whatever its asset type, currency, exchange and dates): {any_row:,} = {pct(any_row, total)}."
        + ("" if passes(any_row, total) else " This is below the bar too, so the call does not hang on how a stock row is told apart."),
        "",
        f"## Tickers that left the index before {after}",
        "",
        f"A ticker has left if it has a member-day in the window and is not in the member list of {result.end}. This is",
        "the survivorship question: free sources tend to drop companies that no longer trade.",
        "",
        md_table(
            ["Tickers", "Count", "With no matched day", "Member-days", "Matched", "Matched share (the rule)", "Ticker only (extra)"],
            [
                [f"Left before {after}", f"{len(gone):,}", f"{sum(1 for tc in gone if not tc.n_matched):,}", f"{g.total:,}",
                 f"{g.matched:,}", pct(g.matched, g.total), pct(g.ticker_only, g.total)],
                [f"Members on {result.end}", f"{len(kept):,}", f"{sum(1 for tc in kept if not tc.n_matched):,}", f"{k.total:,}",
                 f"{k.matched:,}", pct(k.matched, k.total), pct(k.ticker_only, k.total)],
            ],
        ),  # fmt: skip
        "",
        "## Per year",
        "",
        md_table(
            ["Year", "Weekdays", "Member-days", "Matched", "Matched share (the rule)", "Ticker only (extra)"],
            [[year, days_in_year[year], f"{c.total:,}", f"{c.matched:,}", pct(c.matched, c.total), pct(c.ticker_only, c.total)]
             for year, c in yearly(cov).items()],
        ),
        "",
        f"The last year in the table ends on {result.end}.",
        "",
        "## Unmatched member-days by likely reason",
        "",
        "The likely reason is worked out by code from the two files only, so it is a hint, not a finding. It is not",
        "part of the rule.",
        "",
        md_table(
            ["Likely reason", "Tickers", "Unmatched member-days", "Share of all member-days", "What it means"],
            [[kind, f"{len(tcs):,}", f"{sum(tc.n_unmatched for tc in tcs):,}", pct(sum(tc.n_unmatched for tc in tcs), total), meaning]
             for kind, meaning in REASON_KINDS.items() for tcs in [by_kind[kind]]],
        ),  # fmt: skip
        "",
        f"## The {min(TOP_UNMATCHED, len(unmatched))} tickers with the most unmatched member-days",
        "",
        f"The full list is in `{UNMATCHED_FILE}`. \"Possible rename\" means only that another ticker joined the index on",
        "the day this one left; the files do not say whether it is the same company.",
        "",
        md_table(
            ["Ticker", "Unmatched member-days", "Member-days", "First member day", "Last member day", "Likely reason"],
            [[tc.ticker, f"{tc.n_unmatched:,}", f"{tc.n_days:,}", tc.days[0], tc.days[-1], str(result.reasons[tc.ticker])]
             for tc in unmatched[:TOP_UNMATCHED]],
        ),  # fmt: skip
        "",
        "## Inputs",
        "",
        f"- Membership: fja05680/sp500 at commit `{PINNED_SP500_COMMIT}`, file `{PINNED_MEMBERSHIP_FILE}`, sha256",
        f"  `{PINNED_MEMBERSHIP_SHA256}`.",
        f"- Tiingo: `supported_tickers.zip` from `{TIINGO_URL}`,",
        f"  downloaded {PINNED_TIINGO_DOWNLOADED_AT}, sha256 `{PINNED_TIINGO_ZIP_SHA256}`.",
        "- Both stay outside the repo. `scripts/get_us_coverage_data.sh` fetches and verifies them and prints their paths;",
        "  `python -m detmaso.data.ingest.us_coverage --membership <csv> --tiingo <zip>` checks the hashes again and",
        "  writes this report.",
        "",
        "## How a member-day is matched",
        "",
        "- Members on a day: the list of the last fja05680 row dated on or before that day.",
        '- Tickers are compared in upper case with "." and "/" written as "-".',
        f"- A Tiingo row counts as a stock if its asset type is `{STOCK_ASSET_TYPE}`, its currency is `{STOCK_CURRENCY}` and its exchange is",
        f"  not outside the US ({', '.join(sorted(NON_US_EXCHANGES))}). Over-the-counter venues count.",
        "- The day must lie between the row's first and last date, both included. With several rows for one ticker,",
        "  one covering row is enough. A row without dates never matches.",
        "- Shares are rounded down to 2 decimals. The bar is applied to the exact counts.",
        "",
    ]
    return "\n".join(str(line) for line in lines)


def write_outputs(result: Result, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report, table = out_dir / REPORT_FILE, out_dir / UNMATCHED_FILE
    report.write_text(render_report(result), encoding="utf-8")
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "ticker", "unmatched_member_days", "member_days", "first_member_date", "last_member_date",
            "first_unmatched_date", "last_unmatched_date", "tiingo_stock_rows", "likely_reason", "detail",
        ])  # fmt: skip
        for tc in result.unmatched:
            missing = tc.unmatched_days
            writer.writerow([
                tc.ticker, tc.n_unmatched, tc.n_days, tc.days[0], tc.days[-1], missing[0], missing[-1],
                result.stock_rows[tc.ticker], *result.reasons[tc.ticker],
            ])  # fmt: skip
    return report, table


# Command line -------------------------------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins(membership: Path, tiingo_zip: Path) -> None:
    for path, pinned, name in ((membership, PINNED_MEMBERSHIP_SHA256, "membership file"), (tiingo_zip, PINNED_TIINGO_ZIP_SHA256, "Tiingo file")):
        found = sha256_of(path)
        if found != pinned:
            raise CoverageError(
                f"{name} is not the pinned one: sha256 {found}, pinned {pinned}. "
                "See scripts/get_us_coverage_data.sh for what to do; the check does not run on other inputs."
            )


def run_check(membership: Path, tiingo_zip: Path, out_dir: Path) -> Result:
    with membership.open(newline="", encoding="utf-8") as f:
        rows = read_membership(f)
    with zipfile.ZipFile(tiingo_zip) as z, z.open(TIINGO_CSV_NAME) as raw:
        listings = read_tiingo(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
    result = check(rows, listings)
    write_outputs(result, out_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Free US coverage check (docs/RULES.md, Rule UC). Fetches nothing.")
    parser.add_argument("--membership", type=Path, required=True, help=f"fja05680/sp500: '{PINNED_MEMBERSHIP_FILE}'")
    parser.add_argument("--tiingo", type=Path, required=True, help="Tiingo's supported_tickers.zip")
    parser.add_argument("--out-dir", type=Path, default=Path("results/sp500"))
    args = parser.parse_args(argv)
    try:
        verify_pins(args.membership, args.tiingo)
        result = run_check(args.membership, args.tiingo, args.out_dir)
    except CoverageError as e:
        print(f"no verdict: {e}", file=sys.stderr)
        return 2
    counts = totals(result.coverage)
    print(verdict_line(result))
    print(f"ticker only (extra, not the rule): {pct(counts.ticker_only, counts.total)}")
    print(f"written: {args.out_dir / REPORT_FILE}, {args.out_dir / UNMATCHED_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
