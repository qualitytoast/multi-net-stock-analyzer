"""South Korea: FinanceData/marcap daily snapshots -> a Qlib-format dataset (docs/RULES.md, Rules ND, KG and KU).

The source holds one row per stock per trading day since 1995-05-02: raw open, high, low and close, volume, the
change against the exchange's base price, market cap, and the market label. This module

- runs the go/no-go checks KG1-KG3 and writes the numbers behind them (``go-no-go``);
- builds the synthetic top-200 KOSPI universe of Rule KU, adjusts every stock that is ever picked with a factor
  chained from the exchange's base price (Rule KG2), builds the synthetic index series, and writes all of it in
  Qlib's binary layout with Qlib's own ``scripts/dump_bin.py`` (``convert``).

    python -m detmaso.data.ingest.kr go-no-go --source <marcap dir> --manifest scripts/kr_data.sha256 \
        --out results/kr200/go_no_go.md
    python -m detmaso.data.ingest.kr convert --source <marcap dir> --manifest scripts/kr_data.sha256 \
        --dump-bin <dump_bin.py> --qlib-dir ~/.qlib/qlib_data/kr_data_d3b7036 \
        --report results/kr200/conversion_report.md

``scripts/get_kr_data.sh`` fetches and verifies both inputs; the pins it and this module use are written once, in
``scripts/kr_data.pins``. The download is untrusted data: it is only ever read
as parquet, and its path is an argument. The output follows the China archive's conventions, so
``detmaso.data.qlib_io.load_raw_panel`` reads it unchanged:

- adjusted price = raw price x factor, adjusted volume = raw volume / factor (so volume x factor = raw volume);
- the factor is 1 on a stock's first row (the China archive instead scales the first close to 1; only ratios of
  adjusted values are ever used, so the level does not matter);
- a day without trading (zero volume, or no row that day) has no price and no volume (NaN); the factor is carried;
- membership is a Qlib instruments file of point-in-time spans; every stock that is ever picked keeps its whole
  price history, on every board, until its last row.

The rules, and the exact patterns and readings fixed before the universe was first computed, are in docs/RULES.md
(Rule KU, "Details"). This module applies them as written; change them there first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.data.ingest.pins import SCRIPTS_DIR, read_pins
from detmaso.mdtable import md_table

# Pins (scripts/kr_data.pins, the file scripts/get_kr_data.sh sources) ------------------------------------------
PINS = read_pins(SCRIPTS_DIR / "kr_data.pins")
PINNED_COMMIT = PINS["MARCAP_COMMIT"]  # FinanceData/marcap
DUMP_BIN_COMMIT = PINS["QLIB_COMMIT"]  # microsoft/qlib, scripts/dump_bin.py
DUMP_BIN_SHA256 = PINS["DUMP_BIN_SHA256"]

# Rule ND -----------------------------------------------------------------------------------------------------
CHECK_END = pd.Timestamp("2024-03-01")  # every statistic and check uses only dates before the locked test period

# Rule KU -----------------------------------------------------------------------------------------------------
TOP_N = 200
SELECTION_MONTHS = (5, 11)  # the last trading day of May and of November
WINDOW_MONTHS = 6  # the 6 calendar months ending on the selection date
COVERAGE = (4, 5)  # a market cap on at least 4/5 = 80% of the window's trading days (exact integer test)
KOSPI = "KOSPI"
INDEX_BASE = 1000.0
# A stock whose name on the selection date (spaces removed, upper case) matches any pattern is excluded.
EXCLUSION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("SPAC", r"스팩|SPAC|기업인수목적"),
    ("REIT or listed real-estate fund", r"(?<!메)리츠|REIT|부동산투자|코크렙|리얼티|^교보메리츠$|^유레스메리츠|^맥쿼리센트럴|^케이알제\d+호$|^트러스제\d+호$"),
    ("ETF or ETN", r"ETF|ETN|^(KODEX|TIGER|KOSEF|KINDEX|ARIRANG|KBSTAR|HANARO)"),
    ("ship fund", r"^(동북아|아시아|코리아|바다로|거북선|하이골드)\d+호$|^아시아퍼시픽\d+$"),
    ("infrastructure fund", r"인프라$"),
    ("resource, overseas or auction fund", r"유전|니켈\d+호$|베트남|경매|^한국패러랠$|^한국아태특별$"),
    ("mutual or restructuring fund", r"펀드|기금|^박현주|^플래티넘|^플러스\d+호$|^장인환\d+호$|^알바트로스$|^코스파이더$|^SEI에셋$|^트윈스비전일$|^엠피챌린지$|^마이다스성일$|^옵티멈전환일$"),
)  # fmt: skip
_EXCLUSIONS = tuple((label, re.compile(pattern)) for label, pattern in EXCLUSION_PATTERNS)

# Rule KG -----------------------------------------------------------------------------------------------------
MIN_DELISTED_CODES = 200
KNOWN_DELISTINGS = (("117930", "Hanjin Shipping", 2017), ("004940", "Korea Exchange Bank", 2013), ("000830", "Samsung C&T (old)", 2015))
MIN_KNOWN_DELISTINGS = 2
SAMSUNG = "005930"
SAMSUNG_SPLIT_RATIO = 50.0
SAMSUNG_BEFORE_MONTH = "2018-04"  # its last traded day in April 2018
SAMSUNG_AFTER_DAY = "2018-05-04"
OTHER_SPLITS = (("090430", "Amorepacific", 10.0, "2015-05"), ("035420", "Naver", 5.0, "2018-10"), ("035720", "Kakao", 5.0, "2021-04"))
MIN_OTHER_SPLITS = 2
SPLIT_TOLERANCE = 0.01
HALT_DAYS = ("2018-04-30", "2018-05-02", "2018-05-03")
PRICED_DAYS = ("2018-04-27", "2018-05-04")

# Output layout -----------------------------------------------------------------------------------------------
CODE_PREFIX = "KR"  # Qlib lower-cases feature folders, and a letter prefix keeps codes such as 005930 text
INDEX_SYMBOL = "KRTOP200SYN"  # longer than any stock symbol (KR + 6 characters), so it can never collide with one
UNIVERSE_NAME = "kr200"
FIELDS = ("open", "high", "low", "close", "volume", "factor")
PRICE_FIELDS = ("open", "high", "low", "close")
BUILD_INFO = "build_info.json"
NO_BASE_CHECK_FROM = pd.Timestamp("2007-01-01")  # the earliest date any config loads

CAP_COLUMNS = ("Code", "Name", "Market", "Date", "Close", "Marcap")
ROW_COLUMNS = ("Code", "Name", "Market", "Date", "Open", "High", "Low", "Close", "Volume", "Changes", "Marcap")
CHECK_COLUMNS = ("Code", "Market", "Date", "Volume")

# Report thresholds (descriptive only; they decide nothing)
STEP_EPS = 1e-9  # a factor step = a row whose factor differs from the previous row's by more than this (relative)
IMPLAUSIBLE_STEP = 100.0  # a single step above 100x or below 1/100 is listed as implausible


class KrError(ValueError):
    """The source or a tool is not the pinned one, or a rule cannot be applied to the data as written."""


# Source ------------------------------------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def yearly_files(source: Path) -> list[Path]:
    files = sorted((source / "data").glob("marcap-*.parquet"))
    if not files:
        raise KrError(f"no yearly files (data/marcap-*.parquet) under {source}")
    return files


def read_manifest(manifest: Path) -> dict[str, str]:
    """``shasum -a 256`` output: {relative file name: sha256}."""
    out = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest
    if not out:
        raise KrError(f"empty manifest: {manifest}")
    return out


def verify_source(source: Path, manifest: Path) -> str:
    """Every yearly file matches the manifest, and there is no other yearly file. Returns the manifest's own
    sha256, which is what ``data.data_sha256`` means for this dataset."""
    listed = read_manifest(manifest)
    present = {f"data/{p.name}" for p in yearly_files(source)}
    if present != set(listed):
        raise KrError(f"yearly files on disk differ from the manifest: {sorted(present ^ set(listed))}")
    for name, digest in sorted(listed.items()):
        if sha256_of(source / name) != digest:
            raise KrError(f"sha256 mismatch for {name}; re-run scripts/get_kr_data.sh")
    return sha256_of(manifest)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Codes padded to 6 characters (rows up to 2000-05-26 lost their leading zeros) and dates as plain days."""
    df = df.copy()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]").dt.normalize()
    for col in ("Name", "Market"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def is_common(code: pd.Series) -> pd.Series:
    """Common share = the padded code ends in 0 (Rule KU)."""
    return code.astype(str).str[-1] == "0"


def usable(df: pd.DataFrame) -> pd.Series:
    """A source row counts only if its close is a positive number."""
    return df["Close"] > 0


def read_source(source: Path, columns: tuple[str, ...], keep=None) -> pd.DataFrame:
    """The yearly files, standardized; ``keep(df) -> bool rows`` filters each year as it is read."""
    parts = []
    for path in yearly_files(source):
        df = standardize(pd.read_parquet(path, columns=list(columns)))
        parts.append(df if keep is None else df[keep(df)])
    return pd.concat(parts, ignore_index=True)


def source_calendar(source: Path) -> pd.DatetimeIndex:
    """Trading days = the dates present in the source (any row, any board)."""
    days: set = set()
    for path in yearly_files(source):
        days.update(pd.to_datetime(pd.read_parquet(path, columns=["Date"])["Date"]).astype("datetime64[ns]").dt.normalize().unique())
    return pd.DatetimeIndex(sorted(days))


def common_rows(df: pd.DataFrame) -> pd.Series:
    return is_common(df["Code"]) & usable(df)


def _check_unique(df: pd.DataFrame) -> None:
    dup = df.duplicated(["Code", "Date"])
    if dup.any():
        raise KrError(f"{int(dup.sum())} duplicate (code, date) rows in the source, e.g. {df.loc[dup, ['Code', 'Date']].head(3).to_dict('records')}")


# Universe (Rule KU) ------------------------------------------------------------------------------------------
def clean_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name)).upper()


def exclusion(name: str) -> str | None:
    """The kind of listed fund this name matches (None = an ordinary company name)."""
    cleaned = clean_name(name)
    for label, pattern in _EXCLUSIONS:
        if pattern.search(cleaned):
            return label
    return None


@dataclass(frozen=True)
class Selection:
    date: pd.Timestamp  # the last trading day of May or November
    effective: pd.Timestamp  # the next trading day: the first trading day of June or December
    window_start: pd.Timestamp  # the first day of the 6 calendar months ending on the selection date


def window_start(selection_date: pd.Timestamp) -> pd.Timestamp:
    """December to May, or June to November: the first day of the month five months before the selection month."""
    return (selection_date.to_period("M") - (WINDOW_MONTHS - 1)).start_time


def selection_schedule(calendar: pd.DatetimeIndex) -> list[Selection]:
    """Selection dates whose month is complete (a trading day of a later month follows) and whose whole window
    lies inside the source."""
    out = []
    for day, nxt in zip(calendar[:-1], calendar[1:], strict=True):
        if day.month in SELECTION_MONTHS and (nxt.year, nxt.month) != (day.year, day.month):
            start = window_start(day)
            if start >= calendar[0]:
                out.append(Selection(day, nxt, start))
    return out


def pick(caps: pd.DataFrame, calendar: pd.DatetimeIndex, selection: Selection, top_n: int = TOP_N) -> pd.DataFrame:
    """The picks of one selection date, largest first. Reads only rows dated inside the window, which ends on the
    selection date, so later data cannot change the result.

    ``caps``: usable rows of codes ending in 0, on any board (Code, Name, Market, Date, Marcap).
    """
    in_window = (caps["Date"] >= selection.window_start) & (caps["Date"] <= selection.date)
    window = caps.loc[in_window, ["Code", "Name", "Market", "Date", "Marcap"]]
    n_days = int(((calendar >= selection.window_start) & (calendar <= selection.date)).sum())
    with_cap = window[window["Marcap"] > 0]
    stats = with_cap.groupby(with_cap["Code"].astype(str), sort=True)["Marcap"].agg(avg_cap="mean", cap_days="count")
    stats = stats[stats["cap_days"] * COVERAGE[1] >= n_days * COVERAGE[0]]
    today = window[window["Date"] == selection.date].copy()
    today["Code"] = today["Code"].astype(str)
    today = today[(today["Market"].astype(str) == KOSPI) & is_common(today["Code"])]
    today = today[[exclusion(name) is None for name in today["Name"]]].set_index("Code")
    ranked = stats.join(today[["Name"]], how="inner").reset_index(names="Code")
    ranked = ranked.sort_values(["avg_cap", "Code"], ascending=[False, True], kind="mergesort").head(top_n)
    ranked["window_days"] = n_days
    return ranked.reset_index(drop=True)


Spans = dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]


def membership_spans(schedule: list[Selection], picks: dict[pd.Timestamp, list[str]], calendar: pd.DatetimeIndex, last_row: pd.Series) -> Spans:
    """Picks are members from their effective date until the day before the next effective date, or until their
    last row if that comes first. Back-to-back spans of a stock that is picked again are joined."""
    spans: Spans = {}
    for k, sel in enumerate(schedule):
        start = sel.effective
        end = calendar[calendar.get_loc(schedule[k + 1].effective) - 1] if k + 1 < len(schedule) else calendar[-1]
        day_before = calendar[calendar.get_loc(start) - 1]
        for code in picks[sel.date]:
            last = min(end, last_row[code])
            if last < start:
                continue  # no row on or after the effective date: never becomes a member
            mine = spans.setdefault(code, [])
            if mine and mine[-1][1] == day_before:
                mine[-1] = (mine[-1][0], last)
            else:
                mine.append((start, last))
    return spans


def membership_frame(spans: Spans, calendar: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    member = pd.DataFrame(False, index=calendar, columns=codes)
    for code, periods in spans.items():
        for start, end in periods:
            member.loc[start:end, code] = True
    return member


@dataclass
class Universe:
    schedule: list[Selection]
    ranked: dict[pd.Timestamp, pd.DataFrame]  # selection date -> its picks, largest first
    spans: Spans
    last_row: pd.Series  # code -> its last usable row in the source

    @property
    def picked(self) -> list[str]:
        """Every code that is ever picked (a superset of the codes that ever become members)."""
        return sorted({code for frame in self.ranked.values() for code in frame["Code"]})


def build_universe(caps: pd.DataFrame, calendar: pd.DatetimeIndex, top_n: int = TOP_N) -> Universe:
    """``caps``: usable rows of every code ending in 0, on any board (Code, Name, Market, Date, Marcap)."""
    schedule = selection_schedule(calendar)
    if not schedule:
        raise KrError("the source holds no complete selection window")
    ranked = {sel.date: pick(caps, calendar, sel, top_n) for sel in schedule}
    last_row = caps.groupby(caps["Code"].astype(str))["Date"].max()
    spans = membership_spans(schedule, {day: list(frame["Code"]) for day, frame in ranked.items()}, calendar, last_row)
    return Universe(schedule, ranked, spans, last_row)


# Prices (Rule KG2) -------------------------------------------------------------------------------------------
def adjust(rows: pd.DataFrame) -> pd.DataFrame:
    """Adjusted fields for usable rows (any board), one stock after another.

    Factor: 1 on a stock's first row; on every later row it is multiplied by (previous row's close / base price),
    base price = close - change. A row is a source row with a positive close, traded or not. On a row without a
    base price (close - change is not a positive number) the factor is carried over unchanged and the row is
    flagged ``no_base``. A row with zero volume has no price and no volume; a traded row keeps its close, and an
    open, high or low that is zero in the source is missing.

    ``reopened`` flags a factor step on the first traded row after a halt whose base price equals its opening
    price: the exchange set the base price at the opening auction (a re-listing after a spin-off, a capital
    reduction or a long halt), so the move from the last close to that opening price is inside the step and not
    in the returns. A plain split whose first trade is at the base price is flagged too. Report only.
    """
    rows = rows.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    _check_unique(rows)
    code = rows["Code"].astype(str)
    first = code.ne(code.shift(1)).to_numpy()
    close = rows["Close"].to_numpy(dtype="float64")
    prev_close = np.concatenate(([np.nan], close[:-1]))
    base = close - rows["Changes"].to_numpy(dtype="float64")
    has_base = base > 0  # False for a missing change
    step = np.ones(len(rows))
    chained = ~first & has_base
    step[chained] = prev_close[chained] / base[chained]
    factor = pd.Series(step).groupby(code.to_numpy(), sort=False).cumprod().to_numpy()
    volume = rows["Volume"].to_numpy(dtype="float64")
    traded = volume > 0
    after_halt = ~first & (np.concatenate(([np.nan], volume[:-1])) == 0)
    out = pd.DataFrame({"Code": code, "Date": rows["Date"], "Name": rows["Name"].astype(str)})
    for name, col in zip(PRICE_FIELDS, ("Open", "High", "Low", "Close"), strict=True):
        raw = rows[col].to_numpy(dtype="float64")
        out[name] = np.where(traded & (raw > 0), raw * factor, np.nan)
    out["volume"] = np.where(traded, volume / factor, np.nan)
    out["factor"] = factor
    cap = rows["Marcap"].to_numpy(dtype="float64")
    out["marcap"] = np.where(cap > 0, cap, np.nan)
    out["step"] = step
    out["no_base"] = ~first & ~has_base
    out["reopened"] = chained & traded & after_halt & (base == rows["Open"].to_numpy(dtype="float64")) & (np.abs(step - 1.0) > STEP_EPS)
    return out


def stock_frame(adjusted: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One stock's fields on every trading day from its first to its last row; a day without a row has no price
    and carries the factor."""
    days = calendar[calendar.get_loc(adjusted["Date"].iloc[0]) : calendar.get_loc(adjusted["Date"].iloc[-1]) + 1]
    frame = adjusted.set_index("Date")[list(FIELDS)].reindex(days)
    frame["factor"] = frame["factor"].ffill()
    return frame


def wide(adjusted: pd.DataFrame, value: str, calendar: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    return adjusted.pivot(index="Date", columns="Code", values=value).reindex(index=calendar, columns=codes)


def index_series(close: pd.DataFrame, cap: pd.DataFrame, member: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """The synthetic index: 1000 on ``start`` (the first selection date), then chained with the members'
    market-cap-weighted daily return. Weight = market cap on the stock's latest earlier row; return = adjusted
    close over the previous traded adjusted close, minus 1 (0 on a day the stock does not trade)."""
    prev_close = close.ffill().shift(1)
    ret = (close / prev_close - 1.0).where(close.notna() & prev_close.notna(), 0.0)
    weight = cap.ffill().shift(1).where(member)
    total = weight.sum(axis=1)
    day = ((weight * ret).sum(axis=1) / total.where(total > 0)).fillna(0.0)
    level = INDEX_BASE * (1.0 + day.loc[start:]).cumprod()
    level.iloc[0] = INDEX_BASE
    return level.reindex(close.index)


@dataclass
class Converted:
    calendar: pd.DatetimeIndex
    universe: Universe
    adjusted: pd.DataFrame  # every usable row of every picked stock, with its adjusted fields
    frames: dict[str, pd.DataFrame]  # code -> fields on its trading-day range
    member: pd.DataFrame  # (calendar x picked codes) bool
    index: pd.Series  # the synthetic index level on the calendar (NaN before the first selection date)


def convert(rows: pd.DataFrame, calendar: pd.DatetimeIndex, universe: Universe) -> Converted:
    """``rows``: usable source rows, with prices, of at least every picked code (extra codes are ignored)."""
    codes = universe.picked
    rows = rows[rows["Code"].astype(str).isin(codes)]
    missing = sorted(set(codes) - set(rows["Code"].astype(str)))
    if missing:
        raise KrError(f"no price rows for picked codes: {missing[:5]}")
    adjusted = adjust(rows)
    frames = {code: stock_frame(part, calendar) for code, part in adjusted.groupby("Code", sort=True)}
    member = membership_frame(universe.spans, calendar, codes)
    index = index_series(wide(adjusted, "close", calendar, codes), wide(adjusted, "marcap", calendar, codes), member, universe.schedule[0].date)
    return Converted(calendar, universe, adjusted, frames, member, index)


# Qlib output -------------------------------------------------------------------------------------------------
def symbol(code: str) -> str:
    return f"{CODE_PREFIX}{code}".upper()


def verify_dump_bin(dump_bin: Path) -> None:
    if not dump_bin.is_file() or sha256_of(dump_bin) != DUMP_BIN_SHA256:
        raise KrError(f"{dump_bin} is not Qlib's scripts/dump_bin.py at commit {DUMP_BIN_COMMIT}; run scripts/get_kr_data.sh")


def _index_frame(index: pd.Series) -> pd.DataFrame:
    """The index as an instrument on the whole calendar (so the dumped calendar is complete by construction):
    close = the level, factor = 1; open, high, low and volume are unknown."""
    frame = pd.DataFrame(np.nan, index=index.index, columns=list(FIELDS))
    frame["close"] = index
    frame["factor"] = np.where(index.notna(), 1.0, np.nan)
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.rename_axis("date").to_csv(path, float_format="%.17g", date_format="%Y-%m-%d")


def write_qlib(conv: Converted, qlib_dir: Path, dump_bin: Path, workers: int = 4) -> None:
    """Write ``qlib_dir`` (must not exist): calendars/day.txt, instruments/all.txt and kr200.txt, features/."""
    verify_dump_bin(dump_bin)
    if qlib_dir.exists():
        raise KrError(f"{qlib_dir} already exists")
    qlib_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kr_csv_", dir=qlib_dir.parent) as tmp:
        staging = Path(tmp)
        for code, frame in conv.frames.items():
            _write_csv(frame, staging / f"{symbol(code).lower()}.csv")
        _write_csv(_index_frame(conv.index), staging / f"{INDEX_SYMBOL.lower()}.csv")
        cmd = [
            sys.executable, "-I", str(dump_bin), "dump_all", "--data_path", str(staging), "--qlib_dir", str(qlib_dir),
            "--include_fields", ",".join(FIELDS), "--date_field_name", "date", "--file_suffix", ".csv",
            "--max_workers", str(workers),
        ]  # fmt: skip
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise KrError(f"dump_bin failed ({done.returncode}):\n{done.stderr[-2000:]}")
    days = (qlib_dir / "calendars" / "day.txt").read_text().split()
    if days != [d.strftime("%Y-%m-%d") for d in conv.calendar]:
        raise KrError("the dumped calendar is not the source's trading calendar")
    listed = {line.split("\t")[0] for line in (qlib_dir / "instruments" / "all.txt").read_text().splitlines()}
    expected = {symbol(code) for code in conv.frames} | {INDEX_SYMBOL}
    if listed != expected:
        raise KrError(f"dump_bin wrote other instruments than expected: {sorted(listed ^ expected)[:5]}")
    for code, frame in conv.frames.items():
        for name in FIELDS:
            size = (qlib_dir / "features" / symbol(code).lower() / f"{name}.day.bin").stat().st_size
            if size != 4 * (len(frame) + 1):
                raise KrError(f"{symbol(code)} {name}: {size} bytes on disk for {len(frame)} days")
    lines = [
        f"{symbol(code)}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}"
        for code in sorted(conv.universe.spans)
        for start, end in conv.universe.spans[code]
    ]
    (qlib_dir / "instruments" / f"{UNIVERSE_NAME}.txt").write_text("\n".join(lines) + "\n")


def tree_digest(root: Path) -> str:
    """One sha256 over every file's relative path and bytes (the build-info file itself left out)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != BUILD_INFO):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha256_of(path)))
    return digest.hexdigest()


def read_bin(qlib_dir: Path, sym: str, name: str) -> tuple[int, np.ndarray]:
    """One Qlib feature file: (start index in the calendar, float32 values)."""
    data = np.fromfile(qlib_dir / "features" / sym.lower() / f"{name}.day.bin", dtype="<f4")
    return int(data[0]), data[1:]


# Go/no-go (Rule KG) ------------------------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    passed: bool
    lines: list[str]


def check_kg1(meta: pd.DataFrame) -> Check:
    """Delisted stocks are present. ``meta``: every source row before the test period (Code, Market, Date, Volume)."""
    meta = meta[meta["Date"] < CHECK_END]
    last = meta.loc[meta.groupby("Code")["Date"].idxmax()]
    cutoff = CHECK_END - pd.DateOffset(years=1)
    ended = last[(last["Market"] == KOSPI) & (last["Date"] < cutoff)]
    n_ended = len(ended)
    n_common = int(is_common(ended["Code"]).sum())
    rows, found = [], 0
    for code, name, year in KNOWN_DELISTINGS:
        traded = meta[(meta["Code"] == code) & (meta["Volume"] > 0)]
        last_traded = traded["Date"].max() if len(traded) else None
        ok = last_traded is not None and last_traded.year == year
        found += ok
        rows.append([f"{name} ({code})", year, "absent" if last_traded is None else str(last_traded.date()), "yes" if ok else "no"])
    passed = n_ended >= MIN_DELISTED_CODES and found >= MIN_KNOWN_DELISTINGS
    lines = [
        f"- Rows used: dated before {CHECK_END.date()} ({len(meta):,} rows, {meta['Code'].nunique():,} codes).",
        f"- KOSPI codes whose last row is before {cutoff.date()} (more than one year before {CHECK_END.date()}): "
        f"**{n_ended}** (needed: at least {MIN_DELISTED_CODES}). A code counts as KOSPI if its last row is labelled KOSPI.",
        f"- Of these, codes ending in 0 (common shares): {n_common}. Either count clears the bar.",
        f"- Known delistings present with their last traded row in the stated year: **{found} of {len(KNOWN_DELISTINGS)}** "
        f"(needed: at least {MIN_KNOWN_DELISTINGS}).",
        "",
        md_table(["Stock", "Stated year", "Last traded row", "Matches"], rows),
    ]
    return Check("KG1, delisted stocks are present", passed, lines)


def _factor_ratio(adjusted: pd.DataFrame, before: pd.Timestamp, after: pd.Timestamp) -> float:
    by_day = adjusted.set_index("Date")["factor"]
    return float(by_day.loc[after] / by_day.loc[before])


def check_kg2(rows: pd.DataFrame) -> Check:
    """The split adjustment comes out right. ``rows``: usable rows of the four stocks before the test period."""
    rows = rows[rows["Date"] < CHECK_END]
    adjusted = adjust(rows)
    traded = adjusted["close"].notna()

    def within(ratio: float, target: float) -> bool:
        return abs(ratio / target - 1.0) <= SPLIT_TOLERANCE

    sam = adjusted[adjusted["Code"] == SAMSUNG]
    april = sam[(sam["Date"].dt.strftime("%Y-%m") == SAMSUNG_BEFORE_MONTH) & traded[sam.index]]
    after = pd.Timestamp(SAMSUNG_AFTER_DAY)
    sam_ok, sam_line = False, f"- Samsung Electronics ({SAMSUNG}): rows missing"
    if len(april) and (sam["Date"] == after).any():
        before = april["Date"].iloc[-1]
        ratio = _factor_ratio(sam, before, after)
        sam_ok = within(ratio, SAMSUNG_SPLIT_RATIO)
        sam_line = (
            f"- Samsung Electronics ({SAMSUNG}), 50-for-1: factor on {after.date()} / factor on {before.date()} (its last "
            f"traded day in April 2018) = **{ratio:.6f}** (needed: within {SPLIT_TOLERANCE:.0%} of {SAMSUNG_SPLIT_RATIO:g}). "
            f"{'Pass' if sam_ok else 'Fail'}."
        )
    table, n_ok = [], 0
    for code, name, target, month in OTHER_SPLITS:
        one = adjusted[adjusted["Code"] == code]
        in_month = one[one["Date"].dt.strftime("%Y-%m") == month]
        earlier = one[one["Date"] < pd.Timestamp(f"{month}-01")]
        if not len(in_month) or not len(earlier):
            table.append([f"{name} ({code})", f"{target:g}-for-1", month, "rows missing", "", "no"])
            continue
        ratio = _factor_ratio(one, earlier["Date"].iloc[-1], in_month["Date"].iloc[-1])
        biggest = in_month.loc[(in_month["step"] - 1.0).abs().idxmax()]
        ok = within(ratio, target)
        n_ok += ok
        table.append([f"{name} ({code})", f"{target:g}-for-1", month, f"{ratio:.6f}", f"{biggest['step']:.6f} on {biggest['Date'].date()}", "yes" if ok else "no"])
    passed = sam_ok and n_ok >= MIN_OTHER_SPLITS
    lines = [
        "- Factor: 1 on a stock's first row, then multiplied on every row by (previous row's close / base price), base "
        "price = close - change.",
        sam_line,
        f"- Other splits within {SPLIT_TOLERANCE:.0%} of their ratio in the stated month: **{n_ok} of {len(OTHER_SPLITS)}** "
        f"(needed: at least {MIN_OTHER_SPLITS}). Ratio = factor on the month's last row / factor on the last row before the month.",
        "",
        md_table(["Stock", "Split", "Month", "Factor ratio over the month", "Largest one-day step in the month", "Within 1%"], table),
    ]
    return Check("KG2, the split adjustment comes out right", passed, lines)


def check_kg3(rows: pd.DataFrame, calendar: pd.DatetimeIndex) -> Check:
    """Suspended days can be identified, in the converted series of Samsung Electronics."""
    rows = rows[(rows["Code"] == SAMSUNG) & (rows["Date"] < CHECK_END)]
    frame = stock_frame(adjust(rows), calendar[calendar < CHECK_END])
    table, ok = [], True
    for day, want_price in [(d, False) for d in HALT_DAYS] + [(d, True) for d in PRICED_DAYS]:
        ts = pd.Timestamp(day)
        in_frame = ts in frame.index
        has_price = bool(in_frame and frame.loc[ts, list(PRICE_FIELDS)].notna().all())
        no_price = bool(in_frame and frame.loc[ts, [*PRICE_FIELDS, "volume"]].isna().all())
        good = has_price if want_price else no_price
        ok &= good
        close = "missing" if not has_price else f"{frame.loc[ts, 'close'] / frame.loc[ts, 'factor']:,.0f} raw"
        table.append([day, "a price" if want_price else "no price", close if in_frame else "day not in the series", "yes" if good else "no"])
    lines = [
        f"- Converted series of Samsung Electronics ({SAMSUNG}): the same code path that writes the Qlib files.",
        "",
        md_table(["Day", "Expected", "Converted close", "As expected"], sorted(table)),
    ]
    return Check("KG3, suspended days can be identified", ok, lines)


def run_checks(source: Path) -> list[Check]:
    meta = read_source(source, CHECK_COLUMNS, keep=lambda df: df["Date"] < CHECK_END)
    codes = {SAMSUNG, *(code for code, *_ in OTHER_SPLITS)}
    rows = read_source(source, ROW_COLUMNS, keep=lambda df: df["Code"].isin(codes) & usable(df) & (df["Date"] < CHECK_END))
    calendar = pd.DatetimeIndex(sorted(meta["Date"].unique()))
    return [check_kg1(meta), check_kg2(rows), check_kg3(rows, calendar)]


def render_go_no_go(checks: list[Check], data_sha256: str) -> str:
    passed = all(c.passed for c in checks)
    lines = [
        "# South Korea go/no-go (docs/RULES.md, Rule KG)",
        "",
        f"- Source: FinanceData/marcap at commit `{PINNED_COMMIT}`; every yearly file matches `scripts/kr_data.sha256` "
        f"(sha256 of that manifest: `{data_sha256}`).",
        f"- Every number below uses only rows dated before {CHECK_END.date()} (Rule ND).",
        "- Written by `python -m detmaso.data.ingest.kr go-no-go`; the rule was fixed before the data was opened.",
        "",
        f"**Result: {'GO, all three checks pass' if passed else 'NO-GO, the Korea build stops here'}.**",
        "",
        md_table(["Check", "Result"], [[c.name, "PASS" if c.passed else "FAIL"] for c in checks]),
    ]
    for c in checks:
        lines += ["", f"## {c.name}: {'PASS' if c.passed else 'FAIL'}", "", *c.lines]
    lines += [
        "",
        "## Known gap",
        "",
        "Cash dividends are not in this source, so the factor does not include them (Rule KG). Share splits, reverse",
        "splits, rights issues and other changes that move the exchange's base price are in the factor.",
    ]
    return "\n".join(lines) + "\n"


# Conversion report (Rule ND: only dates before the locked test period) -------------------------------------------
def _pct(part: float, whole: float) -> str:
    return "n/a" if not whole else f"{part / whole:.2%}"


def render_conversion_report(conv: Converted, info: dict) -> str:
    cal = conv.calendar[conv.calendar < CHECK_END]
    uni = conv.universe
    member = conv.member.loc[cal]
    codes = list(member.columns)
    priced = wide(conv.adjusted, "close", conv.calendar, codes).loc[cal].notna()
    last_checked = cal[-1]
    schedule = [s for s in uni.schedule if s.date < CHECK_END]
    sizes = [len(uni.ranked[s.date]) for s in schedule]
    picked_before = sorted({c for s in schedule for c in uni.ranked[s.date]["Code"]})
    ended_early = [c for c in picked_before if uni.last_row[c] < last_checked]
    # A member whose last row falls inside one of its spans and before the source ends stopped while a member.
    stopped = {
        code: uni.last_row[code]
        for code, periods in uni.spans.items()
        if uni.last_row[code] < conv.calendar[-1] and any(start <= uni.last_row[code] <= end for start, end in periods)
    }
    year_rows = []
    for year in sorted(set(cal.year)):
        days = cal[cal.year == year]
        m = member.loc[days]
        per_day = m.sum(axis=1)
        if per_day.max() == 0:
            continue
        n_md = int(m.to_numpy().sum())
        n_priced = int((m & priced.loc[days]).to_numpy().sum())
        n_stopped = sum(1 for day in stopped.values() if day.year == year and day < CHECK_END)
        with_members = per_day[per_day > 0]
        year_rows.append([year, len(with_members), int(with_members.min()), f"{with_members.median():.0f}", int(per_day.max()), f"{n_md:,}", _pct(n_priced, n_md), n_stopped])

    adj = conv.adjusted[conv.adjusted["Date"] < CHECK_END]
    is_step = (adj["step"] - 1.0).abs() > STEP_EPS
    step_rows = []
    for year, part in adj.groupby(adj["Date"].dt.year):
        s = part.loc[is_step[part.index], "step"]
        step_rows.append([
            year, f"{len(part):,}", len(s), int((s > 1).sum()), int((s < 1).sum()),
            "" if s.empty else f"{s.max():.4g}", "" if s.empty else f"{s.min():.4g}",
            int(((s > IMPLAUSIBLE_STEP) | (s < 1 / IMPLAUSIBLE_STEP)).sum()), int(part["no_base"].sum()),
        ])  # fmt: skip
    steps = adj[is_step]
    extreme = pd.concat([steps.nlargest(10, "step"), steps.nsmallest(10, "step")])
    extreme_rows = [[symbol(r.Code), r.Name, str(r.Date.date()), f"{r.step:.6g}"] for r in extreme.itertuples()]
    implausible = steps[(steps["step"] > IMPLAUSIBLE_STEP) | (steps["step"] < 1 / IMPLAUSIBLE_STEP)]
    no_base = adj[adj["no_base"]]
    no_base_late = conv.adjusted[conv.adjusted["no_base"] & (conv.adjusted["Date"] >= NO_BASE_CHECK_FROM)]
    traded = adj["close"].notna()
    no_ohl = adj[traded & adj[["open", "high", "low"]].isna().any(axis=1)]
    on_member_day = member.to_numpy()[cal.get_indexer(adj["Date"]), member.columns.get_indexer(adj["Code"])]
    reopened = adj[adj["reopened"] & on_member_day]
    reopened_late = reopened[reopened["Date"] >= NO_BASE_CHECK_FROM]

    lines = [
        "# South Korea conversion report (docs/RULES.md, Rules ND, KG and KU)",
        "",
        f"- Source: FinanceData/marcap at commit `{info['source_commit']}` (manifest sha256 `{info['data_sha256']}`).",
        f"- Qlib files written with Qlib's own `scripts/dump_bin.py` at commit `{info['dump_bin_commit']}` "
        f"(sha256 `{info['dump_bin_sha256']}`); sha256 over the whole output tree: `{info['tree_sha256']}`.",
        f"- Calendar: {conv.calendar[0].date()} .. {conv.calendar[-1].date()} ({len(conv.calendar)} trading days). The conversion is "
        "mechanical and covers all dates.",
        f"- **Every statistic below uses only dates before {CHECK_END.date()}** (last date checked: {last_checked.date()}).",
        f"- Universe `{UNIVERSE_NAME}`; index series `{INDEX_SYMBOL}` (synthetic, {INDEX_BASE:g} on {uni.schedule[0].date.date()}); "
        f"stock symbols are `{CODE_PREFIX}` + the 6-character code.",
        "",
        "## Universe (Rule KU)",
        "",
        f"- Selection dates before {CHECK_END.date()}: {len(schedule)}, from {schedule[0].date.date()} to {schedule[-1].date.date()}; "
        f"picks per selection date: min {min(sizes)}, max {max(sizes)}.",
        f"- Stocks ever picked on those dates: **{len(picked_before)}**; of these, {len(ended_early)} have data that ends before "
        f"{last_checked.date()} (the survivorship check: stocks that were delisted or merged away are kept). The conversion "
        f"covers all dates and holds {len(conv.frames)} stocks, the picks of later selection dates included.",
        f"- Members that stopped for good while a member (last row inside a membership span), before {CHECK_END.date()}: "
        f"{sum(1 for d in stopped.values() if d < CHECK_END)}.",
        "",
        "Per year: members per day, the share of member-days that are priced (have a close that day), and the members",
        "whose last row falls in that year while they are members.",
        "",
        md_table(["Year", "Days with members", "Members min", "Members median", "Members max", "Member-days", "Priced share", "Members that stopped"], year_rows),
        "",
        "## Factor (Rule KG2)",
        "",
        f"Stocks ever picked ({adj['Code'].nunique()} with rows before {CHECK_END.date()}), all their rows on any board. A step is a row "
        "whose factor differs from the previous row's, because the exchange's base price differs from the previous close: "
        "above 1 for a split, bonus issue or rights issue (base price below the last close), below 1 for a reverse split or "
        "capital reduction. A re-listing after a spin-off, merger or long halt can step either way.",
        "",
        md_table(["Year", "Rows", "Steps", "Steps > 1", "Steps < 1", "Largest", "Smallest", f"Beyond {IMPLAUSIBLE_STEP:g}x", "Rows without a base price"], step_rows),
        "",
        f"- Steps beyond {IMPLAUSIBLE_STEP:g}x in either direction (listed as implausible): **{len(implausible)}**.",
        f"- Rows without a base price (factor carried over; docs/RULES.md, Rule KU details): {len(no_base)} before {CHECK_END.date()}, "
        f"{'none' if no_base.empty else f'from {no_base.Date.min().date()} to {no_base.Date.max().date()}'}; "
        f"on or after {NO_BASE_CHECK_FROM.date()} (all dates): **{len(no_base_late)}**.",
        f"- Traded rows with an open, high or low missing in the source (close and volume kept): {len(no_ohl)}.",
        f"- Steps on a member's first traded day after a halt whose base price equals its opening price (the exchange set the "
        f"base price at the opening auction; see the known gaps): {len(reopened)}, of which {len(reopened_late)} from "
        f"{NO_BASE_CHECK_FROM.date()} on. Plain splits whose first trade is at the base price are counted too.",
        "",
        "The 10 largest and 10 smallest steps:",
        "",
        md_table(["Stock", "Name", "Date", "Step"], extreme_rows),
        "",
        "## Known gaps",
        "",
        "- No cash dividends: the factor covers share changes only, so returns are price returns.",
        "- No delisting returns: a delisted stock's history simply ends on its last row.",
        "- Base prices set by an opening auction: when a stock resumes after a capital reduction, a spin-off or merger, or a",
        "  long halt, the exchange sets that day's base price at the opening price. The move from the last close to that",
        "  opening price is then part of the factor step, so it is missing from returns (example: KR042660 re-opened on",
        "  2017-10-30 at half its pre-halt price, a step of 2).",
        "- The index series is synthetic (the members' market-cap-weighted return), not the official KOSPI 200.",
    ]
    return "\n".join(lines) + "\n"


# Commands ----------------------------------------------------------------------------------------------------
def load_caps(source: Path) -> pd.DataFrame:
    caps = read_source(source, CAP_COLUMNS, keep=common_rows).drop(columns="Close")
    _check_unique(caps)
    for col in ("Code", "Name", "Market"):
        caps[col] = caps[col].astype("category")
    return caps.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)


def cmd_go_no_go(args: argparse.Namespace) -> int:
    data_sha256 = verify_source(args.source, args.manifest)
    checks = run_checks(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_go_no_go(checks, data_sha256))
    for c in checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name}")
    print(f"written: {args.out}")
    return 0 if all(c.passed for c in checks) else 1


def cmd_convert(args: argparse.Namespace) -> int:
    qlib_dir = args.qlib_dir.expanduser()
    if qlib_dir.exists() and not args.overwrite:
        raise KrError(f"{qlib_dir} already exists; pass --overwrite to rebuild it")
    data_sha256 = verify_source(args.source, args.manifest)
    verify_dump_bin(args.dump_bin)
    failed = [c.name for c in run_checks(args.source) if not c.passed]
    if failed:
        raise KrError(f"go/no-go failed ({'; '.join(failed)}): the Korea build stops (docs/RULES.md, Rule KG)")
    calendar = source_calendar(args.source)
    universe = build_universe(load_caps(args.source), calendar)
    picked = set(universe.picked)
    rows = read_source(args.source, ROW_COLUMNS, keep=lambda df: df["Code"].isin(picked) & usable(df))
    conv = convert(rows, calendar, universe)
    qlib_dir.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=f".{qlib_dir.name}.build.", dir=qlib_dir.parent))
    try:
        write_qlib(conv, build / "qlib", args.dump_bin)
        info = {
            "source": "FinanceData/marcap",
            "source_commit": PINNED_COMMIT,
            "data_sha256": data_sha256,
            "dump_bin_commit": DUMP_BIN_COMMIT,
            "dump_bin_sha256": DUMP_BIN_SHA256,
            "universe": UNIVERSE_NAME,
            "index_symbol": INDEX_SYMBOL,
            "calendar": [str(calendar[0].date()), str(calendar[-1].date()), len(calendar)],
            "selection_dates": len(universe.schedule),
            "stocks": len(conv.frames),
            "no_base_rows": int(conv.adjusted["no_base"].sum()),
            "no_base_rows_from_2007": int((conv.adjusted["no_base"] & (conv.adjusted["Date"] >= NO_BASE_CHECK_FROM)).sum()),
            "tree_sha256": tree_digest(build / "qlib"),
        }
        (build / "qlib" / BUILD_INFO).write_text(json.dumps(info, indent=2) + "\n")
        if qlib_dir.exists():
            shutil.rmtree(qlib_dir)
        (build / "qlib").rename(qlib_dir)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_conversion_report(conv, info))
        print(f"written: {args.report}")
    print(f"qlib dir: {qlib_dir} ({info['stocks']} stocks, tree sha256 {info['tree_sha256']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="South Korea dataset from FinanceData/marcap (docs/RULES.md, Rules ND, KG, KU).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("go-no-go", cmd_go_no_go), ("convert", cmd_convert)):
        p = sub.add_parser(name)
        p.add_argument("--source", type=Path, required=True, help="the marcap checkout at the pinned commit")
        p.add_argument("--manifest", type=Path, required=True, help="scripts/kr_data.sha256")
        p.set_defaults(fn=fn)
    sub.choices["go-no-go"].add_argument("--out", type=Path, required=True, help="results/kr200/go_no_go.md")
    conv = sub.choices["convert"]
    conv.add_argument("--dump-bin", type=Path, required=True, help="Qlib's scripts/dump_bin.py at the pinned commit")
    conv.add_argument("--qlib-dir", type=Path, required=True, help="output directory, e.g. ~/.qlib/qlib_data/kr_data_d3b7036")
    conv.add_argument("--report", type=Path, default=None, help="results/kr200/conversion_report.md")
    conv.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except KrError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
