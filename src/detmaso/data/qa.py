"""Data QA report (plan Section 5 step 13). Label and missing-value statistics use only dates before the test period; the spot prices are data-integrity checks.

With ``qa.membership_checks`` (docs/RULES.md, Rule ND: every new dataset) the report adds the membership,
pricing and survivorship checks, and nothing in it reads a price or a membership row of the test period: the
spot prices (dated inside the test period) are left out, and label counts stop at the last date whose 20-day
answer ends before the test period. With the flag off the report is exactly the one written for CSI300.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config
from detmaso.data.panel import PreparedData, artifacts_dir
from detmaso.losses.stats import daily_spearman
from detmaso.mdtable import md_frame

PAPER_COVERAGE = {  # Table 4: share of training days per regime
    "csi300": [35, 12, 32, 16, 5],
    "sp500": [40, 15, 28, 13, 4],
}


def write_data_report(data: PreparedData, cfg: Config) -> Path:
    p = data.panel
    nd = cfg.qa.membership_checks  # Rule ND: no price or membership row of the test period is read below
    pre_test = (p.dates < pd.Timestamp(cfg.splits.test[0]))
    if nd:
        stocks = f"stocks that are members on a model date before the test period: {int(data.member.loc[p.dates[pre_test]].to_numpy().any(axis=0).sum())}"
    else:
        stocks = f"stocks ever in the index: {len(p.code_list)}"
    lines = [f"# Data QA report: {cfg.market}", ""]
    lines += [
        f"- Data tag: `{cfg.data.data_tag}` (sha256 `{cfg.data.data_sha256}`)",
        f"- Calendar loaded: {data.raw.calendar[0].date()} .. {data.raw.calendar[-1].date()} ({len(data.raw.calendar)} days)",
        f"- Model dates: {p.dates[0].date()} .. {p.dates[-1].date()} ({len(p.dates)} days); {stocks}",
        "",
        "## Splits",
        "",
        "| Split | First | Last | Days |",
        "|---|---|---|---|",
    ]
    for name, idx in vars(data.splits).items():
        lines.append(f"| {name} | {idx[0].date()} | {idx[-1].date()} | {len(idx)} |")

    uni = p.universe.sum(1).numpy()[pre_test]
    # Rule ND: a label on one of the last `horizon` dates before the test period ends inside it, so those dates are left out.
    labelled = _answer_before_test(data, cfg) if nd else pre_test
    lab = p.label_ok.sum(1).numpy()[labelled]
    lines += [
        "",
        "## Cross-section size (train and validation dates only)",
        "",
        f"- Universe per day (members that traded the day before): min {uni.min()}, median {np.median(uni):.0f}, max {uni.max()}",
        f"- Stocks with a 20-day label per day: min {lab.min()}, median {np.median(lab):.0f}",
        f"- Smallest universe on {p.dates[pre_test][uni.argmin()].date()}",
    ]
    if nd:
        last = p.dates[labelled][-1].date()
        lines.append(f"- Label counts stop at {last}: the {cfg.features.horizon}-day answers of later dates end inside the test period")

    if nd:
        lines += [
            "",
            "## Spot prices",
            "",
            "Left out: this check compares three raw closes dated 2024-03-01, inside the locked test period, and every data",
            "check of a new dataset uses only earlier dates (docs/RULES.md, Rule ND).",
        ]
    else:
        close, factor = data.raw.fields["close"], data.raw.fields["factor"]
        spot = [("SH600519", "2024-03-01", 1685.06), ("SZ000001", "2024-03-01", 10.49), ("SH601318", "2024-03-01", 42.99)]
        lines += ["", "## Spot prices (raw close = $close / $factor)", "", "| Stock | Date | Ours | Expected |", "|---|---|---|---|"]
        for code, d, expected in spot:
            if code in close.columns:
                val = close.loc[d, code] / factor.loc[d, code]
                lines.append(f"| {code} | {d} | {val:.2f} | {expected} |")

    universe = data.masks["universe"]
    lines += ["", "## Missing values per feature (share of universe stock-days before the test period, before filling with 0)", ""]
    model_rows = universe.index.isin(p.dates[pre_test])
    uni_model = universe.loc[model_rows]
    nan_rates = {
        name: float((frame.loc[model_rows].isna() & uni_model).to_numpy().sum() / uni_model.to_numpy().sum())
        for name, frame in data.features_raw.items()
    }
    lines.append(", ".join(f"{k} {v:.1%}" for k, v in nan_rates.items()))

    reg = data.regimes
    lines += ["", "## Regimes", "", "Silhouette score by number of clusters (training period):", ""]
    lines.append(", ".join(f"K={k}: {v:.3f}" for k, v in reg.silhouette.items()))
    best_k = max(reg.silhouette, key=reg.silhouette.get)
    lines += [
        "",
        f"The silhouette-optimal count is **K = {best_k}**; per decision D2 the model uses n = {cfg.regimes.n_agents} regardless.",
        "",
    ]
    tr = p.date_pos(data.splits.train).numpy()
    cov = np.bincount(p.regime[tr].numpy(), minlength=len(reg.names)) / len(tr)
    paper = PAPER_COVERAGE.get(cfg.market)
    if paper is None:  # the paper reports no regime shares for this market
        lines += ["| Regime (agent) | Our coverage of training days |", "|---|---|"]
        lines += [f"| {i + 1} {name} | {cov[i]:.1%} |" for i, name in enumerate(reg.names)]
    else:
        lines += ["| Regime (agent) | Our coverage of training days | Paper (Table 4) |", "|---|---|---|"]
        lines += [f"| {i + 1} {name} | {cov[i]:.1%} | {paper[i]}% |" for i, name in enumerate(reg.names)]
    cent = pd.DataFrame(reg.centroids * reg.std + reg.mean, index=reg.names, columns=reg.state_names)
    lines += ["", "Centroids in raw units (daily returns):", "", md_frame(cent.round(4))]
    v_vol = (cent["sigma20"] + cent["sigma60"]) / 2
    flags = []
    if v_vol["Low-Vol Bear"] > v_vol["High-Vol Bull"]:
        flags.append("the cluster named Low-Vol Bear is more volatile than High-Vol Bull")
    if v_vol["Low-Vol Bull"] > v_vol["High-Vol Bull"]:
        flags.append("the cluster named Low-Vol Bull is more volatile than High-Vol Bull")
    lines += [
        "",
        "Naming-rule check (plan A16 is fixed in advance; names are labels only): "
        + ("; ".join(flags) + "." if flags else "names are consistent with the centroid volatilities."),
    ]

    # Pipeline sanity check (test 30): single-feature ICs on the TRAINING period only.
    lines += ["", "## Single-feature IC on the training period (pipeline sanity check)", ""]
    pos = p.date_pos(data.splits.train_fit)
    y = p.y[pos].numpy()
    m = p.label_ok[pos].numpy()
    rows = []
    for name in ("mom_20", "vol_20", "mom_5", "rturn_20", "logvol20"):
        j = p.feature_names.index(name)
        ic = daily_spearman(p.X[pos, :, j].numpy(), y, m)
        rows.append(f"| {name} | {np.nanmean(ic):+.4f} |")
    fit = data.splits.train_fit
    lines += [f"| Feature | Mean daily IC {fit[0].year}-{fit[-1].year} |", "|---|---|", *rows]
    if cfg.is_walkforward:
        lines += _per_year_history(data, cfg)
    if nd:
        lines += _membership_checks(data, cfg)

    path = artifacts_dir(cfg) / "data_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _answer_before_test(data: PreparedData, cfg: Config) -> np.ndarray:
    """[model dates] bool: the date lies before the test period and so does the day its ``horizon``-day answer ends."""
    cal = data.raw.calendar
    end = cal.get_indexer(data.panel.dates) + cfg.features.horizon
    ok = np.zeros(len(end), dtype=bool)
    inside = end < len(cal)
    ok[inside] = cal[end[inside]] < pd.Timestamp(cfg.splits.test[0])
    return ok


def _per_year_history(data: PreparedData, cfg: Config) -> list[str]:
    """Walk-forward configs: per-year coverage and data-error checks over the whole history (before the test period)."""
    from detmaso.data.returns import returns_one_day

    p = data.panel
    universe = data.masks["universe"]
    ret = returns_one_day(data.raw.fields["close"])
    # A one-day move above the market's largest price limit plus one point (China: 20% + 1 = 21%).
    beyond = round(cfg.trading.max_price_limit + 0.01, 10)
    labelled = p.dates[_answer_before_test(data, cfg)] if cfg.qa.membership_checks else None  # Rule ND
    rows = []
    pre_test = p.dates[p.dates < pd.Timestamp(cfg.splits.test[0])]
    for year in sorted(set(pre_test.year)):
        days = pre_test[pre_test.year == year]
        pos = p.date_pos(days).numpy()
        uni = p.universe[pos].sum(1).numpy()
        lab = p.label_ok[pos if labelled is None else p.date_pos(days[days.isin(labelled)]).numpy()].sum(1).numpy()
        u = universe.loc[days]
        missing = np.mean([float((frame.loc[days].isna() & u).to_numpy().sum()) for frame in data.features_raw.values()]) / max(1, u.to_numpy().sum())
        big = int((ret.loc[days].abs().where(u) > beyond).to_numpy().sum())
        lab_median = f"{np.median(lab):.0f}" if len(lab) else "-"
        rows.append(f"| {year} | {len(days)} | {uni.min()} | {np.median(uni):.0f} | {lab_median} | {missing:.2%} | {big} |")
    return [
        "",
        "## Per year (walk-forward history check)",
        "",
        "Universe = index members that traded the day before; missing = average share of universe stock-days with a missing",
        f"feature (before filling with 0); moves > {beyond:.0%} = universe stock-days whose one-day return exceeds every price limit",
        "(possible data errors, or days without a limit such as a stock's first trading days).",
        "",
        f"| Year | Days | Universe min | Universe median | Labelled median | Missing features | Moves > {beyond:.0%} |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]


# Rule ND: membership, pricing and survivorship ----------------------------------------------------------------
SPAN_BUCKETS = ((1, 5), (6, 20), (21, 60), (61, 250), (251, None))  # unpriced member spans, by length in trading days


def unpriced_spans(member: np.ndarray, priced: np.ndarray) -> np.ndarray:
    """Every run of consecutive rows on which a stock is a member without a close.

    member, priced: [T, N] bool. Returns [K, 3] int rows (stock column, first row, last row), longest first
    (ties: lower column, then earlier start). A run ends when the stock is priced again or stops being a member.
    """
    bad = member & ~priced
    T, N = bad.shape
    padded = np.zeros((T + 2, N), dtype=np.int8)
    padded[1:-1] = bad
    step = np.diff(padded, axis=0).T  # [N, T + 1]: +1 where a run starts, -1 on the row after its last row
    col, first = np.nonzero(step == 1)
    _, after = np.nonzero(step == -1)  # same order as the starts: runs alternate start / end within a column
    order = np.lexsort((first, col, -(after - first)))
    return np.stack([col, first, after - 1], axis=1)[order]


def membership_stats(member: pd.DataFrame, close: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Rule ND numbers over ``dates`` only (rows outside ``dates`` are never read).

    member-day = a stock on a date on which it is a member; priced = it has a close that date. A stock's data
    "ends early" if its last close on ``dates`` falls before the last of ``dates`` (or it has no close on them).
    """
    M = member.loc[dates].to_numpy(dtype=bool)
    P = close.loc[dates].notna().to_numpy()
    T = len(dates)
    ever, has_close = M.any(axis=0), P.any(axis=0)
    last = T - 1 - np.argmax(P[::-1], axis=0)  # row of the last close (meaningless where has_close is False)
    ended = ever & (~has_close | (last < T - 1))
    years = []
    for year in sorted(set(dates.year)):
        idx = np.flatnonzero(dates.year == year)
        m = M[idx]
        per_day = m.sum(axis=1)
        years.append({
            "year": int(year), "days": len(idx), "min": int(per_day.min()), "median": float(np.median(per_day)), "max": int(per_day.max()),
            "member_days": int(m.sum()), "priced": int((m & P[idx]).sum()),
            "ended_members": int((m.any(axis=0) & ended).sum()),
            "last_close_in_year": int((ended & has_close & (last >= idx[0]) & (last <= idx[-1])).sum()),
        })  # fmt: skip
    per_day = M.sum(axis=1)
    return {
        "years": years, "member_days": int(M.sum()), "priced": int((M & P).sum()), "stocks": int(ever.sum()), "ended": int(ended.sum()),
        "never_priced": int((ever & ~has_close).sum()), "min": int(per_day.min()), "median": float(np.median(per_day)), "max": int(per_day.max()),
        "spans": unpriced_spans(M, P),
    }  # fmt: skip


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.2%}" if whole else "-"


def _membership_checks(data: PreparedData, cfg: Config) -> list[str]:
    """Rule ND section: per-year members, priced share and survivorship, warm-up, and the longest unpriced spans."""
    p, close = data.panel, data.raw.fields["close"]
    dates = p.dates[p.dates < pd.Timestamp(cfg.splits.test[0])]  # the dates checked
    last = dates[-1].date()
    st = membership_stats(data.member, close, dates)
    lines = [
        "",
        "## Membership, pricing and survivorship (docs/RULES.md, Rule ND)",
        "",
        f"Dates checked: the {len(dates)} model dates before the test period, {dates[0].date()} .. {last}; no later date is read.",
        "",
        f"- Member = in the `{cfg.data.universe}` list on that day" + (", after the membership filters below." if _filtered(cfg) else "."),
        "- Member-day = one stock on one date on which it is a member. Priced = it has a close on that date.",
        f"- A stock's data ends early if its last close on the dates checked falls before {last} (the last date checked), or it",
        "  has no close on them at all. This is the survivorship check: a dataset that kept only the stocks still trading",
        f"  would show 0 everywhere. A stock that is only suspended on {last} is counted too (nothing later is read).",
        "- Ended members = stocks that are members on at least one date of the year and whose data ends early.",
        "- Last close in year = stocks whose data ends early with their last close in that year.",
        "",
        "| Year | Days | Members min | Members median | Members max | Member-days | Priced | Priced share | Ended members | Last close in year |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for y in st["years"]:
        lines.append(
            f"| {y['year']} | {y['days']} | {y['min']} | {y['median']:g} | {y['max']} | {y['member_days']} | {y['priced']} | "
            f"{_share(y['priced'], y['member_days'])} | {y['ended_members']} | {y['last_close_in_year']} |"
        )
    lines += [
        "",
        f"- All dates checked: members per day min {st['min']}, median {st['median']:g}, max {st['max']}; {st['member_days']} member-days, "
        f"{st['priced']} priced ({_share(st['priced'], st['member_days'])}), {st['member_days'] - st['priced']} unpriced.",
        f"- Survivorship: {st['stocks']} stocks are members on at least one date checked; the data of {st['ended']} of them ends early "
        f"({st['never_priced']} of those {st['ended']} with no close on any date checked).",
    ]
    lines += _filter_lines(data, cfg, dates, st["member_days"])
    lines += _warmup_lines(data, cfg, dates)

    spans = st["spans"]
    length = spans[:, 2] - spans[:, 1] + 1
    lines += [
        "",
        "### Unpriced member spans",
        "",
        "A span = consecutive dates checked on which a stock is a member without a close (it ends when the stock is priced",
        "again or leaves the list). Short spans are ordinary trading halts; long ones are long halts or holes in the data.",
        "",
        "| Span length (trading days) | Spans | Unpriced member-days |",
        "|---|---|---|",
    ]
    for lo, hi in SPAN_BUCKETS:
        k = (length >= lo) if hi is None else (length >= lo) & (length <= hi)
        lines.append(f"| {f'{lo} or more' if hi is None else f'{lo}-{hi}'} | {int(k.sum())} | {int(length[k].sum())} |")
    top = spans[: cfg.qa.worst_spans]
    lines += ["", f"The {len(top)} longest:", "", "| Stock | First unpriced member date | Last | Trading days | Note |", "|---|---|---|---|---|"]
    for col, first, end in top.tolist():
        notes = []
        if first == 0:
            notes.append("from the first date checked")
        if end == len(dates) - 1:
            notes.append("still unpriced on the last date checked")
        lines.append(f"| {close.columns[col]} | {dates[first].date()} | {dates[end].date()} | {end - first + 1} | {'; '.join(notes) or '-'} |")
    return lines


def _filtered(cfg: Config) -> bool:
    return bool(cfg.data.exclude_codes) or cfg.data.min_traded_days > 0


def _filter_lines(data: PreparedData, cfg: Config, dates: pd.DatetimeIndex, kept: int) -> list[str]:
    """What the membership filters (data.exclude_codes, data.min_traded_days) removed on the dates checked."""
    if not _filtered(cfg):
        return []
    listed = data.raw.member.loc[dates]
    out = [f"- Membership filters: the `{cfg.data.universe}` list has {int(listed.to_numpy().sum())} member-days on the dates checked; {kept} are kept."]
    if cfg.data.exclude_codes:
        gone = int(listed[list(cfg.data.exclude_codes)].to_numpy().sum())
        out.append(f"  - Never members (not stocks): {', '.join(cfg.data.exclude_codes)}; {gone} member-days removed.")
        listed = listed.drop(columns=list(cfg.data.exclude_codes))
    if cfg.data.min_traded_days > 0:
        n, total = cfg.data.min_traded_days, int(listed.to_numpy().sum())
        out.append(
            f"  - New-listing filter: a stock is a member only once it has {n} traded days in the loaded data up to the day before "
            f"(traded day {n + 1} on); {total - kept} member-days removed ({_share(total - kept, total)} of the stocks' listed member-days)."
        )
    return out


def _warmup_lines(data: PreparedData, cfg: Config, dates: pd.DatetimeIndex) -> list[str]:
    """Membership on the dates loaded before the first model date, and on the days the first market state reads."""
    from detmaso.data.returns import returns_one_day

    cal, close = data.raw.calendar, data.raw.fields["close"]
    first = int(cal.get_indexer(dates[:1])[0])
    warm = cal[:first]
    if len(warm) == 0:
        return ["- Warm-up: no date is loaded before the first model date."]
    per_day = data.member.loc[warm].sum(axis=1).to_numpy()
    w = max(cfg.regimes.windows)
    span = cal[max(0, first - w) : first]
    m = data.member.loc[span]
    n_span = m.sum(axis=1).to_numpy()
    n_ret = (returns_one_day(close.loc[cal[max(0, first - w - 1)] : span[-1]]).loc[span].notna() & m).sum(axis=1).to_numpy()
    return [
        f"- Warm-up (the {len(warm)} dates loaded before the first model date, {warm[0].date()} .. {warm[-1].date()}): members per day "
        f"min {per_day.min()}, median {np.median(per_day):g}, max {per_day.max()}.",
        f"- Market-state warm-up: the first model date ({dates[0].date()}) reads the member returns of the {len(span)} dates "
        f"{span[0].date()} .. {span[-1].date()} (the longest window is {w} days). Members per day on those dates: min {n_span.min()}, "
        f"median {np.median(n_span):g}, max {n_span.max()}; members with a one-day return: min {n_ret.min()}, median {np.median(n_ret):g}.",
    ]
