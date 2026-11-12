"""Walk-forward comparison of run groups, with the keep rule and the placebo rule applied by code
(`detmaso compare --split walkforward`; docs/RULES.md, Rules WK and WP).

Every run group must be a fully evaluated walk-forward group (every fold, every configured seed) of the same
data, fold layout, backtest and trading settings, and placebo setting. Gains are "variant minus reference":
positive means the variant did better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from detmaso.config import Config
from detmaso.eval import guard
from detmaso.eval.walkforward import run_deflation, run_summary
from detmaso.eval.wf_stats import AFTER_COSTS, N_BOOT, WFError, WFRun, bootstrap, load_wf_run, metric_values, pct, pct_range, seed_metrics, write_json
from detmaso.mdtable import md_table

RULES = ("keep", "placebo")


def check_comparable(runs: list[WFRun]) -> None:
    ref = runs[0]
    for r in runs[1:]:
        if r.cfg.placebo != ref.cfg.placebo:
            raise WFError(f"'{r.tag}' and '{ref.tag}' differ in placebo settings: placebo and real runs are never compared")
        if r.cfg.data_hash() != ref.cfg.data_hash():
            raise WFError(f"'{r.tag}' and '{ref.tag}' use different data (data hash mismatch)")
        for section in ("walkforward", "backtest", "trading"):
            if getattr(r.cfg, section) != getattr(ref.cfg, section):
                raise WFError(f"'{r.tag}' and '{ref.tag}' use different {section} settings")
        if r.seeds != ref.seeds:
            raise WFError(f"'{r.tag}' and '{ref.tag}' use different seeds ({r.seeds} vs {ref.seeds}); seed s is compared with seed s")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise WFError(f"variant names must be unique, got {names}")


def advantage(v: WFRun, ref: WFRun, sms: dict, boot: dict) -> dict:
    """Variant minus reference: yearly IC gains, their average, per-seed gains and the after-cost Sharpe difference."""
    d_year = boot[v.name]["ic_year"] - boot[ref.name]["ic_year"]  # [F, B]
    gv, gr = sms[v.name]["ic_year"], sms[ref.name]["ic_year"]  # [S, F]
    yearly = {}
    for i, y in enumerate(ref.years):
        lo, hi = pct_range(d_year[i])
        yearly[y] = {"gain": float(gv[:, i].mean() - gr[:, i].mean()), "low": lo, "high": hi}
    avg_lo, avg_hi = pct_range(d_year.mean(axis=0))
    sv, sr = metric_values(sms[v.name]["joined"][AFTER_COSTS], "sharpe"), metric_values(sms[ref.name]["joined"][AFTER_COSTS], "sharpe")
    s_lo, s_hi = pct_range(boot[v.name]["sharpe_net"] - boot[ref.name]["sharpe_net"])
    return {
        "yearly": yearly,
        "average": {"gain": float(np.mean([yearly[y]["gain"] for y in ref.years])), "low": avg_lo, "high": avg_hi},
        "seed_gains": {s: float((gv[k] - gr[k]).mean()) for k, s in enumerate(ref.seeds)},
        "sharpe": {"diff": float(sv.mean() - sr.mean()), "low": s_lo, "high": s_hi},
    }


def run_wf_compare(
    runs: list[tuple[Config, str, str]], reference: str | None, out_dir: Path, rule: str | None = None, candidate: str | None = None,
    neighbor: str | None = None, no_setting: bool = False, target_years: list[int] | None = None, ledger_path: Path | None = None,
    n_boot: int = N_BOOT,
) -> Path:  # fmt: skip
    if rule is not None and rule not in RULES:
        raise WFError(f"unknown rule {rule!r}; known: {RULES}")
    wruns = [load_wf_run(cfg, path, tag) for cfg, path, tag in runs]
    check_comparable(wruns)
    by_tag = {r.tag: r for r in wruns}
    if len(by_tag) != len(wruns):
        raise WFError("run group tags must be unique")
    if reference and reference not in by_tag:
        raise WFError(f"reference '{reference}' is not one of the compared run groups {list(by_tag)}")
    ref = by_tag[reference] if reference else wruns[0]
    boot = bootstrap(wruns, n_boot=n_boot)
    sms = {r.name: seed_metrics(r) for r in wruns}
    summaries = {r.name: run_summary(r, sms[r.name], boot[r.name]) for r in wruns}
    placebo = ref.cfg.is_placebo
    deflated = None
    if ledger_path is not None and ledger_path.exists() and not placebo:
        deflated = {r.name: run_deflation(r, sms[r.name], boot[r.name], ledger_path) for r in wruns}
    adv = {r.name: advantage(r, ref, sms, boot) for r in wruns if r is not ref}

    rules = None
    if rule == "keep":
        if candidate not in by_tag or by_tag[candidate] is ref:
            raise WFError("rule keep needs --candidate: the tag of a compared run group other than the reference")
        if neighbor is None and not no_setting:
            raise WFError("rule keep needs --neighbor TAG (the next setting over) or --no-tunable-setting")
        if neighbor is not None and no_setting:
            raise WFError("rule keep takes --neighbor TAG or --no-tunable-setting, not both")
        if neighbor is not None and (neighbor not in by_tag or by_tag[neighbor] in (ref, by_tag[candidate])):
            raise WFError("--neighbor must be the tag of another compared run group")
        cand = by_tag[candidate]
        a = adv[cand.name]
        nb_gain = adv[by_tag[neighbor].name]["average"]["gain"] if neighbor else None
        k = guard.keep_rule(a["yearly"], a["average"], a["seed_gains"], a["sharpe"], has_setting=not no_setting,
                            neighbor_gain=nb_gain, target_years=target_years)  # fmt: skip
        rules = {"rule": "keep", "candidate": cand.name, "reference": ref.name, "neighbor": by_tag[neighbor].name if neighbor else None, **k}
    elif rule == "placebo":
        if not placebo or len(wruns) != 2:
            raise WFError("rule placebo compares exactly two placebo run groups")
        a, b = wruns
        levels = {r.name: pct_range(boot[r.name]["ic_avg"]) for r in wruns}
        verdicts = {}
        for cand, base in ((a, b), (b, a)):
            x = advantage(cand, base, sms, boot)
            verdicts[f"{cand.name} vs {base.name}"] = guard.keep_rule(x["yearly"], x["average"], x["seed_gains"], x["sharpe"], has_setting=False)["verdict"]
        rules = {"rule": "placebo", **guard.rule_wp(levels, verdicts)}

    out_dir.mkdir(parents=True, exist_ok=True)
    from detmaso.eval.wf_plots import compare_figures

    figs = compare_figures(ref.years, {name: a["yearly"] for name, a in adv.items()}, out_dir / "figures")
    (out_dir / "comparison.md").write_text(_markdown(wruns, ref, summaries, adv, deflated, rules, figs, n_boot))
    write_json(out_dir / "comparison.json", {
        "split": "walkforward", "reference": {"variant": ref.name, "tag": ref.tag}, "placebo": placebo, "n_boot": n_boot,
        "runs": [{"variant": r.name, "tag": r.tag, "config": r.config_path, "config_hash": r.cfg.hash(), "data_hash": r.cfg.data_hash(),
                  "model_kind": r.cfg.model.kind, "seeds": r.seeds, "years": r.years} for r in wruns],
        "summaries": summaries, "advantage": adv, "deflated": deflated, "rules": rules,
    })  # fmt: skip
    return out_dir / "comparison.md"


def _yes(flag: bool) -> str:
    return "yes" if flag else "**no**"


def _markdown(runs: list[WFRun], ref: WFRun, summaries: dict, adv: dict, deflated: dict | None, rules: dict | None, figs: list[str], n_boot: int) -> str:
    r0 = runs[0]
    lines = [f"# Walk-forward comparison: {r0.cfg.market}", ""]
    lines += [
        f"- Reference: `{ref.name}` (run group `{ref.tag}`). Gain = variant minus reference, so a positive number means the variant did better.",
        f"- {len(ref.years)} test years ({ref.years[0]}-{ref.years[-1]}), {ref.n_days} test days from {ref.start.date()} to {ref.end.date()}; seeds {ref.seeds}. The locked test period is never used.",
        f"- 95% ranges: {n_boot:,} draws; days redrawn in circular 20-day blocks within each test year (shared by every run group), seeds redrawn per run group from its own random stream (docs/RULES.md, Rule WK).",
        "- \"After costs\" = realistic costs with blocked trades (Rule WT).",
    ]
    if ref.cfg.is_placebo:
        lines.append(f"- **Placebo**: every run group was trained and evaluated on shuffled answers (seed {ref.cfg.placebo.shuffle_seed}); there is nothing real to find.")
    lines += ["", "## Run groups", ""]
    rows = []
    for r in runs:
        try_no = "-" if deflated is None or deflated[r.name]["try_no"] is None else str(deflated[r.name]["try_no"])
        rows.append([f"`{r.name}`", f"`{r.config_path}`", f"`{r.tag}`", f"`{r.cfg.hash()}`", r.cfg.model.kind, try_no])
    lines.append(md_table(["Variant", "Config", "Run group", "Config hash", "Model", "Try #"], rows))

    lines += ["", "## Each run group (all test years joined; mean +- std over seeds [95% range])", ""]
    rows = []
    for r in runs:
        s = summaries[r.name]
        cells = [f"`{r.name}`",
                 f"{s['ic_avg']['mean']:.4f} +- {s['ic_avg']['std']:.4f} [{s['ic_avg']['range'][0]:+.4f}, {s['ic_avg']['range'][1]:+.4f}]",
                 f"{s['sharpe_gross']['mean']:.2f} [{s['sharpe_gross']['range'][0]:+.2f}, {s['sharpe_gross']['range'][1]:+.2f}]",
                 f"{s['sharpe_net']['mean']:.2f} +- {s['sharpe_net']['std']:.2f} [{s['sharpe_net']['range'][0]:+.2f}, {s['sharpe_net']['range'][1]:+.2f}]",
                 f"{s['ar_net']['mean']:.1%}", f"{s['turnover']['mean']:.1%}", f"{1e4 * s['ls_mean']['mean']:.1f}"]  # fmt: skip
        if deflated is not None:
            cells += [pct(deflated[r.name]["dsr"]), pct(deflated[r.name]["dic"])]
        rows.append(cells)
    head = ["Variant", "Average IC", "Sharpe, paper backtest", "Sharpe after costs", "Annual return after costs", "Turnover/day", "Long-short (bps/day)"]
    if deflated is not None:
        head += ["Deflated Sharpe", "Deflated IC"]
    lines.append(md_table(head, rows))
    if deflated is not None:
        d = next(iter(deflated.values()))
        lines += ["", f"Deflated numbers: the probability that the true value beats the best of {d['n_tries']} tries of pure luck (spread of results from {d['source']}). Above about 95% = likely real."]

    others = [r for r in runs if r is not ref]
    lines += ["", f"## Yearly IC gain over `{ref.name}` [95% range]", ""]
    rows = [[str(y)] + [f"{adv[r.name]['yearly'][y]['gain']:+.4f} [{adv[r.name]['yearly'][y]['low']:+.4f}, {adv[r.name]['yearly'][y]['high']:+.4f}]" for r in others]
            for y in ref.years]  # fmt: skip
    lines.append(md_table(["Year", *[f"`{r.name}`" for r in others]], rows))
    lines += ["", "## Gains in brief", ""]
    rows = []
    for r in others:
        a = adv[r.name]
        wins = sum(v["gain"] > 0 for v in a["yearly"].values())
        seed_wins = sum(g > 0 for g in a["seed_gains"].values())
        rows.append([f"`{r.name}`", f"{a['average']['gain']:+.4f} [{a['average']['low']:+.4f}, {a['average']['high']:+.4f}]", f"{wins} of {len(ref.years)}",
                     f"{seed_wins} of {len(a['seed_gains'])}", f"{a['sharpe']['diff']:+.2f} [{a['sharpe']['low']:+.2f}, {a['sharpe']['high']:+.2f}]"])  # fmt: skip
    lines.append(md_table(["Variant", "Average IC gain [95%]", "Years won", "Seeds gaining", "After-cost Sharpe difference [95%]"], rows))

    if rules is not None and rules["rule"] == "keep":
        k = rules
        k2 = k["k2_detail"]
        lines += ["", f"## Rule WK: `{k['candidate']}` vs `{k['reference']}` (docs/RULES.md)", ""]
        lines.append(f"- K1 average gain {k['average']['gain']:+.4f} [{k['average']['low']:+.4f}, {k['average']['high']:+.4f}], range entirely above 0: {_yes(k['k1'])}.")
        if k2["form"] == "general":
            lines.append(f"- K2 years won: {len(k['year_wins'])} of {k['n_years']} (need {k2['wins_needed']}): {_yes(k['k2'])}.")
        else:
            lines.append(f"- K2 (targeted, years {k2['target_years']}): won {len(k2['target_wins'])} (need {k2['target_wins_needed']}); "
                         f"other years with a range entirely below 0: {k2['other_years_worse'] or 'none'}: {_yes(k['k2'])}.")  # fmt: skip
        lines.append(f"- K3 seeds gaining on average: {k['seed_wins']} of {k['seeds']} (need {k['seeds_needed']}): {_yes(k['k3'])}.")
        if k["has_setting"]:
            lines.append(f"- K4 next setting over (`{k['neighbor']}`): average gain {k['neighbor_gain']:+.4f} > 0: {_yes(k['k4'])}.")
        else:
            lines.append("- K4: no tunable setting (declared on the command line).")
        lines.append(f"- K5 after-cost Sharpe difference {k['sharpe']['diff']:+.2f} [{k['sharpe']['low']:+.2f}, {k['sharpe']['high']:+.2f}], not entirely below 0: {_yes(k['k5'])}.")
        lines += [f"- **Verdict: {k['verdict']}.**"]
    if rules is not None and rules["rule"] == "placebo":
        p = rules
        lines += ["", "## Rule WP: placebo (docs/RULES.md)", ""]
        lines += [f"- P1 average IC range not entirely above 0: " + "; ".join(f"`{n}` [{lo:+.4f}, {hi:+.4f}]" for n, (lo, hi) in p["levels"].items()) + f": {_yes(p['p1'])}."]
        lines += [f"- P2 Rule WK never says \"improvement\": " + "; ".join(f"{pair}: {v}" for pair, v in p["keep_verdicts"].items()) + f": {_yes(p['p2'])}."]
        lines += [f"- **Placebo {'passed' if p['pass'] else 'FAILED'}.**"]
    if figs:
        lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    return "\n".join(lines) + "\n"
