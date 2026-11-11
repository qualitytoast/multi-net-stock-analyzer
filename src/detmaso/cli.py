"""Command line entry point: prepare | train | evaluate | usage | compare.

  detmaso prepare  --config configs/csi300.yaml
  detmaso train    --config configs/csi300.yaml --tag final            (all configured seeds)
  detmaso evaluate --config configs/csi300.yaml --tag final --split valid
  detmaso evaluate --config configs/csi300.yaml --tag final --split test --unlock-test
  detmaso usage    --runs configs/csi300.yaml:final [--grid] [--out DIR]   (training days only)
  detmaso compare  --runs configs/csi300.yaml:final configs/csi300_single_mlp.yaml:single_mlp \
                   --reference final --out results/csi300/single_vs_ensemble    (validation only)

Walk-forward (docs/RULES.md, "Walk-forward evaluation"; configs with a `walkforward` section):

  detmaso prepare  --config configs/walkforward/paper.yaml
  detmaso train    --config configs/walkforward/paper.yaml --tag wf [--jobs 8] [--folds 2013 ...]
  detmaso evaluate --config configs/walkforward/paper.yaml --tag wf --split walkforward
  detmaso compare  --split walkforward --runs A.yaml:wf B.yaml:wf --reference wf --out DIR \
                   [--rule keep --candidate TAG (--neighbor TAG | --no-tunable-setting) [--target-years 2015 ...]]
                   [--rule placebo]

A run group (runs/<market>/<tag>/) belongs to one configuration: training refuses to add to a group
trained with a different config hash, and evaluation refuses a config that does not match the group.

Guards (CLAUDE.md): `train` registers every configuration in the try ledger results/<market>/trials.csv
before training (aborted runs included), `evaluate --split valid` over all configured seeds adds its
validation metrics, and `compare` reports each variant's try number, deflated Sharpe and deflated IC
from that ledger.

The test period is locked (plan: "opened once, in Milestone 9, with a frozen config"). Evaluating
it needs --unlock-test and an explicit --tag, uses all configured seeds, and is recorded in the
git-tracked locks/<market>_test.json together with digests of the models, data and code. A later
test evaluation of anything different is refused unless --extra-test-run "reason" records it.
"""

from __future__ import annotations

import os

# cuBLAS needs this for deterministic algorithms on some CUDA/torch versions; set before torch loads.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from detmaso.config import Config, load_config  # noqa: E402

log = logging.getLogger("detmaso")


def _seed_dir(cfg: Config, tag: str, seed: int) -> Path:
    return cfg.run_path / tag / f"seed{seed}"


def _group_hash(cfg: Config, tag: str) -> str | None:
    path = cfg.run_path / tag / "config.json"
    return json.loads(path.read_text())["hash"] if path.exists() else None


def cmd_prepare(cfg: Config, args) -> None:
    from detmaso.data.panel import build, save_prepared
    from detmaso.data.qa import write_data_report

    data = build(cfg, rebuild_raw=args.rebuild_raw)
    save_prepared(data, cfg)
    path = write_data_report(data, cfg)
    log.info("prepared data saved; report at %s", path)


def _write_group_config(cfg: Config, tag: str) -> None:
    """Write the run group's config.json atomically (seeds of one group may train in parallel)."""
    path = cfg.run_path / tag / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".config.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, indent=2, default=str))
    os.replace(tmp, path)


def cmd_train(cfg: Config, args) -> None:
    from detmaso.data.panel import load_prepared
    from detmaso.eval import ledger
    from detmaso.eval.lock import lock_path
    from detmaso.train.trainer import train_seed

    if cfg.is_walkforward:
        from detmaso.walkforward.runner import default_jobs, train_walkforward

        train_walkforward(cfg, args.config, args.tag, args.seeds, args.folds, args.jobs or default_jobs(), ledger.ledger_path(cfg.market))
        return
    if args.folds or args.jobs:
        raise SystemExit("--folds and --jobs are for walk-forward configs only")

    lock = lock_path(cfg)
    if lock.exists() and any(e["tag"] == args.tag for e in json.loads(lock.read_text())["entries"]):
        raise SystemExit(f"run group '{args.tag}' was already evaluated on the test period; train under a new --tag")
    existing = _group_hash(cfg, args.tag)
    if existing is not None and existing != cfg.hash():
        raise SystemExit(f"run group '{args.tag}' was trained with config {existing}, this config is {cfg.hash()}; use a new --tag")
    trials = ledger.ledger_path(cfg.market)
    row = ledger.register(trials, cfg, args.config, args.tag, args.seeds)  # counted before training starts
    log.info("try ledger %s: config %s, kind %s, try %s", trials, cfg.hash(), row["kind"], row["try_no"] or "- (not counted)")
    panel, _, splits = load_prepared(cfg)
    for seed in args.seeds:
        out = _seed_dir(cfg, args.tag, seed)
        _write_group_config(cfg, args.tag)
        try:
            summary = train_seed(cfg, panel, splits, seed, out)
        except BaseException as exc:
            ledger.set_seed_status(trials, cfg, args.tag, seed, f"aborted: {type(exc).__name__}")
            raise
        ledger.set_seed_status(trials, cfg, args.tag, seed, "trained")
        log.info("seed %d done: %s", seed, summary)


def cmd_evaluate(cfg: Config, args) -> None:
    from detmaso.eval.report import evaluate_and_report

    if cfg.is_walkforward or args.split == "walkforward":
        if not (cfg.is_walkforward and args.split == "walkforward"):
            raise SystemExit("walk-forward configs are evaluated with --split walkforward, and only they are")
        from detmaso.eval import ledger
        from detmaso.eval.walkforward import evaluate_walkforward

        path = evaluate_walkforward(cfg, args.config, args.tag, args.seeds, args.folds, ledger.ledger_path(cfg.market))
        log.info("walk-forward report written to %s", path)
        return

    trained = _group_hash(cfg, args.tag)
    if trained != cfg.hash():
        raise SystemExit(f"run group '{args.tag}' was trained with config {trained}, this config is {cfg.hash()}; pass the matching --config")
    if not cfg.train.validate:
        raise SystemExit(f"run group '{args.tag}' is a training-only run (train.validate false): it must never be evaluated")
    lock_record = None
    if args.split == "test":
        from detmaso.eval.lock import TestLockError, open_test_period

        if not args.unlock_test:
            raise SystemExit("the test period is locked; pass --unlock-test (only for the final, frozen run)")
        if args.tag_given is None:
            raise SystemExit("evaluating the test period needs an explicit --tag")
        try:
            lock_record = open_test_period(cfg, args.tag, args.seeds, args.extra_test_run)
        except TestLockError as exc:
            raise SystemExit(f"test period refused: {exc}") from exc
    path = evaluate_and_report(cfg, args.tag, args.seeds, args.split, lock_record=lock_record)
    log.info("report written to %s", path)
    if args.split == "valid":
        from detmaso.eval import ledger

        trials = ledger.ledger_path(cfg.market)
        if not ledger.read(trials)["config_hash"].eq(cfg.hash()).any():
            ledger.register(trials, cfg, args.config, args.tag, [], note="registered at evaluation (trained before the ledger existed)")
        if sorted(args.seeds) != sorted(cfg.train.seeds):
            log.warning("seeds %s are not the configured seeds %s: validation metrics not recorded in the try ledger", args.seeds, list(cfg.train.seeds))
            return
        ledger.record_metrics(trials, cfg, ledger.validation_metrics(cfg.run_path / args.tag))
        log.info("validation metrics recorded in the try ledger %s", trials)


def _parse_runs(specs: list[str]) -> list[tuple[Config, str, str]]:
    runs = []
    for spec in specs:
        config_path, sep, tag = spec.rpartition(":")
        if not sep or not config_path or not tag:
            raise SystemExit(f"--runs entries are CONFIG:TAG, got {spec!r}")
        runs.append((load_config(config_path), config_path, tag))
    return runs


def cmd_usage(args) -> None:
    """Training-day scheduler usage (never the validation period); with --grid, Rule G over pilot runs."""
    from detmaso.eval.guard import PILOT_STRENGTHS, rule_g
    from detmaso.eval.scheduler_diag import summarize, write_usage

    runs = _parse_runs(args.runs)
    if any(cfg.is_walkforward for cfg, _, _ in runs):
        raise SystemExit("usage reads validation-period run groups; walk-forward groups report their moods in `evaluate --split walkforward`")
    summaries = []
    for cfg, config_path, tag in runs:
        path = write_usage(cfg, tag)
        kind = cfg.loss.balance_target if cfg.loss.c_bal > 0 else "regime"
        s = summarize(cfg, tag, kind)
        s["config"] = config_path
        summaries.append(s)
        m = s["rule_m"]
        log.info("%s (c_bal %g): M0 %s M1 %s M2 %s -> Rule M %s; usage file %s", tag, cfg.loss.c_bal, m["m0"], m["m1"], m["m2"], m["pass"], path)
    payload = {"runs": summaries}
    if args.grid:
        pilot_m1 = {}
        for s in summaries:  # Rule G reads the last epoch (epoch 8 of the pilot) in every pilot seed
            pilot_m1[s["c_bal"]] = bool(s["rule_m"]["m0"] and all(c["last"]["m1"] for c in s["seeds"].values()))
        if sorted(pilot_m1) != sorted(PILOT_STRENGTHS):
            raise SystemExit(f"Rule G needs the pilot strengths {PILOT_STRENGTHS}, got {sorted(pilot_m1)}")
        placebo = all(cfg.is_placebo for cfg, _, _ in runs)
        payload["rule_g"] = {"placebo": placebo, "pilot_m1": {f"{c:g}": ok for c, ok in sorted(pilot_m1.items())}, **rule_g(pilot_m1, placebo)}
        log.info("Rule G: %s", payload["rule_g"])
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "usage.json").write_text(json.dumps(payload, indent=2, default=float))
        (out / "usage.md").write_text(_usage_markdown(payload))
        log.info("usage summary written to %s", out)


def _usage_markdown(payload: dict) -> str:
    from detmaso.mdtable import md_table

    lines = ["# Training-day scheduler usage (docs/RULES.md, Rules G and M)", ""]
    first = payload["runs"][0]
    lines += [f"Target ({first['target_kind']} shares of the {sum(first['mood_days'])} training days), %: "
              + ", ".join(f"{100 * t:.1f}" for t in first["target"]), ""]  # fmt: skip
    rows = []
    for s in payload["runs"]:
        for seed, c in sorted(s["seeds"].items()):
            for ck in ("best", "last"):
                if ck in c:
                    rows.append([f"`{s['tag']}`", f"{s['c_bal']:g}", seed, ck, ", ".join(f"{100 * u:.2f}" for u in c[ck]["usage"]),
                                 f"{c[ck]['effective_agents']:.2f}", "yes" if c[ck]["m1"] else "no", "yes" if c[ck]["m2"] else "no"])  # fmt: skip
    lines.append(md_table(["Run group", "c_bal", "Seed", "Model", "Usage per agent, %", "Effective agents", "M1", "M2"], rows))
    if "rule_g" in payload:
        g = payload["rule_g"]
        lines += ["", "## Rule G", "", f"- M1 at the last pilot epoch, both seeds: {g['pilot_m1']}"]
        lines += [f"- Anchor g0: {g['anchor']}", f"- Grid: {g['grid']}", f"- Stop: {g['stop']}"]
    return "\n".join(lines) + "\n"


def cmd_compare(args) -> None:
    from detmaso.eval import ledger
    from detmaso.eval.compare import run_compare

    if args.split == "walkforward":
        from detmaso.eval.wf_compare import run_wf_compare

        runs = _parse_runs(args.runs)
        path = run_wf_compare(runs, args.reference, Path(args.out), rule=args.rule, candidate=args.candidate, neighbor=args.neighbor,
                              no_setting=args.no_tunable_setting, target_years=args.target_years,
                              ledger_path=ledger.ledger_path(runs[0][0].market))  # fmt: skip
        log.info("comparison written to %s", path)
        return
    if args.split != "valid":
        raise SystemExit("comparisons use the validation period or walk-forward folds only; the test period is never used to compare variants")
    if args.rule not in (None, "scheduler_balance") or args.candidate or args.neighbor or args.target_years or args.no_tunable_setting:
        raise SystemExit("--rule keep/placebo, --candidate, --neighbor, --target-years and --no-tunable-setting are for --split walkforward")
    runs = _parse_runs(args.runs)
    if any(cfg.is_walkforward for cfg, _, _ in runs):
        raise SystemExit("walk-forward run groups are compared with --split walkforward")
    path = run_compare(runs, reference=args.reference, out_dir=Path(args.out), rule=args.rule, ledger_path=ledger.ledger_path(runs[0][0].market))
    log.info("comparison written to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detmaso")
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "usage", "compare"])
    parser.add_argument("--config", help="required for prepare, train and evaluate")
    parser.add_argument("--runs", nargs="+", metavar="CONFIG:TAG", help="compare / usage: the run groups")
    parser.add_argument("--reference", help="compare: tag of the reference run group (default: the first)")
    parser.add_argument("--out", help="compare (required) / usage (optional): output directory")
    parser.add_argument("--rule", choices=["scheduler_balance", "keep", "placebo"], default=None, help="compare: apply this docs/RULES.md rule set")
    parser.add_argument("--candidate", help="compare --rule keep: tag of the run group judged against the reference")
    parser.add_argument("--neighbor", help="compare --rule keep: tag of the next setting over (Rule WK, K4)")
    parser.add_argument("--no-tunable-setting", action="store_true", help="compare --rule keep: the change has no tunable setting")
    parser.add_argument("--target-years", type=int, nargs="+", default=None, help="compare --rule keep: years a targeted change declared in advance")
    parser.add_argument("--jobs", type=int, default=None, help="train (walk-forward): parallel jobs (default: CPU count - 2)")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="train / evaluate (walk-forward): only these test years")
    parser.add_argument("--grid", action="store_true", help="usage: apply Rule G to the given pilot runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all seeds in the config")
    parser.add_argument("--tag", default=None, help="run group name, e.g. dev or final (default: dev)")
    parser.add_argument("--split", choices=["valid", "test", "walkforward"], default="valid")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--extra-test-run", default=None, metavar="REASON", help="record an additional test-period opening")
    parser.add_argument("--rebuild-raw", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    if args.command == "compare":
        if not args.runs or len(args.runs) < 2 or not args.out:
            parser.error("compare needs --runs with at least two CONFIG:TAG entries and --out")
        cmd_compare(args)
        return
    if args.command == "usage":
        if not args.runs:
            parser.error("usage needs --runs with CONFIG:TAG entries")
        cmd_usage(args)
        return
    if not args.config:
        parser.error(f"{args.command} needs --config")
    cfg = load_config(args.config)
    args.tag_given = args.tag
    args.tag = args.tag or "dev"
    args.seeds = list(args.seeds) if args.seeds is not None else list(cfg.train.seeds)
    {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate}[args.command](cfg, args)


if __name__ == "__main__":
    main()
