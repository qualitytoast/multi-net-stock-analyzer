"""Command line entry point: prepare | train | evaluate | compare.

  detmaso prepare  --config configs/csi300.yaml
  detmaso train    --config configs/csi300.yaml --tag final            (all configured seeds)
  detmaso evaluate --config configs/csi300.yaml --tag final --split valid
  detmaso evaluate --config configs/csi300.yaml --tag final --split test --unlock-test
  detmaso compare  --runs configs/csi300.yaml:final configs/csi300_single_mlp.yaml:single_mlp \
                   --reference final --out results/csi300/single_vs_ensemble    (validation only)

A run group (runs/<market>/<tag>/) belongs to one configuration: training refuses to add to a group
trained with a different config hash, and evaluation refuses a config that does not match the group.

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


def cmd_train(cfg: Config, args) -> None:
    from detmaso.data.panel import load_prepared
    from detmaso.eval.lock import lock_path
    from detmaso.train.trainer import train_seed

    lock = lock_path(cfg)
    if lock.exists() and any(e["tag"] == args.tag for e in json.loads(lock.read_text())["entries"]):
        raise SystemExit(f"run group '{args.tag}' was already evaluated on the test period; train under a new --tag")
    existing = _group_hash(cfg, args.tag)
    if existing is not None and existing != cfg.hash():
        raise SystemExit(f"run group '{args.tag}' was trained with config {existing}, this config is {cfg.hash()}; use a new --tag")
    panel, _, splits = load_prepared(cfg)
    for seed in args.seeds:
        out = _seed_dir(cfg, args.tag, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / "config.json").write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, indent=2, default=str))
        summary = train_seed(cfg, panel, splits, seed, out)
        log.info("seed %d done: %s", seed, summary)


def cmd_evaluate(cfg: Config, args) -> None:
    from detmaso.eval.report import evaluate_and_report

    trained = _group_hash(cfg, args.tag)
    if trained != cfg.hash():
        raise SystemExit(f"run group '{args.tag}' was trained with config {trained}, this config is {cfg.hash()}; pass the matching --config")
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


def cmd_compare(args) -> None:
    from detmaso.eval.compare import run_compare

    if args.split != "valid":
        raise SystemExit("comparisons use the validation period only; the test period is never used to compare variants")
    runs = []
    for spec in args.runs:
        config_path, sep, tag = spec.rpartition(":")
        if not sep or not config_path or not tag:
            raise SystemExit(f"--runs entries are CONFIG:TAG, got {spec!r}")
        runs.append((load_config(config_path), config_path, tag))
    path = run_compare(runs, reference=args.reference, out_dir=Path(args.out), n_boot=args.bootstrap, block=args.block, rng_seed=args.rng_seed)
    log.info("comparison written to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detmaso")
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "compare"])
    parser.add_argument("--config", help="required for prepare, train and evaluate")
    parser.add_argument("--runs", nargs="+", metavar="CONFIG:TAG", help="compare: the run groups to compare")
    parser.add_argument("--reference", help="compare: tag of the reference run group (default: the first)")
    parser.add_argument("--out", help="compare: output directory")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="compare: bootstrap draws")
    parser.add_argument("--block", type=int, default=20, help="compare: bootstrap block length in trading days")
    parser.add_argument("--rng-seed", type=int, default=0, help="compare: bootstrap random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all seeds in the config")
    parser.add_argument("--tag", default=None, help="run group name, e.g. dev or final (default: dev)")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
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
    if not args.config:
        parser.error(f"{args.command} needs --config")
    cfg = load_config(args.config)
    args.tag_given = args.tag
    args.tag = args.tag or "dev"
    args.seeds = list(args.seeds) if args.seeds is not None else list(cfg.train.seeds)
    {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate}[args.command](cfg, args)


if __name__ == "__main__":
    main()
