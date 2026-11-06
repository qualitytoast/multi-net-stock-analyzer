"""Command line entry point: prepare | train | evaluate.

  detmaso prepare  --config configs/csi300.yaml
  detmaso train    --config configs/csi300.yaml --tag final            (all configured seeds)
  detmaso evaluate --config configs/csi300.yaml --tag final --split valid
  detmaso evaluate --config configs/csi300.yaml --tag final --split test --unlock-test

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
    panel, _, splits = load_prepared(cfg)
    for seed in args.seeds:
        out = _seed_dir(cfg, args.tag, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / "config.json").write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, indent=2, default=str))
        summary = train_seed(cfg, panel, splits, seed, out)
        log.info("seed %d done: %s", seed, summary)


def cmd_evaluate(cfg: Config, args) -> None:
    from detmaso.eval.report import evaluate_and_report

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detmaso")
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all seeds in the config")
    parser.add_argument("--tag", default=None, help="run group name, e.g. dev or final (default: dev)")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--extra-test-run", default=None, metavar="REASON", help="record an additional test-period opening")
    parser.add_argument("--rebuild-raw", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    args.tag_given = args.tag
    args.tag = args.tag or "dev"
    args.seeds = list(args.seeds) if args.seeds is not None else list(cfg.train.seeds)
    {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate}[args.command](cfg, args)


if __name__ == "__main__":
    main()
