"""Command line entry point: prepare | train | evaluate.

  detmaso prepare  --config configs/csi300.yaml
  detmaso train    --config configs/csi300.yaml --seeds 0 1 2 3 4 [--tag dev]
  detmaso evaluate --config configs/csi300.yaml --split valid --seeds 0 [--tag dev]
  detmaso evaluate --config configs/csi300.yaml --split test  --seeds 0 1 2 3 4 --unlock-test

The test period is locked (plan: "opened once, in Milestone 9, with a frozen config").
Evaluating on it requires --unlock-test and records the config hash; a later test evaluation
with a different config hash is refused.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from detmaso.config import Config, load_config

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
    from detmaso.train.trainer import train_seed

    panel, _, splits = load_prepared(cfg)
    for seed in args.seeds:
        out = _seed_dir(cfg, args.tag, seed)
        (out.parent).mkdir(parents=True, exist_ok=True)
        (out.parent / "config.json").write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, indent=2, default=str))
        summary = train_seed(cfg, panel, splits, seed, out)
        log.info("seed %d done: %s", seed, summary)


def _check_test_lock(cfg: Config, tag: str) -> None:
    lock = cfg.run_path / "test_lock.json"
    if lock.exists():
        prior = json.loads(lock.read_text())
        if prior["config_hash"] != cfg.hash() or prior["tag"] != tag:
            raise SystemExit(
                f"test period already opened with config {prior['config_hash']} / tag {prior['tag']}; "
                "refusing to evaluate a different configuration on the test period"
            )
    else:
        lock.write_text(json.dumps({"config_hash": cfg.hash(), "tag": tag, "opened_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


def cmd_evaluate(cfg: Config, args) -> None:
    from detmaso.eval.report import evaluate_and_report

    if args.split == "test":
        if not args.unlock_test:
            raise SystemExit("the test period is locked; pass --unlock-test (only for the final, frozen run)")
        _check_test_lock(cfg, args.tag)
    path = evaluate_and_report(cfg, args.tag, args.seeds, args.split)
    log.info("report written to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="detmaso")
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--tag", default="dev", help="run group name, e.g. dev or final")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--rebuild-raw", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate}[args.command](cfg, args)


if __name__ == "__main__":
    main()
