"""Train a walk-forward run group: every fold x seed, with the paper trainer unchanged (docs/RULES.md, Rule WL).

runs/<market>_wf/<tag>/fold<Y>/seed<s>/ holds each job's model.pt, model_last.pt, history.jsonl and summary.json,
exactly as a validation-period seed folder. A job is skipped when its summary.json carries the config's hash
(so an interrupted group resumes where it stopped). Jobs are independent and deterministic, so running them in
parallel (``jobs`` > 1, one process each, one thread each) gives the same models as running them one by one.

The try ledger gets the configuration before any job starts (one try per configuration, aborted jobs included);
each job's status is stored under "<tag>:Y<year>:s<seed>".
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from detmaso.config import Config
from detmaso.data.panel import Panel, load_prepared
from detmaso.data.regimes import RegimeModel
from detmaso.walkforward.folds import Fold, fold_layout, fold_panel, fold_regimes, load_market_state

log = logging.getLogger(__name__)


def job_dir(cfg: Config, tag: str, year: int, seed: int) -> Path:
    return cfg.run_path / tag / f"fold{year}" / f"seed{seed}"


def job_key(year: int, seed: int) -> str:
    return f"Y{year}:s{seed}"


def group_hash(cfg: Config, tag: str) -> str | None:
    path = cfg.run_path / tag / "config.json"
    return json.loads(path.read_text())["hash"] if path.exists() else None


def job_done(cfg: Config, tag: str, year: int, seed: int) -> bool:
    d = job_dir(cfg, tag, year, seed)
    if not ((d / "summary.json").exists() and (d / "model.pt").exists()):
        return False
    return json.loads((d / "summary.json").read_text()).get("config_hash") == cfg.hash()


def select_folds(folds: list[Fold], years: list[int] | None) -> list[Fold]:
    if years is None:
        return folds
    known = {f.year for f in folds}
    unknown = sorted(set(years) - known)
    if unknown:
        raise ValueError(f"folds {unknown} are not in this config's layout {sorted(known)}")
    return [f for f in folds if f.year in set(years)]


@dataclass
class FoldSet:
    panel: Panel  # the prepared panel (placebo answers applied)
    state: pd.DataFrame  # raw market state
    folds: list[Fold]
    regimes: dict[int, RegimeModel]  # each fold's moods
    panels: dict[int, Panel]  # each fold's panel with its own moods and scheduler inputs


def load_folds(cfg: Config, fit_moods: bool = False) -> FoldSet:
    """The prepared data, the fold layout and each fold's moods (fitted and cached when ``fit_moods``)."""
    panel, _, _ = load_prepared(cfg)
    state = load_market_state(cfg)
    folds = fold_layout(cfg, panel.dates)
    clip = cfg.regimes.scheduler_input_clip
    regimes = {f.year: fold_regimes(cfg, state, f, fit=fit_moods) for f in folds}
    panels = {f.year: fold_panel(panel, state, regimes[f.year], clip) for f in folds}
    return FoldSet(panel, state, folds, regimes, panels)


def _write_group(cfg: Config, tag: str, folds: list[Fold]) -> None:
    run = cfg.run_path / tag
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"hash": cfg.hash(), **cfg.to_dict()}, indent=2, default=str))
    (run / "folds.json").write_text(json.dumps([f.describe() for f in folds], indent=2))


# Worker side (one process per job slot; the panel is loaded once per process) ----------------------------
_WORKER: dict = {}


def _init_worker(cfg: Config) -> None:
    import torch

    torch.set_num_threads(1)
    fs = load_folds(cfg)
    _WORKER.update(cfg=cfg, folds={f.year: f for f in fs.folds}, panels=fs.panels)


def _run_job(tag: str, year: int, seed: int) -> tuple[int, int, str, dict | None]:
    from detmaso.train.trainer import train_seed

    cfg = _WORKER["cfg"]
    try:
        summary = train_seed(cfg, _WORKER["panels"][year], _WORKER["folds"][year].splits, seed, job_dir(cfg, tag, year, seed))
    except Exception as exc:  # recorded in the try ledger by the parent; the other jobs go on
        return year, seed, f"aborted: {type(exc).__name__}: {exc}", None
    return year, seed, "trained", summary


# Parent side -----------------------------------------------------------------------------------------------
def train_walkforward(cfg: Config, config_path: str, tag: str, seeds: list[int], years: list[int] | None = None,
                      jobs: int = 1, ledger_path: Path | None = None) -> dict:  # fmt: skip
    """Train the missing (fold, seed) jobs of a run group. Returns {job key: status}; raises if any job failed."""
    from detmaso.eval import ledger

    if not cfg.is_walkforward:
        raise ValueError("train_walkforward needs a config with a walkforward section")
    existing = group_hash(cfg, tag)
    if existing is not None and existing != cfg.hash():
        raise SystemExit(f"run group '{tag}' was trained with config {existing}, this config is {cfg.hash()}; use a new --tag")
    all_folds = load_folds(cfg, fit_moods=True).folds  # fits and caches every fold's moods before any job starts
    folds = select_folds(all_folds, years)
    _write_group(cfg, tag, all_folds)
    todo = [(f.year, s) for f in folds for s in seeds if not job_done(cfg, tag, f.year, s)]
    if ledger_path is not None:
        row = ledger.register(ledger_path, cfg, config_path, tag, [], jobs=[job_key(y, s) for y, s in todo])
        log.info("try ledger %s: config %s, kind %s, try %s", ledger_path, cfg.hash(), row["kind"], row["try_no"] or "- (not counted)")
    log.info("run group %s: %d of %d jobs to train (%d folds x %d seeds), %d at a time", tag, len(todo), len(folds) * len(seeds), len(folds), len(seeds), jobs)
    statuses: dict[str, str] = {}

    def record(year: int, seed: int, status: str, summary: dict | None) -> None:
        statuses[job_key(year, seed)] = status
        if ledger_path is not None:
            ledger.set_seed_status(ledger_path, cfg, tag, job_key(year, seed), status)
        done = sum(1 for v in statuses.values() if v == "trained")
        extra = f"best epoch {summary['best_epoch']} of {summary['epochs_run']}" if summary else status
        log.info("fold %d seed %d: %s (%d/%d trained)", year, seed, extra, done, len(todo))

    if todo:
        if jobs <= 1:
            _init_worker(cfg)
            for year, seed in todo:
                record(*_run_job(tag, year, seed))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(jobs, len(todo)), mp_context=ctx, initializer=_init_worker, initargs=(cfg,)) as pool:
                futures = [pool.submit(_run_job, tag, year, seed) for year, seed in todo]
                for fut in as_completed(futures):
                    record(*fut.result())
    failed = {k: v for k, v in statuses.items() if v != "trained"}
    if failed:
        raise SystemExit(f"{len(failed)} walk-forward jobs failed: {failed}")
    return statuses


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)
