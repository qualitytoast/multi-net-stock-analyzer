"""Try ledger (CLAUDE.md "Guards: count tries"): every trained configuration, git-tracked.

results/<market>/trials.csv has one row per configuration (keyed by its config hash), in the order the
configurations were first trained. `detmaso train` registers a configuration before training starts, so
runs that later abort are counted too; `detmaso evaluate --split valid` adds the validation metrics that the
deflated Sharpe and deflated IC need.

A configuration counts as a try (gets a try number) when it is trained on the real answers with validation
switched on: every such run checks the validation IC each epoch, so each one is a look at validation.
Identical reruns are the same row. Placebo runs (shuffled answers) and training-only runs (train.validate
false) are recorded but not counted.

Only the CLI writes the real ledger; library code and tests pass explicit paths. Writes take an exclusive
file lock, so parallel training processes cannot lose each other's updates.
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from detmaso.config import Config

ROOT = Path(__file__).resolve().parents[3]
COLUMNS = [
    "try_no", "config_hash", "kind", "first_trained", "config", "variant", "run_groups", "seed_status",
    "valid_days", "ic", "ic_sd_seeds", "sharpe", "excess_sharpe", "excess_sr_daily", "note",
]  # fmt: skip
REAL, PLACEBO, PILOT = "real", "placebo", "training-only"
METRICS = ("valid_days", "ic", "ic_sd_seeds", "sharpe", "excess_sharpe", "excess_sr_daily")


def ledger_path(market: str) -> Path:
    return ROOT / "results" / market / "trials.csv"


def kind_of(cfg: Config) -> str:
    if cfg.is_placebo:
        return PLACEBO
    if not cfg.train.validate:
        return PILOT
    return REAL


@contextmanager
def _locked_rows(path: Path):
    """Yield the ledger rows (list of dicts) under an exclusive lock; write them back on exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        text = fh.read()
        rows = list(csv.DictReader(io.StringIO(text))) if text.strip() else []
        yield rows
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
        fh.seek(0)
        fh.truncate()
        fh.write(out.getvalue())
        fh.flush()
        os.fsync(fh.fileno())


def _row_for(rows: list[dict], key: str) -> dict | None:
    return next((r for r in rows if r["config_hash"] == key), None)


def register(path: Path, cfg: Config, config_path: str, tag: str, seeds: list[int], note: str = "", when: str | None = None) -> dict:
    """Add the configuration (or the run group / seeds to its row) before training starts."""
    with _locked_rows(path) as rows:
        row = _row_for(rows, cfg.hash())
        if row is None:
            kind = kind_of(cfg)
            counted = [int(r["try_no"]) for r in rows if r["try_no"]]
            row = {
                "try_no": str(max(counted, default=0) + 1) if kind == REAL else "",
                "config_hash": cfg.hash(), "kind": kind, "first_trained": when or time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config_path, "variant": cfg.variant, "run_groups": "", "seed_status": "{}", "note": note,
            }  # fmt: skip
            rows.append(row)
        groups = [g for g in row["run_groups"].split(";") if g]
        if tag not in groups:
            row["run_groups"] = ";".join(groups + [tag])
        status = json.loads(row["seed_status"] or "{}")
        for s in seeds:
            status.setdefault(f"{tag}:{s}", "started")
        row["seed_status"] = json.dumps(status, sort_keys=True)
        return dict(row)


def set_seed_status(path: Path, cfg: Config, tag: str, seed: int, status: str) -> None:
    with _locked_rows(path) as rows:
        row = _row_for(rows, cfg.hash())
        if row is None:
            raise KeyError(f"config {cfg.hash()} is not in the ledger {path}")
        s = json.loads(row["seed_status"] or "{}")
        s[f"{tag}:{seed}"] = status
        row["seed_status"] = json.dumps(s, sort_keys=True)


def validation_metrics(run: Path) -> dict:
    """Seed-mean validation metrics of an evaluated run group (from `evaluate --split valid` outputs)."""
    report = json.loads((run / "results_valid.json").read_text())
    series = pd.read_parquet(run / "series_valid.parquet")
    per_seed = report["per_seed"]
    ic = np.array([p["ic"]["ic"] for p in per_seed])
    daily_sr = []
    for _, g in series.groupby("seed", sort=True):
        x = (g["port"] - g["bench"]).to_numpy(float)
        daily_sr.append(x.mean() / x.std(ddof=1))
    return {
        "valid_days": int(series.groupby("seed").size().iloc[0]),
        "ic": float(ic.mean()),
        "ic_sd_seeds": float(ic.std(ddof=1)) if len(ic) > 1 else 0.0,
        "sharpe": float(np.mean([p["port"]["sharpe"] for p in per_seed])),
        "excess_sharpe": float(np.mean([p["excess"]["sharpe"] for p in per_seed])),
        "excess_sr_daily": float(np.mean(daily_sr)),
    }


def record_metrics(path: Path, cfg: Config, metrics: dict) -> None:
    with _locked_rows(path) as rows:
        row = _row_for(rows, cfg.hash())
        if row is None:
            raise KeyError(f"config {cfg.hash()} is not in the ledger {path}; it must be registered by `train`")
        for k in METRICS:
            v = metrics[k]
            row[k] = str(v) if isinstance(v, int) else f"{v:.10g}"


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"try_no": "Int64", "config_hash": str, "note": str}, keep_default_na=False, na_values=[""])
    return df


def tries(path: Path) -> pd.DataFrame:
    """The counted tries (real answers, validation on), in try order."""
    df = read(path)
    return df[df["try_no"].notna()].sort_values("try_no").reset_index(drop=True)


def try_number(path: Path, config_hash: str) -> int | None:
    df = tries(path)
    hit = df[df["config_hash"] == config_hash]
    return int(hit["try_no"].iloc[0]) if len(hit) else None
