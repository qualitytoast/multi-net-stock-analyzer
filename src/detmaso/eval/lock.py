"""Test-period lock (plan: the test period is opened once, with a frozen configuration).

The lock is git-tracked (locks/<market>_test.json) and pins exactly what was evaluated: config and
data hashes, seeds, a fingerprint of the source code, and sha256 digests of every model and of the
prepared panel. Re-running an identical evaluation (for example to regenerate the report) is
allowed; anything else is refused unless an explicit reason is given, which is recorded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from detmaso.config import Config

ROOT = Path(__file__).resolve().parents[3]
PINNED = ("config_hash", "data_hash", "tag", "seeds", "code_sha256", "model_sha256", "panel_sha256", "regimes_sha256", "data_sha256")


class TestLockError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> tuple[str, str | None, bool | None]:
    files = sorted((ROOT / "src").rglob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "configs"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return h.hexdigest()[:16], head, dirty


def lock_path(cfg: Config) -> Path:
    return ROOT / "locks" / f"{cfg.market}_test.json"


def fingerprint(cfg: Config, tag: str, seeds: list[int]) -> dict:
    run = cfg.run_path / tag
    if list(seeds) != list(cfg.train.seeds):
        raise TestLockError(f"the test period is evaluated on all configured seeds {list(cfg.train.seeds)}, got {list(seeds)}")
    models = {}
    for s in seeds:
        d = run / f"seed{s}"
        if not (d / "model.pt").exists() or not (d / "summary.json").exists():
            raise TestLockError(f"missing trained model for seed {s} in {d}")
        summary = json.loads((d / "summary.json").read_text())
        if summary.get("config_hash") != cfg.hash():
            raise TestLockError(f"seed {s} was trained with config {summary.get('config_hash')}, current config is {cfg.hash()}")
        models[str(s)] = _sha256(d / "model.pt")
    prepared = cfg.run_path / "prepared"
    meta = json.loads((prepared / "meta.json").read_text())
    if meta.get("data_hash") != cfg.data_hash():
        raise TestLockError("prepared data does not match the current data config; re-run prepare")
    code, head, dirty = code_fingerprint()
    return {
        "config_hash": cfg.hash(),
        "data_hash": cfg.data_hash(),
        "tag": tag,
        "seeds": list(seeds),
        "code_sha256": code,
        "git_head": head,
        "git_dirty": dirty,
        "model_sha256": models,
        "panel_sha256": _sha256(prepared / "panel.pt"),
        "regimes_sha256": _sha256(prepared / "regimes.json"),
        "data_sha256": cfg.data.data_sha256,
    }


def open_test_period(cfg: Config, tag: str, seeds: list[int], extra_reason: str | None = None) -> dict:
    """Validate and record a test-period evaluation. Returns the lock record (with an opening count)."""
    if cfg.is_placebo:
        raise TestLockError("placebo configs (shuffled answers) can never open the test period")
    if not cfg.train.validate:
        raise TestLockError("training-only runs (train.validate false) can never open the test period")
    current = fingerprint(cfg, tag, seeds)
    path = lock_path(cfg)
    lock = json.loads(path.read_text()) if path.exists() else {"market": cfg.market, "entries": []}
    same = [e for e in lock["entries"] if all(e.get(k) == current[k] for k in PINNED)]
    if same:
        return {**same[0], "openings": len(lock["entries"])}
    if lock["entries"] and not extra_reason:
        raise TestLockError(
            f"the test period was already opened with a different setup ({len(lock['entries'])} recorded in {path}); "
            "pass --extra-test-run 'reason' to record another opening explicitly"
        )
    entry = {**current, "opened_at": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": extra_reason or "first opening"}
    lock["entries"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2) + "\n")
    return {**entry, "openings": len(lock["entries"])}
