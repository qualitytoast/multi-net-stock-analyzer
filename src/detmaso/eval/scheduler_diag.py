"""Training-day scheduler diagnostics for a run group (docs/RULES.md, Rules G and M).

For each seed's restored model (model.pt) and last-epoch model (model_last.pt, when present), the whole
model is run on every training day and the scheduler weights, the day's regime and each agent's dead flag
(output constant across stocks, DEVIATIONS 4) are saved to runs/<market>/<tag>/usage_train.parquet. From
that file: usage (an agent's mean weight, with its dead days counted as 0), effective agents over time, the
regime x agent routing table, and the Rule M checks. Dead agent-days during training come from history.jsonl.

Only training days are used, so this never looks at the validation period (the strength pilot relies on it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detmaso.config import Config
from detmaso.data.panel import load_prepared
from detmaso.eval.guard import m1_holds, m2_holds, rule_m, usage_and_routing
from detmaso.losses.balance import effective_agents, target_shares
from detmaso.train.trainer import load_model, predict

CHECKPOINTS = (("best", "model.pt"), ("last", "model_last.pt"))
USAGE_FILE = "usage_train.parquet"


def write_usage(cfg: Config, tag: str, panel=None, splits=None) -> Path:
    """Run every saved model of the group on the training days and save weights, regimes and dead flags.

    panel / splits default to the prepared data (tests pass a synthetic panel).
    """
    if panel is None or splits is None:
        panel, _, splits = load_prepared(cfg)
    pos = panel.date_pos(splits.train_fit)
    regime = panel.regime[pos].numpy()
    run = cfg.run_path / tag
    frames = []
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    for seed in cfg.train.seeds:
        seed_dir = run / f"seed{seed}"
        for checkpoint, file in CHECKPOINTS:
            if not (seed_dir / file).exists():
                continue
            model = load_model(cfg, panel, seed_dir, file)
            if getattr(model, "n_agents", 1) < 2:
                raise ValueError(f"run group '{tag}' is a single network: it has no scheduler")
            out = predict(model, panel, pos, torch.device("cpu"))
            n = out["w"].shape[1]
            dead = out["raw_std"] < cfg.train.dead_std
            frames.append(pd.DataFrame({
                "seed": seed, "checkpoint": checkpoint, "date": splits.train_fit, "regime": regime,
                **{f"w{i}": out["w"][:, i] for i in range(n)}, **{f"dead{i}": dead[:, i] for i in range(n)},
            }))  # fmt: skip
    if not frames:
        raise FileNotFoundError(f"run group '{tag}' has no trained models in {run}")
    path = run / USAGE_FILE
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def load_checks(run: Path) -> tuple[dict[int, dict[str, dict]], np.ndarray, int]:
    """{seed: {checkpoint: usage_and_routing(...)}}, the training days' regimes, and the number of agents."""
    df = pd.read_parquet(run / USAGE_FILE)
    n = sum(1 for c in df.columns if c.startswith("w") and c[1:].isdigit())
    checks: dict[int, dict[str, dict]] = {}
    regime = None
    for (seed, checkpoint), g in df.groupby(["seed", "checkpoint"], sort=True):
        g = g.sort_values("date")
        regime = g["regime"].to_numpy()
        w = g[[f"w{i}" for i in range(n)]].to_numpy(float)
        live = ~g[[f"dead{i}" for i in range(n)]].to_numpy(bool)
        checks.setdefault(int(seed), {})[checkpoint] = usage_and_routing(w, live, regime, n)
    return checks, regime, n


def history_dead_days(seed_dir: Path) -> dict:
    """Dead agent-days during training: over all epochs, and up to the restored epoch."""
    history = [json.loads(line) for line in (seed_dir / "history.jsonl").read_text().splitlines()]
    best = json.loads((seed_dir / "summary.json").read_text())["best_epoch"]
    total = np.sum([h["dead_days"] for h in history], axis=0)
    upto = np.sum([h["dead_days"] for h in history if h["epoch"] <= best], axis=0)
    return {"all_epochs": total.astype(int).tolist(), "to_restored_epoch": upto.astype(int).tolist()}


def summarize(cfg: Config, tag: str, target_kind: str) -> dict:
    """Usage, routing, effective agents, Rule M and dead agent-days for one run group (needs USAGE_FILE)."""
    run = cfg.run_path / tag
    checks, regime, n = load_checks(run)
    target = target_shares(target_kind, regime, n).numpy()
    m = rule_m(target, checks, list(cfg.train.seeds), routing_required=target_kind == "regime")
    df = pd.read_parquet(run / USAGE_FILE)
    seeds = {}
    for s, by_ck in checks.items():
        seeds[s] = {
            ck: {
                "usage": c["usage"].tolist(),
                "effective_agents": effective_agents(c["usage"] / c["usage"].sum()) if c["usage"].sum() > 0 else float("nan"),
                "routing": c["routing"].tolist(),
                "m1": m1_holds(c["usage"], target),
                "m2": m2_holds(c["routing"], c["mood_days"]),
            }
            for ck, c in by_ck.items()
        }
        seed_dir = run / f"seed{s}"
        if (seed_dir / "history.jsonl").exists():
            seeds[s]["dead_days_training"] = history_dead_days(seed_dir)
        best = df[(df["seed"] == s) & (df["checkpoint"] == "best")]
        seeds[s]["dead_days_restored_model_train"] = [int(best[f"dead{i}"].sum()) for i in range(n)]
    mood_days = np.bincount(regime, minlength=n).tolist()
    return {"tag": tag, "variant": cfg.variant, "c_bal": cfg.loss.c_bal, "target_kind": target_kind, "target": target.tolist(),
            "mood_days": mood_days, "rule_m": m, "seeds": seeds}  # fmt: skip
