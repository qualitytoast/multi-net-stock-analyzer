"""Training loop (paper Algorithm 1; plan Sections 6.2-6.4, A27-A35).

One Adam step per training date, in chronological order; cosine LR stepped per date; global
gradient clipping; skip rules that fail loudly; early stopping on the 3-epoch smoothed
validation IC (only epochs >= 3 can be "best"), restoring the best epoch's weights.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from detmaso.config import Config
from detmaso.data.panel import Panel
from detmaso.data.splits import Splits
from detmaso.losses.objective import DetMASOLoss, SharpeState
from detmaso.losses.stats import daily_spearman
from detmaso.model.factory import build_model

log = logging.getLogger(__name__)


class TrainingAborted(RuntimeError):
    pass


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def lr_at(step: int, t_max: int, lr_max: float, lr_min: float) -> float:
    """Cosine annealing (paper Eq. 21), stepped once per training date."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * step / t_max))


@torch.no_grad()
def predict(model: torch.nn.Module, panel: Panel, positions: torch.Tensor, device: torch.device, chunk: int = 128):
    """Frozen-model outputs for many dates at once (dates are independent at inference)."""
    model.eval()
    outs = defaultdict(list)
    for i in range(0, len(positions), chunk):
        pos = positions[i : i + chunk]
        dtype = next(model.parameters()).dtype
        o = model(panel.X[pos].to(device=device, dtype=dtype), panel.universe[pos].to(device), panel.s_sched[pos].to(device=device, dtype=dtype))
        for k in ("alphas", "alpha_hat", "w", "a", "logits", "raw_std"):
            outs[k].append(getattr(o, k).cpu())
    return {k: torch.cat(v).numpy() for k, v in outs.items()}


def validation_ic(model: torch.nn.Module, panel: Panel, positions: torch.Tensor, device: torch.device) -> float:
    out = predict(model, panel, positions, device)
    ic = daily_spearman(out["alpha_hat"], panel.y[positions].numpy(), panel.label_ok[positions].numpy())
    return float(np.nanmean(ic))


def _term_grad_norms(terms, coefs: dict[str, float], params: list[torch.nn.Parameter]) -> dict[str, float]:
    norms = {}
    for name, (tensor, coef) in {
        "ic": (terms.ic_term, coefs["c_ic"]),
        "sharpe": (terms.sr, coefs["c_sr"]),
        "div": (terms.neg_logdet, coefs["c_div"]),
        "ent": (terms.entropy, coefs["c_ent"]),
    }.items():
        if not tensor.requires_grad:
            norms[name] = 0.0
            continue
        grads = torch.autograd.grad(coef * tensor, params, retain_graph=True, allow_unused=True)
        norms[name] = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads if g is not None)))
    return norms


def train_seed(cfg: Config, panel: Panel, splits: Splits, seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.set_num_threads(1)
    device = resolve_device(cfg.train.device)
    dtype = getattr(torch, cfg.model.dtype)
    model = build_model(cfg, d_in=panel.X.shape[-1]).to(device=device, dtype=dtype)
    params = list(model.parameters())
    opt_cls = {"coupled": torch.optim.Adam, "decoupled": torch.optim.AdamW}[cfg.train.weight_decay_mode]
    opt = opt_cls(params, lr=cfg.train.lr_max, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.train.weight_decay)
    loss_fn = DetMASOLoss(cfg.loss, model.n_agents)
    coefs = {"c_ic": cfg.loss.c_ic, "c_sr": cfg.loss.c_sr, "c_div": cfg.loss.c_div, "c_ent": cfg.loss.c_ent}

    train_pos = panel.date_pos(splits.train_fit)
    val_pos = panel.date_pos(splits.valid_fit)
    t_max = len(train_pos) * cfg.train.max_epochs
    sharpe = SharpeState(window=cfg.loss.sharpe_window)
    tc = cfg.train
    step = 0
    nonfinite_total = 0
    val_ics: list[float] = []
    best_score, best_epoch, best_state, bad = -math.inf, None, None, 0
    history_path = out_dir / "history.jsonl"
    history_path.write_text("")

    for epoch in range(1, tc.max_epochs + 1):
        model.train()
        sharpe.reset()
        acc: dict[str, list] = defaultdict(list)
        grad_norms: dict[str, list] = defaultdict(list)
        skips = {"few_labels": 0, "nonfinite": 0}
        consecutive = 0
        t0 = time.time()
        for j, t in enumerate(train_pos.tolist()):
            lr = lr_at(step, t_max, tc.lr_max, tc.lr_min)
            step += 1  # the schedule advances on skipped dates too (A27)
            for group in opt.param_groups:
                group["lr"] = lr

            def skip(reason: str) -> None:
                nonlocal consecutive
                skips[reason] += 1
                consecutive += 1
                if consecutive >= tc.max_consecutive_skips:
                    raise TrainingAborted(f"{consecutive} consecutive skipped dates (last: {reason})")

            label_ok = panel.label_ok[t]
            if int(label_ok.sum()) < cfg.features.min_label_ok:
                skip("few_labels")
                continue
            out = model(
                panel.X[t : t + 1].to(device=device, dtype=dtype),
                panel.universe[t : t + 1].to(device),
                panel.s_sched[t : t + 1].to(device=device, dtype=dtype),
            )
            terms = loss_fn(
                out.alphas[0], out.alpha_hat[0], out.logits[0], out.w[0],
                panel.y[t].to(device=device, dtype=dtype), label_ok.to(device),
                int(panel.regime[t]) if model.uses_regimes else None, sharpe,
            )  # fmt: skip
            if not torch.isfinite(terms.loss):
                nonfinite_total += 1
                skip("nonfinite")
                continue
            if tc.grad_norm_every and j % tc.grad_norm_every == 0:
                for k, v in _term_grad_norms(terms, coefs, params).items():
                    grad_norms[k].append(v)
            opt.zero_grad(set_to_none=True)
            terms.loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
                nonfinite_total += 1
                opt.zero_grad(set_to_none=True)
                skip("nonfinite")
                continue
            torch.nn.utils.clip_grad_norm_(params, tc.clip_norm)
            opt.step()
            sharpe.push(terms.p_t)
            consecutive = 0
            acc["loss"].append(float(terms.loss.detach()))
            acc["ic_term"].append(float(terms.ic_term.detach()))
            acc["sharpe"].append(float(terms.sr.detach()))
            acc["neg_logdet"].append(float(terms.neg_logdet.detach()))
            acc["entropy"].append(float(terms.entropy.detach()))
            acc["mean_abs_corr"].append(terms.mean_abs_corr)
            acc["logdet_raw"].append(terms.logdet_raw)
            acc["lambda_min"].append(terms.lambda_min)
            acc["affine_share"].append(terms.affine_share)
            acc["chol_retries"].append(terms.cholesky_retries)
            acc["ic_agents"].append(terms.ic)
            acc["w"].append(out.w[0].detach().cpu().numpy())
            acc["raw_std"].append(out.raw_std[0].cpu().numpy())
            acc["dead"].append((out.raw_std[0] < tc.dead_std).cpu().numpy())
            acc["n_live"].append(model.n_agents - terms.n_dead)

        n_skipped = sum(skips.values())
        if n_skipped > tc.max_skip_frac * len(train_pos):
            raise TrainingAborted(f"epoch {epoch}: {n_skipped} skipped dates {skips}")
        if nonfinite_total > tc.max_nonfinite:
            raise TrainingAborted(f"{nonfinite_total} non-finite events")

        val_ic = validation_ic(model, panel, val_pos, device)
        val_ics.append(val_ic)
        smoothed = float(np.mean(val_ics[-tc.smooth :])) if epoch >= tc.smooth else None
        if smoothed is not None:
            if smoothed > best_score:
                best_score, best_epoch, bad = smoothed, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 1),
            "lr": lr,
            "val_ic": val_ic,
            "val_ic_smoothed": smoothed,
            "best_epoch": best_epoch,
            "skips": skips,
            **{k: float(np.mean(v)) for k, v in acc.items() if k not in ("ic_agents", "w", "raw_std", "dead")},
            "dead_days": np.sum(acc["dead"], axis=0).astype(int).tolist(),
            "ic_agents": np.mean(acc["ic_agents"], axis=0).round(5).tolist(),
            "w_mean": np.mean(acc["w"], axis=0).round(4).tolist(),
            "raw_std_min": np.min(acc["raw_std"], axis=0).tolist(),
            "raw_std_median": np.median(acc["raw_std"], axis=0).tolist(),
            "grad_norms": {k: float(np.mean(v)) for k, v in grad_norms.items()},
        }
        with history_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info(
            "seed %d epoch %d | loss %.4f ic_term %.4f sr %.3f -logdet %.3f |corr| %.3f H %.3f | val_ic %.4f smooth %s | %.0fs",
            seed, epoch, record["loss"], record["ic_term"], record["sharpe"], record["neg_logdet"],
            record["mean_abs_corr"], record["entropy"], val_ic,
            "-" if smoothed is None else f"{smoothed:.4f}", record["seconds"],
        )  # fmt: skip
        if bad >= tc.patience:
            break

    if best_state is None:
        raise TrainingAborted("training ended before any epoch could be selected")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    summary = {
        "seed": seed,
        "config_hash": cfg.hash(),
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_ic_smoothed": best_score,
        "val_ic_curve": val_ics,
        "nonfinite_events": nonfinite_total,
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(cfg: Config, panel: Panel, model_dir: Path) -> torch.nn.Module:
    model = build_model(cfg, d_in=panel.X.shape[-1]).to(dtype=getattr(torch, cfg.model.dtype))
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    return model
