"""Scheduler load-balancing penalty (TODO "Fix the scheduler collapse"; docs/RULES.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from detmaso.losses.balance import BalancePenalty, effective_agents, kl_to_target, scheduler_usage, target_shares
from detmaso.model.scheduler import Scheduler
from tests.conftest import synthetic_config
from tests.test_end_to_end import synthetic_panel_and_splits

MOODS = torch.tensor([0.526, 0.079, 0.322, 0.051, 0.022], dtype=torch.float64)


def test_kl_is_zero_at_the_target_and_positive_elsewhere():
    assert float(kl_to_target(MOODS, MOODS.clone())) == pytest.approx(0.0, abs=1e-15)
    assert float(kl_to_target(MOODS, torch.full((5,), 0.2, dtype=torch.float64))) > 0
    # an agent with a zero target is ignored (0 * ln 0 = 0), and a zero usage is floored, not infinite
    t = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    assert float(kl_to_target(t, torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))) == pytest.approx(0.0, abs=1e-15)
    assert np.isfinite(float(kl_to_target(t, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))))


def test_target_shares():
    assert torch.allclose(target_shares("uniform", [], 4), torch.full((4,), 0.25, dtype=torch.float64))
    shares = target_shares("regime", [0, 0, 2, 2, 2, 4], 5)
    assert shares.tolist() == pytest.approx([2 / 6, 0.0, 3 / 6, 0.0, 1 / 6]) and float(shares.sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="agents"):
        target_shares("regime", [0, 5], 5)
    with pytest.raises(ValueError, match="unknown"):
        target_shares("equal", [0], 5)


def test_effective_agents():
    assert effective_agents([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_agents([0.2] * 5) == pytest.approx(5.0)
    assert effective_agents(MOODS.numpy()) == pytest.approx(3.12, abs=0.01)  # perfect specialists on our training days


def test_penalty_gradcheck():
    torch.manual_seed(0)
    sch = Scheduler(5).double()
    torch.nn.init.normal_(sch.head2.weight, std=0.3)  # not the uniform start, so the gradient is non-trivial
    s = torch.randn(30, 12, dtype=torch.float64)
    # the module's penalty equals the same computation written out with the output layer as inputs
    with torch.no_grad():
        direct = float(BalancePenalty(s, MOODS)(sch))
        written_out = float(_penalty_from_head(sch, s, sch.head2.weight, sch.head2.bias))
    assert direct == pytest.approx(written_out, abs=1e-14)
    w0, b0 = sch.head2.weight.detach().clone(), sch.head2.bias.detach().clone()
    assert torch.autograd.gradcheck(lambda w, b: _penalty_from_head(sch, s, w, b), (w0.requires_grad_(), b0.requires_grad_()))


def _penalty_from_head(sch: Scheduler, s: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    logits = torch.nn.functional.linear(torch.relu(sch.head1(_fused(sch, s))), w, b)
    return kl_to_target(MOODS, torch.softmax(logits, -1).mean(0))


def _fused(sch: Scheduler, s: torch.Tensor) -> torch.Tensor:
    parts = s.view(s.shape[0], sch.n_windows, sch.stats_per_window)
    h = torch.stack([enc(parts[:, k]) for k, enc in enumerate(sch.encoders)], dim=1)
    a = torch.softmax(sch.attn(h.flatten(1)), dim=-1)
    return (a.unsqueeze(-1) * h).sum(dim=1)


def test_one_step_on_the_penalty_moves_usage_toward_the_target():
    torch.manual_seed(1)
    sch = Scheduler(5).double()
    s = torch.randn(50, 12, dtype=torch.float64)
    pen = BalancePenalty(s, MOODS)
    before = float(pen(sch).detach())  # uniform start: equal shares, not the mood shares
    opt = torch.optim.SGD(sch.parameters(), lr=0.5)
    opt.zero_grad()
    pen(sch).backward()
    opt.step()
    assert float(pen(sch).detach()) < before


def _recovery(c_bal: float, steps: int = 2000) -> np.ndarray:
    """A saturated scheduler (agents 1, 3, 4 starved) trained with AdamW on an IC-like term that favours
    agents 0 and 2, plus c_bal x the penalty. Returns usage / target."""
    torch.manual_seed(0)
    sch = Scheduler(5).double()
    with torch.no_grad():
        sch.head2.bias.copy_(torch.tensor([6.0, 0.0, 3.0, 0.0, 0.0]))
    s = torch.randn(200, 12, dtype=torch.float64)
    pen = BalancePenalty(s, MOODS)
    opt = torch.optim.AdamW(sch.parameters(), lr=1e-3, weight_decay=1e-4)
    ic = torch.tensor([0.08, 0.01, 0.07, 0.01, 0.01], dtype=torch.float64)
    days = torch.randint(0, 200, (steps,), generator=torch.Generator().manual_seed(1))
    for d in days.tolist():
        logits, _ = sch(s[d : d + 1])
        loss = -(torch.softmax(logits, -1)[0] * ic).sum()
        if c_bal:
            loss = loss + c_bal * pen(sch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (scheduler_usage(sch, s) / MOODS).numpy()


def test_adamw_recovers_starved_agents_with_the_penalty_and_not_without():
    with_penalty, without = _recovery(0.1), _recovery(0.0)
    assert np.all(with_penalty >= 0.5), with_penalty  # Rule M's bar: at least half the target share
    assert np.all(without[[1, 3, 4]] < 0.05), without


def _balance_config(tmp_path, c_bal: float, epochs: int = 3, **train):
    return synthetic_config(
        tmp_path, variant="bal_test", train={"max_epochs": epochs, **train}, model={"dtype": "float64"},
        loss={"c_bal": c_bal, "balance_target": "regime"},
    )  # fmt: skip


def test_training_with_the_penalty_is_deterministic_and_logs_usage(tmp_path):
    from detmaso.train.trainer import train_seed

    panel, splits = synthetic_panel_and_splits()
    a = train_seed(_balance_config(tmp_path / "a", 0.1), panel, splits, seed=2, out_dir=tmp_path / "a" / "s")
    b = train_seed(_balance_config(tmp_path / "b", 0.1), panel, splits, seed=2, out_dir=tmp_path / "b" / "s")
    assert a["val_ic_curve"] == b["val_ic_curve"]
    history = [json.loads(line) for line in (tmp_path / "a" / "s" / "history.jsonl").read_text().splitlines()]
    for h in history:
        assert {"balance", "clip_active", "clip_scale", "usage_train", "effective_agents_train"} <= set(h)
        assert "bal" in h["grad_norms"] and sum(h["usage_train"]) == pytest.approx(1.0, abs=1e-5)
    assert (tmp_path / "a" / "s" / "model_last.pt").exists() and (tmp_path / "a" / "s" / "model.pt").exists()


def test_paper_runs_log_nothing_new(tmp_path):
    from detmaso.train.trainer import train_seed

    panel, splits = synthetic_panel_and_splits()
    cfg = synthetic_config(tmp_path, train={"max_epochs": 3}, model={"dtype": "float64"})
    train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "s")
    history = [json.loads(line) for line in (tmp_path / "s" / "history.jsonl").read_text().splitlines()]
    assert not ({"balance", "clip_active", "usage_train"} & set(history[0])) and "bal" not in history[0]["grad_norms"]
    summary = json.loads((tmp_path / "s" / "summary.json").read_text())
    assert "validated" not in summary and (tmp_path / "s" / "model_last.pt").exists()


def test_a_strong_penalty_brings_usage_to_the_target(tmp_path):
    """The synthetic market's regimes are agents 0 and 2 in equal halves (target 0.5 / 0.5)."""
    from detmaso.train.trainer import train_seed

    panel, splits = synthetic_panel_and_splits()
    train_seed(_balance_config(tmp_path / "on", 1.0, epochs=4), panel, splits, seed=0, out_dir=tmp_path / "on" / "s")
    history = [json.loads(line) for line in (tmp_path / "on" / "s" / "history.jsonl").read_text().splitlines()]
    usage = np.asarray(history[-1]["usage_train"])
    assert usage[0] >= 0.25 and usage[2] >= 0.25, usage  # at least half of each 0.5 target
    assert history[-1]["balance"] < history[0]["balance"] or history[0]["balance"] < 0.05


def test_training_only_runs_never_look_at_validation(tmp_path, monkeypatch):
    import detmaso.train.trainer as trainer_mod

    def forbidden(*args, **kwargs):
        raise AssertionError("validation was evaluated")

    monkeypatch.setattr(trainer_mod, "validation_ic", forbidden)
    panel, splits = synthetic_panel_and_splits()
    cfg = _balance_config(tmp_path, 0.1, epochs=2, validate=False)
    summary = trainer_mod.train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "s")
    assert summary["validated"] is False and summary["val_ic_curve"] == [] and summary["best_epoch"] == 2
    last = torch.load(tmp_path / "s" / "model_last.pt", weights_only=True)
    best = torch.load(tmp_path / "s" / "model.pt", weights_only=True)
    assert all(torch.equal(last[k], best[k]) for k in last)  # without validation the last epoch is the model
