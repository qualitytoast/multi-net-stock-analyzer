"""Plan tests 9-10: scheduler initialization and agent output standardization."""

from __future__ import annotations

import torch

from detmaso.model.detmaso import DetMASO


def _model(seed: int = 0) -> DetMASO:
    torch.manual_seed(seed)
    return DetMASO(n_agents=5, d_in=34, hidden=(16, 8, 4), activation="gelu").double()


# 9 -------------------------------------------------------------------------------------------
def test_scheduler_starts_uniform_and_outputs_are_distributions():
    model = _model()
    X = torch.randn(3, 20, 34, dtype=torch.float64)
    U = torch.ones(3, 20, dtype=torch.bool)
    S = torch.randn(3, 12, dtype=torch.float64)
    out = model(X, U, S)
    assert torch.allclose(out.w, torch.full_like(out.w, 0.2), atol=0, rtol=0)
    assert torch.allclose(out.w.sum(-1), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(out.a.sum(-1), torch.ones(3, dtype=torch.float64))


def test_scheduler_does_not_depend_on_stock_data():
    model = _model()
    for p in model.scheduler.head2.parameters():  # move off the uniform start
        torch.nn.init.normal_(p)
    S = torch.randn(1, 12, dtype=torch.float64)
    U = torch.ones(1, 20, dtype=torch.bool)
    w1 = model(torch.randn(1, 20, 34, dtype=torch.float64), U, S).w
    w2 = model(torch.randn(1, 20, 34, dtype=torch.float64), U, S).w
    assert torch.equal(w1, w2)


# 10 ------------------------------------------------------------------------------------------
def test_agent_outputs_standardized_over_universe_and_padding_does_not_leak():
    model = _model()
    X = torch.randn(1, 30, 34, dtype=torch.float64)
    U = torch.zeros(1, 30, dtype=torch.bool)
    U[0, :24] = True
    S = torch.randn(1, 12, dtype=torch.float64)
    out = model(X, U, S)
    a = out.alphas[0][:, :24]
    assert torch.allclose(a.mean(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12)
    assert torch.allclose((a * a).mean(-1), torch.ones(5, dtype=torch.float64), atol=1e-12)
    assert torch.all(out.alphas[0][:, 24:] == 0)
    X2 = X.clone()
    X2[0, 24:] = 1e6  # garbage in padded rows
    out2 = model(X2, U, S)
    assert torch.equal(out.alphas[0][:, :24], out2.alphas[0][:, :24])
