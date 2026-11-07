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


# Step 1: the single-network baseline ----------------------------------------------------------
def _single(market_state_input: bool = False, seed: int = 0):
    from detmaso.model.single import SingleMLP

    torch.manual_seed(seed)
    return SingleMLP(d_in=34, hidden=(16, 8, 4), activation="gelu", market_state_input=market_state_input).double()


def test_single_network_output_is_standardized_and_padding_does_not_leak():
    model = _single()
    X = torch.randn(2, 30, 34, dtype=torch.float64)
    U = torch.zeros(2, 30, dtype=torch.bool)
    U[:, :24] = True
    S = torch.randn(2, 12, dtype=torch.float64)
    out = model(X, U, S)
    a = out.alphas[:, 0, :24]
    assert out.alphas.shape == (2, 1, 30) and out.a.shape == (2, 0)
    assert torch.allclose(a.mean(-1), torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose((a * a).mean(-1), torch.ones(2, dtype=torch.float64), atol=1e-12)
    assert torch.all(out.alphas[:, 0, 24:] == 0)
    assert torch.equal(out.alpha_hat, out.alphas[:, 0]) and torch.all(out.w == 1) and torch.all(out.logits == 0)
    X2 = X.clone()
    X2[:, 24:] = 1e6
    assert torch.equal(model(X2, U, S).alphas[:, 0, :24], out.alphas[:, 0, :24])


def test_market_state_input_is_shared_by_all_stocks_and_ignored_when_off():
    X = torch.randn(1, 20, 34, dtype=torch.float64)
    X[0, 1] = X[0, 0]  # two identical stocks
    U = torch.ones(1, 20, dtype=torch.bool)
    S1, S2 = torch.randn(1, 12, dtype=torch.float64), torch.randn(1, 12, dtype=torch.float64)
    on = _single(market_state_input=True)
    out = on(X, U, S1)
    assert out.alphas[0, 0, 0] == out.alphas[0, 0, 1]  # the state is the same for every stock
    assert not torch.allclose(out.alphas, on(X, U, S2).alphas)  # but it changes how features are weighed
    off = _single(market_state_input=False)
    assert torch.equal(off(X, U, S1).alphas, off(X, U, S2).alphas)


def test_model_factory_and_weight_counts():
    from detmaso.config import load_config
    from detmaso.model.detmaso import DetMASO
    from detmaso.model.factory import build_model
    from detmaso.model.single import SingleMLP
    from tests.conftest import ROOT

    expected = {"csi300": (DetMASO, 26_634), "csi300_single_mlp": (SingleMLP, 4_865),
                "csi300_single_mlp_wide": (SingleMLP, 26_731), "csi300_single_mlp_state": (SingleMLP, 5_633)}  # fmt: skip
    for name, (cls, n_weights) in expected.items():
        model = build_model(load_config(ROOT / "configs" / f"{name}.yaml"), d_in=34)
        assert isinstance(model, cls) and sum(p.numel() for p in model.parameters()) == n_weights
    assert DetMASO(n_agents=5, d_in=34, hidden=(16, 8, 4)).n_agents == 5 and DetMASO.uses_regimes
    assert SingleMLP.n_agents == 1 and not SingleMLP.uses_regimes
