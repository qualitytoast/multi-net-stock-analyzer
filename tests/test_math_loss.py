"""Plan tests 1-8, 11, 12: log det, gradients, soft rank, Spearman, mask wiring, skip path."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import rankdata, spearmanr

from detmaso.config import LossConfig
from detmaso.losses.objective import DetMASOLoss, SharpeState, neg_logdet, rank_eps
from detmaso.losses.softrank import affine_threshold, soft_rank
from detmaso.losses.stats import daily_spearman, masked_zscore, spearman, zscore_rows



@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _equicorr(n: int, rho: float) -> torch.Tensor:
    return (1 - rho) * torch.eye(n) + rho * torch.ones(n, n)


# 1 -------------------------------------------------------------------------------------------
def test_neg_logdet_identity_is_exactly_zero():
    for shrink in (0.0, 1e-4):
        value, retries = neg_logdet(torch.eye(5), shrink)
        assert value.item() == 0.0 and retries == 0


def test_neg_logdet_two_by_two_matches_closed_form():
    for r in (0.0, 0.3, 0.9):
        sigma = torch.tensor([[1.0, r], [r, 1.0]])
        value, _ = neg_logdet(sigma, 0.0)
        assert value.item() == pytest.approx(-math.log(1 - r * r), abs=1e-12)


# 2 -------------------------------------------------------------------------------------------
def test_identical_agents_give_finite_large_penalty_and_penalty_rises_with_correlation():
    value, retries = neg_logdet(torch.ones(5, 5), 1e-4)  # rank 1: all agents identical
    assert math.isfinite(value.item()) and value.item() > 30
    penalties = [neg_logdet(_equicorr(5, rho), 1e-4)[0].item() for rho in (0.0, 0.3, 0.6, 0.9)]
    assert penalties == sorted(penalties)


# 3 -------------------------------------------------------------------------------------------
def test_gradcheck_masked_zscore():
    x = torch.randn(3, 9, requires_grad=True)
    mask = torch.tensor([True] * 7 + [False] * 2).expand(3, 9)
    assert torch.autograd.gradcheck(lambda v: masked_zscore(v, mask), (x,))


@pytest.mark.parametrize("eps", [5.0, 0.01])
def test_gradcheck_soft_rank_both_regimes(eps):
    # Near-ties so that blocks merge at small eps; well-separated values elsewhere.
    x = torch.tensor([[0.0, 0.001, 0.5, 0.503, 1.0, -0.7], [0.2, -0.1, 0.9, 0.904, -0.5, 0.3]], requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: soft_rank(v, eps), (x,), atol=1e-6)


def _loss_fn(n_agents=3, **overrides):
    cfg = LossConfig(sharpe_min_points=2, **overrides)
    return DetMASOLoss(cfg, n_agents)


def _sharpe_with_history(values=(0.01, -0.02, 0.015)) -> SharpeState:
    state = SharpeState()
    for v in values:
        state.push(v)
    return state


def test_gradcheck_full_loss():
    torch.manual_seed(0)
    n_agents, n_stocks = 3, 8
    y = torch.randn(n_stocks) * 0.05
    ok = torch.ones(n_stocks, dtype=torch.bool)
    loss_fn = _loss_fn(n_agents)

    def f(alphas, logits):
        w = torch.softmax(logits, -1)
        alpha_hat = (w.unsqueeze(-1) * alphas).sum(0)
        return loss_fn(alphas, alpha_hat, logits, w, y, ok, 1, _sharpe_with_history()).loss

    alphas = zscore_rows(torch.randn(n_agents, n_stocks)).requires_grad_()
    logits = torch.randn(n_agents, requires_grad=True)
    assert torch.autograd.gradcheck(f, (alphas, logits), atol=1e-6)


# 4 -------------------------------------------------------------------------------------------
def test_logdet_step_repels_correlated_agents():
    torch.manual_seed(1)
    base = torch.randn(200)
    a = torch.stack([base, base + 0.3 * torch.randn(200)]).requires_grad_()

    def corr(x):
        z = zscore_rows(x)
        return (z[0] * z[1]).mean()

    before = corr(a).item()
    z = zscore_rows(a)
    value, _ = neg_logdet(z @ z.T / z.shape[1], 1e-4)
    value.backward()
    with torch.no_grad():
        after = corr(a - 0.05 * a.grad / a.grad.norm()).item()
    assert abs(after) < abs(before)


# 5 -------------------------------------------------------------------------------------------
def _step_increases(term_name: str, overrides: dict):
    torch.manual_seed(2)
    n_agents, n_stocks = 3, 30
    y = torch.randn(n_stocks) * 0.05
    ok = torch.ones(n_stocks, dtype=torch.bool)
    loss_fn = _loss_fn(n_agents, **overrides)
    alphas = torch.randn(n_agents, n_stocks, requires_grad=True)
    logits = torch.randn(n_agents, requires_grad=True)

    def terms(al, lg):
        z = zscore_rows(al)
        w = torch.softmax(lg, -1)
        return loss_fn(z, (w.unsqueeze(-1) * z).sum(0), lg, w, y, ok, 0, _sharpe_with_history())

    t0 = terms(alphas, logits)
    t0.loss.backward()
    with torch.no_grad():
        t1 = terms(alphas - 1e-3 * alphas.grad, logits - 1e-3 * logits.grad)
    return getattr(t0, term_name).item(), getattr(t1, term_name).item()


@pytest.mark.parametrize(
    "term,overrides",
    [
        ("ic_term", {"c_ic": 1.0, "c_sr": 0.0, "c_div": 0.0, "c_ent": 0.0}),
        ("sr", {"c_ic": 0.0, "c_sr": 1.0, "c_div": 0.0, "c_ent": 0.0}),
        ("entropy", {"c_ic": 0.0, "c_sr": 0.0, "c_div": 0.0, "c_ent": 1.0}),
    ],
)
def test_one_step_on_a_single_term_improves_it(term, overrides):
    before, after = _step_increases(term, overrides)
    assert after > before


def test_one_step_on_the_diversity_term_reduces_neg_logdet():
    before, after = _step_increases("neg_logdet", {"c_ic": 0.0, "c_sr": 0.0, "c_div": 1.0, "c_ent": 0.0})
    assert after < before


def test_rank_eps_uses_the_label_ok_count(monkeypatch):
    import detmaso.losses.objective as obj

    seen = []
    real = obj.soft_rank
    monkeypatch.setattr(obj, "soft_rank", lambda x, eps: (seen.append((x.shape[1], eps)), real(x, eps))[1])
    ok = torch.zeros(40, dtype=torch.bool)
    ok[:30] = True
    al = zscore_rows(torch.randn(3, 40))
    w = torch.full((3,), 1 / 3)
    _loss_fn(3, kappa_soft=0.5)(al, al.mean(0), torch.zeros(3), w, torch.randn(40), ok, 0, SharpeState())
    assert seen == [(30, pytest.approx(0.5 * 30 / 100))]


# 6 -------------------------------------------------------------------------------------------
def test_sharpe_history_is_detached_constants():
    state = _sharpe_with_history()
    assert all(isinstance(v, float) for v in (state.s1, state.s2))
    state.push(torch.tensor(0.5).item())
    n, s1, s2 = state.sums()
    assert n == 4 and isinstance(s1, float)


def test_sharpe_window_is_expanding_by_default_and_rolling_when_set():
    expanding = SharpeState()
    rolling = SharpeState(window=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        expanding.push(v)
        rolling.push(v)
    assert expanding.sums() == (4, 10.0, 30.0)
    assert rolling.sums() == (2, 7.0, 25.0)  # keeps window - 1 past values; today's is added live


# 7 -------------------------------------------------------------------------------------------
def _brute_force_soft_rank(x: np.ndarray, eps: float) -> np.ndarray:
    """Projection of x / eps onto the permutahedron of (1..n), solved as a QP with cvxpy."""
    import itertools

    import cvxpy as cp

    n = len(x)
    w = np.arange(1, n + 1, dtype=float)
    mu = cp.Variable(n)
    constraints = [cp.sum(mu) == w.sum()]
    for k in range(1, n):
        top_k = np.sort(w)[::-1][:k].sum()
        for subset in itertools.combinations(range(n), k):
            constraints.append(cp.sum(mu[list(subset)]) <= top_k)
    cp.Problem(cp.Minimize(cp.sum_squares(mu - x / eps)), constraints).solve()
    return mu.value


@pytest.mark.parametrize("eps", [0.05, 0.3, 2.0])
def test_soft_rank_matches_brute_force_projection(eps):
    rng = np.random.default_rng(3)
    x = rng.normal(size=5)
    ours = soft_rank(torch.tensor(x).unsqueeze(0), eps)[0].numpy()
    np.testing.assert_allclose(ours, _brute_force_soft_rank(x, eps), atol=1e-4)


def test_soft_rank_hard_and_affine_limits():
    x = torch.tensor([[0.3, -1.2, 2.5, 0.9, -0.1]])
    np.testing.assert_allclose(soft_rank(x, 1e-6)[0].numpy(), rankdata(x[0].numpy()))
    big = 100.0
    expected = x / big - (x / big).mean() + (x.shape[1] + 1) / 2
    np.testing.assert_allclose(soft_rank(x, big).numpy(), expected.numpy(), atol=1e-12)
    assert affine_threshold(x).item() <= big


def test_soft_rank_preserves_order_and_ties():
    x = torch.tensor([[0.5, 0.5, -1.0, 2.0, 0.1]])
    r = soft_rank(x, 0.2)[0]
    assert r[0].item() == pytest.approx(r[1].item())
    assert r[2] < r[4] < r[0] < r[3]


def test_rank_eps_formula():
    assert rank_eps(1.0, 300) == pytest.approx(3.0)
    assert rank_eps(0.0015, 500) == pytest.approx(0.0075)


# 8 -------------------------------------------------------------------------------------------
def test_spearman_matches_scipy_with_ties_and_masking():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 5, size=50).astype(float)  # many ties
    b = a + rng.normal(size=50)
    assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)
    scores = np.vstack([a, b])
    labels = np.vstack([b, a])
    labels[0, :10] = np.nan
    mask = np.ones_like(scores, dtype=bool)
    ic = daily_spearman(scores, labels, mask)
    assert ic[0] == pytest.approx(spearmanr(a[10:], b[10:]).statistic, abs=1e-12)


# 11 ------------------------------------------------------------------------------------------
def test_mask_wiring_only_the_regime_agent_learns_from_ic():
    from detmaso.model.detmaso import DetMASO

    torch.manual_seed(5)
    model = DetMASO(n_agents=5, d_in=6, hidden=(8, 4), activation="gelu").double()
    X = torch.randn(1, 40, 6)
    U = torch.ones(1, 40, dtype=torch.bool)
    S = torch.randn(1, 12)
    out = model(X, U, S)
    loss_fn = DetMASOLoss(LossConfig(c_sr=0.0, c_div=0.0, c_ent=0.0, lambda_mask=0.0), 5)
    regime = 2
    terms = loss_fn(out.alphas[0], out.alpha_hat[0], out.logits[0], out.w[0], torch.randn(40) * 0.05, U[0], regime, SharpeState())
    terms.loss.backward()
    for i, agent in enumerate(model.agents):
        grad = sum(float(p.grad.abs().sum()) for p in agent.parameters())
        assert (grad > 0) == (i == regime), (i, grad)


def test_regime_assignment_uses_agent_order():
    from detmaso.data.regimes import RegimeModel

    centroids = np.eye(5, 12) * 3
    model = RegimeModel(mean=np.zeros(12), std=np.ones(12), centroids=centroids, names=list("ABCDE"), state_names=[str(i) for i in range(12)], silhouette={}, fit_end="2020-01-01")
    assert model.assign(centroids).tolist() == [0, 1, 2, 3, 4]


# 12 ------------------------------------------------------------------------------------------
def test_skip_path_aborts_after_consecutive_skips(tmp_path):
    from tests.test_end_to_end import synthetic_panel_and_splits
    from tests.conftest import synthetic_config
    from detmaso.train.trainer import TrainingAborted, train_seed

    cfg = synthetic_config(tmp_path, train={"max_consecutive_skips": 3}, features={"min_label_ok": 10**9})
    panel, splits = synthetic_panel_and_splits()
    with pytest.raises(TrainingAborted, match="3 consecutive skipped dates"):
        train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "seed0")
    assert not (tmp_path / "seed0" / "model.pt").exists()


def test_dead_agent_gets_zero_score_and_others_keep_learning():
    from detmaso.model.detmaso import DetMASO

    torch.manual_seed(6)
    model = DetMASO(n_agents=5, d_in=6, hidden=(8, 4), activation="gelu").double()
    with torch.no_grad():  # make agent 3 output a constant
        last = model.agents[3].net[-1]
        last.weight.zero_()
    X, U, S = torch.randn(1, 40, 6), torch.ones(1, 40, dtype=torch.bool), torch.randn(1, 12)
    out = model(X, U, S)
    assert torch.all(out.alphas[0, 3] == 0)
    terms = DetMASOLoss(LossConfig(), 5)(out.alphas[0], out.alpha_hat[0], out.logits[0], out.w[0], torch.randn(40) * 0.05, U[0], 0, _sharpe_with_history())
    assert torch.isfinite(terms.loss) and terms.n_dead == 1
    terms.loss.backward()
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in model.agents[3].parameters())
    for i in (0, 1, 2, 4):
        assert sum(float(p.grad.abs().sum()) for p in model.agents[i].parameters()) > 0


def test_trainer_survives_a_permanently_dead_agent(tmp_path, monkeypatch):
    """A dead agent (constant output) must not cause non-finite skips; its days are counted."""
    import json as _json

    import detmaso.train.trainer as trainer_mod
    from detmaso.model.detmaso import DetMASO
    from tests.conftest import synthetic_config
    from tests.test_end_to_end import synthetic_panel_and_splits

    real_from_config = DetMASO.from_config

    def dead_from_config(cfg, d_in):
        model = real_from_config(cfg, d_in)
        torch.nn.init.zeros_(model.agents[3].net[-1].weight)
        return model

    monkeypatch.setattr(DetMASO, "from_config", staticmethod(dead_from_config))  # used by model.factory.build_model
    cfg = synthetic_config(tmp_path, train={"max_epochs": 3})
    panel, splits = synthetic_panel_and_splits()
    trainer_mod.train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "s")
    history = [_json.loads(line) for line in (tmp_path / "s" / "history.jsonl").read_text().splitlines()]
    assert all(h["skips"]["nonfinite"] == 0 for h in history)
    assert all(h["dead_days"][3] == len(splits.train_fit) for h in history)


def test_trainer_skips_a_nonfinite_date_and_continues(tmp_path):
    from tests.conftest import synthetic_config
    from tests.test_end_to_end import synthetic_panel_and_splits
    from detmaso.train.trainer import train_seed

    cfg = synthetic_config(tmp_path, train={"max_epochs": 3})
    panel, splits = synthetic_panel_and_splits()
    t = int(panel.date_pos(splits.train_fit[10:11])[0])
    panel.y[t, 0] = float("nan")  # a corrupt label on one training date -> non-finite loss
    summary = train_seed(cfg, panel, splits, seed=0, out_dir=tmp_path / "s")
    import json as _json

    history = [_json.loads(line) for line in (tmp_path / "s" / "history.jsonl").read_text().splitlines()]
    assert [h["skips"]["nonfinite"] for h in history] == [1, 1, 1]
    assert all(np.isfinite(h["loss"]) for h in history) and summary["nonfinite_events"] == 3


# Step 1: one network, no regime mask -----------------------------------------------------------
def _long_sharpe_history(n: int = 25) -> SharpeState:
    rng = np.random.default_rng(1)
    return _sharpe_with_history(tuple(rng.normal(0.01, 0.02, size=n)))


def test_single_network_loss_is_the_ic_and_sharpe_terms_only():
    torch.manual_seed(0)
    n = 40
    alphas = masked_zscore(torch.randn(1, n, dtype=torch.float64, requires_grad=True), torch.ones(1, n, dtype=torch.bool))
    y = torch.randn(n, dtype=torch.float64) * 0.05
    cfg = LossConfig(c_div=0.0, c_ent=0.0)
    terms = DetMASOLoss(cfg, 1)(
        alphas, alphas[0], torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64), y,
        torch.ones(n, dtype=torch.bool), None, _long_sharpe_history(),
    )  # fmt: skip
    sr, ic_term, loss = float(terms.sr.detach()), float(terms.ic_term.detach()), float(terms.loss.detach())
    assert sr != 0.0  # the Sharpe term is active
    assert ic_term == float(terms.ic[0])  # weight 1, no regime mask
    assert float(terms.entropy.detach()) == 0.0 and abs(float(terms.neg_logdet.detach())) < 1e-12
    assert loss == float((-cfg.c_ic * terms.ic_term - cfg.c_sr * terms.sr).detach())


def test_single_network_constant_output_day_scores_zero_without_crashing():
    # DEVIATIONS 4: a constant output scores 0 for that date; with one network no agent is live.
    from detmaso.model.single import SingleMLP

    torch.manual_seed(0)
    model = SingleMLP(d_in=6, hidden=(8, 4), activation="gelu").double()
    with torch.no_grad():
        model.agent.net[-1].weight.zero_()
    X, U, S = torch.randn(1, 40, 6, dtype=torch.float64), torch.ones(1, 40, dtype=torch.bool), torch.randn(1, 12)
    out = model(X, U, S)
    assert torch.all(out.alphas == 0)
    terms = DetMASOLoss(LossConfig(c_div=0.0, c_ent=0.0), 1)(
        out.alphas[0], out.alpha_hat[0], out.logits[0], out.w[0], torch.randn(40, dtype=torch.float64) * 0.05, U[0], None,
        _long_sharpe_history(),
    )  # fmt: skip
    assert terms.n_dead == 1 and math.isnan(terms.lambda_min)
    assert float(terms.ic[0]) == 0.0 and torch.isfinite(terms.loss)
    terms.loss.backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in model.parameters())


def test_no_regime_mask_weights_every_agent_ic_fully():
    torch.manual_seed(0)
    n, k = 40, 5
    alphas = masked_zscore(torch.randn(k, n, dtype=torch.float64), torch.ones(k, n, dtype=torch.bool))
    logits = torch.randn(k, dtype=torch.float64)
    w = torch.softmax(logits, -1)
    y = torch.randn(n, dtype=torch.float64)
    args = (alphas, (w[:, None] * alphas).sum(0), logits, w, y, torch.ones(n, dtype=torch.bool))
    loss_fn = DetMASOLoss(LossConfig(lambda_mask=0.1), k)
    unmasked = loss_fn(*args, None, SharpeState())
    masked = loss_fn(*args, 2, SharpeState())
    ic = torch.as_tensor(unmasked.ic, dtype=torch.float64)
    assert float(unmasked.ic_term) == pytest.approx(float((w * ic).sum()), abs=1e-15)
    m = torch.full((k,), 0.1, dtype=torch.float64)
    m[2] = 1.0
    assert float(masked.ic_term) == pytest.approx(float((w * m * ic).sum()), abs=1e-15)
