import numpy as np
import pytest

from mmsim import Params, AvellanedaStoikov, Symmetric, exact_moments
from mmsim.control import optimal_policy
from mmsim.strategies import GLFT, TablePolicy

P = Params()
SMALL = Params(T=0.6, dt=0.1, sigma=2.0, A=6.0, k=1.5)   # 6 steps with large fill probabilities


@pytest.mark.parametrize("kw", [{}, {"impact": 0.3}, {"fill_cost": 0.2}])
@pytest.mark.parametrize("gamma", [0.1, 1.0])
@pytest.mark.parametrize("base", [SMALL, P], ids=["small", "paper"])
def test_optimizer_value_matches_evaluation(base, gamma, kw):
    """The CE the optimiser reports is the exact CE of the policy it returns."""
    p = base.with_(**kw)
    pol, ce = optimal_policy(p, gamma)
    assert exact_moments(pol, p, gamma_ce=gamma)["ce"] == pytest.approx(ce, abs=1e-9)


@pytest.mark.parametrize("kw", [{}, {"impact": 0.3}, {"fill_cost": 0.2}, {"noise": "gaussian"}])
@pytest.mark.parametrize("gamma", [0.1, 1.0, 3.0])
def test_no_perturbation_improves_the_optimum(gamma, kw):
    """Moving the quotes at any reachable state, one side or both, never raises the exact CE."""
    p = SMALL.with_(**kw)
    pol, ce = optimal_policy(p, gamma)
    n = p.n_steps
    rng = np.random.default_rng(0)
    for _ in range(60):
        i = int(rng.integers(n))
        j = int(rng.integers(-i, i + 1)) + n
        da, db = pol.delta_a.copy(), pol.delta_b.copy()
        da[i, j] += rng.choice([0.0, -0.3, -0.03, 0.03, 0.3])
        db[i, j] += rng.choice([-0.3, -0.03, 0.03, 0.3])
        assert exact_moments(TablePolicy(da, db), p, gamma_ce=gamma)["ce"] <= ce + 1e-10


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_optimum_beats_the_heuristics(gamma):
    pol, ce = optimal_policy(P, gamma)
    for strat in [AvellanedaStoikov(gamma), GLFT(gamma), Symmetric.matching(gamma, P)]:
        assert exact_moments(strat, P, gamma_ce=gamma)["ce"] <= ce + 1e-9


@pytest.mark.parametrize("fill_cost", [0.0, 0.3])
def test_risk_neutral_limit(fill_cost):
    """As gamma -> 0 (no impact), every reachable quote tends to 1/k plus the per-fill cost."""
    pol, _ = optimal_policy(P.with_(fill_cost=fill_cost), 1e-7)
    n = P.n_steps
    for i in range(n):
        np.testing.assert_allclose(pol.delta_a[i, n - i:n + i + 1], 1 / P.k + fill_cost, atol=1e-4)
        np.testing.assert_allclose(pol.delta_b[i, n - i:n + i + 1], 1 / P.k + fill_cost, atol=1e-4)


@pytest.mark.parametrize("kw", [{}, {"impact": 0.3}, {"fill_cost": 0.2}])
def test_optimal_quotes_are_symmetric(kw):
    """Long q on the ask side mirrors short q on the bid side."""
    pol, _ = optimal_policy(P.with_(**kw), 0.1)
    np.testing.assert_allclose(pol.delta_a, pol.delta_b[:, ::-1], atol=1e-9)


def test_optimum_is_stationary_and_close_to_glft_far_from_the_horizon():
    g, n = 0.1, P.n_steps
    pol, _ = optimal_policy(P, g)
    near_zero = slice(n - 10, n + 11)
    np.testing.assert_allclose(pol.delta_a[0, near_zero], pol.delta_a[100, near_zero], atol=1e-3)
    q = np.arange(-3.0, 4.0)
    _, bid, ask = GLFT(g).quote(np.zeros(q.size), q, 0, P)
    np.testing.assert_allclose(pol.delta_a[0, n - 3:n + 4], ask, atol=0.02)
    np.testing.assert_allclose(pol.delta_b[0, n - 3:n + 4], -bid, atol=0.02)
