import numpy as np
import pytest

from mmsim import (Params, AvellanedaStoikov, AdjustedAS, Symmetric, simulate, draw_randomness,
                   symmetric_moments, symmetric_moments_continuous, average_as_spread, paired_diff)

P = Params()


@pytest.mark.parametrize("impact,fill_cost", [(0.0, 0.0), (0.3, 0.0), (0.0, 0.2)])
@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("which", ["inventory", "symmetric"])
def test_pnl_decomposition_is_exact(gamma, which, impact, fill_cost):
    p = P.with_(impact=impact, fill_cost=fill_cost)
    strat = AvellanedaStoikov(gamma) if which == "inventory" else Symmetric.matching(gamma, P)
    r = simulate(strat, p, n_paths=500, seed=1)
    np.testing.assert_allclose(r.pnl, r.spread_pnl + r.inventory_pnl, atol=1e-8)


def test_trade_log_consistent_with_totals():
    r = simulate(AvellanedaStoikov(0.1), P, n_paths=300, seed=2, log_trades=True)
    tr = r.trades
    assert tr["path"].size == r.n_trades.sum()
    edge = tr["side"] * (tr["mid"] - tr["price"])  # buy below mid / sell above mid => positive edge
    per_path = np.bincount(tr["path"], weights=edge, minlength=300)
    np.testing.assert_allclose(per_path, r.spread_pnl, atol=1e-8)
    final_q = np.bincount(tr["path"], weights=tr["side"], minlength=300)
    np.testing.assert_allclose(final_q, r.q)


def test_same_seed_is_reproducible():
    a = simulate(AvellanedaStoikov(0.1), P, n_paths=200, seed=5)
    b = simulate(AvellanedaStoikov(0.1), P, n_paths=200, seed=5)
    np.testing.assert_array_equal(a.pnl, b.pnl)


def test_common_random_numbers_share_price_path():
    a = simulate(AvellanedaStoikov(0.1), P, n_paths=50, seed=3, record=True)
    b = simulate(Symmetric.matching(0.1, P), P, n_paths=50, seed=3, record=True)
    np.testing.assert_array_equal(a.paths["s"], b.paths["s"])


def test_reservation_price_moves_against_inventory():
    strat = AvellanedaStoikov(0.1)
    s = np.array([100.0, 100.0, 100.0])
    r, bid, ask = strat.quote(s, np.array([-3.0, 0.0, 3.0]), 0, P)
    assert r[0] > 100.0 and r[1] == 100.0 and r[2] < 100.0
    np.testing.assert_allclose(ask - bid, ask[1] - bid[1])  # spread does not depend on inventory


def test_average_spread_matches_paper():
    assert average_as_spread(0.1, P) == pytest.approx(1.49, abs=0.01)
    assert average_as_spread(0.01, P) == pytest.approx(1.35, abs=0.01)
    assert average_as_spread(1.0, P) == pytest.approx(3.02, abs=0.015)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_symmetric_matches_closed_form(gamma):
    strat = Symmetric.matching(gamma, P)
    th = symmetric_moments(strat.spread, P)
    n = 40000
    r = simulate(strat, P, n_paths=n, seed=11)
    assert abs(r.pnl.mean() - th["mean_pnl"]) < 4 * th["std_pnl"] / np.sqrt(n)
    assert abs(r.pnl.std() - th["std_pnl"]) < 4 * th["std_pnl"] / np.sqrt(2 * n)
    assert abs(r.q.std() - th["std_q"]) < 4 * th["std_q"] / np.sqrt(2 * n)


@pytest.mark.parametrize("which", ["inventory", "symmetric"])
def test_inventory_pnl_has_zero_mean_without_adverse_selection(which):
    strat = AvellanedaStoikov(0.1) if which == "inventory" else Symmetric.matching(0.1, P)
    r = simulate(strat, P, n_paths=20000, seed=7)
    se = r.inventory_pnl.std() / np.sqrt(r.inventory_pnl.size)
    assert abs(r.inventory_pnl.mean()) < 4 * se


def test_small_gamma_converges_to_symmetric():
    g = 1e-6
    rnd = draw_randomness(P, 2000, seed=9)
    a = simulate(AvellanedaStoikov(g), P, n_paths=2000, randomness=rnd)
    b = simulate(Symmetric.matching(g, P), P, n_paths=2000, randomness=rnd)
    assert np.mean(np.abs(a.pnl - b.pnl)) < 0.05


def test_paired_diff_beats_independent_se():
    rnd = draw_randomness(P, 2000, seed=4)
    a = simulate(AvellanedaStoikov(0.1), P, n_paths=2000, randomness=rnd)
    b = simulate(Symmetric.matching(0.1, P), P, n_paths=2000, randomness=rnd)
    d = paired_diff(a.pnl, b.pnl)
    assert d["se"] < d["se_independent"]


def test_discrete_moments_converge_to_continuous_limit():
    spread = Symmetric.matching(0.1, P).spread
    c = symmetric_moments_continuous(spread, P)
    d = symmetric_moments(spread, P.with_(dt=1e-5))
    for key in ["mean_pnl", "std_q", "std_pnl"]:
        assert d[key] == pytest.approx(c[key], rel=1e-3)


# ---------- exact dynamic-programming solution ----------
import itertools
from mmsim import exact_moments, sampling_se, cara_ce


def brute_force(strategy, p, gamma):
    """Enumerate every fill/price outcome sequence of a tiny model and compute exact statistics directly."""
    n = p.n_steps
    codes = np.array(list(itertools.product(range(8), repeat=n)))   # 4 fill outcomes x 2 price moves per step
    m = codes.shape[0]
    s, q, x, w = np.full(m, p.s0), np.zeros(m), np.zeros(m), np.ones(m)
    sdt = p.sigma * np.sqrt(p.dt)
    for i in range(n):
        fill, up = codes[:, i] // 2, codes[:, i] % 2
        sold, bought = np.isin(fill, [1, 3]), np.isin(fill, [2, 3])
        _, bid, ask = strategy.quote(s, q, i, p)
        pa = np.minimum(p.A * np.exp(-p.k * (ask - s)) * p.dt, 1.0)
        pb = np.minimum(p.A * np.exp(-p.k * (s - bid)) * p.dt, 1.0)
        w *= np.where(sold, pa, 1 - pa) * np.where(bought, pb, 1 - pb) * 0.5
        x += sold * (ask - p.fill_cost) - bought * (bid + p.fill_cost)
        q += bought.astype(float) - sold
        s += np.where(up == 1, sdt, -sdt) + p.impact * (sold.astype(float) - bought)
    pnl = x + q * s
    mean = np.sum(w * pnl)
    var = np.sum(w * (pnl - mean) ** 2)
    kurt = np.sum(w * (pnl - mean) ** 4) / var**2
    ce = -np.log(np.sum(w * np.exp(-gamma * pnl))) / gamma
    q_dist = {v: np.sum(w[q == v]) for v in np.unique(q)}
    return mean, np.sqrt(var), kurt, ce, q_dist


@pytest.mark.parametrize("impact,fill_cost", [(0.0, 0.0), (0.3, 0.0), (0.0, 0.2), (0.3, 0.2)])
@pytest.mark.parametrize("gamma", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("which", ["inventory", "symmetric", "adjusted"])
def test_dp_matches_brute_force_enumeration(gamma, which, impact, fill_cost):
    # 6 steps, large fill probs, clipping when |q| is large
    p = Params(T=0.6, dt=0.1, sigma=2.0, A=6.0, k=1.5, impact=impact, fill_cost=fill_cost)
    strat = {"inventory": AvellanedaStoikov(gamma), "symmetric": Symmetric.matching(gamma, p),
             "adjusted": AdjustedAS.frozen_inventory(gamma, impact)}[which]
    ex = exact_moments(strat, p, gamma_ce=gamma)
    mean, sd, kurt, ce, q_dist = brute_force(strat, p, gamma)
    assert ex["mean_pnl"] == pytest.approx(mean, abs=1e-9)
    assert ex["std_pnl"] == pytest.approx(sd, abs=1e-9)
    assert ex["kurt_pnl"] == pytest.approx(kurt, rel=1e-9)
    assert ex["ce"] == pytest.approx(ce, abs=1e-9)
    for v, prob in q_dist.items():
        assert ex["q_dist"][int(v) + p.n_steps] == pytest.approx(prob, abs=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_dp_matches_closed_form_for_symmetric(gamma):
    strat = Symmetric.matching(gamma, P)
    ex, th = exact_moments(strat, P), symmetric_moments(strat.spread, P)
    for key in ["mean_pnl", "std_pnl", "std_q"]:
        assert ex[key] == pytest.approx(th[key], rel=1e-10)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_dp_matches_monte_carlo_for_inventory(gamma):
    strat = AvellanedaStoikov(gamma)
    ex = exact_moments(strat, P, gamma_ce=gamma)
    n = 40000
    r = simulate(strat, P, n_paths=n, seed=21)
    se = sampling_se(ex, n)
    assert abs(r.pnl.mean() - ex["mean_pnl"]) < 4 * se["mean_pnl"]
    assert abs(r.pnl.std() - ex["std_pnl"]) < 4 * se["std_pnl"]
    assert abs(r.q.std() - ex["std_q"]) < 4 * se["std_q"]
    assert abs(r.n_trades.mean() - ex["mean_trades"]) < 0.2


# ---------- adverse selection: each fill moves the mid by p.impact in the trade's direction ----------


def single_fill_steps(r, n_paths, n_steps):
    """Number of steps with exactly one fill, per path, from the trade log."""
    tr = r.trades
    per_step = np.bincount(tr["path"] * n_steps + tr["step"], minlength=n_paths * n_steps)
    return (per_step.reshape(n_paths, n_steps) == 1).sum(axis=1)


@pytest.mark.parametrize("which", ["inventory", "symmetric", "adjusted"])
def test_impact_cost_identity(which):
    """sum_i q_{i+1} * eps * (sold_i - bought_i) = -(eps/2) * (#single-fill steps + q_T^2), path by path."""
    eps, n = 0.3, 300
    p = P.with_(impact=eps)
    strat = {"inventory": AvellanedaStoikov(0.1), "symmetric": Symmetric.matching(0.1, P),
             "adjusted": AdjustedAS.frozen_inventory(0.1, eps)}[which]
    r = simulate(strat, p, n_paths=n, seed=2, log_trades=True)
    np.testing.assert_allclose(r.impact_pnl, -eps / 2 * (single_fill_steps(r, n, p.n_steps) + r.q**2), atol=1e-8)


def test_impact_only_adds_the_impact_term():
    """Quotes are set relative to the mid, so fills do not depend on eps: under common random numbers the PnL
    with impact equals the PnL without it plus impact_pnl, path by path."""
    rnd = draw_randomness(P, 300, seed=3)
    a = simulate(AvellanedaStoikov(0.1), P, 300, randomness=rnd)
    b = simulate(AvellanedaStoikov(0.1), P.with_(impact=0.3), 300, randomness=rnd)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.n_trades, b.n_trades)
    np.testing.assert_allclose(b.pnl - a.pnl, b.impact_pnl, atol=1e-8)
    assert np.all(a.impact_pnl == 0)


def test_adjusted_as_reduces_to_as_without_adjustment():
    s, q = np.full(5, 100.0), np.arange(-2.0, 3.0)
    for strat in [AdjustedAS(0.1), AdjustedAS.frozen_inventory(0.1, 0.0)]:
        for i in [0, 100, 199]:
            for got, want in zip(strat.quote(s, q, i, P), AvellanedaStoikov(0.1).quote(s, q, i, P)):
                np.testing.assert_allclose(got, want)


@pytest.mark.parametrize("kind", ["impact", "fill_cost"])
@pytest.mark.parametrize("which", ["inventory", "adjusted"])
def test_dp_matches_monte_carlo_with_adverse_selection(which, kind):
    eps = 0.25
    p = P.with_(**{kind: eps})
    strat = AvellanedaStoikov(0.1) if which == "inventory" else AdjustedAS.frozen_inventory(0.1, eps)
    ex = exact_moments(strat, p, gamma_ce=0.1)
    n = 40000
    r = simulate(strat, p, n_paths=n, seed=31)
    se = sampling_se(ex, n)
    assert abs(r.pnl.mean() - ex["mean_pnl"]) < 4 * se["mean_pnl"]
    assert abs(r.pnl.std() - ex["std_pnl"]) < 4 * se["std_pnl"]
    assert abs(r.q.std() - ex["std_q"]) < 4 * se["std_q"]


@pytest.mark.parametrize("kind", ["impact", "fill_cost"])
def test_closed_forms_reject_adverse_selection(kind):
    for f in [symmetric_moments, symmetric_moments_continuous]:
        with pytest.raises(ValueError):
            f(1.49, P.with_(**{kind: 0.1}))
