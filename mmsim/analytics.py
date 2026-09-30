"""Closed-form moments used to validate the simulator."""
import numpy as np

from .params import Params


def symmetric_moments(spread: float, p: Params) -> dict:
    """
    Exact moments of the symmetric (constant-spread, mid-centred) strategy in the discrete model.
    Let d = spread/2 and lam = A*exp(-k*d)*dt (fill prob per side per step), n steps, v = 2*lam*(1-lam).
      PnL = spread_pnl + inventory_pnl, with spread_pnl = d * (#fills), inventory_pnl = sum_i q_{i+1} dS_i.
      E[PnL]            = 2 n lam d            (dS is a martingale increment independent of fills)
      Var(q_T)          = n v
      Var(spread_pnl)   = 2 n d^2 lam (1-lam)
      Var(inventory_pnl)= sigma^2 dt * sum_i E[q_{i+1}^2] = sigma^2 dt * v * n(n+1)/2
      Cov(spread_pnl, inventory_pnl) = 0     (each dS_i has mean zero and is independent of all fills)
    """
    d = spread / 2
    lam = p.A * np.exp(-p.k * d) * p.dt
    if lam > 1:
        raise ValueError("fill probability > 1; formulas assume no clipping")
    n = p.n_steps
    v = 2 * lam * (1 - lam)
    var_spread = 2 * n * d**2 * lam * (1 - lam)
    var_inv = p.sigma**2 * p.dt * v * n * (n + 1) / 2
    return dict(half_spread=d, fill_prob=lam, mean_pnl=2 * n * lam * d, std_q=np.sqrt(n * v),
                std_spread_pnl=np.sqrt(var_spread), std_inventory_pnl=np.sqrt(var_inv),
                std_pnl=np.sqrt(var_spread + var_inv))


def symmetric_moments_continuous(spread: float, p: Params) -> dict:
    """
    Continuous-time (dt -> 0) limit of symmetric_moments: fills become Poisson with rate L = A*exp(-k*d) per side.
      E[PnL] = 2 L T d,  Var(q_T) = 2 L T,  Var(spread_pnl) = 2 L T d^2,  Var(inventory_pnl) = sigma^2 L T^2.
    The discrete Bernoulli model has an extra (1 - lam) factor in every variance, so it understates risk at finite dt.
    """
    d = spread / 2
    L = p.A * np.exp(-p.k * d)
    var_spread = 2 * L * p.T * d**2
    var_inv = p.sigma**2 * L * p.T**2
    return dict(half_spread=d, rate=L, mean_pnl=2 * L * p.T * d, std_q=np.sqrt(2 * L * p.T),
                std_spread_pnl=np.sqrt(var_spread), std_inventory_pnl=np.sqrt(var_inv),
                std_pnl=np.sqrt(var_spread + var_inv))


def exact_moments(strategy, p: Params, gamma_ce: float = None) -> dict:
    """
    Exact distributional quantities of terminal PnL for any strategy whose quote offsets depend only on
    (q, step), computed by backward dynamic programming on the (inventory, time-step) grid -- no simulation.

    Mark-to-market wealth M_i = x_i + q_i s_i has one-step increment
        dM = e + q' * dS,   e = sold * delta_a + bought * delta_b,   q' = q - sold + bought,
    where (sold, bought) are independent Bernoulli(p_a), Bernoulli(p_b) given (q, i), and dS is independent
    of everything else. PnL = M_n - M_0. Let R(q, i) be the PnL accumulated from step i on, starting at q.
    Given the step-i outcome, R(q, i) = e + q' dS + R(q', i+1) with the three pieces independent, so
        E[R(q,i)^k] = sum_outcomes P * sum_j C(k,j) E[(e + R(q',i+1))^(k-j)] * q'^j * E[dS^j]      (k = 1..4)
        Phi(q,i) = E[exp(-g R(q,i))] = sum_outcomes P * exp(-g e) * E[exp(-g q' dS)] * Phi(q',i+1)   (CARA)
    Since |q_i| <= i, a grid q in [-n, n] makes the recursion exact (no truncation).
    The distribution of q_T is propagated forward exactly.
    """
    from math import comb

    n = p.n_steps
    qs = np.arange(-n, n + 1, dtype=float)
    nq = qs.size
    s_arr = np.full(nq, p.s0)
    sdt = p.sigma * np.sqrt(p.dt)
    K = 4
    dS_moments = [1.0, 0.0, sdt**2, 0.0, (1.0 if p.noise == "binomial" else 3.0) * sdt**4]

    def offsets(i):
        _, bid, ask = strategy.quote(s_arr, qs, i, p)
        da, db = ask - s_arr, s_arr - bid
        da, db = np.maximum(da, -50.0), np.maximum(db, -50.0)  # avoid exp overflow; prob is clipped to 1 anyway
        pa = np.minimum(p.A * np.exp(-p.k * da) * p.dt, 1.0)
        pb = np.minimum(p.A * np.exp(-p.k * db) * p.dt, 1.0)
        # outcomes: none, sell, buy, both -> (probability, edge, inventory shift, number of fills)
        return [((1 - pa) * (1 - pb), np.zeros(nq), 0, 0),
                (pa * (1 - pb), da, -1, 1),
                ((1 - pa) * pb, db, +1, 1),
                (pa * pb, da + db, 0, 2)]

    def at(arr, shift):
        """arr evaluated at q + shift (edge-padded; padded values are never reached from q_0 = 0)."""
        padded = np.pad(arr, 1, mode="edge")
        return padded[1 + shift: 1 + shift + nq]

    if p.noise == "binomial":
        log_mgf = lambda x: np.logaddexp(x, -x) - np.log(2.0)   # log E[exp(-x * dS/sdt)] = log cosh(x)
    else:
        log_mgf = lambda x: 0.5 * x**2

    mu = [np.ones(nq)] + [np.zeros(nq) for _ in range(K)]   # raw moments of R(q, i)
    trades = np.zeros(nq)
    log_phi = np.zeros(nq)
    with np.errstate(divide="ignore"):
        for i in range(n - 1, -1, -1):
            new_mu = [np.ones(nq)] + [np.zeros(nq) for _ in range(K)]
            new_trades = np.zeros(nq)
            terms = []
            for pr, e, sh, cnt in offsets(i):
                qn = qs + sh
                mu_next = [at(mu[l], sh) for l in range(K + 1)]
                # E[(e + R')^r] for r = 0..K
                c_mom = [sum(comb(r, l) * e ** (r - l) * mu_next[l] for l in range(r + 1)) for r in range(K + 1)]
                for k in range(1, K + 1):
                    new_mu[k] += pr * sum(comb(k, j) * c_mom[k - j] * qn**j * dS_moments[j] for j in range(k + 1))
                new_trades += pr * (cnt + at(trades, sh))
                if gamma_ce is not None:
                    terms.append(np.log(pr) - gamma_ce * e + log_mgf(gamma_ce * qn * sdt) + at(log_phi, sh))
            mu, trades = new_mu, new_trades
            if gamma_ce is not None:
                log_phi = np.logaddexp.reduce(np.stack(terms), axis=0)

    # forward pass: exact distribution of q_T starting from q_0 = 0
    pi = np.zeros(nq)
    pi[n] = 1.0
    for i in range(n):
        new = np.zeros(nq)
        for pr, _, sh, _ in offsets(i):
            mass = pi * pr
            if sh == 0:
                new += mass
            elif sh > 0:
                new[1:] += mass[:-1]
            else:
                new[:-1] += mass[1:]
        pi = new

    def central(m1, m2, m3, m4):
        var = m2 - m1**2
        c4 = m4 - 4 * m3 * m1 + 6 * m2 * m1**2 - 3 * m1**4
        return var, c4 / var**2

    c = n  # grid index of q = 0
    m1, m2, m3, m4 = (float(mu[k][c]) for k in range(1, 5))
    var, kurt = central(m1, m2, m3, m4)
    qm = [float(np.sum(pi * qs**k)) for k in range(1, 5)]
    var_q, kurt_q = central(*qm)
    out = dict(mean_pnl=m1, std_pnl=float(np.sqrt(var)), kurt_pnl=float(kurt),
               mean_trades=float(trades[c]), mean_q=qm[0], std_q=float(np.sqrt(var_q)), kurt_q=float(kurt_q),
               q_values=qs, q_dist=pi)
    if gamma_ce is not None:
        out["ce"] = float(-log_phi[c] / gamma_ce)
    return out


def sampling_se(exact: dict, n_paths: int) -> dict:
    """Standard errors of an n_paths Monte Carlo estimate of mean / std, from the exact moments.
    SE(sample std) ~ sigma * sqrt((kurtosis - 1) / (4 n)) (delta method; = sigma/sqrt(2n) for normal data)."""
    return dict(mean_pnl=exact["std_pnl"] / np.sqrt(n_paths),
                std_pnl=exact["std_pnl"] * np.sqrt((exact["kurt_pnl"] - 1) / (4 * n_paths)),
                mean_q=exact["std_q"] / np.sqrt(n_paths),
                std_q=exact["std_q"] * np.sqrt((exact["kurt_q"] - 1) / (4 * n_paths)))
