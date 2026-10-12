"""Exact CARA-optimal quotes for the discrete model, by backward induction on the (inventory, step) grid."""
import numpy as np

from .params import Params
from .strategies import TablePolicy


def optimal_policy(p: Params, gamma: float, sweeps: int = 200, tol: float = 1e-12):
    """
    Quotes that maximise E[-exp(-gamma * PnL)] exactly, with the same dynamics as `simulate` and `exact_moments`
    (including p.impact and p.fill_cost). Returns (TablePolicy, certainty equivalent from q = 0 at t = 0).

    Let Phi(q, i) = min E[exp(-gamma * R(q, i))] over the quotes from step i on, and W(q') = E[exp(-gamma q' dS)]
    * Phi(q', i+1). Grouping the four fill outcomes by whether the ask fills, the step-i objective is
        (1 - p_a) U + p_a exp(-gamma delta_a) V,      p_a = min(1, A exp(-k delta_a) dt),
    where U and V depend on the bid only. For a fixed bid this is minimised at
        delta_a = ln(1 + gamma/k) / gamma + ln(V / U) / gamma,
    or at the largest distance with p_a = 1 if that is further out; the bid is symmetric. The two first-order
    conditions are iterated to a fixed point (they are coupled only through the rare step where both sides fill).
    Since |q_i| <= i, the grid q in [-n, n] is exact, as in exact_moments.
    """
    n = p.n_steps
    qs = np.arange(-n, n + 1, dtype=float)
    nq = qs.size
    sdt = p.sigma * np.sqrt(p.dt)
    if p.noise == "binomial":
        log_mgf = lambda x: np.logaddexp(x, -x) - np.log(2.0)
    else:
        log_mgf = lambda x: 0.5 * x**2
    base = np.log1p(gamma / p.k) / gamma
    d_clip = np.log(p.A * p.dt) / p.k          # at or inside this distance a quote fills with probability 1

    def at(arr, shift):
        padded = np.pad(arr, 1, mode="edge")
        return padded[1 + shift: 1 + shift + nq]

    def log_probs(d):
        lp = np.minimum(np.log(p.A * p.dt) - p.k * d, 0.0)
        return lp, np.log1p(-np.exp(lp))

    # log of exp(-gamma * extra edge) per outcome: per-fill cost c, and the mid move after the fill (impact eps)
    c, eps = p.fill_cost, p.impact
    l10 = gamma * (c - eps * (qs - 1))          # sold only:   q' = q - 1, edge += -c + eps * q'
    l01 = gamma * (c + eps * (qs + 1))          # bought only: q' = q + 1, edge += -c - eps * q'
    l11 = 2 * gamma * c                         # both:        q' = q
    DA, DB = np.empty((n, nq)), np.empty((n, nq))
    da, db = np.full(nq, base), np.full(nq, base)
    log_phi = np.zeros(nq)
    with np.errstate(divide="ignore"):
        for i in range(n - 1, -1, -1):
            lw = log_mgf(gamma * qs * sdt) + log_phi          # log W(q') on the grid
            lw_dn, lw_up = at(lw, -1), at(lw, +1)             # log W(q - 1), log W(q + 1)
            for _ in range(sweeps):
                lpb, l1pb = log_probs(db)
                log_u = np.logaddexp(l1pb + lw, lpb - gamma * db + l01 + lw_up)
                log_v = np.logaddexp(l1pb + l10 + lw_dn, lpb - gamma * db + l11 + lw)
                da_new = np.maximum(base + (log_v - log_u) / gamma, d_clip)
                lpa, l1pa = log_probs(da_new)
                log_u = np.logaddexp(l1pa + lw, lpa - gamma * da_new + l10 + lw_dn)
                log_v = np.logaddexp(l1pa + l01 + lw_up, lpa - gamma * da_new + l11 + lw)
                db_new = np.maximum(base + (log_v - log_u) / gamma, d_clip)
                change = max(np.abs(da_new - da).max(), np.abs(db_new - db).max())
                da, db = da_new, db_new
                if change < tol:
                    break
            lpa, l1pa = log_probs(da)
            lpb, l1pb = log_probs(db)
            log_phi = np.logaddexp.reduce(np.stack([
                l1pa + l1pb + lw,
                lpa + l1pb - gamma * da + l10 + lw_dn,
                l1pa + lpb - gamma * db + l01 + lw_up,
                lpa + lpb - gamma * (da + db) + l11 + lw]), axis=0)
            DA[i], DB[i] = da, db
    return TablePolicy(DA, DB), float(-log_phi[n] / gamma)
