"""Vectorised Monte Carlo market-making simulator (Avellaneda & Stoikov 2008, Section 3.3)."""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .params import Params
from .strategies import Strategy


@dataclass
class SimResult:
    pnl: np.ndarray            # terminal PnL per path: cash + q_T * S_T
    q: np.ndarray              # terminal inventory per path
    n_trades: np.ndarray       # number of fills per path
    spread_pnl: np.ndarray     # sum over fills of the edge vs the mid at quote time
    inventory_pnl: np.ndarray  # sum_i q_{i+1} * dS_i (PnL from holding inventory while the mid moves)
    impact_pnl: np.ndarray     # part of inventory_pnl caused by fill-triggered mid moves (zero when p.impact == 0)
    clip_frac: float           # share of (path, step, side) where A*exp(-k*delta)*dt > 1 and was clipped
    both_fill_frac: float      # share of (path, step) where bid and ask both filled
    t: Optional[np.ndarray] = None
    paths: Optional[Dict[str, np.ndarray]] = None   # (n_steps+1, n_paths) arrays, if record=True
    trades: Optional[Dict[str, np.ndarray]] = None  # flat trade log, if log_trades=True


def draw_randomness(p: Params, n_paths: int, seed: int) -> Dict[str, np.ndarray]:
    """All random inputs for one experiment. Reusing them across strategies gives common random numbers."""
    rng = np.random.default_rng(seed)
    n = p.n_steps
    u_ask = rng.random((n, n_paths))
    u_bid = rng.random((n, n_paths))
    if p.noise == "binomial":
        dS = p.sigma * np.sqrt(p.dt) * np.where(rng.random((n, n_paths)) < 0.5, 1.0, -1.0)
    elif p.noise == "gaussian":
        dS = p.sigma * np.sqrt(p.dt) * rng.standard_normal((n, n_paths))
    else:
        raise ValueError(p.noise)
    return dict(u_ask=u_ask, u_bid=u_bid, dS=dS)


def simulate(strategy: Strategy, p: Params = Params(), n_paths: int = 1000, seed: int = 0,
             record: bool = False, log_trades: bool = False,
             randomness: Optional[Dict[str, np.ndarray]] = None) -> SimResult:
    """
    Each step i: the strategy quotes (bid, ask) given (s_i, q_i); the ask fills with prob A*exp(-k*delta_a)*dt,
    the bid with prob A*exp(-k*delta_b)*dt (independently, clipped to [0, 1]); fills execute at the quoted
    prices; then the mid moves by dS_i, plus p.impact * (sold - bought) (adverse selection: a filled ask means
    a buyer, and the mid moves up). Terminal inventory is marked at the mid.
    Same seed (or same `randomness`) => same mid increments dS and same fill uniforms for every strategy
    (with p.impact > 0 the mid path itself also depends on the strategy's fills).
    """
    rnd = randomness if randomness is not None else draw_randomness(p, n_paths, seed)
    n = p.n_steps
    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    x = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)
    spread_pnl = np.zeros(n_paths)
    inventory_pnl = np.zeros(n_paths)
    impact_pnl = np.zeros(n_paths)
    n_clipped = 0
    n_both = 0

    if record:
        keys = ["s", "r", "bid", "ask", "q"]
        rec = {kk: np.full((n + 1, n_paths), np.nan) for kk in keys}
    if log_trades:
        log = {kk: [] for kk in ["path", "step", "side", "price", "mid", "q_before"]}

    for i in range(n):
        r, bid, ask = strategy.quote(s, q, i, p)
        if record:
            for kk, v in zip(keys, [s, r, bid, ask, q]):
                rec[kk][i] = v

        delta_a, delta_b = ask - s, s - bid
        lam_a = p.A * np.exp(-p.k * delta_a) * p.dt
        lam_b = p.A * np.exp(-p.k * delta_b) * p.dt
        n_clipped += int((lam_a > 1).sum() + (lam_b > 1).sum())
        sold = rnd["u_ask"][i] < np.minimum(lam_a, 1.0)
        bought = rnd["u_bid"][i] < np.minimum(lam_b, 1.0)
        n_both += int((sold & bought).sum())

        if log_trades:
            for mask, side, price in [(sold, -1, ask), (bought, +1, bid)]:
                idx = np.nonzero(mask)[0]
                log["path"].append(idx)
                log["step"].append(np.full(idx.size, i))
                log["side"].append(np.full(idx.size, side))
                log["price"].append(price[idx] if np.ndim(price) else np.full(idx.size, price))
                log["mid"].append(s[idx])
                log["q_before"].append(q[idx])

        spread_pnl += sold * delta_a + bought * delta_b
        x += sold * ask - bought * bid
        q = q - sold + bought
        n_trades += sold.astype(float) + bought
        jump = p.impact * (sold.astype(float) - bought)
        impact_pnl += q * jump
        inventory_pnl += q * (rnd["dS"][i] + jump)
        s = s + rnd["dS"][i] + jump

    out = SimResult(pnl=x + q * s, q=q, n_trades=n_trades, spread_pnl=spread_pnl,
                    inventory_pnl=inventory_pnl, impact_pnl=impact_pnl, clip_frac=n_clipped / (2 * n * n_paths),
                    both_fill_frac=n_both / (n * n_paths))
    if record:
        rec["s"][-1], rec["q"][-1] = s, q
        out.paths, out.t = rec, np.arange(n + 1) * p.dt
    if log_trades:
        out.trades = {kk: np.concatenate(v) for kk, v in log.items()}
    return out
