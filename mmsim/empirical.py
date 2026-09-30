"""Calibration and backtesting on real trades (Binance spot aggTrades archives, see notebook 03)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .params import Params
from .simulator import SimResult

AGG_COLUMNS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "buyer_maker", "best_match"]


def load_aggtrades(path) -> pd.DataFrame:
    """One day of Binance spot aggTrades (zip or csv, no header) -> DataFrame with price, qty, t (seconds since
    00:00 UTC of that day) and buy (True when the buyer was the aggressor, i.e. took liquidity)."""
    df = pd.read_csv(path, header=None, names=AGG_COLUMNS, usecols=["price", "qty", "ts", "buyer_maker"])
    ts = df["ts"].to_numpy()
    t = ts / (1e6 if ts.max() > 1e14 else 1e3)   # spot archives use microseconds since 2025, milliseconds before
    day0 = np.floor(t.min() / 86400) * 86400
    return pd.DataFrame({"price": df["price"].to_numpy(float), "qty": df["qty"].to_numpy(float),
                         "t": t - day0, "buy": ~df["buyer_maker"].to_numpy(bool)})


@dataclass
class Seconds:
    """Per-second view of a trade record: what a market maker who re-quotes once a second needs."""
    s: np.ndarray       # last trade price before the second starts (the mid proxy quotes are based on)
    s_end: np.ndarray   # last trade price at the end of the second (= s of the next second)
    hi: np.ndarray      # highest aggressive-buy price during the second (nan if there was none)
    lo: np.ndarray      # lowest aggressive-sell price during the second (nan if there was none)


def to_seconds(trades: pd.DataFrame, n: int = 86400) -> Seconds:
    """Aggregate a day of trades into n one-second bins. Seconds before the first trade take its price."""
    sec = np.floor(trades["t"].to_numpy()).astype(np.int64)
    keep = (sec >= 0) & (sec < n)
    sec, px, buy = sec[keep], trades["price"].to_numpy()[keep], trades["buy"].to_numpy()[keep]
    idx = np.arange(n)
    s_end = pd.Series(px).groupby(sec).last().reindex(idx).ffill().fillna(px[0]).to_numpy()
    hi = pd.Series(px[buy]).groupby(sec[buy]).max().reindex(idx).to_numpy()
    lo = pd.Series(px[~buy]).groupby(sec[~buy]).min().reindex(idx).to_numpy()
    return Seconds(s=np.r_[px[0], s_end[:-1]], s_end=s_end, hi=hi, lo=lo)


def fill_probability(sec: Seconds, deltas):
    """Empirical probability that a quote at distance delta from s is reached during the second: (asks, bids)."""
    with np.errstate(invalid="ignore"):   # seconds without trades on a side compare as False
        pa = np.array([np.mean(sec.hi >= sec.s + d) for d in deltas])
        pb = np.array([np.mean(sec.lo <= sec.s - d) for d in deltas])
    return pa, pb


def fit_intensity(deltas, prob, p_min=1e-3, p_max=0.5):
    """Least-squares fit of log P(delta) = log(A * dt) - k * delta with dt = 1 s, on the points with
    p_min < P < p_max. Returns (A, k)."""
    deltas, prob = np.asarray(deltas, float), np.asarray(prob, float)
    use = (prob > p_min) & (prob < p_max)
    slope, intercept = np.polyfit(deltas[use], np.log(prob[use]), 1)
    return float(np.exp(intercept)), float(-slope)


def backtest(strategy, sec: Seconds, p: Params, tick: float = 0.01, log_trades: bool = False) -> SimResult:
    """
    Run `strategy` on consecutive episodes of p.n_steps seconds of real data (p.dt must be 1). Each second it quotes
    from the price at the start of the second and its inventory, with the ask rounded up and the bid rounded down
    to the tick. The ask is filled (one unit, at the quoted price) if an aggressive buy prints at or above it during
    the second, the bid if an aggressive sell prints at or below it. Inventory is marked at the last price of each
    episode. Our orders are assumed not to move the market, and queue position is ignored.
    The accounting matches `simulate`: pnl = spread_pnl + inventory_pnl on every episode.
    """
    if p.dt != 1:
        raise ValueError("backtest re-quotes once a second: use Params(dt=1)")
    n = p.n_steps
    m = len(sec.s) // n
    view = lambda a: a[: m * n].reshape(m, n).T      # (step, episode)
    S, S_end, HI, LO = view(sec.s), view(sec.s_end), view(sec.hi), view(sec.lo)
    q, x = np.zeros(m), np.zeros(m)
    n_trades, spread_pnl, inventory_pnl = np.zeros(m), np.zeros(m), np.zeros(m)
    n_both = 0
    if log_trades:
        log = {kk: [] for kk in ["path", "step", "side", "price", "mid", "q_before"]}

    with np.errstate(invalid="ignore"):
        for i in range(n):
            s = S[i]
            _, bid, ask = strategy.quote(s, q, i, p)
            ask = np.ceil(np.round(ask / tick, 6)) * tick * np.ones(m)
            bid = np.floor(np.round(bid / tick, 6)) * tick * np.ones(m)
            sold, bought = HI[i] >= ask, LO[i] <= bid
            n_both += int((sold & bought).sum())
            if log_trades:
                for mask, side, price in [(sold, -1, ask), (bought, +1, bid)]:
                    idx = np.nonzero(mask)[0]
                    log["path"].append(idx)
                    log["step"].append(np.full(idx.size, i))
                    log["side"].append(np.full(idx.size, side))
                    log["price"].append(price[idx])
                    log["mid"].append(s[idx])
                    log["q_before"].append(q[idx])
            spread_pnl += sold * (ask - s) + bought * (s - bid)
            x += sold * ask - bought * bid
            q = q - sold + bought
            n_trades += sold.astype(float) + bought
            inventory_pnl += q * (S_end[i] - s)

    out = SimResult(pnl=x + q * S_end[-1], q=q, n_trades=n_trades, spread_pnl=spread_pnl,
                    inventory_pnl=inventory_pnl, impact_pnl=np.zeros(m), clip_frac=np.nan,
                    both_fill_frac=n_both / (n * m))
    if log_trades:
        out.trades = {kk: np.concatenate(v) for kk, v in log.items()}
    return out


def adverse_move(sec: Seconds, trades: dict, n_steps: int, h: int) -> float:
    """Mean price move against the market maker over the h seconds after the start of each fill's second:
    up after its ask was hit, down after its bid was hit (trades as logged by `backtest`)."""
    t = trades["path"] * n_steps + trades["step"]
    price_at = np.r_[sec.s, sec.s_end[-1]]           # price at the start of second j, j = 0..n
    ok = t + h < price_at.size
    return float(np.mean(-trades["side"][ok] * (price_at[t[ok] + h] - price_at[t[ok]])))
