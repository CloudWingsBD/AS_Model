"""Summary statistics with Monte Carlo standard errors."""
import numpy as np


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(x.size)


def std_se(x):
    """Sample std and its approximate standard error std/sqrt(2(n-1)) (normal approximation)."""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return sd, sd / np.sqrt(2 * (x.size - 1))


def cara_ce(pnl, gamma):
    """Certainty equivalent under exponential utility: -(1/gamma) * ln E[exp(-gamma * PnL)] (log-sum-exp stable)."""
    z = -gamma * np.asarray(pnl, float)
    m = z.max()
    return -(m + np.log(np.mean(np.exp(z - m)))) / gamma


def mv_ce(pnl, gamma):
    """Second-order approximation of the CARA certainty equivalent: mean - gamma/2 * variance."""
    pnl = np.asarray(pnl, float)
    return pnl.mean() - 0.5 * gamma * pnl.var(ddof=1)


def paired_diff(a, b, z=1.96):
    """Mean of a - b over paired paths (common random numbers), its SE and CI,
    plus the SE an independent-samples comparison would have had."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m, se = mean_se(a - b)
    se_indep = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return dict(diff=m, se=se, ci_lo=m - z * se, ci_hi=m + z * se, se_independent=se_indep)


def summarize(res, gamma):
    m, se = mean_se(res.pnl)
    sd, sd_se = std_se(res.pnl)
    qm, q_se = mean_se(res.q)
    return {"profit": m, "profit SE": se, "std profit": sd, "std profit SE": sd_se,
            "final q": qm, "final q SE": q_se, "std q": res.q.std(ddof=1),
            "trades": res.n_trades.mean(), "CE (CARA)": cara_ce(res.pnl, gamma),
            "CE (mean-var)": mv_ce(res.pnl, gamma)}
