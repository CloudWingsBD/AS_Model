# Avellaneda–Stoikov market-making simulator

A replication of Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*, Quantitative Finance 8(3).

## Layout

```
mmsim/            simulator package
  params.py       Params dataclass (defaults are the paper's parameters)
  strategies.py   AvellanedaStoikov, Symmetric; a new strategy implements quote(s, q, i, p) -> (centre, bid, ask)
  simulator.py    vectorised Monte Carlo: common random numbers, PnL decomposition, per-fill trade log, discretisation diagnostics
  analytics.py    exact solutions: closed-form moments of the symmetric strategy; exact moments, kurtosis, CARA CE and inventory distribution of any (q, t) strategy by dynamic programming
  stats.py        standard errors, paired differences, CARA certainty equivalent
tests/            unit tests
01_replication.ipynb   replication experiments, figures and discussion
```

## Running

```
pip install -r requirements.txt
pytest -q
jupyter notebook 01_replication.ipynb
```

## Usage

```python
from mmsim import Params, AvellanedaStoikov, Symmetric, simulate, draw_randomness

p = Params()
rnd = draw_randomness(p, n_paths=1000, seed=0)          # both strategies share the same random numbers
inv = simulate(AvellanedaStoikov(gamma=0.1), p, 1000, randomness=rnd)
sym = simulate(Symmetric.matching(0.1, p), p, 1000, randomness=rnd)
print(inv.pnl.mean(), inv.pnl.std(), sym.pnl.mean(), sym.pnl.std())
```

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3), 217–224. Journal-version PDF: https://math.nyu.edu/inmemoriam/avellaneda//HighFrequencyTrading.pdf (this project compares against Tables 1–3 of this version)
- 2006 preprint: https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf (its tables differ from the published version; see Appendix B of the notebook)
