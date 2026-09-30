# Avellaneda–Stoikov market-making simulator

[![tests](https://github.com/CloudWingsBD/AS_Model/actions/workflows/tests.yml/badge.svg)](https://github.com/CloudWingsBD/AS_Model/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A replication of Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*, Quantitative Finance 8(3). It pairs a vectorised Monte Carlo simulator with an exact dynamic-programming solution of the paper's discrete model, which is used to validate both the simulator and the paper's tables. The model is then extended with adverse selection (notebook 02) and tested against a week of real BTC and ETH trades (notebook 03).

## Results

- **Matches the paper.** The dynamic program gives the exact PnL moments of the discrete model, free of simulation noise. All 24 numbers in the paper's Tables 1–3 are within 1.9 standard errors of the exact values (χ² = 19.4 with 24 degrees of freedom, p = 0.73), so the differences are just the sampling noise of the paper's 1000 paths.
- **Validated three ways.** The dynamic program agrees with brute-force enumeration of every path on a small model (to 1e−9), with the closed form for the symmetric strategy (to 1e−10), and with 40000-path Monte Carlo at the paper's parameters.
- **The trade-off.** At the same spread, skewing quotes by inventory keeps 70–100% of the symmetric strategy's mean profit and cuts its PnL standard deviation to 45–65% (γ = 0.01–1).
- **Monte Carlo cannot estimate the CARA certainty equivalent** of the symmetric strategy. At γ = 0.1 the exact value is 39.8, while Monte Carlo with 10³–10⁶ paths gives 44.5–54.0: expected utility is dominated by rare paths on which inventory runs away.
- **Results depend on dt.** At the paper's dt = 0.005, the symmetric strategy's PnL standard deviation is about 12% below its continuous-time limit.

![Mean vs standard deviation of terminal PnL as gamma varies, and the inventory/symmetric ratios](figures/mean_std_frontier.png)

*Left: mean vs standard deviation of terminal PnL as γ varies; at each γ both strategies use the same spread. Right: inventory / symmetric ratios. Above γ ≈ 2 the same-average-spread benchmark stops being meaningful.*

Published values vs exact values:

| γ | Strategy | Profit (paper / exact) | Std of profit (paper / exact) | Std of final q (paper / exact) |
|---|---|---|---|---|
| 0.1 | inventory | 65.0 / 64.89 | 6.6 / 6.54 | 2.9 / 2.93 |
| 0.1 | symmetric | 68.4 / 68.22 | 12.7 / 13.46 | 8.4 / 8.40 |
| 0.01 | inventory | 68.6 / 68.40 | 8.7 / 8.96 | 5.1 / 5.21 |
| 0.01 | symmetric | 68.8 / 68.67 | 12.8 / 13.68 | 8.7 / 8.71 |
| 1 | inventory | 31.4 / 31.45 | 5.0 / 4.84 | 1.7 / 1.64 |
| 1 | symmetric | 44.0 / 43.69 | 11.0 / 10.73 | 5.1 / 5.17 |

The full analysis, more figures and the derivations are in [`01_replication.ipynb`](01_replication.ipynb).

### Extension: adverse selection

[`02_adverse_selection.ipynb`](02_adverse_selection.ipynb) makes fills informative: each fill moves the mid permanently by ε in the direction of the trade (`Params(impact=ε)`), and the exact dynamic program is extended to match.

- **An exact accounting identity.** On every path the PnL caused by these moves is −(ε/2)(N₁ + q_T²), where N₁ is the number of steps with exactly one fill: each fill costs ε/2, and only terminal inventory is penalised.
- **Inventory control halves the cost.** As dt → 0 the symmetric strategy pays ε per fill and A-S about 0.55ε. At γ = 0.1, A-S overtakes the symmetric strategy on mean profit once ε > 0.12.
- **The textbook adjustment over-reacts.** Widening by ε and skewing by an extra qε (frozen-inventory indifference prices) is worse than not adjusting at all (certainty equivalent 48.6 vs 52.4 at ε = 0.25). The best adjustment on a grid is small and gains little.

![Mean and standard deviation of terminal PnL as the adverse move per fill grows](figures/adverse_selection.png)

### Real data: BTC and ETH on Binance

[`03_real_data.ipynb`](03_real_data.ipynb) calibrates the model on a week of Binance spot trades (BTCUSDT and ETHUSDT, 23–29 September 2026). It then compares the model's predictions with a backtest of the same strategies against the real order flow, re-quoting once a second in 10-minute episodes.

- **The fill model works.** The exponential fill curve fits the data, and the number of fills is predicted to within 11% on every day.
- **The paper's profit prediction fails.** It predicts a profit every day, while the backtest loses money every day. After a fill the price moves permanently against the market maker, by about $16 for BTC and $0.55 for ETH: two to three times the spread earned.
- **Every fill costs one full ε, for both strategies.** Inventory control does not cut the adverse-selection cost, as notebook 02's single-dealer model assumed it would. Charging ε per fill predicts mean PnL to within about one standard error.
- **Risk is understated, and forecasts need frequent recalibration.** The model predicts 37–72% of the actual PnL standard deviation. Calibrating on the previous hour gives a 0.5–0.6 correlation with the next hour's PnL, against about zero when calibrating on the previous day.

![Predicted vs backtested mean PnL per day for three versions of the model](figures/real_data_predictions.png)

## Layout

```
mmsim/            simulator package
  params.py       Params dataclass (defaults are the paper's parameters; impact adds adverse selection)
  strategies.py   AvellanedaStoikov, Symmetric, AdjustedAS; a new strategy implements quote(s, q, i, p) -> (centre, bid, ask)
  simulator.py    vectorised Monte Carlo: common random numbers, PnL decomposition, per-fill trade log, discretisation diagnostics
  analytics.py    exact solutions: closed-form moments of the symmetric strategy; exact moments, kurtosis, CARA CE and inventory distribution of any (q, t) strategy by dynamic programming, with or without adverse selection
  stats.py        standard errors, paired differences, CARA certainty equivalent
  empirical.py    real data: load Binance aggTrades, per-second view, fill-curve calibration, backtest, adverse moves
tests/            unit tests (run by CI on every push)
scripts/          download_binance.py: fetches the trade archives used by notebook 03 into data/ (not tracked)
figures/          README figures, exported from the notebooks
01_replication.ipynb         replication experiments, figures and discussion
02_adverse_selection.ipynb   adverse selection: what it costs the paper's strategies, and how to respond
03_real_data.ipynb           the model against a week of real BTC and ETH trades
```

## Running

```
pip install -r requirements.txt
pytest -q
python scripts/download_binance.py BTCUSDT ETHUSDT --start 2026-09-23 --end 2026-09-29   # data for notebook 03, about 150 MB
jupyter notebook
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

## License

[MIT](LICENSE)
