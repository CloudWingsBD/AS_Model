"""Model parameters (Avellaneda & Stoikov 2008, Section 3.3 defaults)."""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    s0: float = 100.0       # initial mid price
    T: float = 1.0          # horizon
    sigma: float = 2.0      # mid-price volatility (price units / sqrt(time))
    dt: float = 0.005       # time step
    k: float = 1.5          # fill-intensity decay: lambda(delta) = A * exp(-k * delta)
    A: float = 140.0        # fill-intensity level
    noise: str = "binomial"  # mid increments: "binomial" (+/- sigma*sqrt(dt), as in the paper) or "gaussian"
    impact: float = 0.0     # adverse selection: each fill moves the mid by this much in the trade's direction

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    def tau(self, i: int) -> float:
        """Time to horizon T - t_i at the start of step i."""
        return self.T - i * self.dt

    def tau_grid(self) -> np.ndarray:
        return self.T - np.arange(self.n_steps) * self.dt

    def with_(self, **kwargs) -> "Params":
        return replace(self, **kwargs)
