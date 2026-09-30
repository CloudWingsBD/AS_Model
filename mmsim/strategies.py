"""Quoting strategies. Each strategy maps (mid, inventory, step) -> (centre, bid, ask), vectorised over paths."""
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np

from .params import Params

Quote = Tuple[np.ndarray, np.ndarray, np.ndarray]


class Strategy(Protocol):
    name: str

    def quote(self, s: np.ndarray, q: np.ndarray, i: int, p: Params) -> Quote:
        """Return (reference price, bid, ask) at the start of step i."""
        ...


def as_spread(gamma: float, tau, sigma: float, k: float):
    """Total A-S spread: gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)."""
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log1p(gamma / k)


def average_as_spread(gamma: float, p: Params) -> float:
    """Time-average of the A-S spread over the simulation grid (the symmetric benchmark's spread)."""
    return float(np.mean(as_spread(gamma, p.tau_grid(), p.sigma, p.k)))


@dataclass(frozen=True)
class AvellanedaStoikov:
    """Inventory strategy: quote the A-S spread around the reservation price r = s - q*gamma*sigma^2*(T-t)."""
    gamma: float
    name: str = "inventory"

    def quote(self, s, q, i, p):
        tau = p.tau(i)
        r = s - q * self.gamma * p.sigma**2 * tau
        half = as_spread(self.gamma, tau, p.sigma, p.k) / 2
        return r, r - half, r + half


@dataclass(frozen=True)
class AdjustedAS:
    """A-S with each side widened by `widen` and the centre moved by an extra -q*skew (the defaults give plain A-S).
    Used to respond to adverse selection, i.e. fill-triggered mid moves of size eps (Params.impact)."""
    gamma: float
    widen: float = 0.0
    skew: float = 0.0
    name: str = "adjusted"

    @classmethod
    def frozen_inventory(cls, gamma: float, eps: float) -> "AdjustedAS":
        """Adjustment implied by the frozen-inventory indifference prices under impact eps,
        r^a = s + eps*(1 - q) + gamma*sigma^2*(T-t)*(1 - 2q)/2 and r^b = s - eps*(1 + q) - gamma*sigma^2*(T-t)*(1 + 2q)/2:
        widen each side by eps and skew by an extra -q*eps."""
        return cls(gamma, widen=eps, skew=eps)

    def quote(self, s, q, i, p):
        tau = p.tau(i)
        r = s - q * (self.gamma * p.sigma**2 * tau + self.skew)
        half = as_spread(self.gamma, tau, p.sigma, p.k) / 2 + self.widen
        return r, r - half, r + half


@dataclass(frozen=True)
class Symmetric:
    """Benchmark: constant spread centred on the mid, ignoring inventory."""
    spread: float
    name: str = "symmetric"

    @classmethod
    def matching(cls, gamma: float, p: Params) -> "Symmetric":
        """Symmetric strategy with the same (time-averaged) spread as A-S with this gamma, as in the paper."""
        return cls(spread=average_as_spread(gamma, p))

    def quote(self, s, q, i, p):
        half = self.spread / 2
        return s, s - half, s + half
