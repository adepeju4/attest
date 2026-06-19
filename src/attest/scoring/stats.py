"""Wilson confidence intervals and two-proportion significance for pass rates."""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.96


@dataclass(frozen=True)
class Proportion:
    rate: float
    low: float
    high: float
    n: int

    def pct(self) -> str:
        return f"{self.rate:.0%} (95% CI {self.low:.0%}–{self.high:.0%}, n={self.n})"


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Proportion:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Proportion(0.0, 0.0, 0.0, 0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Proportion(rate=p, low=max(0.0, center - margin),
                      high=min(1.0, center + margin), n=n)


def two_proportion_z(s_a: int, n_a: int, s_b: int, n_b: int) -> float:
    """Two-proportion z-statistic for the difference in pass rates A vs B."""
    if n_a == 0 or n_b == 0:
        return 0.0
    p_a, p_b = s_a / n_a, s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0
    return (p_a - p_b) / se


def difference_is_real(s_a: int, n_a: int, s_b: int, n_b: int,
                       z_threshold: float = Z_95) -> bool:
    """True only if A and B differ beyond noise at ~95%."""
    return abs(two_proportion_z(s_a, n_a, s_b, n_b)) >= z_threshold
