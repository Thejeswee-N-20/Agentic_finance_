"""Statistical rigor for backtest results: bootstrap CIs and the deflated Sharpe.

Bailey & López de Prado document how backtest overfitting and multiple testing
inflate apparent skill. Two remedies are implemented here, both dependency-free:

- ``bootstrap_confidence_interval`` — a moving-block bootstrap (blocks preserve
  the short-range autocorrelation of daily returns) yielding a percentile CI
  for any metric of the return series.
- ``deflated_sharpe_ratio`` — the probability that the observed Sharpe ratio
  exceeds the expected maximum Sharpe of ``n_trials`` unskilled strategies,
  correcting for non-normality (skew, kurtosis) and selection bias
  [Bailey & López de Prado, 2014]. Values near 1 indicate skill unlikely to be
  a multiple-testing artefact; values near 0.5 or below indicate the result is
  consistent with selection from noise.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Callable, Iterable, List, Sequence, Tuple

__all__ = ["bootstrap_confidence_interval", "deflated_sharpe_ratio"]

_EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def bootstrap_confidence_interval(
    returns: Iterable[float],
    metric: Callable[[Sequence[float]], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    block: int = 5,
    seed: int = 0,
) -> Tuple[float, float]:
    """Moving-block-bootstrap percentile CI for ``metric(returns)``.

    Resamples whole blocks of ``block`` consecutive returns (preserving local
    autocorrelation) to build ``n_boot`` pseudo-series, evaluates the metric on
    each, and returns the ``ci`` percentile interval. Deterministic for a given
    ``seed``.
    """
    rets: List[float] = list(returns)
    n = len(rets)
    if n == 0:
        return (0.0, 0.0)
    block = max(1, min(block, n))
    rng = random.Random(seed)
    n_blocks = math.ceil(n / block)
    stats: List[float] = []
    for _ in range(n_boot):
        sample: List[float] = []
        for _ in range(n_blocks):
            start = rng.randrange(0, max(1, n - block + 1))
            sample.extend(rets[start:start + block])
        stats.append(float(metric(sample[:n])))
    stats.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, min(n_boot - 1, int(alpha * n_boot)))
    hi_idx = max(0, min(n_boot - 1, int((1.0 - alpha) * n_boot) - 1))
    return (stats[lo_idx], stats[hi_idx])


def _moments(rets: Sequence[float]) -> Tuple[float, float, float, float]:
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    std = math.sqrt(var)
    if std == 0.0:
        return mean, 0.0, 0.0, 3.0
    skew = sum((r - mean) ** 3 for r in rets) / (n * std ** 3)
    kurt = sum((r - mean) ** 4 for r in rets) / (n * std ** 4)
    return mean, std, skew, kurt


def deflated_sharpe_ratio(returns: Iterable[float], n_trials: int) -> float:
    """P(true Sharpe > 0 | observed Sharpe, ``n_trials`` tested strategies).

    Implements DSR = Phi( (SR - SR0) * sqrt(T-1) / sqrt(1 - skew*SR +
    (kurt-1)/4 * SR^2) ) with SR the per-period Sharpe and SR0 the expected
    maximum per-period Sharpe of ``n_trials`` independent unskilled trials.
    Returns 0.0 for series too short or degenerate to assess.
    """
    rets: List[float] = list(returns)
    n = len(rets)
    if n < 2 or n_trials < 1:
        return 0.0
    mean, std, skew, kurt = _moments(rets)
    if std == 0.0:
        return 0.0
    sr = mean / std  # per-period Sharpe

    # Expected max Sharpe under H0 across n_trials (selection-bias benchmark).
    if n_trials == 1:
        sr0 = 0.0
    else:
        e = math.e
        z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * e))
        sr0 = math.sqrt(1.0 / (n - 1)) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)

    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0.0:
        return 0.0
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_NORMAL.cdf(z))
