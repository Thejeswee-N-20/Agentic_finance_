"""Position sizing rules.

All sizers return a **fraction of portfolio equity** to allocate to a single
position, in ``[0, max_leverage]``. They are pure functions: deterministic, no
mutation, fully unit-testable. The risk manager composes them with VaR/CVaR
limits and per-position / portfolio caps.
"""
from __future__ import annotations

__all__ = [
    "volatility_target_size",
    "fractional_kelly_size",
    "apply_caps",
]


def volatility_target_size(
    target_vol: float,
    asset_vol: float,
    max_leverage: float = 1.0,
) -> float:
    """Inverse-volatility sizing: ``size = target_vol / asset_vol``.

    Scales exposure so the position contributes roughly ``target_vol`` of
    volatility. Lower asset volatility -> larger size, clamped to
    ``max_leverage``. A zero/near-zero ``asset_vol`` returns ``max_leverage``.
    """
    if target_vol < 0:
        raise ValueError(f"target_vol must be >= 0, got {target_vol}")
    if asset_vol < 0:
        raise ValueError(f"asset_vol must be >= 0, got {asset_vol}")
    if asset_vol == 0:
        return max_leverage
    size = target_vol / asset_vol
    return min(size, max_leverage)


def fractional_kelly_size(
    win_prob: float,
    win_loss_ratio: float = 1.0,
    fraction: float = 0.5,
) -> float:
    """Fractional-Kelly sizing.

    Full Kelly for a payoff of ``b`` to 1 with win probability ``p`` is
    ``f* = (b*p - (1 - p)) / b``. We scale by ``fraction`` (default half-Kelly,
    the common risk-managed choice) and clamp negatives (no edge) to 0.
    """
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")
    if win_loss_ratio <= 0:
        raise ValueError(f"win_loss_ratio must be > 0, got {win_loss_ratio}")
    if fraction < 0:
        raise ValueError(f"fraction must be >= 0, got {fraction}")
    b = win_loss_ratio
    full_kelly = (b * win_prob - (1.0 - win_prob)) / b
    return max(0.0, fraction * full_kelly)


def apply_caps(
    desired_size: float,
    per_position_cap: float,
    remaining_budget: float = 1.0,
) -> float:
    """Clamp a desired size to ``[0, min(per_position_cap, remaining_budget)]``.

    ``remaining_budget`` is the unused fraction of portfolio equity, so a new
    position can never push total allocation past 100% (or whatever budget the
    portfolio manager passes in).
    """
    ceiling = min(per_position_cap, remaining_budget)
    return max(0.0, min(desired_size, ceiling))
