"""Counterfactual explanations over the decision surrogate.

Answers the regulator's question SHAP cannot: *"what is the smallest change to
the inputs that would have flipped this decision?"* [Wachter et al., 2017].
Because the surrogate's fidelity to the real policy is measured (see
``surrogate_fidelity``), a counterfactual on the surrogate is an auditable,
quantified statement about the decision boundary — e.g. *"the BUY becomes a
HOLD if 20-day momentum falls below +2.1%"*.

Method: a per-feature line search. Each feature is swept across its observed
range (holding all others fixed); the smallest move that changes the
surrogate's prediction — optionally to a specific ``target_action`` — is
recorded. Results are ranked by *relative* change (fraction of the feature's
range), so the first counterfactual is the "nearest" alternative world.
Single-feature search keeps the explanation human-readable and is the standard
first-order approach; multi-feature search is a noted extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agentic_finance.xai.surrogate import SurrogateModel

__all__ = ["Counterfactual", "find_counterfactuals", "counterfactual_report"]


@dataclass(frozen=True)
class Counterfactual:
    """The smallest found single-feature change that flips the decision."""

    feature: str
    original_value: float
    counterfactual_value: float
    delta: float
    original_action: str
    counterfactual_action: str


def _ranges_from_matrix(matrix: Sequence[Sequence[float]],
                        names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    cols = list(zip(*matrix))
    return {name: (min(col), max(col)) for name, col in zip(names, cols)}


def find_counterfactuals(
    surrogate: SurrogateModel,
    row: Sequence[float],
    target_action: Optional[str] = None,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    training_matrix: Optional[Sequence[Sequence[float]]] = None,
    steps: int = 25,
) -> List[Counterfactual]:
    """Per-feature minimal-change counterfactuals for one decision row.

    ``feature_ranges`` (name -> (lo, hi)) bounds the search; if absent it is
    derived from ``training_matrix``. Returns counterfactuals sorted by
    relative change (nearest first); empty when no single-feature change flips
    the decision (e.g. a single-class surrogate) or the row already has the
    target action.
    """
    names = surrogate.feature_names
    if feature_ranges is None:
        if training_matrix is None:
            raise ValueError("provide feature_ranges or training_matrix")
        feature_ranges = _ranges_from_matrix(training_matrix, names)

    original_action = surrogate.predict([list(row)])[0]
    if target_action is not None and original_action == target_action:
        return []
    if len(set(surrogate.classes_)) < 2:
        return []

    found: List[Counterfactual] = []
    for idx, name in enumerate(names):
        lo, hi = feature_ranges.get(name, (None, None))
        if lo is None or hi is None or hi == lo:
            continue
        original_value = float(row[idx])
        best: Optional[Counterfactual] = None
        for step in range(steps + 1):
            candidate_value = lo + (hi - lo) * step / steps
            if candidate_value == original_value:
                continue
            candidate = list(row)
            candidate[idx] = candidate_value
            action = surrogate.predict([candidate])[0]
            if action == original_action:
                continue
            if target_action is not None and action != target_action:
                continue
            delta = candidate_value - original_value
            if best is None or abs(delta) < abs(best.delta):
                best = Counterfactual(
                    feature=name, original_value=original_value,
                    counterfactual_value=candidate_value, delta=delta,
                    original_action=original_action, counterfactual_action=action,
                )
        if best is not None:
            found.append(best)

    def _relative(cf: Counterfactual) -> float:
        lo, hi = feature_ranges[cf.feature]
        span = hi - lo
        return abs(cf.delta) / span if span else float("inf")

    return sorted(found, key=_relative)


def counterfactual_report(cfs: Sequence[Counterfactual], top: int = 3) -> str:
    """Human-readable summary of the nearest counterfactuals."""
    if not cfs:
        return "No counterfactual found: no single-feature change flips this decision."
    lines = ["Counterfactuals (smallest changes that flip the decision):"]
    for cf in list(cfs)[:top]:
        lines.append(
            f"  - {cf.original_action} -> {cf.counterfactual_action} if "
            f"`{cf.feature}` moves from {cf.original_value:+.4f} to "
            f"{cf.counterfactual_value:+.4f} (Δ {cf.delta:+.4f})"
        )
    return "\n".join(lines)
