"""Faithful surrogate of the decision policy.

SHAP attributes feature contributions of a *model*, but the trading decision is
produced by a multi-agent + risk-engine pipeline, not a single differentiable
model. The standard resolution is to fit a faithful **surrogate** (here a random
forest) that maps the tabular feature vector to the agent's action, then explain
the surrogate — and to report how faithfully it reproduces the policy
(``surrogate_fidelity``). An explanation is only as trustworthy as that fidelity.

scikit-learn is imported lazily so the rest of the framework runs without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

__all__ = ["SurrogateModel", "train_surrogate", "surrogate_fidelity"]


@dataclass
class SurrogateModel:
    """A fitted surrogate classifier plus the feature names it was trained on."""

    model: object  # sklearn RandomForestClassifier (kept loose to avoid hard import)
    feature_names: List[str]

    @property
    def classes_(self) -> List[str]:
        return list(self.model.classes_)  # type: ignore[attr-defined]

    def predict(self, matrix: Sequence[Sequence[float]]) -> List[str]:
        import numpy as np

        preds = self.model.predict(np.asarray(matrix, dtype=float))  # type: ignore[attr-defined]
        return [str(p) for p in preds]


def train_surrogate(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    feature_names: Sequence[str] | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 0,
) -> SurrogateModel:
    """Fit a random-forest surrogate mapping feature rows to action labels."""
    if len(matrix) != len(labels):
        raise ValueError(f"matrix rows ({len(matrix)}) != labels ({len(labels)})")
    if len(matrix) == 0:
        raise ValueError("cannot train on an empty dataset")

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.asarray(matrix, dtype=float)
    y = np.asarray([str(label) for label in labels])
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise ValueError(f"feature_names ({len(names)}) != n_features ({X.shape[1]})")

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
    )
    clf.fit(X, y)
    return SurrogateModel(model=clf, feature_names=names)


def surrogate_fidelity(
    model: SurrogateModel,
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
) -> float:
    """Agreement (accuracy) between surrogate predictions and the agent labels."""
    if len(matrix) == 0:
        raise ValueError("cannot score fidelity on an empty dataset")
    preds = model.predict(matrix)
    correct = sum(1 for p, t in zip(preds, labels) if p == str(t))
    return correct / len(labels)
