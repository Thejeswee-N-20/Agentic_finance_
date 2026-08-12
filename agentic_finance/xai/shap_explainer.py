"""TreeSHAP feature attributions over the decision surrogate.

- ``local_attributions``: per-feature SHAP values for a single decision, for the
  surrogate's predicted class (the auditable "why this call").
- ``global_importance``: mean |SHAP| per feature across samples and classes (the
  beeswarm/summary view).

Handles the differing shapes that ``shap.TreeExplainer.shap_values`` returns
across versions (list-of-arrays vs 3-D ndarray) and degrades gracefully on a
single-class surrogate (no discriminative signal -> zero attributions).
``shap``/``numpy`` are imported lazily.
"""
from __future__ import annotations

from typing import Dict, Sequence

from agentic_finance.xai.surrogate import SurrogateModel

__all__ = ["local_attributions", "global_importance"]


def _normalized_shap(model_obj, X):
    """Return SHAP values as an ndarray of shape (n_samples, n_features, n_classes)."""
    import numpy as np
    import shap

    explainer = shap.TreeExplainer(model_obj)
    sv = explainer.shap_values(X)

    if isinstance(sv, list):  # list per class -> stack on last axis
        return np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 3:          # already (n_samples, n_features, n_classes)
        return sv
    if sv.ndim == 2:          # (n_samples, n_features) -> single output
        return sv[:, :, np.newaxis]
    raise ValueError(f"unexpected SHAP value shape: {sv.shape}")


def local_attributions(surrogate: SurrogateModel, row: Sequence[float]) -> Dict[str, float]:
    """Per-feature SHAP attribution for ``row``, for the predicted class."""
    names = surrogate.feature_names
    if len(surrogate.classes_) < 2:
        return {name: 0.0 for name in names}

    import numpy as np

    X = np.asarray([list(row)], dtype=float)
    sv = _normalized_shap(surrogate.model, X)  # (1, n_features, n_classes)
    predicted = surrogate.predict([list(row)])[0]
    class_idx = surrogate.classes_.index(predicted)
    values = sv[0, :, class_idx]
    return {name: float(v) for name, v in zip(names, values)}


def global_importance(
    surrogate: SurrogateModel,
    matrix: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """Mean |SHAP| per feature across all samples and classes."""
    names = surrogate.feature_names
    if len(surrogate.classes_) < 2:
        return {name: 0.0 for name in names}

    import numpy as np

    X = np.asarray(matrix, dtype=float)
    sv = _normalized_shap(surrogate.model, X)        # (n, f, c)
    importance = np.abs(sv).mean(axis=(0, 2))         # average over samples + classes
    return {name: float(v) for name, v in zip(names, importance)}
