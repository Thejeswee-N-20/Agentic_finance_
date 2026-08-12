"""Explainable-AI (XAI) layer for the Agentic Finance framework.

A multi-level, *measured* explainability stack rather than a single SHAP plot:

- ``risk_attribution`` : which risk constraint bound each enforced position
  (pure-Python, from the ``RiskedOrder`` sequence) — quantifies the
  "risk-mitigated" claim. (no extra deps)
- ``surrogate``        : a faithful gradient-boosted surrogate of the decision
  policy + a fidelity score. (scikit-learn)
- ``shap_explainer``   : local + global TreeSHAP feature attributions. (shap)
- ``report``           : human-readable decision audit report.

Heavier deps (scikit-learn, shap) are imported lazily inside their modules so
``risk_attribution`` and the rest of the framework work without them installed.
"""

from agentic_finance.xai.risk_attribution import (
    attribute_constraints,
    binding_summary,
    explain_order,
)
from agentic_finance.xai.counterfactual import (
    Counterfactual,
    counterfactual_report,
    find_counterfactuals,
)
from agentic_finance.xai.report import decision_report, decision_trace

# Surrogate + SHAP pull in scikit-learn / shap; expose lazily so importing the
# package (e.g. for risk_attribution / report) never forces those heavy deps.
__all__ = [
    # no-dep
    "attribute_constraints",
    "binding_summary",
    "explain_order",
    "decision_report",
    "decision_trace",
    "Counterfactual",
    "find_counterfactuals",
    "counterfactual_report",
    # lazy (scikit-learn / shap)
    "train_surrogate",
    "surrogate_fidelity",
    "SurrogateModel",
    "local_attributions",
    "global_importance",
]


def __getattr__(name):  # PEP 562 lazy import for the heavy-dep symbols
    if name in {"train_surrogate", "surrogate_fidelity", "SurrogateModel"}:
        from agentic_finance.xai import surrogate as _s
        return getattr(_s, name)
    if name in {"local_attributions", "global_importance"}:
        from agentic_finance.xai import shap_explainer as _x
        return getattr(_x, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
