"""
SHAP Explainability — answers WHY the model flagged a location as high risk.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution score
for a specific prediction. Critical for scientific credibility and debugging.

Install: pip install shap
Docs:    https://shap.readthedocs.io
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """SHAP contribution for one feature in one prediction."""
    feature: str
    value: float          # actual feature value fed to model
    shap_value: float     # how much this feature pushed the score up/down
    direction: str        # 'increases_risk' | 'decreases_risk' | 'neutral'

    @property
    def abs_impact(self) -> float:
        return abs(self.shap_value)


@dataclass
class PredictionExplanation:
    """Full explanation for a single prediction."""
    risk_score: float
    base_value: float                          # model's average prediction
    contributions: list[FeatureContribution]   # sorted by abs impact
    top_driver: str                            # most impactful feature name

    def summary(self) -> str:
        lines = [
            f"Risk Score: {self.risk_score:.3f}  (base rate: {self.base_value:.3f})",
            f"Top driver: {self.top_driver}",
            "",
            "Feature contributions (+ = increases risk, - = decreases risk):",
        ]
        for c in self.contributions[:5]:
            sign = "+" if c.shap_value > 0 else ""
            lines.append(f"  {c.feature:<30} val={c.value:.3f}   SHAP={sign}{c.shap_value:.4f}")
        return "\n".join(lines)


class SHAPExplainer:
    """
    Wraps a trained BaseModel and computes SHAP values per prediction.
    Supports TreeExplainer (fast, exact) for tree-based models.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._explainer = None
        self._background: pd.DataFrame | None = None

    def fit_background(self, X_background: pd.DataFrame) -> None:
        """
        Fit the SHAP explainer on background data (typically training set sample).
        Call this once after training before calling explain().
        """
        try:
            import shap
        except ImportError:
            logger.error("shap not installed. Run: pip install shap")
            return

        sklearn_model = getattr(self._model, "_model", self._model)
        model_type = type(sklearn_model).__name__

        try:
            if "XGB" in model_type or "RandomForest" in model_type or "GradientBoosting" in model_type:
                self._explainer = shap.TreeExplainer(sklearn_model)
                logger.info("Using TreeExplainer for %s", model_type)
            else:
                # Fallback: KernelExplainer works on any model but is slower
                background = shap.sample(X_background, 100)
                predict_fn = lambda x: sklearn_model.predict_proba(x)[:, 1]
                self._explainer = shap.KernelExplainer(predict_fn, background)
                logger.info("Using KernelExplainer for %s", model_type)

            self._background = X_background
            logger.info("SHAP explainer ready on %d background samples", len(X_background))
        except Exception as e:
            logger.error("SHAP explainer init failed: %s", e)

    def explain(self, X: pd.DataFrame, risk_scores: np.ndarray) -> list[PredictionExplanation]:
        """
        Compute SHAP explanations for a batch of predictions.
        Returns one PredictionExplanation per row in X.
        """
        if self._explainer is None:
            logger.warning("Explainer not fitted — call fit_background() first.")
            return []

        try:
            import shap
            shap_values = self._explainer.shap_values(X)

            # For binary classifiers TreeExplainer returns [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # take class=1 (fire risk)

            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1])

            explanations = []
            for i, row in enumerate(X.itertuples(index=False)):
                contributions = []
                for j, feat in enumerate(X.columns):
                    sv = float(shap_values[i, j])
                    fv = float(X.iloc[i, j])
                    direction = "increases_risk" if sv > 0.005 else \
                                "decreases_risk" if sv < -0.005 else "neutral"
                    contributions.append(FeatureContribution(
                        feature=feat,
                        value=fv,
                        shap_value=sv,
                        direction=direction,
                    ))

                contributions.sort(key=lambda c: c.abs_impact, reverse=True)
                explanations.append(PredictionExplanation(
                    risk_score=float(risk_scores[i]),
                    base_value=base_value,
                    contributions=contributions,
                    top_driver=contributions[0].feature if contributions else "unknown",
                ))

            return explanations

        except Exception as e:
            logger.error("SHAP computation failed: %s", e)
            return []

    def plot_summary(self, X: pd.DataFrame, output_path: Path | None = None) -> None:
        """
        Renders a SHAP beeswarm summary plot.
        Shows global feature importance across all predictions.
        """
        if self._explainer is None:
            logger.warning("Call fit_background() before plotting.")
            return
        try:
            import shap
            import matplotlib.pyplot as plt
            shap_values = self._explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap.summary_plot(shap_values, X, show=False)
            if output_path:
                plt.savefig(output_path, bbox_inches="tight", dpi=150)
                logger.info("SHAP summary plot saved to %s", output_path)
            else:
                plt.show()
        except Exception as e:
            logger.error("SHAP plot failed: %s", e)

    def plot_waterfall(
        self,
        explanation: PredictionExplanation,
        output_path: Path | None = None,
    ) -> None:
        """
        Renders a waterfall chart for a single prediction.
        Shows exactly how each feature pushed the score up or down from base.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            features = [c.feature for c in explanation.contributions[:8]]
            values = [c.shap_value for c in explanation.contributions[:8]]
            colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(features, values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (contribution to risk score)")
            ax.set_title(
                f"Prediction Explanation — Risk Score: {explanation.risk_score:.3f}\n"
                f"Top driver: {explanation.top_driver}"
            )
            red = mpatches.Patch(color="#d73027", label="Increases risk")
            blue = mpatches.Patch(color="#4575b4", label="Decreases risk")
            ax.legend(handles=[red, blue])
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, bbox_inches="tight", dpi=150)
            else:
                plt.show()
        except Exception as e:
            logger.error("Waterfall plot failed: %s", e)
