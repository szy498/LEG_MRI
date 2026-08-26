"""Reproducible, headless figures from computed outputs only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import decision_curve, roc_band


def prediction_plots(predictions, output, n_bootstrap=1000, seed=42):
    out = Path(output) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for (strategy, model), d in predictions.groupby(["strategy", "model"]):
        y, p = d.y.to_numpy(), d.probability.to_numpy()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        fpr, tpr, _ = roc_curve(y, p)
        grid, bounds = roc_band(y, p, n_bootstrap, seed)
        axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, p):.3f}")
        axes[0].fill_between(grid, *bounds, alpha=0.18)
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
        axes[0].legend()
        observed, predicted = calibration_curve(y, p, n_bins=5, strategy="quantile")
        axes[1].plot(predicted, observed, "o", label="Quintiles")
        if len(np.unique(p)) > 2:
            loess = sm.nonparametric.lowess(y, p, frac=0.75, it=0)
            axes[1].plot(loess[:, 0], loess[:, 1], label="LOESS")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1].set(
            xlabel="Predicted probability", ylabel="Observed proportion", xlim=(0, 1), ylim=(0, 1)
        )
        histogram = axes[1].inset_axes([0.02, 0.02, 0.96, 0.17])
        histogram.hist(p, bins=np.linspace(0, 1, 11), color="gray", alpha=0.3)
        histogram.axis("off")
        axes[1].legend(loc="upper left")
        dca = decision_curve(y, p)
        for col in ("model", "all", "none"):
            axes[2].plot(dca.threshold, dca[col], label=col)
        axes[2].set(
            xlabel="Threshold probability",
            ylabel="Net benefit",
            ylim=(-0.1, max(0.1, y.mean() + 0.05)),
        )
        axes[2].legend()
        fig.suptitle(f"{strategy} / {model}")
        fig.savefig(out / f"{strategy}_{model}.png", dpi=200)
        plt.close(fig)
        dca.to_csv(out / f"{strategy}_{model}_dca.csv", index=False)


def shap_plots(shap_values, output, n_predictions):
    """Fold-held-out SHAP: bar denominator includes zero contributions in unselected folds."""
    out = Path(output) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for (strategy, model), group in shap_values.groupby(["strategy", "model"]):
        importance = group.assign(abs_shap=group.shap.abs()).groupby("feature").abs_shap.sum()
        importance = (importance / n_predictions).nlargest(20).sort_values()
        fig, axes = plt.subplots(
            1, 2, figsize=(12, max(4, len(importance) * 0.28)), constrained_layout=True
        )
        axes[0].barh(importance.index, importance.values, color="#2878b5")
        axes[0].set_xlabel("Mean |SHAP| across held-out predictions")
        rng = np.random.default_rng(42)
        for i, feature in enumerate(importance.index):
            values = group.loc[group.feature == feature]
            # Values shown are fold-specific standardized values, never global scaling.
            colors = values.value_standardized.clip(-2, 2)
            axes[1].scatter(
                values.shap,
                i + rng.uniform(-0.22, 0.22, len(values)),
                c=colors,
                cmap="coolwarm",
                vmin=-2,
                vmax=2,
                s=9,
                alpha=0.65,
            )
        axes[1].axvline(0, color="gray", lw=0.7)
        axes[1].set_yticks(range(len(importance)), labels=importance.index)
        axes[1].set_xlabel("SHAP value (red: higher feature value)")
        fig.suptitle(f"{strategy} / {model} / {group.output_scale.iloc[0]}")
        fig.savefig(out / f"{strategy}_{model}_shap.png", dpi=200)
        plt.close(fig)
