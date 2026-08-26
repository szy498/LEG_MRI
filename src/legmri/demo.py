"""Generate synthetic feature tables for software testing."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_json
from .modeling import QUANTITATIVE, STRATEGIES, run_nested_cv


def synthetic_cohort(n=80, seed=42):
    rng = np.random.default_rng(seed)
    age, sex, bmi = rng.normal(45, 8, n), rng.integers(0, 2, n), rng.normal(25, 3, n)
    phenotype, random_intercept = rng.normal(size=n), rng.normal(0, 0.5, n)
    df = pd.DataFrame(
        {"Patient_ID": [f"synthetic-{i:04d}" for i in range(n)], "Age": age, "Sex": sex, "BMI": bmi}
    )
    for j, name in enumerate(QUANTITATIVE):
        df[name] = (
            (0.1 + 0.015 * phenotype + rng.normal(0, 0.003, n))
            if j == 0
            else rng.uniform(0.01, 0.3, n)
        )
    df["IMAT_to_SoftTissue"] = 0.05 + 0.009 * phenotype + rng.normal(0, 0.006, n)
    for time, mean, interaction in ((0, 5.5, 0.15), (60, 8.2, 1.0), (120, 7.6, 0.8)):
        df[f"Glucose_{time}"] = (
            mean + interaction * phenotype + random_intercept + rng.normal(0, 0.5, n)
        )
        df[f"Insulin_{time}"] = np.expm1(
            2.4
            + (time != 0)
            + interaction * phenotype * 0.3
            + random_intercept * 0.3
            + rng.normal(0, 0.2, n)
        )
    df["HbA1c"] = 5.5 + 0.2 * phenotype + rng.normal(0, 0.15, n)
    df["Diabetes_Status"] = ((df.Glucose_0 >= 6.1) | (df.Glucose_120 >= 7.8)).astype(int)
    radiomics = [
        f"Muscle_Fat_original_firstorder_{name}"
        for name in ("Mean", "90Percentile", "Kurtosis", "Skewness", "Median", "Variance")
    ]
    for i, name in enumerate(radiomics):
        df[name] = phenotype * (0.8 if i < 2 else 0.1) + rng.normal(size=n)
    return df, list(QUANTITATIVE), radiomics


def demo_config(config):
    cfg = deepcopy(config)
    cfg["cv"].update(
        outer_folds=3,
        outer_repeats=2,
        inner_folds=3,
        inner_repeats=1,
        bootstrap=50,
        shap=True,
        models=["LR", "RF", "XGB"],
        strategies=list(STRATEGIES),
    )
    cfg["cv"]["grids"] = {
        "LR": {"C": [1]},
        "RF": {"n_estimators": [20], "max_depth": [3]},
        "XGB": {"n_estimators": [10], "max_depth": [2]},
    }
    cfg["selection"].update(lasso_c=[1], integration_folds=3, texture_stability_splits=3)
    return cfg


def run_demo(config, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    df, q, r = synthetic_cohort()
    cfg = demo_config(config)
    df.to_csv(out / "synthetic.csv", index=False)
    df[["Patient_ID", "Diabetes_Status", "Age", "Sex", "BMI"]].to_csv(
        out / "clinical.csv", index=False
    )
    df[["Patient_ID"] + q].to_csv(out / "quantitative.csv", index=False)
    df[["Patient_ID"] + r].to_csv(out / "radiomics.csv", index=False)
    write_json(
        out / "SYNTHETIC_DATA_ONLY.json", {"purpose": "software integration test", "n": len(df)}
    )
    result = run_nested_cv(df, q, r, cfg, out / "prediction")
    return result
