"""Seven feature strategies evaluated on identical repeated nested CV splits."""

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .io import read_table, record_run, write_json
from .metrics import binary_metrics, bootstrap_metrics, participant_predictions
from .selection import StudySelector

STRATEGIES = {
    "Clinical_Only": (False, False, True),
    "Quantitative_Only": (True, False, False),
    "Radiomics_Only": (False, True, False),
    "Image_Only": (True, True, False),
    "Quantitative_Clinical": (True, False, True),
    "Radiomics_Clinical": (False, True, True),
    "Image_Clinical": (True, True, True),
}

# Supplementary Table S6. Volumes remain available in the extraction output,
# but are not silently added to the 14-feature quantitative modeling panel.
QUANTITATIVE = [
    "IMAT_to_Muscle",
    "Subcutaneous_to_Muscle",
    "IMAT_to_IMAT_Muscle",
    "Muscle_to_SoftTissue",
    "Fat_to_SoftTissue",
    "IMAT_to_SoftTissue",
    "SubFat_to_SoftTissue",
    "Fat_to_Muscle",
    "SMFF_mean",
    "SMFF_std",
    "SMFF_median",
    "SMFF_25perc",
    "SMFF_75perc",
    "SMFF_wf",
]


def load_model_data(clinical_path, quantitative_path, radiomics_path, config):
    sid, target = config["id_col"], config["target"]
    df = read_table(clinical_path, sid)
    if not set(df[target].unique()) <= {0, 1} or df[target].nunique() != 2:
        raise ValueError("Target must be 0=normal, 1=dysglycemia, with both classes present")
    if not set(df.Sex.dropna().unique()) <= {0, 1}:
        raise ValueError("Encode Sex consistently as 0/1")
    q, r = [], []
    # Never expose OGTT measurements or endpoints to feature discovery.
    df = df[[sid, target] + config["clinical"]].copy()
    for path, domain in ((quantitative_path, "quantitative"), (radiomics_path, "radiomics")):
        if path is None:
            continue
        features = read_table(path, sid)
        aliases = {k: v for k, v in config.get("feature_aliases", {}).items() if k in features}
        features = features.rename(columns=aliases)
        if features.columns.duplicated().any():
            raise ValueError("Feature aliases created duplicate columns")
        if domain == "quantitative":
            cols = config.get("quantitative_features", QUANTITATIVE)
            missing = set(cols) - set(features)
            if missing:
                raise ValueError(f"Missing quantitative columns: {sorted(missing)}")
            q = list(cols)
        else:
            cols = [c for c in features if "_original_" in c and not c.startswith("diagnostics_")]
            if not cols:
                raise ValueError("Radiomics columns must contain '_original_'")
            r = cols
        if not set(df[sid]) <= set(features[sid]):
            raise ValueError(f"{domain}: missing participants from clinical cohort")
        if set(cols) & {sid, target, *config["clinical"]}:
            raise ValueError(
                "Imaging features cannot include identifiers, targets or clinical covariates"
            )
        df = df.merge(features[[sid] + list(cols)], on=sid, how="left", validate="one_to_one")
    df = df.sort_values(sid).reset_index(drop=True)
    for col in q + r + config["clinical"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df, q, r


def make_search(strategy, model_name, q, r, config, cache=None):
    use_q, use_r, use_c = STRATEGIES[strategy]
    if use_q and not q or use_r and not r:
        raise ValueError(f"{strategy} requires a missing feature domain")
    seed, cfg = config["seed"], config["cv"]
    selector = StudySelector(
        q if use_q else [],
        r if use_r else [],
        config["clinical"] if use_c else [],
        config["selection"],
        random_state=seed,
    )
    if model_name == "LR":
        model = LogisticRegression(
            class_weight="balanced", max_iter=5000, solver="liblinear", random_state=seed
        )
    elif model_name == "RF":
        model = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
    elif model_name == "XGB":
        from .boosting import FoldBalancedXGB

        model = FoldBalancedXGB(eval_metric="logloss", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown model {model_name}")
    pipeline = Pipeline([("select", selector), ("model", model)], memory=cache)
    grid = {f"model__{k}": v for k, v in cfg["grids"][model_name].items()}
    if use_q or use_r:
        grid["select__lasso_c"] = config["selection"]["lasso_c"]
    inner = RepeatedStratifiedKFold(
        n_splits=cfg["inner_folds"],
        n_repeats=cfg["inner_repeats"],
        random_state=cfg.get("inner_seed", 7),
    )
    return GridSearchCV(
        pipeline,
        grid,
        cv=inner,
        scoring="roc_auc",
        n_jobs=cfg["n_jobs"],
        error_score="raise",
        refit=True,
        return_train_score=False,
    )


def heldout_shap(pipeline, train, test, ids, metadata):
    import shap

    selector, model = pipeline["select"], pipeline["model"]
    background, x = selector.transform(train), selector.transform(test)
    if isinstance(model, LogisticRegression):
        explanation = shap.LinearExplainer(model, background)(x)
    else:
        explanation = shap.TreeExplainer(model)(x)
    values = explanation.values
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        base_values = base_values[:, 1]
    base_values = np.broadcast_to(base_values.squeeze(), (len(x),))
    rows = []
    for i, sid in enumerate(ids):
        for j, feature in enumerate(selector.get_feature_names_out()):
            rows.append(
                {
                    **metadata,
                    "Patient_ID": sid,
                    "feature": feature,
                    "value_standardized": x[i, j],
                    "shap": values[i, j],
                    "base_value": base_values[i],
                    "output_scale": "probability"
                    if isinstance(model, RandomForestClassifier)
                    else "log_odds",
                }
            )
    return rows


def run_nested_cv(frame, q, r, config, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cfg, seed, target = config["cv"], config["seed"], config["target"]
    if frame[config["id_col"]].duplicated().any():
        raise ValueError("One row per participant is required for subject-level CV")
    if (out / "outer_predictions.csv").exists():
        raise FileExistsError("Output already contains a CV run; choose a new output directory")
    X, y, ids = frame[q + r + config["clinical"]], frame[target].to_numpy(), frame[config["id_col"]]
    outer = RepeatedStratifiedKFold(
        n_splits=cfg["outer_folds"], n_repeats=cfg["outer_repeats"], random_state=seed
    )
    splits = list(outer.split(X, y))
    write_json(
        out / "outer_splits.json",
        [
            {
                "repeat": i // cfg["outer_folds"],
                "fold": i % cfg["outer_folds"],
                "train": ids.iloc[a].tolist(),
                "test": ids.iloc[b].tolist(),
            }
            for i, (a, b) in enumerate(splits)
        ],
    )
    predictions, fold_metrics, selections, shap_rows = [], [], [], []
    for i, (train, test) in enumerate(splits):
        repeat, fold = divmod(i, cfg["outer_folds"])
        for strategy, model in itertools.product(cfg["strategies"], cfg["models"]):
            print(f"repeat={repeat} fold={fold} {strategy}/{model}", flush=True)
            meta = {"repeat": repeat, "fold": fold, "strategy": strategy, "model": model}
            search = make_search(strategy, model, q, r, config, cache=str(out / "_cache"))
            search.fit(X.iloc[train], y[train])
            estimator = search.best_estimator_
            prob = estimator.predict_proba(X.iloc[test])[:, 1]
            predictions.extend(
                {**meta, "Patient_ID": ids.iloc[j], "y": int(y[j]), "probability": float(p)}
                for j, p in zip(test, prob)
            )
            fold_metrics.append(
                {
                    **meta,
                    "inner_auc": search.best_score_,
                    **binary_metrics(y[test], prob, cfg["threshold"]),
                }
            )
            selections.extend(
                {**meta, "feature": name} for name in estimator["select"].selected_features_
            )
            folder = out / "fold_models" / strategy / model / f"repeat_{repeat}_fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            # Strip only transient joblib caching, preserving the fitted selector/model.
            estimator.memory = None
            joblib.dump(estimator, folder / "pipeline.joblib")
            write_json(
                folder / "fit.json",
                {
                    **meta,
                    "best_params": search.best_params_,
                    "training_subjects": ids.iloc[train].tolist(),
                    "test_subjects": ids.iloc[test].tolist(),
                    "integration_frequency": estimator["select"].integration_frequencies_,
                    "selected_features": estimator["select"].selected_features_,
                },
            )
            if cfg.get("shap", False):
                shap_rows.extend(
                    heldout_shap(estimator, X.iloc[train], X.iloc[test], ids.iloc[test], meta)
                )
            pd.DataFrame(predictions).to_csv(out / "outer_predictions.csv", index=False)
    predictions = pd.DataFrame(predictions)
    expected = cfg["outer_repeats"]
    participants = participant_predictions(predictions)
    if not participants.n_predictions.eq(expected).all():
        raise RuntimeError(
            "Incomplete outer predictions: every participant must appear once per repeat"
        )
    participants.to_csv(out / "participant_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(fold_metrics).groupby(["strategy", "model"]).inner_auc.agg(
        ["mean", "std", "count"]
    ).to_csv(out / "inner_auc_summary.csv")
    summary = []
    for (strategy, model), group in participants.groupby(["strategy", "model"]):
        metrics = bootstrap_metrics(
            group.y, group.probability, cfg["bootstrap"], seed, cfg["threshold"]
        )
        for metric, values in metrics.items():
            summary.append({"strategy": strategy, "model": model, "metric": metric, **values})
    pd.DataFrame(summary).to_csv(out / "performance.csv", index=False)
    if selections:
        selected = pd.DataFrame(selections)
        selected.to_csv(out / "selected_features.csv", index=False)
        frequency = (
            selected.groupby(["strategy", "model", "feature"]).size().rename("count").reset_index()
        )
        frequency["frequency"] = frequency["count"] / len(splits)
        frequency.to_csv(out / "selection_stability.csv", index=False)
    if shap_rows:
        shap_frame = pd.DataFrame(shap_rows)
        shap_frame.to_csv(out / "heldout_shap.csv", index=False)
        # Missing feature in a fold has zero attribution; denominator includes all predictions.
        importance = shap_frame.assign(abs_shap=shap_frame.shap.abs()).groupby(
            ["strategy", "model", "feature"]
        ).abs_shap.sum() / (len(frame) * expected)
        importance.rename("mean_abs_shap").reset_index().to_csv(
            out / "shap_importance.csv", index=False
        )
        from .plots import shap_plots

        shap_plots(shap_frame, out, len(frame) * expected)
    record_run(out, config)
    from .plots import prediction_plots

    prediction_plots(participants, out, cfg["bootstrap"], seed)
    return participants


def fit_final(frame, q, r, config, strategy, model, output):
    """Fit a user-selected model after CV; this step produces no new test estimate."""
    search = make_search(strategy, model, q, r, config)
    X = frame[q + r + config["clinical"]]
    search.fit(X, frame[config["target"]])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, out / "pipeline.joblib")
    write_json(
        out / "model.json",
        {
            "strategy": strategy,
            "model": model,
            "id_col": config["id_col"],
            "best_params": search.best_params_,
            "input_columns": list(X),
            "training_subjects": frame[config["id_col"]].tolist(),
        },
    )
    record_run(out, config)
