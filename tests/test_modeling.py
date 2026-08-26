import numpy as np
import pandas as pd
import pytest

from legmri.demo import demo_config, synthetic_cohort
from legmri.modeling import heldout_shap, make_search, run_nested_cv


def test_nested_cv_complete_predictions_and_fitted_artifacts(tmp_path, config):
    df, q, r = synthetic_cohort(60)
    cfg = demo_config(config)
    cfg["cv"].update(models=["LR"], strategies=["Clinical_Only"], bootstrap=10, shap=False)
    result = run_nested_cv(df, q, r, cfg, tmp_path)
    assert len(result) == 60
    assert result.n_predictions.eq(2).all()
    outer = pd.read_csv(tmp_path / "outer_predictions.csv")
    assert len(outer) == 120
    assert not outer.duplicated(["repeat", "Patient_ID"]).any()
    assert len(list(tmp_path.rglob("pipeline.joblib"))) == 6


@pytest.mark.parametrize("name", ["LR", "RF", "XGB"])
def test_optional_models_and_heldout_shap(name, config):
    pytest.importorskip("shap")
    if name == "XGB":
        pytest.importorskip("xgboost")
    df, q, r = synthetic_cohort(60)
    cfg = demo_config(config)
    cfg["cv"]["grids"]["XGB"] = {"n_estimators": [5], "max_depth": [2]}
    search = make_search("Clinical_Only", name, q, r, cfg)
    X, y = df[q + r + cfg["clinical"]], df.Diabetes_Status
    search.fit(X.iloc[:45], y.iloc[:45])
    result = heldout_shap(
        search.best_estimator_, X.iloc[:45], X.iloc[45:], df.Patient_ID.iloc[45:], {"model": name}
    )
    assert len(result) == 45
    assert np.isfinite([x["shap"] for x in result]).all()
    table = pd.DataFrame(result)
    reconstructed = (
        table.groupby("Patient_ID").shap.sum() + table.groupby("Patient_ID").base_value.first()
    )
    model = search.best_estimator_["model"]
    transformed = search.best_estimator_["select"].transform(X.iloc[45:])
    if name == "LR":
        expected = model.decision_function(transformed)
    elif name == "RF":
        expected = model.predict_proba(transformed)[:, 1]
    else:
        expected = model.predict(transformed, output_margin=True)
    assert np.allclose(reconstructed.to_numpy(), expected, atol=1e-5)


def test_id_merge_aliases_and_no_label_features(tmp_path, config):
    from legmri.modeling import load_model_data

    df, q, r = synthetic_cohort(20)
    clinical, quantitative, radiomics = (tmp_path / name for name in ("c.csv", "q.csv", "r.csv"))
    df.to_csv(clinical, index=False)
    df[["Patient_ID"] + q].rename(
        columns={"IMAT_to_IMAT_Muscle": "IMAT_to_（IMAT+Muscle）"}
    ).to_csv(quantitative, index=False)
    df[["Patient_ID"] + r].sample(frac=1, random_state=7).to_csv(radiomics, index=False)
    merged, qnames, rnames = load_model_data(clinical, quantitative, radiomics, config)
    assert "Glucose_0" not in merged and qnames == q and rnames == r
    assert merged.Patient_ID.tolist() == df.Patient_ID.tolist()
    assert np.allclose(merged[r], df[r])


def test_duplicate_participants_are_rejected(tmp_path, config):
    df, q, r = synthetic_cohort(20)
    df.loc[1, "Patient_ID"] = df.loc[0, "Patient_ID"]
    with pytest.raises(ValueError, match="One row per participant"):
        run_nested_cv(df, q, r, demo_config(config), tmp_path)


def test_search_fits_selector_only_on_inner_training_rows(config, monkeypatch):
    from legmri.selection import StudySelector

    df, q, r = synthetic_cohort(60)
    cfg = demo_config(config)
    search = make_search("Quantitative_Clinical", "LR", q, r, cfg)
    X, y = df[q + r + cfg["clinical"]], df.Diabetes_Status
    expected = [set(a) for a, _ in search.cv.split(X, y)] + [set(range(len(X)))]
    observed = []
    original = StudySelector.fit

    def traced(self, X, y):
        observed.append(set(X.index))
        return original(self, X, y)

    monkeypatch.setattr(StudySelector, "fit", traced)
    search.fit(X, y)
    assert observed == expected


def test_refit_serialization_and_cli_prediction(tmp_path, config):
    import joblib
    from legmri.cli import main
    from legmri.modeling import fit_final

    df, q, r = synthetic_cohort(60)
    cfg = demo_config(config)
    model_dir = tmp_path / "model"
    fit_final(df.iloc[:45], q, r, cfg, "Quantitative_Clinical", "RF", model_dir)
    features = tmp_path / "new.csv"
    df.iloc[45:].to_csv(features, index=False)
    output = tmp_path / "prediction.csv"
    main(["model-predict", str(model_dir), str(features), "--output", str(output)])
    loaded = joblib.load(model_dir / "pipeline.joblib")
    expected = loaded.predict_proba(df.iloc[45:])[:, 1]
    predictions = pd.read_csv(output)
    assert predictions.Patient_ID.tolist() == df.Patient_ID.iloc[45:].tolist()
    assert np.allclose(predictions.probability, expected)


@pytest.mark.parametrize("name", ["LR", "RF", "XGB"])
def test_empty_imaging_selection_produces_predictions(name, config):
    if name == "XGB":
        pytest.importorskip("xgboost")
    df, _, _ = synthetic_cohort(60)
    df["constant"] = 1.0
    cfg = demo_config(config)
    search = make_search("Quantitative_Only", name, ["constant"], [], cfg)
    search.fit(df.iloc[:45], df.Diabetes_Status.iloc[:45])
    assert search.best_estimator_["select"].selected_features_ == []
    probability = search.predict_proba(df.iloc[45:])[:, 1]
    assert len(probability) == 15 and np.isfinite(probability).all()
