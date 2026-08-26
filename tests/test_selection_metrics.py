import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from legmri.metrics import bootstrap_metrics, decision_curve, participant_predictions
from legmri.selection import StudySelector, correlation_keep


def test_correlation_is_absolute_and_respects_priority():
    x = pd.DataFrame({"less_interpretable": [1, 2, 3, 4], "preferred": [-1, -2, -3, -4]})
    assert correlation_keep(x, 0.9, ["preferred"]) == ["preferred"]


def test_selector_transform_does_not_learn_test_statistics():
    x = pd.DataFrame({"Age": [1.0, 2.0, 3.0, np.nan], "BMI": [20.0, 21.0, 22.0, 23.0]})
    selector = StudySelector(clinical=["Age", "BMI"])
    clone(selector).fit(x, [0, 1, 0, 1])
    selector.fit(x, [0, 1, 0, 1])
    medians, means = selector.medians_.copy(), selector.means_.copy()
    output = selector.transform(pd.DataFrame({"Age": [np.nan, 10000.0], "BMI": [30.0, 300.0]}))
    pd.testing.assert_series_equal(selector.medians_, medians)
    pd.testing.assert_series_equal(selector.means_, means)
    assert output[0, 0] == 0
    assert output[1, 0] > 1000


def test_no_selected_features_keeps_fold():
    x = pd.DataFrame({"constant": np.ones(20)})
    selected = StudySelector(quantitative=["constant"]).fit(x, [0, 1] * 10)
    assert selected.transform(x).shape == (20, 1)
    assert selected.get_feature_names_out().tolist() == ["__intercept_only__"]


def test_combined_selection_frequency_is_local(config):
    rng = np.random.default_rng(3)
    y = np.array([0, 1] * 30)
    x = pd.DataFrame({"q": y * 5 + rng.normal(size=60), "r": rng.normal(size=60)})
    settings = {**config["selection"], "integration_folds": 3}
    selector = StudySelector(["q"], ["r"], [], settings, lasso_c=10).fit(x, y)
    assert selector.integration_frequencies_["q"] == 1
    assert "q" in selector.selected_features_


def test_repeated_predictions_average_before_bootstrap():
    rows = [
        {
            "strategy": "s",
            "model": "m",
            "Patient_ID": str(i),
            "y": i % 2,
            "probability": 0.2 + 0.1 * repeat + 0.4 * (i % 2),
        }
        for repeat in range(2)
        for i in range(8)
    ]
    result = participant_predictions(pd.DataFrame(rows))
    assert len(result) == 8
    assert result.n_predictions.eq(2).all()
    assert result.probability.iloc[0] == pytest.approx(0.25)
    ci = bootstrap_metrics(result.y, result.probability, 30)
    assert ci["AUC"] == {"estimate": 1.0, "low": 1.0, "high": 1.0}


def test_decision_curve_formula():
    dca = decision_curve([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9], [0.5])
    assert dca.model.iloc[0] == 0.25
    assert dca["all"].iloc[0] == 0
    with pytest.raises(ValueError):
        decision_curve([0, 1], [0.1, 0.9], [1])


def test_texture_stability_and_excluded_shapes(config):
    rng = np.random.default_rng(4)
    y = np.array([0, 1] * 40)
    texture = "Muscle_Fat_original_glcm_Contrast"
    shape = "Muscle_Fat_original_shape_SurfaceVolumeRatio"
    X = pd.DataFrame({texture: y * 3 + rng.normal(size=80), shape: y * 3})
    selector = StudySelector(radiomics=[texture, shape], settings=config["selection"], lasso_c=10)
    selector.fit(X, y)
    assert texture in selector.selected_features_
    assert shape not in selector.selected_features_


def test_integration_requires_strictly_more_than_eighty_percent(config, monkeypatch):
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"q": rng.normal(size=50), "r": rng.normal(size=50)})
    calls = {"q": 0, "r": 0}

    def single_domain(self, frame, y, columns):
        name = columns[0]
        calls[name] += 1
        return [name] if name == "r" or calls[name] <= 4 else []

    monkeypatch.setattr(StudySelector, "_single_domain", single_domain)
    monkeypatch.setattr(StudySelector, "_lasso", lambda self, raw, y, columns: columns)
    selector = StudySelector(["q"], ["r"], [], config["selection"]).fit(X, [0, 1] * 25)
    assert selector.integration_frequencies_["q"] == 0.8
    assert selector.integration_frequencies_["r"] == 1
    assert selector.selected_features_ == ["r"]
