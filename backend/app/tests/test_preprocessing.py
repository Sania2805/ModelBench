"""
Preprocessing, splitting, training and evaluation — pure Python, no MySQL needed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.core.errors import DataValidationError
from app.services.evaluation_service import evaluate_regression, pick_best_model
from app.services.preprocessing_service import (
    build_preprocessor,
    prepare_experiment_data,
    select_feature_columns,
)
from app.services.training_service import MODEL_REGISTRY, train_and_evaluate
from app.tests.conftest import make_classification_df, make_regression_df


def test_missing_values_are_imputed():
    X = pd.DataFrame({"num": [1.0, np.nan, 3.0, 5.0], "cat": ["a", np.nan, "a", "b"]})
    pre = build_preprocessor(["num"], ["cat"], [], scale_numeric=False)
    out = pre.fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()
    assert out[1, 0] == 3.0  # median of 1, 3, 5
    # The missing category becomes the most frequent value "a" -> one-hot [1, 0].
    assert list(out[1, 1:]) == [1.0, 0.0]


def test_numeric_and_categorical_columns_are_handled():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["x", "y", "z", "x"], "flag": [True, False, True, True]})
    pre = build_preprocessor(["num"], ["cat"], ["flag"], scale_numeric=True)
    out = pre.fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape == (4, 1 + 3 + 2)  # scaled num + 3 categories + 2 bool values
    assert abs(out[:, 0].mean()) < 1e-9  # standardised to mean 0


def test_unknown_category_at_prediction_does_not_crash():
    X = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    pre = build_preprocessor([], ["cat"], [], scale_numeric=False).fit(X)
    out = pre.transform(pd.DataFrame({"cat": ["never-seen"]}))
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.sum() == 0  # all zeros instead of an error


def test_final_preprocessing_is_refitted_on_all_non_test_rows_only():
    df = make_regression_df(100)
    data = prepare_experiment_data(df, "price", "regression", 0.2, 1)
    result = train_and_evaluate(MODEL_REGISTRY["linear_regression"], data, "regression", 1)
    imputer = result["pipeline"].named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    train_median = pd.concat([data.X_train, data.X_validation])[data.numeric_columns].median().to_numpy()
    assert np.allclose(imputer.statistics_, train_median)


def test_target_is_excluded_and_copies_of_target_are_dropped():
    df = make_classification_df()
    df["label_copy"] = df["label"]
    features, excluded, warnings = select_feature_columns(df, "label")
    assert "label" not in features
    assert "label_copy" not in features
    assert "label_copy" in excluded
    assert any("label_copy" in w for w in warnings)


def test_identifier_is_kept_with_warning():
    df = make_classification_df()
    df["row_id"] = range(len(df))
    features, _, warnings = select_feature_columns(df, "label")
    assert "row_id" in features
    assert any("row_id" in w for w in warnings)


def test_classification_split_is_stratified():
    data = prepare_experiment_data(make_classification_df(100), "label", "classification", 0.25, 3)
    assert data.metadata["stratified_split"] is True
    assert data.metadata["train_rows"] == 75 and data.metadata["test_rows"] == 25
    assert set(data.y_test.unique()) == set(data.y_train.unique())


def test_rare_class_falls_back_to_random_split():
    df = make_classification_df(40)
    df.loc[0, "label"] = "rare"
    data = prepare_experiment_data(df, "label", "classification", 0.2, 0)
    assert data.metadata["stratified_split"] is False
    assert any("Stratified" in w for w in data.warnings)


def test_rows_with_missing_target_are_dropped():
    df = make_regression_df(50)
    df.loc[[0, 1], "price"] = np.nan
    data = prepare_experiment_data(df, "price", "regression", 0.2, 0)
    assert data.metadata["rows_dropped_missing_target"] == 2
    assert data.metadata["rows_used"] == 48


@pytest.mark.parametrize(
    "target,task,message",
    [
        ("missing_col", "classification", "does not exist"),
        ("price", "classification", "continuous"),  # decimal target for classification
        ("city", "regression", "non-numeric"),  # text target for regression
    ],
)
def test_unsuitable_targets_are_rejected(target, task, message):
    with pytest.raises(DataValidationError, match=message):
        prepare_experiment_data(make_regression_df(), target, task, 0.2, 0)


def test_constant_target_is_rejected():
    df = make_regression_df(30)
    df["price"] = 5.0
    with pytest.raises(DataValidationError, match="constant"):
        prepare_experiment_data(df, "price", "regression", 0.2, 0)


def test_too_small_dataset_is_rejected():
    df = make_regression_df(8)
    with pytest.raises(DataValidationError, match="at least 10"):
        prepare_experiment_data(df, "price", "regression", 0.2, 0)


def test_classification_training_produces_valid_metrics():
    data = prepare_experiment_data(make_classification_df(120), "label", "classification", 0.25, 0)
    for name in ("logistic_regression", "decision_tree_classifier", "random_forest_classifier"):
        result = train_and_evaluate(MODEL_REGISTRY[name], data, "classification", 0)
        metrics = result["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0
        matrix = result["chart_data"]["confusion_matrix"]
        assert matrix["labels"] == ["no", "yes"]
        assert sum(map(sum, matrix["matrix"])) == len(data.y_test)
        assert result["training_time_seconds"] >= 0
        assert isinstance(result["pipeline"], Pipeline)


def test_multiclass_roc_auc_is_na_when_class_missing_from_test():
    df = make_classification_df(60)
    df.loc[:2, "label"] = "maybe"  # 3 rows of a third class
    data = prepare_experiment_data(df, "label", "classification", 0.2, 0)
    # Force the rare class out of the test set to simulate the edge case.
    keep = data.y_test != "maybe"
    data.X_test, data.y_test = data.X_test[keep], data.y_test[keep]
    if "maybe" not in set(data.y_train):
        pytest.skip("random split put no rare rows in training")
    result = train_and_evaluate(MODEL_REGISTRY["logistic_regression"], data, "classification", 0)
    assert result["metrics"]["roc_auc"] is None
    assert "missing" in result["metric_notes"]["roc_auc"]


def test_regression_training_produces_valid_metrics():
    data = prepare_experiment_data(make_regression_df(120), "price", "regression", 0.2, 0)
    result = train_and_evaluate(MODEL_REGISTRY["linear_regression"], data, "regression", 0)
    m = result["metrics"]
    assert m["mae"] > 0 and m["mse"] > 0
    assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]))
    assert m["r2"] > 0.8  # the synthetic data is close to linear
    assert len(result["chart_data"]["actual_vs_predicted"]) == len(data.y_test)


def test_r2_is_na_for_constant_test_targets():
    metrics, notes, _ = evaluate_regression(pd.Series([3.0, 3.0, 3.0]), np.array([2.0, 3.0, 4.0]), 0)
    assert metrics["r2"] is None
    assert "r2" in notes


def test_pick_best_model_ignores_failed_and_missing_metrics():
    results = [
        {"model_name": "a", "status": "success", "metrics": {"rmse": 5.0}},
        {"model_name": "b", "status": "success", "metrics": {"rmse": 3.0}},
        {"model_name": "c", "status": "failed", "metrics": None},
        {"model_name": "d", "status": "success", "metrics": {"rmse": None}},
    ]
    assert pick_best_model(results, "regression") == "b"
    assert pick_best_model([results[2]], "regression") is None


def test_infinite_numeric_features_are_treated_as_missing():
    df = make_regression_df(100)
    df.loc[0, "size"] = np.inf
    df["rooms"] = df["rooms"].astype(float)
    df.loc[1, "rooms"] = -np.inf
    data = prepare_experiment_data(df, "price", "regression", 0.2, 2)
    assert any("infinite numeric feature" in warning for warning in data.warnings)
    result = train_and_evaluate(MODEL_REGISTRY["linear_regression"], data, "regression", 2)
    assert np.isfinite(result["metrics"]["rmse"])


def test_date_columns_are_engineered_and_raw_pipeline_predicts():
    df = make_regression_df(100)
    df["signup_date"] = pd.date_range("2024-01-01", periods=len(df), freq="D").astype(str)
    data = prepare_experiment_data(df, "price", "regression", 0.2, 3)
    assert "signup_date" in data.datetime_columns
    result = train_and_evaluate(MODEL_REGISTRY["linear_regression"], data, "regression", 3)
    raw_features = df.drop(columns=["price"]).iloc[:3]
    predictions = result["pipeline"].predict(raw_features)
    assert len(predictions) == 3 and np.isfinite(predictions).all()


def test_user_can_choose_binary_positive_class():
    df = make_classification_df(120)
    data = prepare_experiment_data(df, "label", "classification", 0.2, 4, positive_class="no")
    result = train_and_evaluate(MODEL_REGISTRY["logistic_regression"], data, "classification", 4)
    assert "positive class 'no'" in result["metric_notes"]["averaging"]
    with pytest.raises(DataValidationError, match="not present"):
        prepare_experiment_data(df, "label", "classification", 0.2, 4, positive_class="unknown")


def test_validation_score_is_used_for_model_selection_not_test_score():
    results = [
        {"model_name": "test_winner", "status": "success", "metrics": {"f1": 0.99}, "selection_metric_value": 0.4},
        {"model_name": "validation_winner", "status": "success", "metrics": {"f1": 0.6}, "selection_metric_value": 0.8},
    ]
    assert pick_best_model(results, "classification") == "validation_winner"


def test_user_can_exclude_identifier_feature():
    df = make_regression_df(80)
    df["customer_id"] = range(len(df))
    data = prepare_experiment_data(df, "price", "regression", 0.2, 8, exclude_columns=["customer_id"])
    assert "customer_id" not in data.metadata["feature_columns"]
    assert data.metadata["excluded_columns"]["customer_id"] == "excluded by user"



def test_chronological_split_keeps_later_dates_out_of_training():
    df = make_regression_df(100)
    df["event_date"] = pd.date_range("2024-01-01", periods=len(df), freq="D").astype(str)
    data = prepare_experiment_data(
        df, "price", "regression", 0.2, 12, split_strategy="chronological", time_column="event_date"
    )
    train_max = pd.to_datetime(pd.concat([data.X_train, data.X_validation])["event_date"]).max()
    test_min = pd.to_datetime(data.X_test["event_date"]).min()
    validation_min = pd.to_datetime(data.X_validation["event_date"]).min()
    fit_max = pd.to_datetime(data.X_train["event_date"]).max()
    assert fit_max < validation_min < test_min
    assert train_max < test_min


def test_class_balancing_option_is_applied_to_classifier():
    data = prepare_experiment_data(make_classification_df(100), "label", "classification", 0.2, 13)
    result = train_and_evaluate(
        MODEL_REGISTRY["logistic_regression"], data, "classification", 13, balance_classes=True
    )
    assert result["pipeline"].named_steps["model"].class_weight == "balanced"
    assert result["hyperparameters"]["class_weight"] == "balanced"
