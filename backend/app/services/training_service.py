"""
The model registry and the code that trains one model.

Each model is wrapped in a single scikit-learn Pipeline:

    Pipeline([("preprocessor", ColumnTransformer(...)), ("model", Estimator(...))])

Keeping preprocessing and the estimator together means:
  * `fit` learns preprocessing from training data only (no leakage), and
  * the saved .joblib file can take *raw* feature columns and predict directly,
    because it carries its own fitted imputers, scaler and encoders.

The MVP uses one train/test split with sensible default hyperparameters — no
hyperparameter tuning or cross-validation. That keeps runs fast and easy to
explain, at the cost of somewhat noisier comparisons.
"""

import time

import pandas as pd

import pandas as pd
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.evaluation_service import evaluate_classification, evaluate_regression
from app.services.preprocessing_service import PreparedData, build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str  # stable key used in the API and database
    display_name: str
    task_type: str
    description: str
    # Linear models are sensitive to feature scale; tree models are not.
    needs_scaling: bool
    hyperparameters: dict[str, Any]
    factory: Callable[..., BaseEstimator]

    def create(self, random_seed: int, balance_classes: bool = False) -> BaseEstimator:
        params = dict(self.hyperparameters)
        if "random_state" in params:
            params["random_state"] = random_seed
        if balance_classes and self.task_type == "classification":
            params["class_weight"] = "balanced"
        return self.factory(**params)


# Default hyperparameters, chosen to be reasonable on typical tabular data:
#   * LogisticRegression max_iter=1000: the default 100 often stops before converging.
#   * DecisionTree max_depth=None, min_samples_leaf=1: scikit-learn defaults. Trees can
#     overfit; comparing them with Random Forest shows that effect nicely.
#   * RandomForest n_estimators=100: a common balance of accuracy and speed.
#     n_jobs=-1 uses all CPU cores.
#   * random_state is replaced by the experiment's seed so runs are reproducible.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(
            name="logistic_regression",
            display_name="Logistic Regression",
            task_type="classification",
            description="Linear model that predicts class probabilities. Fast, interpretable baseline.",
            needs_scaling=True,
            hyperparameters={"C": 1.0, "max_iter": 1000, "random_state": 42},
            factory=LogisticRegression,
        ),
        ModelSpec(
            name="decision_tree_classifier",
            display_name="Decision Tree Classifier",
            task_type="classification",
            description="Learns if/else rules. Captures non-linear patterns but can overfit.",
            needs_scaling=False,
            hyperparameters={"max_depth": None, "min_samples_leaf": 1, "random_state": 42},
            factory=DecisionTreeClassifier,
        ),
        ModelSpec(
            name="random_forest_classifier",
            display_name="Random Forest Classifier",
            task_type="classification",
            description="Averages many decision trees. Usually strong on tabular data.",
            needs_scaling=False,
            hyperparameters={"n_estimators": 100, "max_depth": None, "n_jobs": -1, "random_state": 42},
            factory=RandomForestClassifier,
        ),
        ModelSpec(
            name="linear_regression",
            display_name="Linear Regression",
            task_type="regression",
            description="Fits a straight-line relationship. Simple, interpretable baseline.",
            needs_scaling=True,
            hyperparameters={"fit_intercept": True},
            factory=LinearRegression,
        ),
        ModelSpec(
            name="decision_tree_regressor",
            display_name="Decision Tree Regressor",
            task_type="regression",
            description="Predicts with learned if/else rules. Non-linear but can overfit.",
            needs_scaling=False,
            hyperparameters={"max_depth": None, "min_samples_leaf": 1, "random_state": 42},
            factory=DecisionTreeRegressor,
        ),
        ModelSpec(
            name="random_forest_regressor",
            display_name="Random Forest Regressor",
            task_type="regression",
            description="Averages many regression trees. Robust, strong default.",
            needs_scaling=False,
            hyperparameters={"n_estimators": 100, "max_depth": None, "n_jobs": -1, "random_state": 42},
            factory=RandomForestRegressor,
        ),
    ]
}


def models_for_task(task_type: str) -> list[ModelSpec]:
    return [spec for spec in MODEL_REGISTRY.values() if spec.task_type == task_type]


def display_name_for(model_name: str) -> str:
    spec = MODEL_REGISTRY.get(model_name)
    return spec.display_name if spec else model_name


def build_pipeline(spec: ModelSpec, data: PreparedData, random_seed: int, balance_classes: bool = False) -> Pipeline:
    preprocessor = build_preprocessor(
        data.numeric_columns, data.categorical_columns, data.boolean_columns,
        datetime_columns=data.datetime_columns, scale_numeric=spec.needs_scaling
    )
    return Pipeline([("preprocessor", preprocessor), ("model", spec.create(random_seed, balance_classes))])


def train_and_evaluate(
    spec: ModelSpec, data: PreparedData, task_type: str, random_seed: int, balance_classes: bool = False
) -> dict[str, Any]:
    """
    Fit one pipeline on the training split and evaluate it on the test split.

    Returns a dict with the fitted pipeline, metrics, notes, chart data and
    timings. Exceptions propagate so the caller can record the model as failed.
    """
    pipeline = build_pipeline(spec, data, random_seed, balance_classes)

    # Record (rather than hide) warnings such as "lbfgs failed to converge"
    # so they can be shown to the user next to the model's results.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        pipeline.fit(data.X_train, data.y_train)
        # Validation score is used ONLY to choose the winner. The final test set
        # remains untouched until every model has been selected/refitted.
        validation_pred = pipeline.predict(data.X_validation)
        if task_type == "classification":
            validation_metrics, _, _ = evaluate_classification(
                pipeline, data.X_validation, data.y_validation, validation_pred, data.positive_class
            )
        else:
            validation_metrics, _, _ = evaluate_regression(data.y_validation, validation_pred, random_seed)
        selection_metric = "f1" if task_type == "classification" else "rmse"
        selection_metric_value = validation_metrics.get(selection_metric)

        # Refit on all non-test rows after model selection.
        X_final = pd.concat([data.X_train, data.X_validation], axis=0)
        y_final = pd.concat([data.y_train, data.y_validation], axis=0)
        pipeline.fit(X_final, y_final)
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = pipeline.predict(data.X_test)
        prediction_time = time.perf_counter() - start

    if task_type == "classification":
        metrics, notes, chart_data = evaluate_classification(
            pipeline, data.X_test, data.y_test, y_pred, data.positive_class
        )
    else:
        metrics, notes, chart_data = evaluate_regression(data.y_test, y_pred, random_seed)

    fit_warnings = sorted({str(w.message).splitlines()[0] for w in caught if issubclass(w.category, Warning)})
    if fit_warnings:
        notes["training_warnings"] = " | ".join(fit_warnings[:3])

    hyperparameters = dict(spec.hyperparameters)
    if "random_state" in hyperparameters:
        hyperparameters["random_state"] = random_seed
    if balance_classes and task_type == "classification":
        hyperparameters["class_weight"] = "balanced"

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "selection_metric_value": selection_metric_value,
        "selection_metric_value": selection_metric_value,
        "metric_notes": notes,
        "chart_data": chart_data,
        "hyperparameters": hyperparameters,
        "training_time_seconds": round(training_time, 6),
        "prediction_time_seconds": round(prediction_time, 6),
    }
