"""Model registry, user-configurable hyperparameters, and model training."""

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.services.evaluation_service import evaluate_classification, evaluate_regression
from app.services.preprocessing_service import PreparedData, build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    task_type: str
    description: str
    needs_scaling: bool
    hyperparameters: dict[str, Any]
    factory: Callable[..., BaseEstimator]
    # UI metadata for safe, bounded user-configurable parameters.
    parameter_schema: dict[str, dict[str, Any]]
    supports_class_weight: bool = False
    requires_dense: bool = False

    def resolve_hyperparameters(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(self.hyperparameters)
        overrides = overrides or {}
        unknown = set(overrides) - set(self.parameter_schema)
        if unknown:
            raise ValueError(f"Unsupported hyperparameter(s) for {self.display_name}: {', '.join(sorted(unknown))}.")
        for key, value in overrides.items():
            definition = self.parameter_schema[key]
            kind = definition["type"]
            if kind == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{key} must be a whole number.")
                if value < definition["min"] or value > definition["max"]:
                    raise ValueError(f"{key} must be between {definition['min']} and {definition['max']}.")
            elif kind == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{key} must be a number.")
                if value < definition["min"] or value > definition["max"]:
                    raise ValueError(f"{key} must be between {definition['min']} and {definition['max']}.")
                value = float(value)
            elif kind == "boolean":
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be true or false.")
            elif kind == "select":
                if value not in definition["options"]:
                    raise ValueError(f"Invalid value for {key}.")
            params[key] = value
        return params

    def create(
        self,
        random_seed: int,
        balance_classes: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> BaseEstimator:
        params = self.resolve_hyperparameters(overrides)
        if "random_state" in params:
            params["random_state"] = random_seed
        if balance_classes and self.task_type == "classification" and self.supports_class_weight:
            params["class_weight"] = "balanced"
        return self.factory(**params)


def _int(default: int, minimum: int, maximum: int, step: int = 1, label: str | None = None):
    return {"type": "integer", "default": default, "min": minimum, "max": maximum, "step": step, "label": label}


def _num(default: float, minimum: float, maximum: float, step: float = 0.1, label: str | None = None):
    return {"type": "number", "default": default, "min": minimum, "max": maximum, "step": step, "label": label}


def _select(default: Any, options: list[Any], label: str | None = None):
    return {"type": "select", "default": default, "options": options, "label": label}


def _bool(default: bool, label: str | None = None):
    return {"type": "boolean", "default": default, "label": label}


def _spec(name, display_name, task_type, description, factory, defaults, schema, *, scaling=False, class_weight=False, dense=False):
    return ModelSpec(name, display_name, task_type, description, scaling, defaults, factory, schema,
                     supports_class_weight=class_weight, requires_dense=dense)


# Parameters exposed here are deliberately limited to useful, bounded controls.
# Defaults remain pre-filled, so users can simply run an experiment unchanged.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    s.name: s for s in [
        _spec("logistic_regression", "Logistic Regression", "classification",
              "Linear probabilistic classifier; a fast, interpretable baseline.", LogisticRegression,
              {"C": 1.0, "max_iter": 1000, "random_state": 42},
              {"C": _num(1.0, 0.01, 10.0, 0.1, "Regularization strength (C)"), "max_iter": _int(1000, 100, 5000, 100, "Maximum iterations")}, scaling=True, class_weight=True),
        _spec("decision_tree_classifier", "Decision Tree Classifier", "classification",
              "Learns if/else rules; captures non-linear patterns but can overfit.", DecisionTreeClassifier,
              {"max_depth": None, "min_samples_leaf": 1, "random_state": 42},
              {"max_depth": _select(None, [None, 3, 5, 10, 20], "Maximum tree depth"), "min_samples_leaf": _int(1, 1, 20, 1, "Minimum samples per leaf")}, class_weight=True),
        _spec("random_forest_classifier", "Random Forest Classifier", "classification",
              "Ensemble of decision trees; a strong general-purpose tabular baseline.", RandomForestClassifier,
              {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1, "random_state": 42},
              {"n_estimators": _int(100, 50, 500, 50, "Number of trees"), "max_depth": _select(None, [None, 5, 10, 20], "Maximum tree depth"), "min_samples_leaf": _int(1, 1, 20, 1, "Minimum samples per leaf")}, class_weight=True),
        _spec("knn_classifier", "K-Nearest Neighbors Classifier", "classification",
              "Classifies using nearby training examples; sensitive to feature scaling.", KNeighborsClassifier,
              {"n_neighbors": 5, "weights": "uniform", "p": 2},
              {"n_neighbors": _int(5, 1, 30, 1, "Number of neighbors"), "weights": _select("uniform", ["uniform", "distance"], "Neighbor weighting"), "p": _select(2, [1, 2], "Distance metric (1 = Manhattan, 2 = Euclidean)")}, scaling=True),
        _spec("svc", "Support Vector Classifier (SVC)", "classification",
              "Finds separating boundaries; supports linear and non-linear kernels.", SVC,
              {"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42},
              {"C": _num(1.0, 0.01, 10.0, 0.1, "Regularization strength (C)"), "kernel": _select("rbf", ["rbf", "linear", "poly", "sigmoid"], "Kernel"), "gamma": _select("scale", ["scale", "auto"], "Kernel coefficient")}, scaling=True, class_weight=True),
        _spec("gaussian_nb", "Gaussian Naive Bayes", "classification",
              "Probabilistic baseline based on Gaussian feature likelihoods.", GaussianNB,
              {"var_smoothing": 1e-9},
              {"var_smoothing": _select(1e-9, [1e-12, 1e-10, 1e-9, 1e-8, 1e-6], "Variance smoothing")}, dense=True),
        _spec("gradient_boosting_classifier", "Gradient Boosting Classifier", "classification",
              "Builds trees sequentially to correct earlier prediction errors.", GradientBoostingClassifier,
              {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
              {"n_estimators": _int(100, 50, 400, 50, "Number of boosting stages"), "learning_rate": _num(0.1, 0.01, 1.0, 0.05, "Learning rate"), "max_depth": _int(3, 1, 8, 1, "Tree depth")}),
        _spec("adaboost_classifier", "AdaBoost Classifier", "classification",
              "Combines weak learners, focusing successive learners on harder examples.", AdaBoostClassifier,
              {"n_estimators": 100, "learning_rate": 0.5, "random_state": 42},
              {"n_estimators": _int(100, 50, 400, 50, "Number of estimators"), "learning_rate": _num(0.5, 0.01, 2.0, 0.05, "Learning rate")}),
        _spec("extra_trees_classifier", "Extra Trees Classifier", "classification",
              "Randomized tree ensemble that can be fast and effective on tabular data.", ExtraTreesClassifier,
              {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1, "random_state": 42},
              {"n_estimators": _int(100, 50, 500, 50, "Number of trees"), "max_depth": _select(None, [None, 5, 10, 20], "Maximum tree depth"), "min_samples_leaf": _int(1, 1, 20, 1, "Minimum samples per leaf")}, class_weight=True),
        _spec("linear_regression", "Linear Regression", "regression",
              "Fits a linear relationship; a simple, interpretable baseline.", LinearRegression,
              {"fit_intercept": True}, {"fit_intercept": _bool(True, "Fit intercept")}, scaling=True),
        _spec("decision_tree_regressor", "Decision Tree Regressor", "regression",
              "Predicts using learned decision rules; can overfit without constraints.", DecisionTreeRegressor,
              {"max_depth": None, "min_samples_leaf": 1, "random_state": 42},
              {"max_depth": _select(None, [None, 3, 5, 10, 20], "Maximum tree depth"), "min_samples_leaf": _int(1, 1, 20, 1, "Minimum samples per leaf")}),
        _spec("random_forest_regressor", "Random Forest Regressor", "regression",
              "Averages many regression trees; robust for many tabular problems.", RandomForestRegressor,
              {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1, "random_state": 42},
              {"n_estimators": _int(100, 50, 500, 50, "Number of trees"), "max_depth": _select(None, [None, 5, 10, 20], "Maximum tree depth"), "min_samples_leaf": _int(1, 1, 20, 1, "Minimum samples per leaf")}),
        _spec("ridge_regression", "Ridge Regression", "regression",
              "Linear regression with L2 regularization to reduce coefficient magnitude.", Ridge,
              {"alpha": 1.0, "fit_intercept": True},
              {"alpha": _num(1.0, 0.001, 100.0, 0.1, "Regularization strength (alpha)"), "fit_intercept": _bool(True, "Fit intercept")}, scaling=True),
        _spec("lasso_regression", "Lasso Regression", "regression",
              "Linear regression with L1 regularization; can shrink coefficients to zero.", Lasso,
              {"alpha": 1.0, "max_iter": 5000},
              {"alpha": _num(1.0, 0.001, 100.0, 0.1, "Regularization strength (alpha)"), "max_iter": _int(5000, 500, 10000, 500, "Maximum iterations")}, scaling=True),
        _spec("elasticnet_regression", "ElasticNet Regression", "regression",
              "Combines L1 and L2 regularization.", ElasticNet,
              {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 5000},
              {"alpha": _num(1.0, 0.001, 100.0, 0.1, "Regularization strength (alpha)"), "l1_ratio": _num(0.5, 0.0, 1.0, 0.05, "L1 ratio"), "max_iter": _int(5000, 500, 10000, 500, "Maximum iterations")}, scaling=True),
        _spec("knn_regressor", "K-Nearest Neighbors Regressor", "regression",
              "Predicts from nearby examples; sensitive to feature scaling.", KNeighborsRegressor,
              {"n_neighbors": 5, "weights": "uniform", "p": 2},
              {"n_neighbors": _int(5, 1, 30, 1, "Number of neighbors"), "weights": _select("uniform", ["uniform", "distance"], "Neighbor weighting"), "p": _select(2, [1, 2], "Distance metric (1 = Manhattan, 2 = Euclidean)")}, scaling=True),
        _spec("svr", "Support Vector Regressor (SVR)", "regression",
              "Support-vector regression with configurable kernel and error tolerance.", SVR,
              {"C": 1.0, "kernel": "rbf", "epsilon": 0.1, "gamma": "scale"},
              {"C": _num(1.0, 0.01, 10.0, 0.1, "Regularization strength (C)"), "kernel": _select("rbf", ["rbf", "linear", "poly", "sigmoid"], "Kernel"), "epsilon": _num(0.1, 0.01, 1.0, 0.01, "Epsilon-insensitive margin"), "gamma": _select("scale", ["scale", "auto"], "Kernel coefficient")}, scaling=True),
        _spec("gradient_boosting_regressor", "Gradient Boosting Regressor", "regression",
              "Builds regression trees sequentially to reduce prediction errors.", GradientBoostingRegressor,
              {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
              {"n_estimators": _int(100, 50, 400, 50, "Number of boosting stages"), "learning_rate": _num(0.1, 0.01, 1.0, 0.05, "Learning rate"), "max_depth": _int(3, 1, 8, 1, "Tree depth")}),
        _spec("adaboost_regressor", "AdaBoost Regressor", "regression",
              "Combines weak regressors, emphasizing examples with larger errors.", AdaBoostRegressor,
              {"n_estimators": 100, "learning_rate": 0.5, "random_state": 42},
              {"n_estimators": _int(100, 50, 400, 50, "Number of estimators"), "learning_rate": _num(0.5, 0.01, 2.0, 0.05, "Learning rate")}),
        _spec("extra_trees_regressor", "Extra Trees Regressor", "regression",
              "Randomized tree ensemble for non-linear regression.", ExtraTreesRegressor,
              {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1, "random_state": 42},
              {"n_estimators": _int(100, 50, 500, 50, "Number of trees"), "max_depth": _select(None, [None, 5, 10, 20], "Maximum tree depth"), "min_samples_leaf": _int(1, 1, 20, 1, "Minimum samples per leaf")}),
    ]
}


def models_for_task(task_type: str) -> list[ModelSpec]:
    return [spec for spec in MODEL_REGISTRY.values() if spec.task_type == task_type]


def display_name_for(model_name: str) -> str:
    spec = MODEL_REGISTRY.get(model_name)
    return spec.display_name if spec else model_name


def _ensure_dense(matrix):
    """Convert scipy sparse preprocessing output for estimators requiring dense arrays."""
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def build_pipeline(
    spec: ModelSpec,
    data: PreparedData,
    random_seed: int,
    balance_classes: bool = False,
    hyperparameters: dict[str, Any] | None = None,
) -> Pipeline:
    preprocessor = build_preprocessor(
        data.numeric_columns, data.categorical_columns, data.boolean_columns,
        datetime_columns=data.datetime_columns, scale_numeric=spec.needs_scaling,
    )
    estimator = spec.create(random_seed, balance_classes, hyperparameters)
    if spec.requires_dense:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer
        # GaussianNB expects dense matrices; use a top-level function so joblib can pickle it.
        estimator = Pipeline([("to_dense", FunctionTransformer(_ensure_dense)), ("estimator", estimator)])
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train_and_evaluate(
    spec: ModelSpec,
    data: PreparedData,
    task_type: str,
    random_seed: int,
    balance_classes: bool = False,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline = build_pipeline(spec, data, random_seed, balance_classes, hyperparameters)
    resolved_hyperparameters = spec.resolve_hyperparameters(hyperparameters)
    if "random_state" in resolved_hyperparameters:
        resolved_hyperparameters["random_state"] = random_seed
    if balance_classes and task_type == "classification" and spec.supports_class_weight:
        resolved_hyperparameters["class_weight"] = "balanced"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        pipeline.fit(data.X_train, data.y_train)
        validation_pred = pipeline.predict(data.X_validation)
        if task_type == "classification":
            validation_metrics, _, _ = evaluate_classification(
                pipeline, data.X_validation, data.y_validation, validation_pred, data.positive_class
            )
        else:
            validation_metrics, _, _ = evaluate_regression(data.y_validation, validation_pred, random_seed)
        selection_metric = "f1" if task_type == "classification" else "rmse"
        selection_metric_value = validation_metrics.get(selection_metric)

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

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "selection_metric_value": selection_metric_value,
        "metric_notes": notes,
        "chart_data": chart_data,
        "hyperparameters": resolved_hyperparameters,
        "training_time_seconds": round(training_time, 6),
        "prediction_time_seconds": round(prediction_time, 6),
    }
