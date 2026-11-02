"""
Preparing data for training: target validation, feature selection, the
train/test split, and the scikit-learn preprocessing pipeline.

THE most important rule in this file — avoid data leakage:
    We split into train and test FIRST, and only then fit any transformation.
    The imputers, scaler and one-hot encoder live inside a scikit-learn
    Pipeline, so `pipeline.fit(X_train, y_train)` learns medians, means and
    categories from the training rows only. The test rows are only ever
    *transformed* with what was learned from training — exactly what will
    happen with genuinely new data after deployment.

If we imputed or scaled before splitting, statistics from the test set would
leak into training and the reported scores would be too optimistic.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.errors import DataValidationError
from app.services.dataset_service import infer_column_kind, is_likely_identifier

MAX_CLASSES = 50
MIN_TEST_ROWS = 2
VALIDATION_FRACTION = 0.2
VALIDATION_FRACTION = 0.2


@dataclass
class PreparedData:
    """Training, validation and untouched test partitions plus preprocessing metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    X_validation: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]
    positive_class: Any = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

def _is_integer_valued(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and bool((series.dropna() % 1 == 0).all())


def validate_classification_target(y: pd.Series) -> tuple[pd.Series, list[str]]:
    """
    Check that the target is a sensible set of class labels.

    Returns the (possibly tidied) target and a list of warnings. Raises
    DataValidationError when classification clearly does not make sense.
    """
    warnings: list[str] = []
    if pd.api.types.is_float_dtype(y) and not _is_integer_valued(y):
        raise DataValidationError(
            "The target column contains continuous decimal values. "
            "Classification needs categories — choose regression instead."
        )
    # A column like [0, 1, NaN] is read by pandas as float (0.0, 1.0). After
    # dropping missing targets we turn it back into integers for cleaner labels.
    if pd.api.types.is_float_dtype(y):
        y = y.astype("int64")

    class_counts = y.value_counts()
    n_classes = len(class_counts)
    if n_classes < 2:
        raise DataValidationError(
            "The target column has only one distinct value. A classifier needs at least two classes."
        )
    if n_classes > MAX_CLASSES or (n_classes > 20 and n_classes > 0.5 * len(y)):
        raise DataValidationError(
            f"The target column has {n_classes} distinct values, which is too many for classification. "
            "If it is a numeric quantity, choose regression."
        )
    rare = [str(label) for label, count in class_counts.items() if count < 2]
    if rare:
        warnings.append(
            f"Class(es) {', '.join(rare[:5])} have fewer than 2 examples. Metrics for these "
            "classes are unreliable and stratified splitting is not possible."
        )
    imbalance = class_counts.max() / class_counts.min()
    if imbalance >= 10:
        warnings.append(
            "The classes are highly imbalanced. Accuracy can look high even for a weak model; "
            "pay more attention to F1-score, precision and recall."
        )
    return y, warnings


def validate_regression_target(y: pd.Series) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        raise DataValidationError(
            "The target column contains non-numeric values. Regression needs a numeric target — "
            "choose classification for categories."
        )
    if not bool(pd.Series(y).apply(math.isfinite).all()):
        raise DataValidationError("The target column contains infinite values.")
    unique_count = y.nunique()
    if unique_count < 2:
        raise DataValidationError("The target column is constant, so there is nothing to predict.")
    if unique_count <= 10 and _is_integer_valued(y):
        warnings.append(
            f"The target has only {unique_count} distinct whole-number values. "
            "If these represent categories, classification may be more appropriate."
        )
    return y.astype("float64"), warnings


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_feature_columns(df: pd.DataFrame, target_column: str) -> tuple[list[str], dict[str, str], list[str]]:
    """
    Decide which columns become model features.

    Returns (feature_columns, excluded_columns_with_reason, warnings).

    We only *exclude* columns that are useless or obvious leaks:
      * completely empty columns (nothing to learn from),
      * exact copies of the target (the model would just read the answer).
    Likely identifiers are kept but produce a warning — the user knows their
    data better than a heuristic does.
    """
    target = df[target_column]
    features, excluded, warnings = [], {}, []
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        if series.isna().all():
            excluded[column] = "column is completely empty"
        elif series.equals(target):
            excluded[column] = "column is an exact copy of the target (data leakage)"
        else:
            features.append(column)
            if is_likely_identifier(str(column), series):
                warnings.append(
                    f"Feature '{column}' looks like an identifier and was kept. "
                    "If it is just a row ID, remove it from the CSV to avoid misleading results."
                )
    for column, reason in excluded.items():
        warnings.append(f"Column '{column}' was excluded from training: {reason}.")
    if not features:
        raise DataValidationError("No usable feature columns remain after removing the target.")
    return features, excluded, warnings


def is_datetime_column(series: pd.Series) -> bool:
    """Detect actual datetime dtypes and mostly parseable date-like text columns.

    Numeric columns are deliberately excluded: values such as 20240101 or 1700000000
    may be IDs/measurements, and guessing their unit would be unsafe.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().astype(str).head(100)
    if len(sample) < 3:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed", utc=True)
    return float(parsed.notna().mean()) >= 0.8


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert raw date columns into numeric calendar features at fit/predict time."""
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.feature_names_in_)
        outputs = []
        for column in frame.columns:
            values = pd.to_datetime(frame[column], errors="coerce", format="mixed", utc=True)
            parts = pd.DataFrame({
                f"{column}__year": values.dt.year,
                f"{column}__month": values.dt.month,
                f"{column}__day": values.dt.day,
                f"{column}__weekday": values.dt.dayofweek,
                f"{column}__hour": values.dt.hour,
                f"{column}__minute": values.dt.minute,
            }, index=frame.index)
            outputs.append(parts.astype(float))
        if not outputs:
            return np.empty((len(frame), 0), dtype=float)
        return pd.concat(outputs, axis=1).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{name}__{part}" for name in names for part in ("year", "month", "day", "weekday", "hour", "minute")], dtype=object)


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return numeric, categorical, boolean and datetime column lists."""
    numeric, categorical, boolean, datetime_columns = [], [], [], []
    for column in X.columns:
        series = X[column]
        if is_datetime_column(series):
            datetime_columns.append(column)
        elif pd.api.types.is_bool_dtype(series):
            boolean.append(column)
        elif infer_column_kind(series) == "numeric":
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical, boolean, datetime_columns


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame, y: pd.Series, task_type: str, test_size: float, random_seed: int,
    split_strategy: str = "random", time_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, bool, bool, list[str]]:
    """Create train/validation/test partitions without using test data for model selection."""
    warnings: list[str] = []
    n_rows = len(y)
    n_test = math.ceil(test_size * n_rows)
    n_train_pool = n_rows - n_test
    if n_test < MIN_TEST_ROWS or n_train_pool < 5:
        raise DataValidationError(
            f"With {n_rows} usable rows and a test size of {test_size:.0%}, there are too few rows "
            "to create training, validation and test sets. Add more rows or adjust the test size."
        )

    validation_fraction = min(VALIDATION_FRACTION, max(0.15, 2 / n_train_pool))
    n_validation = math.ceil(validation_fraction * n_train_pool)
    stratified = False
    validation_stratified = False

    if split_strategy == "chronological":
        if not time_column or time_column not in X.columns:
            raise DataValidationError("Chronological splitting requires a valid time_column feature.")
        ordered = X[time_column].sort_values(kind="stable")
        X = X.loc[ordered.index]
        y = y.loc[ordered.index]
        X_train_pool, X_test = X.iloc[:n_train_pool], X.iloc[n_train_pool:]
        y_train_pool, y_test = y.iloc[:n_train_pool], y.iloc[n_train_pool:]
        X_train = X_train_pool.iloc[:-n_validation]
        X_validation = X_train_pool.iloc[-n_validation:]
        y_train = y_train_pool.iloc[:-n_validation]
        y_validation = y_train_pool.iloc[-n_validation:]
        warnings.append("Chronological split: earlier rows train the model; later rows are reserved for validation and testing.")
    else:
        stratify_test = None
        if task_type == "classification":
            counts = y.value_counts()
            if counts.min() >= 2 and n_test >= len(counts) and n_train_pool >= len(counts):
                stratify_test = y
                stratified = True
            else:
                warnings.append("Stratified train/test splitting was not possible; a random split was used.")

        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratify_test
        )
        stratify_validation = None
        if task_type == "classification":
            counts = y_train_pool.value_counts()
            if counts.min() >= 2 and n_validation >= len(counts) and (len(y_train_pool)-n_validation) >= len(counts):
                stratify_validation = y_train_pool
                validation_stratified = True
            else:
                warnings.append("Stratified validation splitting was not possible; a random validation split was used.")

        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_pool, y_train_pool, test_size=validation_fraction, random_state=random_seed + 1,
            stratify=stratify_validation
        )
    if task_type == "classification" and y_train.nunique() < 2:
        raise DataValidationError("The model-fitting subset contains only one class. Add more data for the other classes.")
    if task_type == "regression" and y_test.nunique() < 2:
        warnings.append("All test targets are identical, so R² cannot be calculated for this split.")
    return (X_train, X_validation, X_test, y_train, y_validation, y_test,
            stratified, validation_stratified, warnings)


# ---------------------------------------------------------------------------
# The preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    boolean_columns: list[str],
    scale_numeric: bool = False,
    datetime_columns: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build the ColumnTransformer that turns raw columns into model-ready numbers.

    Numeric columns:
        1. Fill missing values with the MEDIAN (robust to outliers, unlike the mean).
        2. Optionally standardise to mean 0 / std 1. This matters for Logistic and
           Linear Regression (and their regularisation), but trees split on
           thresholds and do not care about scale, so we skip it for them.

    Categorical (text) columns:
        1. Fill missing values with the MOST FREQUENT category.
        2. One-hot encode. handle_unknown="ignore" means a category that never
           appeared in training (e.g. a new city) becomes all-zeros instead of
           crashing prediction.

    Boolean columns (True/False with no gaps) are one-hot encoded directly;
    SimpleImputer does not accept the bool dtype and they have no missing values.

    Any column not listed is dropped (remainder="drop"), so the target can
    never sneak in as a feature.
    """
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if boolean_columns:
        transformers.append(("boolean", OneHotEncoder(handle_unknown="ignore"), boolean_columns))
    if datetime_columns:
        date_pipeline = Pipeline([
            ("extract", DateTimeFeatureExtractor()),
            ("imputer", SimpleImputer(strategy="median")),
            *(([("scaler", StandardScaler())] if scale_numeric else [])),
        ])
        transformers.append(("datetime", date_pipeline, datetime_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def prepare_experiment_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float,
    random_seed: int,
    min_rows: int = 10,
    positive_class: str | None = None,
    exclude_columns: list[str] | None = None,
    split_strategy: str = "random",
    time_column: str | None = None,
) -> PreparedData:
    """Validate the target, choose features and split. Nothing is fitted here."""
    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' does not exist in this dataset.")

    warnings: list[str] = []
    missing_target = int(df[target_column].isna().sum())
    if missing_target:
        # Rows without a label cannot be used for supervised learning.
        df = df.loc[df[target_column].notna()]
        warnings.append(f"{missing_target} row(s) with a missing target value were dropped.")
    if len(df) < min_rows:
        raise DataValidationError(
            f"Only {len(df)} rows have a target value; at least {min_rows} are needed."
        )

    if task_type == "classification":
        y, target_warnings = validate_classification_target(df[target_column])
    elif task_type == "regression":
        y, target_warnings = validate_regression_target(df[target_column])
    else:
        raise DataValidationError(f"Unsupported task type '{task_type}'.")
    warnings.extend(target_warnings)

    resolved_positive_class = None
    if task_type == "classification" and y.nunique() == 2:
        labels = list(y.unique())
        if positive_class is not None:
            matches = [label for label in labels if str(label) == str(positive_class)]
            if not matches:
                available = ", ".join(str(label) for label in sorted(labels, key=str))
                raise DataValidationError(f"Positive class '{positive_class}' is not present in the target. Choose one of: {available}.")
            resolved_positive_class = matches[0]
        else:
            resolved_positive_class = sorted(labels, key=str)[-1]
    elif positive_class is not None:
        raise DataValidationError("Positive-class selection is only available for binary classification.")

    feature_columns, excluded, feature_warnings = select_feature_columns(df, target_column)
    warnings.extend(feature_warnings)
    requested_exclusions = set(exclude_columns or [])
    unknown_exclusions = requested_exclusions - set(df.columns)
    if unknown_exclusions:
        raise DataValidationError(f"Unknown feature column(s) to exclude: {', '.join(sorted(unknown_exclusions))}.")
    if target_column in requested_exclusions:
        raise DataValidationError("The target column cannot be excluded from the experiment.")
    for column in requested_exclusions:
        if column in feature_columns:
            feature_columns.remove(column)
            excluded[column] = "excluded by user"
            warnings.append(f"Feature '{column}' was excluded by your selection.")
    if not feature_columns:
        raise DataValidationError("No usable feature columns remain after applying your exclusions.")
    X = df[feature_columns].copy()
    if split_strategy not in {"random", "chronological"}:
        raise DataValidationError("split_strategy must be 'random' or 'chronological'.")
    if split_strategy == "chronological":
        if not time_column or time_column not in X.columns:
            raise DataValidationError("Choose a date/time feature as time_column for chronological splitting.")
        parsed_time = pd.to_datetime(X[time_column], errors="coerce", format="mixed", utc=True)
        if parsed_time.isna().any():
            raise DataValidationError(f"Time column '{time_column}' contains missing or invalid date values.")
        X.loc[:, time_column] = parsed_time
    elif time_column is not None:
        raise DataValidationError("time_column can only be provided when split_strategy is 'chronological'.")
    # Infinity is not a missing value to sklearn imputers. Normalize it before splitting
    # so all numeric pipelines can impute it using training-only statistics.
    numeric_feature_cols = X.select_dtypes(include=["number"]).columns
    if len(numeric_feature_cols):
        inf_mask = X[numeric_feature_cols].isin([np.inf, -np.inf])
        inf_count = int(inf_mask.sum().sum())
        if inf_count:
            X.loc[:, numeric_feature_cols] = X[numeric_feature_cols].replace([np.inf, -np.inf], np.nan)
            warnings.append(f"{inf_count} infinite numeric feature value(s) were treated as missing and will be imputed.")

    (X_train, X_validation, X_test, y_train, y_validation, y_test,
     stratified, validation_stratified, split_warnings) = split_data(
        X, y, task_type, test_size, random_seed, split_strategy, time_column
    )
    warnings.extend(split_warnings)
    numeric, categorical, boolean, datetime_columns = split_feature_types(X)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        X_validation=X_validation,
        numeric_columns=numeric,
        categorical_columns=categorical,
        boolean_columns=boolean,
        datetime_columns=datetime_columns,
        positive_class=resolved_positive_class,
        warnings=warnings,
        metadata={
            "rows_used": int(len(y)),
            "rows_dropped_missing_target": missing_target,
            "train_rows": int(len(y_train) + len(y_validation)),
            "model_fit_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
            "test_rows": int(len(y_test)),
            "stratified_split": stratified,
            "validation_stratified_split": validation_stratified,
            "split_strategy": split_strategy,
            "time_column": time_column,
            "feature_columns": feature_columns,
            "numeric_features": numeric,
            "categorical_features": categorical + boolean,
            "excluded_columns": excluded,
            "class_labels": [str(v) for v in sorted(y.unique(), key=str)] if task_type == "classification" else None,
            "datetime_features": datetime_columns,
            "positive_class": str(resolved_positive_class) if resolved_positive_class is not None else None,
        },
    )
