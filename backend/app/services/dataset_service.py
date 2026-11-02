"""
Dataset upload, validation and profiling.

Flow for an upload:
    1. Check the extension (.csv only).
    2. Read the bytes in chunks and stop as soon as the size limit is exceeded,
       so a huge file is never loaded fully into memory.
    3. Parse with pandas to prove the content really is a usable CSV.
    4. Save the ORIGINAL bytes unchanged under a UUID filename in uploads/.
    5. Store metadata (not the file contents) in MySQL.

Profiling is split into pure functions (no database, no disk) so they are easy
to unit test.
"""

import io
import json
import logging
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import DataValidationError, NotFoundError, PayloadTooLargeError
from app.core.security import (
    has_allowed_extension,
    new_id,
    resolve_safe_path,
    sanitize_display_filename,
)
from app.database.models import Dataset, DatasetInsight, utc_now

logger = logging.getLogger(__name__)

PREVIEW_ROW_COUNT = 10
READ_CHUNK_BYTES = 1024 * 1024
# UTF-8 first (with or without a BOM, as Excel adds one). latin-1 is a fallback
# that can decode any byte sequence, which covers CSVs exported by older tools.
CANDIDATE_ENCODINGS = ("utf-8-sig", "latin-1")
IDENTIFIER_NAMES = {"id", "index", "uuid", "guid", "key", "row", "row_id", "rownum", "row_number"}


# ---------------------------------------------------------------------------
# Reading and validating raw CSV bytes
# ---------------------------------------------------------------------------

def read_limited(file: BinaryIO, max_bytes: int) -> bytes:
    """Read an uploaded file, raising PayloadTooLargeError once it exceeds max_bytes."""
    buffer = bytearray()
    while chunk := file.read(READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            limit_mb = max_bytes / (1024 * 1024)
            raise PayloadTooLargeError(f"The file is larger than the {limit_mb:g} MB upload limit.")
    return bytes(buffer)


def parse_csv_bytes(content: bytes, settings: Settings) -> tuple[pd.DataFrame, str]:
    """
    Parse CSV bytes into a DataFrame and return it with the encoding that worked.

    Raises DataValidationError with a human-readable reason for anything that is
    not a usable table: empty files, binary files, broken quoting, duplicate
    column names, too few rows, and so on.
    """
    if not content.strip():
        raise DataValidationError("The file is empty.")
    # Text files never contain NUL bytes; Excel .xlsx, images, etc. do.
    if b"\x00" in content[:8192]:
        raise DataValidationError(
            "The file does not look like a text CSV. If it is an Excel file, "
            "save it as 'CSV (Comma delimited)' first."
        )

    last_error: Exception | None = None
    for encoding in CANDIDATE_ENCODINGS:
        try:
            # Read the header row separately: pandas silently renames duplicate
            # headers ("price", "price.1"), which would hide a real data problem.
            header = pd.read_csv(io.BytesIO(content), encoding=encoding, header=None, nrows=1)
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError as error:
            last_error = error
            continue
        except pd.errors.EmptyDataError:
            raise DataValidationError("The file has no columns or rows.") from None
        except pd.errors.ParserError as error:
            # pandas messages here are short and safe, e.g. "Expected 3 fields in line 7, saw 5".
            raise DataValidationError(f"The CSV could not be parsed: {error}".strip()) from None
    else:
        raise DataValidationError("The file encoding is not supported.") from last_error

    header_names = [str(name) for name in header.iloc[0].tolist()]
    duplicates = sorted({name for name in header_names if header_names.count(name) > 1})
    if duplicates:
        raise DataValidationError(f"Column names must be unique. Duplicated: {', '.join(duplicates)}.")

    validate_dataframe_shape(df, settings)
    return df, encoding


def validate_dataframe_shape(df: pd.DataFrame, settings: Settings) -> None:
    """Reject tables that are too small, too wide, or completely empty."""
    if df.shape[1] < 2:
        raise DataValidationError(
            "The dataset needs at least 2 columns: one or more features and a target. "
            "If your file uses ';' or tabs as separators, re-export it with commas."
        )
    if df.shape[1] > settings.max_columns:
        raise DataValidationError(f"The dataset has {df.shape[1]} columns; the limit is {settings.max_columns}.")
    if len(df) < settings.min_dataset_rows:
        raise DataValidationError(
            f"The dataset has only {len(df)} data rows. At least {settings.min_dataset_rows} "
            "rows are needed to split into training and test sets."
        )
    if df.dropna(axis=1, how="all").shape[1] == 0:
        raise DataValidationError("Every column in the dataset is empty.")


# ---------------------------------------------------------------------------
# Profiling (pure functions)
# ---------------------------------------------------------------------------

def infer_column_kind(series: pd.Series) -> str:
    """
    Classify a column as "numeric" or "categorical" for preprocessing.

    Booleans are treated as categorical: True/False are labels, not quantities.
    Text, dates and mixed columns are categorical too (one-hot encoded later).
    """
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def is_likely_identifier(name: str, series: pd.Series) -> bool:
    """
    Heuristic: does this column look like a row ID rather than a real feature?

    IDs are a classic source of leakage/overfitting: a model can memorise them
    but they carry no meaning for new data. We only WARN — the user decides.
    """
    row_count = len(series)
    if row_count == 0:
        return False
    unique_ratio = series.nunique(dropna=True) / row_count
    lower = name.strip().lower()
    id_like_name = (
        lower in IDENTIFIER_NAMES
        or lower.endswith("_id")
        or lower.endswith(" id")
        or lower.startswith("unnamed:")  # pandas index exported to CSV
    )
    if id_like_name and unique_ratio >= 0.9:
        return True
    # Every value unique and not a float measurement -> probably an ID or free text.
    all_unique = unique_ratio == 1.0 and series.notna().all() and row_count >= 20
    return bool(all_unique and (pd.api.types.is_integer_dtype(series) or infer_column_kind(series) == "categorical"))


def build_column_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    row_count = len(df)
    columns = []
    for name in df.columns:
        series = df[name]
        missing = int(series.isna().sum())
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "kind": infer_column_kind(series),
                "missing_count": missing,
                "missing_percent": round(100 * missing / row_count, 2) if row_count else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "likely_identifier": is_likely_identifier(str(name), series),
            }
        )
    return columns


def _json_safe_number(value: Any) -> float | None:
    """Convert numpy numbers to plain floats; NaN/inf become None (JSON has no NaN)."""
    if value is None:
        return None
    number = float(value)
    return round(number, 6) if np.isfinite(number) else None


def build_profile_data(df: pd.DataFrame) -> dict[str, Any]:
    """Compute everything shown on the dataset preview page."""
    columns = build_column_info(df)
    numeric_cols = [c["name"] for c in columns if c["kind"] == "numeric"]
    categorical_cols = [c["name"] for c in columns if c["kind"] == "categorical"]

    numeric_statistics: dict[str, dict[str, float | None]] = {}
    if numeric_cols:
        described = df[numeric_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
        for col, row in described.iterrows():
            numeric_statistics[str(col)] = {stat: _json_safe_number(v) for stat, v in row.items()}

    categorical_statistics: dict[str, dict[str, Any]] = {}
    for col in categorical_cols:
        counts = df[col].astype("object").value_counts(dropna=True)
        categorical_statistics[col] = {
            "unique": int(df[col].nunique(dropna=True)),
            "top": str(counts.index[0]) if not counts.empty else None,
            "top_frequency": int(counts.iloc[0]) if not counts.empty else 0,
        }

    # to_json handles NaN -> null and numpy types -> plain JSON values for us.
    preview = json.loads(df.head(PREVIEW_ROW_COUNT).to_json(orient="records", date_format="iso"))

    warnings = []
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate row(s) found. Duplicates can end up in both the "
            "training and test sets and make scores look better than they really are."
        )
    for col in columns:
        if col["likely_identifier"]:
            warnings.append(
                f"Column '{col['name']}' looks like an identifier. IDs rarely help a model and "
                "can cause overfitting — consider removing it from the CSV before training."
            )
        if col["missing_count"] == len(df):
            warnings.append(f"Column '{col['name']}' is completely empty and will be ignored during training.")
        elif col["kind"] == "categorical" and col["unique_count"] > 50 and col["unique_count"] > 0.5 * len(df):
            warnings.append(
                f"Column '{col['name']}' has {col['unique_count']} distinct text values. "
                "One-hot encoding it creates many features and may slow training."
            )

    missing_by_column = {
        col["name"]: {
            "missing_count": col["missing_count"],
            "missing_percent": col["missing_percent"],
        }
        for col in columns
    }

    # Keep the matrix useful and bounded for wide datasets. Non-finite values
    # (e.g. constant columns) are converted to null for MySQL JSON compatibility.
    correlation_columns = numeric_cols[:40]
    correlation_matrix: dict[str, dict[str, float | None]] = {}
    if len(correlation_columns) >= 2:
        corr = df[correlation_columns].corr(numeric_only=True)
        for row_name in corr.index:
            correlation_matrix[str(row_name)] = {
                str(col_name): _json_safe_number(corr.loc[row_name, col_name])
                for col_name in corr.columns
            }

    recommendations = []
    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing_percent = 100 * int(df.isna().sum().sum()) / total_cells
    if missing_percent > 0:
        recommendations.append(
            f"Review missing values: {int(df.isna().sum().sum()):,} cells ({missing_percent:.1f}% of the dataset) are empty."
        )
    if duplicate_rows:
        recommendations.append(
            f"Consider removing or investigating {duplicate_rows:,} duplicate rows before training."
        )
    for col in columns:
        if col["likely_identifier"]:
            recommendations.append(
                f"Review '{col['name']}' because it appears identifier-like and may not generalize as a predictive feature."
            )
        if col["missing_percent"] >= 50:
            recommendations.append(
                f"'{col['name']}' has {col['missing_percent']:.1f}% missing values; consider whether it should be retained."
            )
    if len(numeric_cols) > 40:
        recommendations.append("Correlation analysis is limited to the first 40 numeric columns to keep the saved report compact.")
    if not recommendations:
        recommendations.append("No major data-quality warnings were detected by the automated checks.")

    return {
        "columns": columns,
        "numeric_column_count": len(numeric_cols),
        "categorical_column_count": len(categorical_cols),
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_row_count": duplicate_rows,
        "preview": preview,
        "numeric_statistics": numeric_statistics,
        "categorical_statistics": categorical_statistics,
        "missing_by_column": missing_by_column,
        "correlation_matrix": correlation_matrix,
        "correlation_columns_analyzed": len(correlation_columns),
        "recommendations": recommendations,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Database + disk operations
# ---------------------------------------------------------------------------

def create_dataset_from_upload(
    db: Session, settings: Settings, original_filename: str | None, file: BinaryIO, owner_id: str
) -> tuple[Dataset, pd.DataFrame]:
    """Validate an uploaded CSV, save it to disk and record it in MySQL."""
    if not has_allowed_extension(original_filename):
        raise DataValidationError("Only .csv files are accepted.")

    content = read_limited(file, settings.max_upload_size_bytes)
    df, encoding = parse_csv_bytes(content, settings)

    dataset_id = new_id()
    stored_filename = f"{dataset_id}.csv"  # server-generated; the client's name is never used on disk
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    file_path = settings.upload_path / stored_filename
    # Write the original bytes untouched — we never modify the user's data.
    file_path.write_bytes(content)

    dataset = Dataset(
        id=dataset_id,
        owner_id=owner_id,
        original_filename=sanitize_display_filename(original_filename),
        stored_filename=stored_filename,
        file_size_bytes=len(content),
        file_encoding=encoding,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns_metadata=build_column_info(df),
    )
    try:
        db.add(dataset)
        db.commit()
    except Exception:
        # Keep disk and database consistent: no orphan CSV without a DB row.
        db.rollback()
        file_path.unlink(missing_ok=True)
        raise
    logger.info("Stored dataset %s (%d rows, %d columns)", dataset_id, df.shape[0], df.shape[1])
    return dataset, df


def get_dataset(db: Session, dataset_id: str, owner_id: str) -> Dataset:
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == owner_id))
    if dataset is None:
        raise NotFoundError("Dataset not found.")
    return dataset


def load_dataframe(dataset: Dataset, settings: Settings) -> pd.DataFrame:
    """Read a stored dataset back from uploads/ using a safe, server-side path."""
    path = resolve_safe_path(settings.upload_path, dataset.stored_filename)
    return pd.read_csv(path, encoding=dataset.file_encoding, low_memory=False)


def list_datasets(db: Session, limit: int, offset: int, owner_id: str) -> tuple[int, list[Dataset]]:
    owned = Dataset.owner_id == owner_id
    total = db.scalar(select(func.count()).select_from(Dataset).where(owned)) or 0
    rows = db.scalars(select(Dataset).where(owned).order_by(Dataset.uploaded_at.desc()).limit(limit).offset(offset)).all()
    return total, list(rows)


# ---------------------------------------------------------------------------
# Persistent insight snapshots
# ---------------------------------------------------------------------------

def save_insight_snapshot(db: Session, dataset_id: str, profile: dict[str, Any]) -> DatasetInsight:
    """Save a new immutable version of a dataset's analysis."""
    latest_version = db.scalar(
        select(func.max(DatasetInsight.version)).where(DatasetInsight.dataset_id == dataset_id)
    ) or 0
    snapshot = DatasetInsight(
        id=new_id(),
        dataset_id=dataset_id,
        version=int(latest_version) + 1,
        schema_version=1,
        insights_json=profile,
        generated_at=utc_now(),
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def get_latest_insight(db: Session, dataset_id: str) -> DatasetInsight | None:
    return db.scalars(
        select(DatasetInsight)
        .where(DatasetInsight.dataset_id == dataset_id)
        .order_by(DatasetInsight.version.desc())
        .limit(1)
    ).first()


def list_insight_snapshots(db: Session, dataset_id: str) -> list[DatasetInsight]:
    return list(db.scalars(
        select(DatasetInsight)
        .where(DatasetInsight.dataset_id == dataset_id)
        .order_by(DatasetInsight.version.desc())
    ).all())
