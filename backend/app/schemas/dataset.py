"""
Pydantic schemas for dataset responses.

These are separate from the SQLAlchemy models on purpose: the database model
describes how data is *stored*, while these describe what the API *returns*.
For example, the stored filename on disk is never included here.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from app.schemas.common import UTCDateTime


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    kind: str  # "numeric" or "categorical"
    missing_count: int
    missing_percent: float
    unique_count: int
    likely_identifier: bool = False


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    uploaded_at: UTCDateTime


class DatasetProfile(BaseModel):
    dataset: DatasetSummary
    columns: list[ColumnInfo]
    numeric_column_count: int
    categorical_column_count: int
    total_missing_values: int
    duplicate_row_count: int
    # First rows of the file; values are JSON-safe (NaN becomes null).
    preview: list[dict[str, Any]]
    # Describe-style stats keyed by column name.
    numeric_statistics: dict[str, dict[str, float | None]]
    categorical_statistics: dict[str, dict[str, Any]]
    warnings: list[str]


class DatasetListResponse(BaseModel):
    total: int
    items: list[DatasetSummary]
