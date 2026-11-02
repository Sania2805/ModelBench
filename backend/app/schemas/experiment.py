"""Pydantic schemas for experiment requests and responses."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import UTCDateTime

TaskType = Literal["classification", "regression"]


class ExperimentCreate(BaseModel):
    """Body of POST /api/experiments. Pydantic rejects bad types/ranges with HTTP 422."""

    dataset_id: str = Field(min_length=1, max_length=36)
    target_column: str = Field(min_length=1, max_length=255)
    task_type: TaskType
    model_names: list[str] = Field(min_length=1, max_length=20)
    # Optional per-model overrides. Omitted values retain each model's defaults.
    model_hyperparameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # 10%–50% test data. Outside this range results become unreliable or the
    # model has too little data to learn from.
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_seed: int = Field(default=42, ge=0, le=2**31 - 1)
    positive_class: str | None = Field(default=None, max_length=255, description="Positive label for binary classification; defaults to the last sorted class.")
    exclude_columns: list[str] = Field(default_factory=list, max_length=100)
    balance_classes: bool = False
    split_strategy: Literal["random", "chronological"] = "random"
    time_column: str | None = Field(default=None, max_length=255)

    @field_validator("model_names")
    @classmethod
    def no_duplicate_models(cls, names: list[str]) -> list[str]:
        if len(set(names)) != len(names):
            raise ValueError("Each model can only be selected once.")
        return names


class AvailableModel(BaseModel):
    name: str
    display_name: str
    task_type: TaskType
    description: str
    hyperparameters: dict[str, Any]
    parameter_schema: dict[str, dict[str, Any]]


class ModelResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    display_name: str
    status: str
    metrics: dict[str, float | None] | None
    metric_notes: dict[str, str] | None
    chart_data: dict[str, Any] | None
    hyperparameters: dict[str, Any] | None
    training_time_seconds: float | None
    prediction_time_seconds: float | None
    has_artifact: bool
    error_message: str | None
    is_best: bool = False


class DatasetBrief(BaseModel):
    id: str
    original_filename: str
    row_count: int
    column_count: int


class ExperimentDetail(BaseModel):
    id: str
    dataset: DatasetBrief
    target_column: str
    task_type: TaskType
    selected_models: list[str]
    test_size: float
    random_seed: int
    positive_class: str | None = None
    status: str
    primary_metric: str
    primary_metric_higher_is_better: bool
    best_model_name: str | None
    run_metadata: dict[str, Any] | None
    error_message: str | None
    created_at: UTCDateTime
    completed_at: UTCDateTime | None
    results: list[ModelResultOut]
    # [{"model": "Random Forest", "f1": 0.91, "accuracy": 0.93, ...}] ready for a bar chart.
    comparison_chart: list[dict[str, Any]]


class ExperimentListItem(BaseModel):
    id: str
    dataset_id: str
    dataset_name: str
    target_column: str
    task_type: TaskType
    selected_models: list[str]
    status: str
    primary_metric: str
    best_model_name: str | None
    best_metric_value: float | None
    created_at: UTCDateTime


class ExperimentListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ExperimentListItem]
