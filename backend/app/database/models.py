"""
SQLAlchemy ORM models — the MySQL tables used by ModelBench.

    datasets 1 ──< experiments 1 ──< model_results

* A dataset can be used by many experiments.
* Each model result belongs to exactly one experiment.

Design notes:
* IDs are UUID strings (CHAR(36)). They are unguessable, so a user cannot
  enumerate other experiments by counting 1, 2, 3...
* The CSV contents and .joblib files are NOT stored in the database. They live
  on disk; the database only stores their server-generated filenames.
* Structured data (column info, metrics, chart data) uses the JSON type, which
  MySQL 5.7+ supports natively.
* Timestamps are stored as UTC. MySQL DATETIME has no timezone, so we always
  write UTC and the API marks them as UTC when sending them out.
* DATETIME(6) keeps microseconds. Plain DATETIME rounds to whole seconds, so
  two experiments started in the same second could not be ordered reliably.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base

TASK_TYPES = ("classification", "regression")
EXPERIMENT_STATUSES = ("running", "completed", "partial", "failed")
RESULT_STATUSES = ("success", "failed")


# MySQL DATETIME with microsecond precision.
Timestamp = DATETIME(fsp=6)


def utc_now() -> datetime:
    # Store naive UTC datetimes because MySQL DATETIME columns are timezone-less.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    # Server-generated name inside uploads/, e.g. "3f2c...e1.csv". Never a client path.
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    file_encoding: Mapped[str] = mapped_column(String(20), default="utf-8")
    row_count: Mapped[int] = mapped_column(Integer)
    column_count: Mapped[int] = mapped_column(Integer)
    # List of {"name", "dtype", "kind", "missing_count", ...} computed at upload time.
    columns_metadata: Mapped[list] = mapped_column(JSON)
    uploaded_at: Mapped[datetime] = mapped_column(Timestamp, default=utc_now, index=True)

    experiments: Mapped[list["Experiment"]] = relationship(back_populates="dataset")


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint(f"task_type IN {TASK_TYPES}", name="ck_experiment_task_type"),
        CheckConstraint(f"status IN {EXPERIMENT_STATUSES}", name="ck_experiment_status"),
        CheckConstraint("test_size > 0 AND test_size < 1", name="ck_experiment_test_size"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # RESTRICT: a dataset cannot be deleted while experiments still reference it.
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), index=True
    )
    target_column: Mapped[str] = mapped_column(String(255))
    task_type: Mapped[str] = mapped_column(String(20))
    selected_models: Mapped[list] = mapped_column(JSON)
    test_size: Mapped[float] = mapped_column(Float)
    random_seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="running", index=True)
    # "f1" for classification, "rmse" for regression.
    primary_metric: Mapped[str] = mapped_column(String(20))
    best_model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Extra facts about the run: train/test sizes, feature lists, warnings shown to the user.
    run_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utc_now, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(Timestamp, nullable=True)

    dataset: Mapped[Dataset] = relationship(back_populates="experiments")
    # cascade="all, delete-orphan": deleting an experiment also deletes its results.
    results: Mapped[list["ModelResult"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ModelResult(Base):
    __tablename__ = "model_results"
    __table_args__ = (
        # One row per (experiment, model): the same model cannot appear twice in one run.
        UniqueConstraint("experiment_id", "model_name", name="uq_result_experiment_model"),
        CheckConstraint(f"status IN {RESULT_STATUSES}", name="ck_result_status"),
        Index("ix_model_results_experiment_status", "experiment_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))
    # e.g. {"accuracy": 0.91, "f1": 0.89, "roc_auc": null, ...}
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Explanations for metrics that are N/A, e.g. {"roc_auc": "Only one class in test set"}
    metric_notes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Confusion matrix (classification) or actual-vs-predicted points (regression).
    chart_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Relative path inside artifacts/, e.g. "<experiment_id>/random_forest_classifier.joblib"
    artifact_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utc_now)

    experiment: Mapped[Experiment] = relationship(back_populates="results")
