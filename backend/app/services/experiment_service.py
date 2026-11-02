"""
Running, saving and loading experiments.

run_experiment() is the heart of the app:

    1. Validate the request (dataset exists, models match the task).
    2. Load the CSV and prepare/split the data. If the data is unsuitable we
       stop here and do NOT create an experiment row — nothing misleading is saved.
    3. Insert the experiment with status "running" and commit, so even a crash
       mid-training leaves a visible, honest record.
    4. Train each model. One model failing does not stop the others; the
       failure is stored as a "failed" ModelResult with its error message.
    5. Save each fitted pipeline to artifacts/<experiment_id>/<model>.joblib.
    6. Commit all results and the final status in ONE transaction:
         completed - every model succeeded
         partial   - some succeeded, some failed
         failed    - none succeeded (never reported as a success)
       If that commit fails, the artifact files written in step 5 are deleted.

Training runs synchronously inside the request. That is fine for an MVP with
small datasets; a production system would move it to a background worker.
"""

import logging
import shutil
from typing import Any

import joblib
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.core.errors import ConflictError, DataValidationError, NotFoundError
from app.core.security import new_id, resolve_safe_path
from app.database.models import Experiment, ModelResult, utc_now
from app.schemas.experiment import ExperimentCreate
from app.services import dataset_service
from app.services.evaluation_service import PRIMARY_METRICS, pick_best_model
from app.services.preprocessing_service import prepare_experiment_data
from app.services.training_service import MODEL_REGISTRY, display_name_for, train_and_evaluate

logger = logging.getLogger(__name__)


def validate_model_selection(task_type: str, model_names: list[str]) -> None:
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown:
        raise DataValidationError(f"Unknown model(s): {', '.join(unknown)}.")
    incompatible = [name for name in model_names if MODEL_REGISTRY[name].task_type != task_type]
    if incompatible:
        raise DataValidationError(
            f"Model(s) {', '.join(display_name_for(n) for n in incompatible)} cannot be used for {task_type}."
        )


def run_experiment(db: Session, settings: Settings, request: ExperimentCreate) -> Experiment:
    validate_model_selection(request.task_type, request.model_names)
    dataset = dataset_service.get_dataset(db, request.dataset_id)
    df = dataset_service.load_dataframe(dataset, settings)

    # Pure validation + split. Raises DataValidationError before anything is saved.
    data = prepare_experiment_data(
        df,
        request.target_column,
        request.task_type,
        request.test_size,
        request.random_seed,
        min_rows=settings.min_dataset_rows,
        positive_class=request.positive_class,
        exclude_columns=request.exclude_columns,
        split_strategy=request.split_strategy,
        time_column=request.time_column,
    )
    if request.balance_classes and request.task_type != "classification":
        raise DataValidationError("Class balancing is available only for classification experiments.")

    primary_metric, _ = PRIMARY_METRICS[request.task_type]
    experiment = Experiment(
        id=new_id(),
        dataset_id=dataset.id,
        target_column=request.target_column,
        task_type=request.task_type,
        selected_models=request.model_names,
        test_size=request.test_size,
        random_seed=request.random_seed,
        status="running",
        primary_metric=primary_metric,
        run_metadata={**data.metadata, "warnings": data.warnings, "model_selection_method": "validation holdout; final metrics reported on untouched test set", "primary_metric": primary_metric, "balance_classes": request.balance_classes},
    )
    db.add(experiment)
    db.commit()

    artifact_dir = settings.artifact_path / experiment.id
    try:
        result_rows = _train_all_models(experiment, data, artifact_dir)
        successes = [r for r in result_rows if r["status"] == "success"]

        for row in result_rows:
            experiment.results.append(
                ModelResult(id=new_id(), **{k: v for k, v in row.items() if k not in {"pipeline", "selection_metric_value"}})
            )
        experiment.best_model_name = pick_best_model(result_rows, request.task_type)
        if not successes:
            experiment.status = "failed"
            experiment.error_message = "No model could be trained successfully. See each model's error."
        elif len(successes) < len(result_rows):
            experiment.status = "partial"
        else:
            experiment.status = "completed"
        experiment.completed_at = utc_now()
        db.commit()
    except Exception:
        # Something unexpected (e.g. database or disk error). Undo the partial
        # transaction, delete any artifacts, and mark the experiment as failed.
        logger.exception("Experiment %s failed unexpectedly", experiment.id)
        db.rollback()
        shutil.rmtree(artifact_dir, ignore_errors=True)
        _mark_failed(db, experiment.id, "The experiment failed due to an internal error.")
        raise

    logger.info("Experiment %s finished with status %s", experiment.id, experiment.status)
    return experiment


def _train_all_models(experiment: Experiment, data, artifact_dir) -> list[dict[str, Any]]:
    """Train every selected model, saving artifacts for those that succeed."""
    rows: list[dict[str, Any]] = []
    for model_name in experiment.selected_models:
        spec = MODEL_REGISTRY[model_name]
        try:
            outcome = train_and_evaluate(
                spec, data, experiment.task_type, experiment.random_seed,
                balance_classes=bool((experiment.run_metadata or {}).get("balance_classes", False)),
            )
        except Exception as error:  # one model failing must not stop the others
            logger.warning("Model %s failed in experiment %s: %s", model_name, experiment.id, error)
            rows.append({"model_name": model_name, "status": "failed", "error_message": str(error)[:2000]})
            continue

        artifact_dir.mkdir(parents=True, exist_ok=True)
        # The folder name is the experiment's UUID, so artifacts from different
        # experiments can never overwrite each other.
        relative_path = f"{experiment.id}/{model_name}.joblib"
        joblib.dump(outcome["pipeline"], artifact_dir / f"{model_name}.joblib")
        rows.append(
            {
                "model_name": model_name,
                "status": "success",
                "artifact_filename": relative_path,
                **{k: v for k, v in outcome.items() if k != "pipeline"},
            }
        )
    return rows


def _mark_failed(db: Session, experiment_id: str, message: str) -> None:
    try:
        experiment = db.get(Experiment, experiment_id)
        if experiment is not None:
            experiment.status = "failed"
            experiment.error_message = message
            experiment.completed_at = utc_now()
            db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not mark experiment %s as failed", experiment_id)


# ---------------------------------------------------------------------------
# Reading experiments back
# ---------------------------------------------------------------------------

def get_experiment(db: Session, experiment_id: str) -> Experiment:
    # selectinload fetches the dataset and results in a couple of extra queries
    # up front, instead of one query per result later (the "N+1" problem).
    experiment = db.scalar(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.results), selectinload(Experiment.dataset))
    )
    if experiment is None:
        raise NotFoundError("Experiment not found.")
    return experiment


def build_experiment_detail(experiment: Experiment) -> dict[str, Any]:
    """Shape an experiment into the API response, including chart-ready data."""
    metric, higher_is_better = PRIMARY_METRICS[experiment.task_type]

    def sort_key(result: ModelResult):
        value = (result.metrics or {}).get(metric)
        # Successful results first, ordered best-to-worst; failures at the end.
        if result.status != "success" or value is None:
            return (1, 0.0)
        return (0, -value if higher_is_better else value)

    ordered = sorted(
        experiment.results,
        key=lambda result: (0 if result.model_name == experiment.best_model_name else 1, *sort_key(result)),
    )
    results = [
        {
            "id": r.id,
            "model_name": r.model_name,
            "display_name": display_name_for(r.model_name),
            "status": r.status,
            "metrics": r.metrics,
            "metric_notes": r.metric_notes,
            "chart_data": r.chart_data,
            "hyperparameters": r.hyperparameters,
            "training_time_seconds": r.training_time_seconds,
            "prediction_time_seconds": r.prediction_time_seconds,
            "has_artifact": bool(r.artifact_filename),
            "error_message": r.error_message,
            "is_best": r.model_name == experiment.best_model_name,
        }
        for r in ordered
    ]
    comparison_chart = [
        {
            "model": display_name_for(r.model_name),
            **(r.metrics or {}),
            "training_time_seconds": r.training_time_seconds,
        }
        for r in ordered
        if r.status == "success"
    ]
    dataset = experiment.dataset
    return {
        "id": experiment.id,
        "dataset": {
            "id": dataset.id,
            "original_filename": dataset.original_filename,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
        },
        "target_column": experiment.target_column,
        "task_type": experiment.task_type,
        "selected_models": experiment.selected_models,
        "test_size": experiment.test_size,
        "random_seed": experiment.random_seed,
        "positive_class": (experiment.run_metadata or {}).get("positive_class"),
        "status": experiment.status,
        "primary_metric": metric,
        "primary_metric_higher_is_better": higher_is_better,
        "best_model_name": experiment.best_model_name,
        "run_metadata": experiment.run_metadata,
        "error_message": experiment.error_message,
        "created_at": experiment.created_at,
        "completed_at": experiment.completed_at,
        "results": results,
        "comparison_chart": comparison_chart,
    }


def list_experiments(db: Session, page: int, page_size: int) -> dict[str, Any]:
    total = db.scalar(select(func.count()).select_from(Experiment)) or 0
    experiments = db.scalars(
        select(Experiment)
        .options(selectinload(Experiment.results), selectinload(Experiment.dataset))
        .order_by(Experiment.created_at.desc())
        .limit(page_size)
        .offset((page - 1) * page_size)
    ).all()

    items = []
    for exp in experiments:
        best_value = None
        if exp.best_model_name:
            best = next((r for r in exp.results if r.model_name == exp.best_model_name), None)
            best_value = (best.metrics or {}).get(exp.primary_metric) if best else None
        items.append(
            {
                "id": exp.id,
                "dataset_id": exp.dataset_id,
                "dataset_name": exp.dataset.original_filename,
                "target_column": exp.target_column,
                "task_type": exp.task_type,
                "selected_models": exp.selected_models,
                "status": exp.status,
                "primary_metric": exp.primary_metric,
                "best_model_name": exp.best_model_name,
                "best_metric_value": best_value,
                "created_at": exp.created_at,
            }
        )
    return {"total": total, "page": page, "page_size": page_size, "items": items}


def get_artifact_path(db: Session, settings: Settings, experiment_id: str, model_name: str):
    """
    Find the .joblib file for one model of one experiment.

    The client only sends IDs/names. We look the stored relative path up in
    MySQL and resolve it safely inside artifacts/ — the client can never
    choose a filesystem path.
    """
    result = db.scalar(
        select(ModelResult).where(
            ModelResult.experiment_id == experiment_id, ModelResult.model_name == model_name
        )
    )
    if result is None:
        raise NotFoundError("This model was not part of the experiment.")
    if result.status != "success" or not result.artifact_filename:
        raise ConflictError("This model failed to train, so there is no pipeline to download.")
    return resolve_safe_path(settings.artifact_path, result.artifact_filename)
