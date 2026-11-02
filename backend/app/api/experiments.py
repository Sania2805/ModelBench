"""Experiment, model-catalogue, report and artifact-download endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.connection import get_db
from app.schemas.experiment import (
    AvailableModel,
    ExperimentCreate,
    ExperimentDetail,
    ExperimentListResponse,
)
from app.services import experiment_service, report_service
from app.services.training_service import MODEL_REGISTRY, models_for_task

router = APIRouter(tags=["experiments"])


@router.get("/models", response_model=list[AvailableModel])
def list_available_models(task_type: Literal["classification", "regression"] | None = None) -> list[AvailableModel]:
    """The models the frontend can offer, optionally filtered by task type."""
    specs = models_for_task(task_type) if task_type else list(MODEL_REGISTRY.values())
    return [
        AvailableModel(
            name=s.name,
            display_name=s.display_name,
            task_type=s.task_type,
            description=s.description,
            hyperparameters=s.hyperparameters,
        )
        for s in specs
    ]


@router.post("/experiments", response_model=ExperimentDetail, status_code=status.HTTP_201_CREATED)
def create_experiment(
    request: ExperimentCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    experiment = experiment_service.run_experiment(db, settings, request)
    # Re-load with relationships so the response contains dataset info and results.
    return experiment_service.build_experiment_detail(experiment_service.get_experiment(db, experiment.id))


@router.get("/experiments", response_model=ExperimentListResponse)
def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict:
    return experiment_service.list_experiments(db, page, page_size)


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)) -> dict:
    return experiment_service.build_experiment_detail(experiment_service.get_experiment(db, experiment_id))


@router.get("/experiments/{experiment_id}/report", response_class=FileResponse)
def download_report(
    experiment_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    experiment = experiment_service.get_experiment(db, experiment_id)
    path = report_service.generate_report(experiment, settings)
    return FileResponse(path, media_type="text/html", filename=path.name)


@router.get("/experiments/{experiment_id}/models/{model_name}/download", response_class=FileResponse)
def download_model(
    experiment_id: str,
    model_name: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    path = experiment_service.get_artifact_path(db, settings, experiment_id, model_name)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"modelbench_{model_name}_{experiment_id[:8]}.joblib",
    )
