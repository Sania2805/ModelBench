"""
Dataset endpoints. Routes stay thin: they parse the request, call a service,
and return a schema. All real work lives in app/services/dataset_service.py.

Routes are plain `def` (not `async def`) on purpose: pandas work is CPU-bound
and blocking, and FastAPI runs sync routes in a thread pool so they do not
freeze the server's event loop.
"""

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.connection import get_db
from app.schemas.dataset import DatasetListResponse, DatasetProfile, DatasetSummary
from app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetProfile, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    file: UploadFile = File(..., description="A CSV file"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DatasetProfile:
    dataset, df = dataset_service.create_dataset_from_upload(db, settings, file.filename, file.file)
    return DatasetProfile(
        dataset=DatasetSummary.model_validate(dataset), **dataset_service.build_profile_data(df)
    )


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DatasetListResponse:
    total, datasets = dataset_service.list_datasets(db, limit, offset)
    return DatasetListResponse(total=total, items=[DatasetSummary.model_validate(d) for d in datasets])


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
def get_dataset_profile(
    dataset_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DatasetProfile:
    dataset = dataset_service.get_dataset(db, dataset_id)
    df = dataset_service.load_dataframe(dataset, settings)
    return DatasetProfile(
        dataset=DatasetSummary.model_validate(dataset), **dataset_service.build_profile_data(df)
    )


@router.get("/{dataset_id}/target-classes")
def get_target_classes(
    dataset_id: str,
    target_column: str = Query(..., min_length=1, max_length=255),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return distinct labels for a classification target so the UI can offer positive-class selection."""
    dataset = dataset_service.get_dataset(db, dataset_id)
    df = dataset_service.load_dataframe(dataset, settings)
    if target_column not in df.columns:
        from app.core.errors import DataValidationError
        raise DataValidationError(f"Target column '{target_column}' does not exist in this dataset.")
    values = df[target_column].dropna().unique().tolist()
    if len(values) != 2:
        return {"target_column": target_column, "task_type": "classification", "classes": [str(v) for v in sorted(values, key=str)], "positive_class_selection": False, "message": "Positive-class selection is available only for binary classification."}
    return {"target_column": target_column, "task_type": "classification", "classes": [str(v) for v in sorted(values, key=str)], "positive_class_selection": True}
