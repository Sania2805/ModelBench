"""Collects all routers under the /api prefix."""

from fastapi import APIRouter, Depends

from app.api import auth, datasets, experiments, health
from app.api.auth import require_current_user

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(datasets.router, dependencies=[Depends(require_current_user)])
api_router.include_router(experiments.router, dependencies=[Depends(require_current_user)])
