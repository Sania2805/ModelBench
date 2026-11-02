"""Collects all routers under the /api prefix."""

from fastapi import APIRouter

from app.api import datasets, experiments, health

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(experiments.router)
