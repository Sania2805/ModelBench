"""GET /api/health — is the API up, and can it reach MySQL?"""

from fastapi import APIRouter

from app.database.connection import check_database_connection

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    # Only a yes/no for the database: no host, user or error details are exposed.
    database_ok = check_database_connection()
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "connected" if database_ok else "unavailable",
    }
