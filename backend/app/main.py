"""
FastAPI application entry point.

Start the server from the `backend` folder:

    uvicorn app.main:app --reload --port 8000

Interactive API docs are then available at http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from app.api import api_router
from app.core.config import ensure_storage_dirs, get_settings
from app.core.errors import AppError
from app.database.connection import check_database_connection, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("modelbench")

REQUIRED_TABLES = {"datasets", "experiments", "model_results", "users"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # get_settings() raises a clear ConfigurationError if .env is incomplete,
    # which stops the server immediately instead of failing on first request.
    settings = get_settings()
    ensure_storage_dirs(settings)
    logger.info("Using database %s", settings.safe_database_description)  # never logs the password

    # We do NOT fall back to another database: if MySQL is down we say so loudly.
    if not check_database_connection():
        logger.error("Cannot connect to MySQL. Is the server running and is backend/.env correct?")
    else:
        missing = REQUIRED_TABLES - set(inspect(get_engine()).get_table_names())
        if missing:
            logger.error("Missing tables %s. Run: python -m app.database.init_db", sorted(missing))
    yield
    get_engine().dispose()


app = FastAPI(
    title="ModelBench API",
    description="Upload a CSV, train several scikit-learn models and compare them.",
    version="1.0.0",
    lifespan=lifespan,
)

# Only the configured frontend origins may call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    # Lets the frontend read the download filename from the response headers.
    expose_headers=["Content-Disposition"],
)


# --- Consistent error responses: always {"detail": "<message>"} ---

@app.exception_handler(AppError)
async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": error.message})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    # Turn pydantic's list of errors into one readable sentence, keeping the raw list too.
    messages = []
    for err in error.errors():
        location = ".".join(str(part) for part in err["loc"] if part not in ("body", "query", "path"))
        messages.append(f"{location}: {err['msg']}" if location else err["msg"])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "; ".join(messages), "errors": [{"loc": e["loc"], "msg": e["msg"]} for e in error.errors()]},
    )


@app.exception_handler(SQLAlchemyError)
async def handle_database_error(request: Request, error: SQLAlchemyError) -> JSONResponse:
    # Log the error class only; SQLAlchemy messages can include SQL and parameters.
    logger.error("Database error on %s %s: %s", request.method, request.url.path, error.__class__.__name__)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The database is unavailable or returned an error. Please try again."},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception("Unexpected error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred."},
    )


app.include_router(api_router)
