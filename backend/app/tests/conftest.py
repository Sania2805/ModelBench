"""
Shared pytest fixtures.

Safety first: the tests must NEVER touch your real `modelbench_db`.
Before the app is imported we switch DATABASE_NAME to TEST_DATABASE_NAME
(default `modelbench_test_db`) and refuse to run if that name does not contain
"test" or equals the development database name. Uploads, artifacts and
reports go to a temporary folder, not the project's folders.

Tests that need MySQL use the `client` or `db_session` fixtures. If the test
database is unreachable those tests are SKIPPED with a clear reason (they are
never silently run against another database). Pure ML/preprocessing tests run
without MySQL.
"""

import io
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings, get_settings

# --- 1. Point the whole app at the test database and temp folders -----------
_base_settings = get_settings()  # raises a readable error if backend/.env is incomplete
TEST_DB_NAME = _base_settings.test_database_name
if "test" not in TEST_DB_NAME.lower() or TEST_DB_NAME == _base_settings.database_name:
    raise RuntimeError(
        f"Refusing to run tests against database '{TEST_DB_NAME}'. "
        "Set TEST_DATABASE_NAME to a separate database whose name contains 'test'."
    )
_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="modelbench_tests_"))
os.environ["DATABASE_NAME"] = TEST_DB_NAME
os.environ["UPLOAD_DIR"] = str(_TEMP_ROOT / "uploads")
os.environ["ARTIFACT_DIR"] = str(_TEMP_ROOT / "artifacts")
os.environ["REPORT_DIR"] = str(_TEMP_ROOT / "reports")
get_settings.cache_clear()

# Imported only after the environment above is set.
from sqlalchemy import text  # noqa: E402

from app.database import models  # noqa: E402,F401
from app.database.connection import Base, get_engine, get_session_factory  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


# --- 2. Database fixtures -----------------------------------------------------

@pytest.fixture(scope="session")
def settings() -> Settings:
    current = get_settings()
    assert current.database_name == TEST_DB_NAME  # double-check the safety switch
    return current


@pytest.fixture(scope="session")
def db_engine(settings):
    engine = get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as error:
        pytest.skip(
            f"MySQL test database '{TEST_DB_NAME}' is not reachable "
            f"({error.__class__.__name__}). Create it and check backend/.env — see README."
        )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    session = get_session_factory()()
    yield session
    session.close()
    _clear_tables(db_engine)


def _clear_tables(engine) -> None:
    # Delete children before parents to respect the foreign keys.
    with engine.begin() as connection:
        for table in ("model_results", "experiments", "datasets"):
            connection.execute(text(f"DELETE FROM {table}"))


@pytest.fixture
def client(db_engine):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    _clear_tables(db_engine)


# --- 3. Small synthetic datasets ---------------------------------------------

def make_classification_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    color = rng.choice(["red", "green", "blue"], n)
    label = np.where(x1 + 0.5 * x2 + (color == "red") > 0.3, "yes", "no")
    df = pd.DataFrame({"x1": x1, "x2": x2, "color": color, "label": label})
    df.loc[[1, 5, 9], "x1"] = np.nan
    df.loc[[2, 6], "color"] = np.nan
    return df


def make_regression_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.uniform(50, 200, n)
    rooms = rng.integers(1, 5, n)
    city = rng.choice(["A", "B"], n)
    price = 3 * size + 20 * rooms + np.where(city == "A", 50, 0) + rng.normal(0, 10, n)
    df = pd.DataFrame({"size": size, "rooms": rooms, "city": city, "price": price})
    df.loc[[3, 7], "size"] = np.nan
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


@pytest.fixture
def classification_csv() -> bytes:
    return to_csv_bytes(make_classification_df())


@pytest.fixture
def regression_csv() -> bytes:
    return to_csv_bytes(make_regression_df())


@pytest.fixture
def upload(client):
    """Helper: upload CSV bytes and return the parsed JSON response."""

    def _upload(content: bytes, filename: str = "data.csv"):
        response = client.post("/api/datasets/upload", files={"file": (filename, content, "text/csv")})
        assert response.status_code == 201, response.text
        return response.json()

    return _upload
