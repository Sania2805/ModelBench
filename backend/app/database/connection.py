"""
MySQL connection management with SQLAlchemy 2.0.

Key ideas:
* ONE engine for the whole process. The engine owns a connection pool, so
  creating a new engine per request would waste connections and be slow.
* A short-lived Session per request, provided by the `get_db` dependency.
  The session is always closed in `finally`, even if the request fails.
* `pool_pre_ping=True` tests a pooled connection before using it. MySQL drops
  idle connections after `wait_timeout`, and pre-ping transparently replaces
  dead ones instead of failing the request with "MySQL server has gone away".
"""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models (see app/database/models.py)."""


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=3600,  # recycle connections hourly, below MySQL's default 8h timeout
        pool_size=5,
        max_overflow=10,
        # Never echo SQL here: logs could end up containing data values.
        echo=False,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    # expire_on_commit=False lets us keep reading ORM objects after commit
    # (e.g. to build the API response) without triggering extra queries.
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """
    FastAPI dependency that yields a database session.

    Services decide when to commit. If an exception escapes, we roll back any
    half-finished transaction so nothing partial is left behind.
    """
    session = get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    """Return True if a trivial query succeeds. Used by /api/health and at startup."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
