"""
Create the ModelBench tables in MySQL.

Run from the `backend` folder:

    python -m app.database.init_db

This uses SQLAlchemy's `create_all()`, which only creates tables that do not
exist yet — it never drops or alters existing tables, so it is safe to re-run.

It does NOT create the database itself. Create `modelbench_db` first (see the
README); that keeps the application's MySQL user free of admin privileges.

For a production project you would switch to a migration tool such as Alembic
so schema changes can be applied step by step. All models are registered on
`Base.metadata`, which is exactly what Alembic needs, so that upgrade is easy.
"""

import sys

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import ConfigurationError, get_settings
from app.database import models  # noqa: F401  (importing registers the tables on Base.metadata)
from app.database.connection import Base, get_engine


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def main() -> int:
    try:
        settings = get_settings()
        print(f"Creating tables in {settings.safe_database_description} ...")
        init_db()
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return 1
    except SQLAlchemyError as error:
        # Print only the error class and the driver's short message — no connection URL.
        reason = getattr(error, "orig", None) or error.__class__.__name__
        print(f"Could not create tables: {reason}")
        print("Check that MySQL is running, the database exists, and backend/.env is correct.")
        return 1
    print("Done. Tables: " + ", ".join(sorted(Base.metadata.tables)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
