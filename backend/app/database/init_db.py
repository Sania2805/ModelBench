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

import shutil
import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import ConfigurationError, get_settings
from app.database import models  # noqa: F401  (importing registers the tables on Base.metadata)
from app.database.connection import Base, get_engine


def init_db() -> None:
    engine = get_engine()
    # Create any brand-new tables first (notably users on older installations).
    Base.metadata.create_all(bind=engine)

    # create_all() does not alter existing tables. Older ModelBench databases
    # have datasets without an owner_id, so safely discard legacy user data
    # before adding a NOT NULL ownership column. The user accounts themselves
    # are retained and can sign in normally; they simply start with empty data.
    columns = {column["name"] for column in inspect(engine).get_columns("datasets")}
    if "owner_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM model_results"))
            connection.execute(text("DELETE FROM experiments"))
            connection.execute(text("DELETE FROM dataset_insights"))
            connection.execute(text("DELETE FROM datasets"))
            connection.execute(text(
                "ALTER TABLE datasets "
                "ADD COLUMN owner_id VARCHAR(36) NOT NULL, "
                "ADD CONSTRAINT fk_datasets_owner_id_users FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE, "
                "ADD INDEX ix_datasets_owner_id (owner_id)"
            ))
        # Remove legacy uploaded CSVs, trained artifacts and generated reports
        # too: they no longer have database records and belong to no account.
        settings = get_settings()
        for storage_path in (settings.upload_path, settings.artifact_path, settings.report_path):
            shutil.rmtree(storage_path, ignore_errors=True)
            storage_path.mkdir(parents=True, exist_ok=True)


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
