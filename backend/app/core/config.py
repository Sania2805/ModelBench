"""
Central configuration for the ModelBench backend.

All settings are read from environment variables (and from `backend/.env`
during local development) using pydantic-settings. Nothing secret is
hardcoded here — the database password only ever lives in `.env` or in the
real environment.

Why a Settings class?
    It gives us one place to validate configuration at startup. If a required
    variable such as DATABASE_USER is missing, the app fails immediately with a
    readable message instead of crashing later with a confusing database error.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# backend/app/core/config.py -> parents[2] is backend/, parents[3] is the project root.
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    # --- MySQL connection (required; no defaults for credentials) ---
    database_host: str = "localhost"
    database_port: int = 3306
    database_name: str
    database_user: str
    # An empty password is technically allowed by MySQL, so we only require
    # that the variable is *present*, not that it is non-empty.
    database_password: str
    # Separate database used only by pytest (see app/tests/conftest.py).
    test_database_name: str = "modelbench_test_db"

    # Authentication. AUTH_SECRET_KEY must be a unique random value (32+ characters).
    auth_secret_key: str = Field(min_length=32)
    auth_token_expire_minutes: int = Field(default=60, gt=0, le=1440)
    auth_cookie_secure: bool = False
    auth_cookie_samesite: Literal["lax", "strict", "none"] = "lax"

    # --- Upload limits ---
    max_upload_size_mb: float = Field(default=10, gt=0)
    max_columns: int = Field(default=500, gt=0)
    min_dataset_rows: int = Field(default=10, gt=1)

    # --- Storage folders (relative paths are resolved against the project root) ---
    upload_dir: Path = Path("uploads")
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")

    # --- CORS: comma-separated list of allowed frontend origins ---
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Real environment variables take priority over values in .env.
        # The test suite relies on this to point at a separate test database.
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def _resolve_dir(self, folder: Path) -> Path:
        return folder if folder.is_absolute() else (PROJECT_ROOT / folder).resolve()

    @property
    def upload_path(self) -> Path:
        return self._resolve_dir(self.upload_dir)

    @property
    def artifact_path(self) -> Path:
        return self._resolve_dir(self.artifact_dir)

    @property
    def report_path(self) -> Path:
        return self._resolve_dir(self.report_dir)

    @property
    def database_url(self) -> URL:
        """
        Build the SQLAlchemy URL from separate parts.

        URL.create() escapes special characters (like @, :, / or #) in the
        username and password, so a password such as "p@ss:word" will not
        break the connection string the way naive string formatting would.
        """
        return URL.create(
            drivername="mysql+pymysql",
            username=self.database_user,
            password=self.database_password,
            host=self.database_host,
            port=self.database_port,
            database=self.database_name,
            query={"charset": "utf8mb4"},
        )

    @property
    def safe_database_description(self) -> str:
        """A log-friendly description of the database that never includes the password."""
        return f"mysql://{self.database_host}:{self.database_port}/{self.database_name}"


@lru_cache
def get_settings() -> Settings:
    """
    Load settings once and cache them for the life of the process.

    Converts pydantic's validation error into a short, beginner-friendly
    message that names the missing variables (but never prints their values).
    """
    try:
        return Settings()
    except ValidationError as error:
        missing = [str(err["loc"][0]).upper() for err in error.errors()]
        raise ConfigurationError(
            "ModelBench configuration is missing or invalid for: "
            + ", ".join(missing)
            + ". Copy backend/.env.example to backend/.env and fill in your MySQL details."
        ) from None


def ensure_storage_dirs(settings: Settings) -> None:
    """Create the uploads/, artifacts/ and reports/ folders if they do not exist yet."""
    for folder in (settings.upload_path, settings.artifact_path, settings.report_path):
        folder.mkdir(parents=True, exist_ok=True)
