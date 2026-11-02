"""
File-safety helpers.

The golden rule: never trust a filename or path that came from the client.

* Uploaded files are stored under a server-generated name (a UUID), not the
  name the user's browser sent.
* Whenever we turn a stored relative name back into a real path, we resolve it
  and check that it is still inside the expected folder. This blocks "path
  traversal" tricks such as `../../Windows/system.ini`.
"""

import re
import uuid
from pathlib import Path

from app.core.errors import NotFoundError

ALLOWED_DATASET_EXTENSIONS = {".csv"}

# Characters we allow in a *display* name. Anything else is replaced with "_".
_UNSAFE_DISPLAY_CHARS = re.compile(r"[^A-Za-z0-9._\- ()]")


def new_id() -> str:
    """Generate a random, unguessable ID used for datasets, experiments and results."""
    return str(uuid.uuid4())


def sanitize_display_filename(filename: str | None, fallback: str = "dataset.csv") -> str:
    """
    Clean the original filename so it is safe to show in the UI and in reports.

    We keep only the final path component (browsers on some systems send full
    paths) and strip unusual characters. This name is *only* for display — the
    file on disk always uses a generated name.
    """
    if not filename:
        return fallback
    name = Path(filename.replace("\\", "/")).name
    name = _UNSAFE_DISPLAY_CHARS.sub("_", name).strip()
    return name[:200] or fallback


def has_allowed_extension(filename: str | None) -> bool:
    return bool(filename) and Path(filename).suffix.lower() in ALLOWED_DATASET_EXTENSIONS


def resolve_safe_path(base_dir: Path, relative_name: str) -> Path:
    """
    Turn a stored relative filename into an absolute path inside `base_dir`.

    Raises NotFoundError if the name would escape the folder (path traversal)
    or if the file does not exist. Callers never receive a path outside base_dir.
    """
    base = base_dir.resolve()
    candidate = (base / relative_name).resolve()
    if not candidate.is_relative_to(base):
        # Deliberately vague message: we do not echo paths back to the client.
        raise NotFoundError("The requested file is not available.")
    if not candidate.is_file():
        raise NotFoundError("The requested file is no longer available on the server.")
    return candidate
