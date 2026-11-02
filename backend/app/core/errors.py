"""
Application-level exceptions.

Services raise these instead of FastAPI's HTTPException, so the service layer
does not depend on the web framework. `app/main.py` registers handlers that
turn each one into a consistent JSON response: {"detail": "<message>"}.
"""


class AppError(Exception):
    """Base class for errors that are safe to show to the user."""

    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DataValidationError(AppError):
    """The uploaded data or the requested experiment is not usable (HTTP 422)."""

    status_code = 422


class NotFoundError(AppError):
    """A dataset, experiment, model result or file does not exist (HTTP 404)."""

    status_code = 404


class ConflictError(AppError):
    """The request is valid but cannot run in the resource's current state (HTTP 409)."""

    status_code = 409


class PayloadTooLargeError(AppError):
    """The uploaded file exceeds the configured size limit (HTTP 413)."""

    status_code = 413
