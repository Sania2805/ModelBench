"""Small helpers shared by the Pydantic schemas."""

from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator


def _as_utc(value: datetime) -> datetime:
    # The database stores naive UTC datetimes. Attaching tzinfo makes the JSON
    # output end in "+00:00", so the browser converts it to local time correctly.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]
