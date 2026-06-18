"""The first built-in tool: get_current_time.

Deliberately trivial. Its only job is to prove the full round trip works — the model
decides it needs the current time, emits a tool call, the harness executes this, and
the result flows back so the model can answer. A model cannot know the wall-clock time
from its training data, so this is a clean, unfakeable demonstration of tool use.
"""

from __future__ import annotations

from datetime import datetime, timezone as _tz
from zoneinfo import ZoneInfo

from cornac.tools.base import tool


@tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given IANA timezone (e.g. "UTC", "America/New_York", "Asia/Kolkata")."""
    if timezone.upper() == "UTC":
        tz = _tz.utc
    else:
        tz = ZoneInfo(timezone)  # raises if the name is invalid; the registry reports it
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
