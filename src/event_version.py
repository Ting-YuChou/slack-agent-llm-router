"""Which ``event_version`` values a consumer of the chat pipeline accepts (A12b).

Producers stamp ``event_version`` (``EVENT_SCHEMA_VERSION``) on every event.
Consumers accept any version with the same major, and events without one
(written before the field existed). Anything else is a schema they cannot
read: the sink and the policy materializer dead-letter it, and the Flink jobs
drop and count it.

Stdlib only: the Flink jobs import it from the mounted ``src/``.
"""

from __future__ import annotations

import re
from typing import Any

EVENT_SCHEMA_VERSION = "1.0"
SUPPORTED_EVENT_MAJOR_VERSION = int(EVENT_SCHEMA_VERSION.split(".", 1)[0])
UNSUPPORTED_EVENT_VERSION_REASON = "unsupported_event_version"

# ASCII digits only and a short major: str.isdigit() also accepts characters
# int() rejects ("1²"), and int() refuses very long digit strings.
_MAJOR_PATTERN = re.compile(r"[0-9]{1,6}")


def is_supported_event_version(value: Any) -> bool:
    """True for a missing version or one whose major part is supported.

    Never raises: any value it cannot read is unsupported.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        major = str(value).strip().split(".", 1)[0]
    except Exception:
        return False
    if not _MAJOR_PATTERN.fullmatch(major):
        return False
    return int(major) == SUPPORTED_EVENT_MAJOR_VERSION


class UnsupportedEventVersionError(ValueError):
    """An event whose ``event_version`` has an unsupported major version.

    The message names only the reason and the value's type: the value comes
    from the event and lands in DLQ payloads and logs.
    """

    reason = UNSUPPORTED_EVENT_VERSION_REASON

    def __init__(self, version: Any):
        super().__init__(
            f"{UNSUPPORTED_EVENT_VERSION_REASON} "
            f"(event_version of type {type(version).__name__})"
        )


def check_event_version(event: Any) -> None:
    """Raise UnsupportedEventVersionError for a dict with a bad version."""
    if isinstance(event, dict) and not is_supported_event_version(
        event.get("event_version")
    ):
        raise UnsupportedEventVersionError(event.get("event_version"))
