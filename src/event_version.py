"""Which ``event_version`` values a consumer of the chat pipeline accepts (A12b).

Producers stamp ``event_version`` (``"1.0"``) on every event. Consumers accept
any version with the supported major, and events without one (written before
the field existed). Anything else is a schema they cannot read: the sink and
the policy materializer dead-letter it, and the Flink jobs drop and count it.

Stdlib only: the Flink jobs import it from the mounted ``src/``.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_EVENT_MAJOR_VERSION = 1
UNSUPPORTED_EVENT_VERSION_REASON = "unsupported_event_version"


def is_supported_event_version(value: Any) -> bool:
    """True for a missing version or one whose major part is supported."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    major = str(value).strip().split(".", 1)[0]
    if not major.isdigit():
        return False
    return int(major) == SUPPORTED_EVENT_MAJOR_VERSION


def describe_event_version(value: Any) -> str:
    """A short, value-bounded rendering of a rejected version for logs."""
    return repr(value)[:32]


class UnsupportedEventVersionError(ValueError):
    """An event whose ``event_version`` has an unsupported major version."""

    reason = UNSUPPORTED_EVENT_VERSION_REASON

    def __init__(self, version: Any):
        self.version = version
        super().__init__(
            f"{UNSUPPORTED_EVENT_VERSION_REASON}: event_version "
            f"{describe_event_version(version)}"
        )


def check_event_version(event: Any) -> None:
    """Raise UnsupportedEventVersionError for a dict with a bad version."""
    if isinstance(event, dict) and not is_supported_event_version(
        event.get("event_version")
    ):
        raise UnsupportedEventVersionError(event.get("event_version"))
