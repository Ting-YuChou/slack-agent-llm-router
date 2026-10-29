"""Guards applied at the public HTTP boundary of the router.

The Slack bot calls ``InferenceEngine.process_query`` in-process and builds
server-side metadata itself, so nothing here runs on that path.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from src.utils.schema import QueryRequest, UserTier

Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Dict[str, Any], Receive, Send], Awaitable[None]]

# Metadata keys the server writes for itself: retrieved context and style
# instructions go straight into the prompt, and the Slack and routing keys
# describe where the request came from. External callers must not set them.
SERVER_ONLY_METADATA_KEYS = frozenset(
    {
        "rag_context",
        "web_search_context",
        "response_style_instructions",
        "routing_intent_source",
        "slack",
    }
)
SERVER_ONLY_METADATA_PREFIXES: Tuple[str, ...] = ("slack_",)


# Metadata that moves a request up the provider queue or into the fast lane.
URGENCY_METADATA_KEYS: Tuple[str, ...] = (
    "requires_low_latency",
    "route_to_fast_lane",
    "latency_sla",
    "fast_lane",
)
_TIER_RANK = {UserTier.FREE: 0, UserTier.PREMIUM: 1, UserTier.ENTERPRISE: 2}
UNMAPPED_KEY_FINGERPRINT_CHARS = 12


@dataclass(frozen=True)
class ApiPrincipal:
    """The authenticated caller of the HTTP API."""

    client_id: str
    max_tier: str = UserTier.FREE.value
    rpm: int = 0
    burst: int = 0
    active: int = 0
    allow_priority: bool = False
    mapped: bool = False


def api_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _same_secret(provided: str, expected: str) -> bool:
    # Compare bytes: compare_digest rejects non-ASCII str arguments.
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _client_key_matches(
    client: Dict[str, Any],
    provided_key: str,
    fingerprint: str,
    getenv: Callable[..., Optional[str]],
) -> bool:
    key_env = client.get("key_env")
    if key_env:
        expected = (getenv(str(key_env)) or "").strip()
        return bool(expected) and _same_secret(provided_key, expected)
    expected_digest = str(client.get("sha256") or "").lower()
    return bool(expected_digest) and secrets.compare_digest(
        fingerprint, expected_digest
    )


def resolve_api_principal(
    provided_key: Optional[str],
    api_keys_config: Dict[str, Any],
    anonymous_keys: Sequence[str],
    *,
    getenv: Callable[..., Optional[str]] = os.getenv,
) -> Optional[ApiPrincipal]:
    """Map a presented API key to its principal, or None when it is not valid.

    Keys mapped under ``api_keys.clients`` get their configured entitlements.
    Any other accepted key (``api_keys.env_var``) is an unmapped principal named
    after its fingerprint, limited to the free tier and default priority.
    """
    if not provided_key:
        return None
    fingerprint = api_key_fingerprint(provided_key)
    for client in api_keys_config.get("clients") or []:
        if _client_key_matches(client, provided_key, fingerprint, getenv):
            rpm = int(client.get("rpm") or 0)
            return ApiPrincipal(
                client_id=str(client["client_id"]),
                max_tier=str(client.get("max_tier") or UserTier.FREE.value),
                rpm=rpm,
                burst=int(client.get("burst") or rpm),
                active=int(client.get("active") or 0),
                allow_priority=bool(client.get("allow_priority", False)),
                mapped=True,
            )
    for expected_key in anonymous_keys:
        if _same_secret(provided_key, expected_key):
            rpm = int(api_keys_config.get("unmapped_rpm") or 0)
            return ApiPrincipal(
                client_id=f"key:{fingerprint[:UNMAPPED_KEY_FINGERPRINT_CHARS]}",
                rpm=rpm,
                burst=rpm,
                active=int(api_keys_config.get("unmapped_active") or 0),
            )
    return None


def api_key_clients_configured(
    api_keys_config: Dict[str, Any],
    *,
    getenv: Callable[..., Optional[str]] = os.getenv,
) -> bool:
    """Return whether any mapped client can currently authenticate."""
    for client in api_keys_config.get("clients") or []:
        if client.get("sha256"):
            return True
        key_env = client.get("key_env")
        if key_env and (getenv(str(key_env)) or "").strip():
            return True
    return False


def apply_principal_entitlements(
    request: QueryRequest, principal: ApiPrincipal
) -> List[str]:
    """Clamp what the request asks for to what the principal may have.

    Returns a description of each change for logging.
    """
    changes: List[str] = []
    max_tier = UserTier(principal.max_tier)
    if _TIER_RANK[UserTier(request.user_tier)] > _TIER_RANK[max_tier]:
        changes.append(
            f"user_tier {UserTier(request.user_tier).value}->{max_tier.value}"
        )
        request.user_tier = max_tier
    if principal.allow_priority:
        return changes
    default_priority = QueryRequest.model_fields["priority"].default
    if request.priority != default_priority:
        changes.append(f"priority {request.priority}->{default_priority}")
        request.priority = default_priority
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    for key in URGENCY_METADATA_KEYS:
        if key in metadata:
            metadata.pop(key)
            changes.append(f"metadata.{key} dropped")
    return changes


def is_server_only_metadata_key(key: Any) -> bool:
    name = str(key)
    return name in SERVER_ONLY_METADATA_KEYS or name.startswith(
        SERVER_ONLY_METADATA_PREFIXES
    )


def strip_server_only_metadata(request: Any) -> List[str]:
    """Remove server-only metadata keys in place; return the removed keys."""
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    removed = sorted(str(key) for key in metadata if is_server_only_metadata_key(key))
    for key in removed:
        metadata.pop(key, None)
    return removed


class AttachmentRejected(ValueError):
    def __init__(self, *, status_code: int, error: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message


def validate_external_attachments(request: Any, *, max_attachment_bytes: int) -> None:
    """Reject attachments whose size is inconsistent or above the limit."""
    for index, attachment in enumerate(getattr(request, "attachments", None) or []):
        content = getattr(attachment, "content", None)
        size_bytes = int(getattr(attachment, "size_bytes", 0) or 0)
        actual = len(content) if content is not None else None
        if max(size_bytes, actual or 0) > max_attachment_bytes:
            raise AttachmentRejected(
                status_code=413,
                error="attachment_too_large",
                message=(
                    f"Attachment {index} exceeds the {max_attachment_bytes}-byte "
                    "limit"
                ),
            )
        if actual is not None and actual != size_bytes:
            raise AttachmentRejected(
                status_code=422,
                error="invalid_attachment",
                message=(
                    f"Attachment {index} declares size_bytes={size_bytes} but "
                    f"carries {actual} bytes"
                ),
            )


class RequestBodyLimitMiddleware:
    """Pure ASGI middleware that caps request bodies before anything parses them.

    A declared Content-Length above the limit is rejected without reading the
    body. Otherwise the body is read up to the limit (this also covers chunked
    bodies without Content-Length) and replayed to the app.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        exempt_paths: Iterable[str] = (),
    ):
        self.app = app
        self.max_body_bytes = int(max_body_bytes)
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(
        self, scope: Dict[str, Any], receive: Receive, send: Send
    ) -> None:
        if (
            scope.get("type") != "http"
            or self.max_body_bytes <= 0
            or scope.get("path") in self.exempt_paths
        ):
            await self.app(scope, receive, send)
            return

        declared = _declared_content_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            await self._reject(send)
            return

        chunks: List[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                return  # the client went away before sending the whole body
            body = message.get("body", b"") or b""
            received += len(body)
            if received > self.max_body_bytes:
                await self._reject(send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        buffered: Optional[Message] = {
            "type": "http.request",
            "body": b"".join(chunks),
            "more_body": False,
        }

        async def replay() -> Message:
            nonlocal buffered
            if buffered is not None:
                message, buffered = buffered, None
                return message
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, send: Send) -> None:
        payload = json.dumps(
            {
                "error": "request_too_large",
                "message": (
                    f"Request body exceeds the {self.max_body_bytes}-byte limit"
                ),
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _declared_content_length(scope: Dict[str, Any]) -> Optional[int]:
    for name, value in scope.get("headers") or []:
        if name.lower() == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None
