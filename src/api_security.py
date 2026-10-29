"""Guards applied at the public HTTP boundary of the router.

The Slack bot calls ``InferenceEngine.process_query`` in-process and builds
server-side metadata itself, so nothing here runs on that path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
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

# Metadata that moves a request up the provider queue or into the fast lane.
# Clients may send it, but only principals with allow_priority keep it.
URGENCY_METADATA_KEYS: Tuple[str, ...] = (
    "requires_low_latency",
    "route_to_fast_lane",
    "latency_sla",
    "fast_lane",
)

# The only metadata keys an external POST /route caller may set; every other
# key is dropped. Everything the server reads from request.metadata is either
# listed here or written by the server itself (retrieved context, prompt style
# instructions, cache bypass flags, RAG fallback models and warnings, derived
# routing features, session counters, Slack and workspace keys), so a caller
# cannot forge it. Add a key here only if a client may legitimately choose it.
CLIENT_METADATA_KEYS = frozenset(
    {
        # Routing preferences, still bounded by the tier's allowed models.
        "preferred_models",
        "avoid_models",
        "avoid_providers",
        "cost_sensitivity",
        "error_sensitivity",
        "requires_high_reasoning",
        # Tool and retrieval toggles, like tool_policy and rag_policy.
        "enable_web_search",
        "web_search_auto",
        "enable_rag",
        "rag_auto",
        *URGENCY_METADATA_KEYS,
    }
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


def filter_external_metadata(request: Any) -> List[str]:
    """Keep only client-settable metadata keys; return the dropped keys."""
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    dropped = sorted(str(key) for key in metadata if key not in CLIENT_METADATA_KEYS)
    for key in dropped:
        metadata.pop(key, None)
    return dropped


class AttachmentRejected(ValueError):
    def __init__(self, *, status_code: int, error: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message


def normalize_external_attachments(request: Any, *, max_attachment_bytes: int) -> None:
    """Cap inline attachment content and make size_bytes match it.

    The cap applies to the content actually sent. A caller's size_bytes is
    only a declaration (JSON carries the content as text, so it may count
    characters or the original file size): when content is present it is
    replaced by the real length so downstream sees one consistent value.
    URL-only attachments carry no content and keep their declared size.
    """
    for index, attachment in enumerate(getattr(request, "attachments", None) or []):
        content = getattr(attachment, "content", None)
        if content is None:
            continue
        if len(content) > max_attachment_bytes:
            raise AttachmentRejected(
                status_code=413,
                error="attachment_too_large",
                message=(
                    f"Attachment {index} exceeds the {max_attachment_bytes}-byte "
                    "limit"
                ),
            )
        attachment.size_bytes = len(content)


# JSON fields around an inline document (filename, metadata, ids).
RAG_JSON_DOCUMENT_OVERHEAD_BYTES = 64 * 1024
# JSON encoders may legally escape base64's "+" (.NET: +, +5 bytes) and
# "/" (PHP: \/, +1 byte). Each is 1/64 of the characters of base64 of arbitrary
# bytes, so escaped text runs about 9.4% longer; allow 10%.
RAG_JSON_ESCAPING_PERCENT = 110
# Multipart boundaries, part headers and the other form fields.
RAG_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
# A batch carries one inline document at most this large plus up to 1000
# document entries; larger sets go through presigned /rag/uploads.
RAG_BATCH_OVERHEAD_BYTES = 1024 * 1024
RAG_BODY_LIMIT_HINTS = {
    "/rag/documents": (
        "Upload larger documents with POST /rag/uploads (presigned upload)."
    ),
    "/rag/batches": (
        "Upload large or many documents with POST /rag/uploads (presigned "
        "upload) and reference them by storage_ref in the batch."
    ),
}


def _base64_length(decoded_bytes: int) -> int:
    return 4 * math.ceil(decoded_bytes / 3)


def _escaped_json_base64_length(decoded_bytes: int) -> int:
    encoded = _base64_length(decoded_bytes)
    return -(-encoded * RAG_JSON_ESCAPING_PERCENT // 100)


def rag_request_body_limits(upload_config: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Body caps for the RAG ingestion paths, sized from their own limits."""
    json_document = _escaped_json_base64_length(
        int(upload_config.get("json_max_decoded_bytes") or 10_000_000)
    )
    multipart_document = int(upload_config.get("multipart_max_bytes") or 100_000_000)
    return {
        "/rag/documents": {
            "application/json": json_document + RAG_JSON_DOCUMENT_OVERHEAD_BYTES,
            "multipart/form-data": multipart_document + RAG_MULTIPART_OVERHEAD_BYTES,
        },
        "/rag/batches": {"application/json": json_document + RAG_BATCH_OVERHEAD_BYTES},
    }


class RequestBodyLimitMiddleware:
    """Pure ASGI middleware that caps request bodies before anything parses them.

    The cap is ``path_limits[path][media type]`` (``"*"`` matches any media
    type) and otherwise ``max_body_bytes``. A declared Content-Length above the
    cap is rejected without reading the body. Otherwise bytes are counted as
    the app reads them (this also covers chunked bodies without
    Content-Length): past the cap the client gets 413, the app sees a
    disconnect and its own response is dropped.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        path_limits: Optional[Dict[str, Dict[str, int]]] = None,
        hints: Optional[Dict[str, str]] = None,
    ):
        self.app = app
        self.max_body_bytes = int(max_body_bytes)
        self.path_limits = {
            path: {str(media).lower(): int(limit) for media, limit in limits.items()}
            for path, limits in (path_limits or {}).items()
        }
        # Per-path advice appended to the 413 message.
        self.hints = dict(hints or {})

    def limit_for(self, scope: Dict[str, Any]) -> int:
        by_media_type = self.path_limits.get(str(scope.get("path") or ""))
        if not by_media_type:
            return self.max_body_bytes
        media_type = _media_type(scope)
        if media_type in by_media_type:
            return by_media_type[media_type]
        return by_media_type.get("*", self.max_body_bytes)

    async def __call__(
        self, scope: Dict[str, Any], receive: Receive, send: Send
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope)

        hint = self.hints.get(str(scope.get("path") or ""))
        declared = _declared_content_length(scope)
        if declared is not None and declared > limit:
            await self._reject(send, limit, hint)
            return

        # Count the body as the app pulls it instead of reading it up front:
        # a request that authentication rejects is never read at all.
        received = 0
        overflowed = False
        rejected = False
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received, overflowed, rejected, response_started
            if overflowed:
                return {"type": "http.disconnect"}
            message = await receive()
            if message.get("type") != "http.request":
                return message
            received += len(message.get("body", b"") or b"")
            if received <= limit:
                return message
            overflowed = True
            if not response_started:
                response_started = rejected = True
                await self._reject(send, limit, hint)
            # Stop the app reading; whatever it answers is dropped below.
            return {"type": "http.disconnect"}

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

    async def _reject(self, send: Send, limit: int, hint: Optional[str]) -> None:
        message = f"Request body exceeds the {limit}-byte limit"
        if hint:
            message = f"{message}. {hint}"
        payload = json.dumps({"error": "request_too_large", "message": message}).encode(
            "utf-8"
        )
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


def parse_media_type(content_type: Optional[str]) -> str:
    """Return the media type of a Content-Type value, without parameters.

    Use this wherever a body parser is chosen, so it agrees with the cap the
    body-limit middleware applied.
    """
    return (content_type or "").split(";", 1)[0].strip().lower()


def _media_type(scope: Dict[str, Any]) -> str:
    for name, value in scope.get("headers") or []:
        if name.lower() == b"content-type":
            return parse_media_type(value.decode("latin-1"))
    return ""


def _declared_content_length(scope: Dict[str, Any]) -> Optional[int]:
    for name, value in scope.get("headers") or []:
        if name.lower() == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None
