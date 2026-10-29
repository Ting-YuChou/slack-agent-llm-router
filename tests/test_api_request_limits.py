"""Field limits on the public chat request schema (R14)."""

import pytest
from pydantic import ValidationError

from src.api_security import RequestBodyLimitMiddleware, rag_request_body_limits
from src.utils.schema import (
    Attachment,
    AttachmentType,
    PlatformConfig,
    QueryRequest,
)


@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("user_id", 256),
        ("request_id", 128),
        ("session_id", 256),
        ("conversation_id", 256),
    ],
)
def test_query_request_id_fields_are_length_bounded(field, limit):
    QueryRequest(query="hi", **{"user_id": "u1", field: "x" * limit})

    with pytest.raises(ValidationError):
        QueryRequest(query="hi", **{"user_id": "u1", field: "x" * (limit + 1)})


@pytest.mark.parametrize("field", ["allowed_tools", "knowledge_base_ids"])
def test_query_request_list_fields_are_capped(field):
    QueryRequest(query="hi", user_id="u1", **{field: [f"item-{i}" for i in range(32)]})

    with pytest.raises(ValidationError):
        QueryRequest(
            query="hi", user_id="u1", **{field: [f"item-{i}" for i in range(33)]}
        )
    with pytest.raises(ValidationError):
        QueryRequest(query="hi", user_id="u1", **{field: ["x" * 257]})


@pytest.mark.parametrize(
    ("field", "limit"),
    [("id", 128), ("name", 1024), ("mime_type", 255), ("url", 4096)],
)
def test_attachment_string_fields_are_length_bounded(field, limit):
    base = {
        "name": "notes.txt",
        "type": AttachmentType.DOCUMENT,
        "size_bytes": 3,
        "mime_type": "text/plain",
    }
    Attachment(**{**base, field: "x" * limit})

    with pytest.raises(ValidationError):
        Attachment(**{**base, field: "x" * (limit + 1)})


def test_slack_shaped_request_fits_the_limits():
    request = QueryRequest(
        query="hello",
        user_id="U0123456789",
        session_id="5f0c8f7e-2f7b-4f0a-9d7c-0c2b9d1d1e11",
        conversation_id="U0123456789:C0123456789:1712345678.123456",
        metadata={"rag_context": "server-built context", "slack_context_source": "t"},
    )

    # In-process callers (the Slack bot) keep their server-built metadata.
    assert request.metadata["rag_context"] == "server-built context"


def test_attachment_size_is_not_checked_against_content_in_the_schema():
    # The /route boundary rejects mismatches; in-process callers may build them.
    attachment = Attachment(
        name="report.csv",
        type=AttachmentType.DOCUMENT,
        size_bytes=11,
        mime_type="text/csv",
        content=b"a,b\n1,2\n",
    )

    assert attachment.size_bytes == 11


def test_security_config_defaults_cap_request_bodies():
    security = PlatformConfig.model_validate({}).security

    assert security.max_request_body_bytes == 2 * 1024 * 1024
    assert security.max_attachment_bytes == 1024 * 1024
    assert security.request_body_limits == {}
    assert not hasattr(security, "request_body_limit_exempt_paths")


@pytest.mark.parametrize(
    "limits",
    [{"rag/batches": {"*": 10}}, {"/rag/batches": {"application/json": 0}}],
)
def test_request_body_limits_reject_bad_entries(limits):
    with pytest.raises(ValidationError):
        PlatformConfig.model_validate({"security": {"request_body_limits": limits}})


def test_request_body_limits_normalize_media_types():
    security = PlatformConfig.model_validate(
        {"security": {"request_body_limits": {"/x": {" Application/JSON ": 10}}}}
    ).security

    assert security.request_body_limits == {"/x": {"application/json": 10}}


def test_rag_body_limits_derive_from_the_rag_upload_limits():
    limits = rag_request_body_limits(
        {"json_max_decoded_bytes": 3000, "multipart_max_bytes": 500_000}
    )

    assert limits == {
        "/rag/documents": {
            "application/json": 4000 + 64 * 1024,
            "multipart/form-data": 500_000 + 1024 * 1024,
        },
        "/rag/batches": {"application/json": 4000 + 1024 * 1024},
    }


def test_rag_body_limits_use_the_rag_defaults():
    limits = rag_request_body_limits({})

    # base64 of a 10 MB document is 13_333_336 bytes.
    assert limits["/rag/documents"]["application/json"] == 13_333_336 + 64 * 1024
    assert limits["/rag/documents"]["multipart/form-data"] == (
        100_000_000 + 1024 * 1024
    )


async def _status_for(middleware, path, body, content_type):
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [
            (b"content-type", content_type.encode()),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    await middleware(scope, receive, send)
    return sent[0]["status"]


async def _ok_app(scope, receive, send):
    while (await receive()).get("more_body"):
        pass
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "content_type", "size", "status"),
    [
        ("/route", "application/json", 100, 200),
        ("/route", "application/json", 101, 413),
        ("/upload", "application/json; charset=utf-8", 300, 200),
        ("/upload", "application/json; charset=utf-8", 301, 413),
        ("/upload", "MULTIPART/FORM-DATA; boundary=x", 500, 200),
        ("/upload", "multipart/form-data; boundary=x", 501, 413),
        ("/upload", "text/plain", 101, 413),
        ("/any-type", "text/plain", 700, 200),
    ],
)
async def test_body_limit_is_chosen_by_path_and_media_type(
    path, content_type, size, status
):
    middleware = RequestBodyLimitMiddleware(
        _ok_app,
        max_body_bytes=100,
        path_limits={
            "/upload": {"application/json": 300, "multipart/form-data": 500},
            "/any-type": {"*": 700},
        },
    )

    assert await _status_for(middleware, path, b"x" * size, content_type) == status
