"""Field limits on the public chat request schema (R14)."""

import pytest
from pydantic import ValidationError

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
    assert "/route" not in security.request_body_limit_exempt_paths
    assert "/rag/documents" in security.request_body_limit_exempt_paths
