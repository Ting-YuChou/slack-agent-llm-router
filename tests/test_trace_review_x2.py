"""Round-1 review fixes for X2: chat_requests key, Slack metrics, row
timestamps, Slack file names."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part3_pipeline import (
    ClickHouseManager,
    build_chat_feedback_row,
    build_chat_request_row,
    build_request_raw_event,
)
from src.utils.schema import Attachment, AttachmentType, QueryRequest

from test_main import (  # noqa: F401
    DummyEventProducer,
    _admission_platform,
    patched_platform_deps,
)


# chat_requests keeps distinct calls that reuse an id


@pytest.mark.asyncio
async def test_chat_requests_orders_by_ts_after_the_request_id():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    statements = [call.args[1] for call in manager._run_blocking.await_args_list]
    (ddl,) = [sql for sql in statements if "llm_router.chat_requests (" in sql]
    assert "ORDER BY (toStartOfHour(ts), request_id, ts)" in ddl


# Slack has its own metric labels


@pytest.mark.asyncio
async def test_slack_metrics_are_not_route_traffic(tmp_path, patched_platform_deps):
    platform = _admission_platform(tmp_path)
    producer = DummyEventProducer({})
    platform.services["event_producer"] = producer
    inference = platform.services["inference"]
    inference.response = inference.response.model_copy(
        update={"error": "inference_failed", "error_code": "timeout"}
    )
    request = QueryRequest(
        query="hi", user_id="U1", metadata={"source": "slack", "surface": "thread"}
    )

    await platform.chat_ingress.handle(request, source="slack")

    for event in producer.metric_events:
        labels = event["labels"]
        assert labels["endpoint"] == "slack"
        assert labels["method"] not in {"thread", "mention", "slash", "dm"}
        assert labels["method"] not in {"GET", "POST", "PUT", "DELETE"}
    errors = [e for e in producer.metric_events if e["metric_name"] == "errors_total"]
    assert errors
    assert all(e["labels"]["error_type"] != "RouteError" for e in errors)
    assert all(e["labels"]["component"] != "api" for e in errors)


# Row timestamps come from the event


def _completed_event(**fields):
    event = {
        "event_type": "inference.completed",
        "request_id": "req-1",
        "selected_model": "gpt-5",
        "provider": "openai",
        "status": "success",
    }
    event.update(fields)
    return event


def test_a_chat_request_row_falls_back_to_the_completion_timestamp():
    event = _completed_event(
        request_timestamp="not a time",
        completion_timestamp="2026-10-01T12:00:00Z",
    )

    first = build_chat_request_row(event)
    second = build_chat_request_row(event)

    assert first["ts"] == second["ts"]
    assert first["ts"].isoformat().startswith("2026-10-01T12:00:00")
    assert first["version"] == second["version"]


def test_a_chat_request_row_without_any_timestamp_is_rejected():
    with pytest.raises(ValueError):
        build_chat_request_row(_completed_event())


def test_a_chat_feedback_row_without_a_timestamp_is_rejected():
    with pytest.raises(ValueError):
        build_chat_feedback_row({"request_id": "req-1", "rating": "up"})


# Slack file names stay out of requests.raw


def test_slack_attachment_names_stay_out_of_requests_raw():
    request = QueryRequest(
        query="summarize",
        user_id="U1",
        metadata={
            "source": "slack",
            "surface": "mention",
            "slack": {
                "channel_id": "C1",
                "attachment_count": 1,
                "attachment_names": ["salary-review-jane.pdf"],
            },
        },
        attachments=[
            Attachment(
                name="salary-review-jane.pdf",
                type=AttachmentType.DOCUMENT,
                size_bytes=11,
                mime_type="application/pdf",
                content=b"file detail",
            )
        ],
    )

    event = build_request_raw_event(request)

    assert "salary-review-jane" not in str(event)
    (attachment,) = event["attachments"]
    assert attachment["size_bytes"] == 11
    assert attachment["type"] == "document"
    assert attachment["mime_type"] == "application/pdf"
    assert event["attachments_count"] == 1
    assert event["metadata"]["slack"]["attachment_count"] == 1
