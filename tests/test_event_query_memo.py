"""CH4: the redacted query that events carry is computed once per request."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.llm_router_part3_pipeline as pipeline
from src.llm_router_part3_pipeline import (
    KafkaProducerManager,
    QueryLogEntry,
    _event_query_text,
    build_inference_completed_event,
    build_request_raw_event,
)
from src.redaction import redact_text
from src.utils.schema import (
    Attachment,
    AttachmentType,
    InferenceResponse,
    QueryRequest,
)

OPENAI_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
SECRET_QUERY = f"why does {OPENAI_KEY} fail?"


@pytest.fixture
def redactions(monkeypatch):
    """Every text the pipeline module itself hands to redact_text."""
    calls = []

    def counting(text):
        calls.append(text)
        return redact_text(text)

    monkeypatch.setattr(pipeline, "redact_text", counting)
    return calls


def _response():
    return InferenceResponse(
        response_text="world",
        model_name="gpt-5",
        provider="openai",
        token_count_input=3,
        token_count_output=5,
        total_tokens=8,
        latency_ms=12,
        tokens_per_second=120.0,
        cost_usd=0.01,
    )


def _producer():
    producer = KafkaProducerManager.__new__(KafkaProducerManager)
    producer._enqueue_message = MagicMock()
    return producer


async def _published_row(request):
    producer = _producer()
    await producer.produce_query_log(request, _response(), SimpleNamespace())
    return producer._enqueue_message.call_args.kwargs["value"]


def _attachment(name):
    return Attachment(
        type=AttachmentType.DOCUMENT,
        name=name,
        mime_type="application/pdf",
        size_bytes=10,
    )


def _entry(query_text):
    return QueryLogEntry(
        query_id="q1",
        timestamp=datetime.now(timezone.utc),
        user_id="u1",
        user_tier="free",
        query_text=query_text,
        query_type="general",
        selected_model="gpt-5",
        token_count_input=1,
        token_count_output=1,
        latency_ms=1,
        cost_usd=0.0,
        status="success",
    )


@pytest.mark.asyncio
async def test_the_three_events_redact_a_requests_query_once(redactions):
    request = QueryRequest(query=SECRET_QUERY, user_id="u1")

    raw = build_request_raw_event(request)
    completed = build_inference_completed_event(request, _response())
    row = await _published_row(request)

    assert redactions == [SECRET_QUERY]
    expected = redact_text(SECRET_QUERY)
    assert OPENAI_KEY not in expected
    assert raw["query_text"] == expected
    assert completed["query_text"] == expected
    assert row["query_text"] == expected


def test_a_changed_query_source_or_attachment_name_is_redacted_again(redactions):
    request = QueryRequest(
        query="what is in plan.pdf and notes.pdf?",
        user_id="u1",
        metadata={"source": "slack"},
        attachments=[_attachment("plan.pdf")],
    )

    first = _event_query_text(request)
    assert first == "what is in [attachment] and notes.pdf?"

    # The copies share the request's memo; each changed input must miss it.
    other_query = request.model_copy(update={"query": "is plan.pdf final?"})
    assert _event_query_text(other_query) == "is [attachment] final?"

    other_source = request.model_copy(update={"metadata": {"source": "api"}})
    assert _event_query_text(other_source) == "what is in plan.pdf and notes.pdf?"

    other_names = request.model_copy(
        update={"attachments": [_attachment("plan.pdf"), _attachment("notes.pdf")]}
    )
    assert _event_query_text(other_names) == "what is in [attachment] and [attachment]?"

    assert len(redactions) == 4
    assert _event_query_text(request) == first
    assert len(redactions) == 4


def test_a_copy_with_only_max_tokens_changed_reuses_the_text(redactions):
    request = QueryRequest(query=SECRET_QUERY, user_id="u1")
    first = _event_query_text(request)

    capped = request.model_copy(update={"max_tokens": 64})

    assert _event_query_text(capped) == first
    assert redactions == [SECRET_QUERY]


@pytest.mark.asyncio
async def test_the_query_log_row_is_still_redacted_and_cut_to_10000_characters(
    redactions,
):
    query = f"{SECRET_QUERY} " + "x" * 12000
    request = QueryRequest(query=query, user_id="u1")

    row = await _published_row(request)

    assert row["query_text"] == redact_text(query)[:10000]
    assert len(row["query_text"]) == 10000
    assert OPENAI_KEY not in row["query_text"]
    assert redactions == [query]


def test_an_entry_marked_redacted_is_not_redacted_again(redactions):
    redacted = redact_text(SECRET_QUERY)
    entry = _entry(redacted)

    marked = entry.to_dict(query_text_redacted=True)

    assert redactions == []
    assert marked == entry.to_dict()
    assert "query_text_redacted" not in marked


def test_an_unmarked_entry_is_still_redacted(redactions):
    row = _entry(SECRET_QUERY).to_dict()

    assert redactions == [SECRET_QUERY]
    assert OPENAI_KEY not in row["query_text"]


def test_an_object_without_a_memo_still_gets_redacted_text(redactions):
    request = SimpleNamespace(query=SECRET_QUERY, metadata=None, attachments=[])

    assert _event_query_text(request) == redact_text(SECRET_QUERY)
    assert _event_query_text(request) == redact_text(SECRET_QUERY)
    assert redactions == [SECRET_QUERY, SECRET_QUERY]


# --- one pass must be final: the query log no longer gets a second one ----------

# Inputs where masking one credential used to uncover the next one, so a
# single pass left a secret that only a second pass would have masked.
CHAINED_SECRETS = [
    ("token: --password --password s3cr3t!9", "s3cr3t!9"),
    ("set API_KEY= --token --token abc123XYZ789abc", "abc123XYZ789abc"),
]


@pytest.mark.parametrize("text, secret", CHAINED_SECRETS)
def test_one_redaction_pass_leaves_nothing_for_a_second(text, secret):
    once = redact_text(text)

    assert secret not in once
    assert redact_text(once) == once


@pytest.mark.parametrize("text, secret", CHAINED_SECRETS)
def test_the_query_log_message_carries_no_chained_secret(text, secret):
    request = QueryRequest(query=text, user_id="u1")

    assert secret not in _event_query_text(request)
