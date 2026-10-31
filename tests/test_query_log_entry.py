"""The query log row that produce_query_log publishes."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm_router_part3_pipeline import KafkaProducerManager
from src.utils.schema import Attachment, AttachmentType, QueryRequest


def _producer():
    producer = KafkaProducerManager.__new__(KafkaProducerManager)
    producer._enqueue_message = MagicMock()
    return producer


def _response(**overrides):
    values = dict(
        timestamp=datetime.now(timezone.utc),
        model_name="m",
        token_count_input=1,
        token_count_output=2,
        latency_ms=3,
        cost_usd=0.0,
        error=None,
        cached=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


async def _published_row(request, response):
    producer = _producer()
    await producer.produce_query_log(request, response)
    return producer._enqueue_message.call_args.kwargs["value"]


@pytest.mark.asyncio
async def test_compression_ratio_is_no_longer_made_up():
    request = QueryRequest(query="hi", user_id="U1")

    row = await _published_row(request, _response(compressed_context=True))

    assert row["compression_ratio"] == 0.0


# T4: a Slack file's name can say what is in it (X6).


def _slack_request(source="slack"):
    return QueryRequest(
        query="what is in quarterly-layoffs-plan.pdf?",
        user_id="U1",
        metadata={"source": source},
        attachments=[
            Attachment(
                type=AttachmentType.DOCUMENT,
                name="quarterly-layoffs-plan.pdf",
                size_bytes=10,
                mime_type="application/pdf",
            )
        ],
    )


@pytest.mark.asyncio
async def test_slack_query_log_strips_attachment_names():
    row = await _published_row(_slack_request(), _response())

    assert "quarterly-layoffs-plan" not in str(row)
    assert row["query_text"] == "what is in [attachment]?"


@pytest.mark.asyncio
async def test_api_query_log_keeps_its_existing_text():
    row = await _published_row(_slack_request(source="api"), _response())

    assert row["query_text"] == "what is in quarterly-layoffs-plan.pdf?"
