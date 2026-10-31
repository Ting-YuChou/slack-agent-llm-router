"""The query log row that produce_query_log publishes."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm_router_part3_pipeline import KafkaProducerManager
from src.utils.schema import QueryRequest


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
