"""T2: a Slack chat query carries how long it waited in the bot's queue."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot, SlackWorkItem
from src.api_security import CLIENT_METADATA_KEYS


def chat_bot(process_query):
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=SimpleNamespace(process_query=process_query),
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "999.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    return bot


def mention(ts="100.0", text="<@UBOT> first question"):
    return {
        "type": "app_mention",
        "channel": "C1",
        "user": "U1",
        "text": text,
        "ts": ts,
    }


def thread_reply(ts="101.0", text="a follow-up"):
    return {
        "type": "message",
        "channel": "C1",
        "user": "U1",
        "text": text,
        "thread_ts": "100.0",
        "ts": ts,
    }


@pytest.fixture
def clock(monkeypatch):
    """Moves the bot's monotonic clock forward; asyncio's moves with it."""
    real = time.monotonic
    offset = {"seconds": 0.0}
    monkeypatch.setattr(
        "slack.bot_real.time.monotonic", lambda: real() + offset["seconds"]
    )
    return offset


def waits(bot):
    return [
        call.args[0].metadata.get("ingress_queue_wait_ms")
        for call in bot.inference_engine.process_query.await_args_list
    ]


@pytest.mark.asyncio
async def test_a_queued_query_carries_its_wait_as_internal_metadata(
    inference_response_factory, clock
):
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok"))
    )
    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=mention(), name="mention")
    )
    clock["seconds"] += 1.5

    await bot._handle_chat_work(bot._work_queue.get_nowait())

    (wait,) = waits(bot)
    assert isinstance(wait, int) and not isinstance(wait, bool)
    assert 1500 <= wait < 2500
    assert "ingress_queue_wait_ms" not in CLIENT_METADATA_KEYS


@pytest.mark.asyncio
async def test_a_parked_query_counts_the_time_it_spent_parked(
    inference_response_factory, clock
):
    release = asyncio.Event()

    async def process_query(request):
        if "first" in request.query:
            await release.wait()
        return inference_response_factory(response_text="ok")

    bot = chat_bot(AsyncMock(side_effect=process_query))
    bot._mark_thread_active("C1", "100.0")
    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=mention(), name="mention")
    )
    owner = asyncio.create_task(bot._handle_chat_work(bot._work_queue.get_nowait()))
    while not bot.inference_engine.process_query.await_count:
        await asyncio.sleep(0)

    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=thread_reply(), name="message")
    )
    await bot._handle_chat_work(bot._work_queue.get_nowait())
    assert bot._thread_pending == 1
    clock["seconds"] += 4.0
    release.set()
    await asyncio.wait_for(owner, timeout=1)

    first, second = waits(bot)
    assert 0 <= first < 1000
    assert 4000 <= second < 5000


@pytest.mark.asyncio
async def test_an_item_that_was_never_queued_carries_no_wait(
    inference_response_factory,
):
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok"))
    )

    await bot._handle_chat_work(SlackWorkItem("event", mention(), "mention"))

    assert waits(bot) == [None]


@pytest.mark.asyncio
async def test_an_event_cannot_supply_its_own_wait(inference_response_factory):
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok"))
    )
    event = dict(mention(), _ingress_queue_wait_ms=99)

    await bot._handle_chat_work(SlackWorkItem("event", event, "mention"))

    assert waits(bot) == [None]


@pytest.mark.asyncio
async def test_a_queued_slash_query_carries_its_wait(inference_response_factory, clock):
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok"))
    )
    bot._post_command_response = AsyncMock()
    command = {
        "text": "what is 2+2?",
        "user_id": "U1",
        "channel_id": "C1",
        "trigger_id": "T123",
    }
    assert await bot._enqueue_work(
        SlackWorkItem(kind="slash", payload=command, name="slack_slash_command")
    )
    clock["seconds"] += 0.25
    bot._start_work_workers()
    try:
        await asyncio.wait_for(bot._work_queue.join(), timeout=1)
    finally:
        await bot._stop_work_workers()

    (wait,) = waits(bot)
    assert 250 <= wait < 1250
