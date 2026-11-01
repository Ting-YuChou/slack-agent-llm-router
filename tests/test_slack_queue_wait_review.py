"""T2 review: the queue wait travels with the work, not on the event."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot, SlackQueryTrace, SlackWorkItem


def _bot(process_query=None):
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=SimpleNamespace(process_query=process_query or AsyncMock()),
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "999.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    return bot


def _mention():
    return {
        "type": "app_mention",
        "channel": "C1",
        "user": "U1",
        "text": "<@UBOT> first question",
        "ts": "100.0",
    }


@pytest.fixture
def clock(monkeypatch):
    real = time.monotonic
    offset = {"seconds": 0.0}
    monkeypatch.setattr(
        "slack.bot_real.time.monotonic", lambda: real() + offset["seconds"]
    )
    return offset


@pytest.mark.asyncio
async def test_the_event_payload_is_left_as_slack_sent_it(
    inference_response_factory, clock
):
    bot = _bot(AsyncMock(return_value=inference_response_factory(response_text="ok")))
    event = _mention()
    assert await bot._enqueue_work(SlackWorkItem("event", event, "mention"))
    clock["seconds"] += 0.5

    await bot._handle_chat_work(bot._work_queue.get_nowait())

    request = bot.inference_engine.process_query.await_args.args[0]
    assert 500 <= request.metadata["ingress_queue_wait_ms"] < 1500
    assert "_ingress_queue_wait_ms" not in event


@pytest.mark.asyncio
async def test_an_agent_request_sees_its_queue_wait_too(clock):
    bot = _bot()
    seen = []

    async def handle_event(event):
        seen.append(SlackQueryTrace.for_message(event, team_id=None).queue_wait_ms)

    bot._is_agent_request = lambda event: True
    bot._handle_event = handle_event
    assert await bot._enqueue_work(SlackWorkItem("event", _mention(), "mention"))
    clock["seconds"] += 0.25
    bot._start_work_workers()
    try:
        await asyncio.wait_for(bot._work_queue.join(), timeout=1)
        await asyncio.gather(*list(bot.background_tasks))
    finally:
        await bot._stop_work_workers()

    (wait,) = seen
    assert 250 <= wait < 1250
