"""T3 review: per-send confirmation, unconfirmed clicks, off-worker waits."""

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import (
    CHAT_FEEDBACK_NOT_RECORDED_TEXT,
    CHAT_FEEDBACK_THANKS_TEXT,
    CHAT_FEEDBACK_UNCONFIRMED_TEXT,
    SlackBot,
)
from src.llm_router_part3_pipeline import KafkaProducerManager

REQUEST_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "slack:T1:C1:100.0"))


class SlowSiblingKafka:
    """Acks chat.feedback at once; holds every other topic until released."""

    def __init__(self):
        self.release = asyncio.Event()
        self.sent = []

    async def send_and_wait(self, topic, key, value):
        if topic != "chat.feedback":
            await self.release.wait()
        self.sent.append(topic)
        return SimpleNamespace(offset=1)

    async def stop(self):
        return None


def producer_with(kafka, **producer_config):
    manager = KafkaProducerManager(
        {"producer": {"retries": 0, "retry_backoff_ms": 0, **producer_config}}
    )
    manager.producer = kafka
    manager._accepting = True
    return manager


async def stop(manager):
    manager._accepting = False
    task = manager._dispatcher_task
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def feedback(**overrides):
    return {
        "request_id": REQUEST_ID,
        "user_id": "U2",
        "rating": "up",
        "click_id": "1700.1",
        **overrides,
    }


@pytest.mark.asyncio
async def test_a_slow_sibling_in_the_batch_does_not_delay_the_feedback():
    kafka = SlowSiblingKafka()
    manager = producer_with(kafka)
    # Both land in one dispatcher batch: queue them before it starts.
    assert manager._enqueue_message(
        "requests_raw", "requests.raw", key="k", value={"a": 1}
    )
    waiting = asyncio.create_task(
        manager.produce_chat_feedback(**feedback(), wait_for_delivery=True, timeout=1)
    )
    await asyncio.sleep(0)
    manager._dispatcher_task = asyncio.create_task(manager._dispatch_loop())
    started = time.monotonic()
    try:
        delivered = await asyncio.wait_for(waiting, timeout=2)
    finally:
        kafka.release.set()
        await stop(manager)

    assert delivered is True
    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_an_unconfirmed_send_is_reported_as_unknown_not_failed():
    kafka = SlowSiblingKafka()
    manager = producer_with(kafka)
    manager._dispatcher_task = asyncio.create_task(manager._dispatch_loop())
    try:
        manager.topics = {"chat_feedback": "held"}  # never acked in time
        delivered = await manager.produce_chat_feedback(
            **feedback(), wait_for_delivery=True, timeout=0.05
        )
    finally:
        kafka.release.set()
        await stop(manager)

    assert delivered is None


def test_only_a_producer_that_waits_for_acks_confirms_delivery():
    assert KafkaProducerManager({"producer": {}}).confirms_chat_feedback_delivery
    assert not KafkaProducerManager(
        {"producer": {"wait_for_ack": False}}
    ).confirms_chat_feedback_delivery


# The bot


def _bot(producer):
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(),
        services={"event_producer": producer},
    )
    bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    return bot


def _click(action_ts="1700.1"):
    return {
        "type": "block_actions",
        "user": {"id": "U2"},
        "channel": {"id": "C1"},
        "actions": [
            {
                "action_id": "chat_feedback_up",
                "value": REQUEST_ID,
                "action_ts": action_ts,
            }
        ],
    }


def _texts(bot):
    return [
        call.kwargs["text"]
        for call in bot.web_client.chat_postEphemeral.await_args_list
    ]


async def _settle(bot):
    await asyncio.gather(*list(bot.background_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_the_worker_does_not_wait_for_the_confirmation():
    gate = asyncio.Event()

    async def produce(**fields):
        await gate.wait()
        return True

    producer = SimpleNamespace(
        confirms_chat_feedback_delivery=True,
        produce_chat_feedback=AsyncMock(side_effect=produce),
    )
    bot = _bot(producer)

    await asyncio.wait_for(bot._handle_agent_interactive(_click()), timeout=0.5)
    assert _texts(bot) == []
    assert bot.background_tasks

    gate.set()
    await _settle(bot)
    assert _texts(bot) == [CHAT_FEEDBACK_THANKS_TEXT]
    assert "1700.1" in bot._chat_feedback_clicks


@pytest.mark.asyncio
async def test_an_unconfirmed_click_stays_claimed_and_is_not_retried():
    producer = SimpleNamespace(
        confirms_chat_feedback_delivery=True,
        produce_chat_feedback=AsyncMock(return_value=None),
    )
    bot = _bot(producer)

    await bot._handle_agent_interactive(_click())
    await _settle(bot)
    await bot._handle_agent_interactive(_click())
    await _settle(bot)

    assert producer.produce_chat_feedback.await_count == 1
    assert _texts(bot) == [CHAT_FEEDBACK_UNCONFIRMED_TEXT]
    assert "try again" not in CHAT_FEEDBACK_UNCONFIRMED_TEXT.lower()
    assert "Thanks" not in CHAT_FEEDBACK_UNCONFIRMED_TEXT


@pytest.mark.asyncio
async def test_a_failed_confirmed_send_can_still_be_retried():
    producer = SimpleNamespace(
        confirms_chat_feedback_delivery=True,
        produce_chat_feedback=AsyncMock(side_effect=[False, True]),
    )
    bot = _bot(producer)

    await bot._handle_agent_interactive(_click())
    await _settle(bot)
    await bot._handle_agent_interactive(_click())
    await _settle(bot)

    assert _texts(bot) == [CHAT_FEEDBACK_NOT_RECORDED_TEXT, CHAT_FEEDBACK_THANKS_TEXT]
