"""T3: "Thanks" for chat feedback only once Kafka confirmed its delivery."""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import (
    CHAT_FEEDBACK_NOT_RECORDED_TEXT,
    CHAT_FEEDBACK_THANKS_TEXT,
    SlackBot,
)
from src.llm_router_part3_pipeline import KafkaProducerManager

REQUEST_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "slack:T1:C1:100.0"))


class FakeKafka:
    """A single-phase Kafka producer double: send_and_wait only."""

    def __init__(self, outcome="ok"):
        self.outcome = outcome
        self.sent = []
        self.release = asyncio.Event()

    async def send_and_wait(self, topic, key, value):
        if self.outcome == "hang":
            await self.release.wait()
        if self.outcome == "fail":
            raise RuntimeError("broker down")
        self.sent.append((topic, key, value))
        return SimpleNamespace(offset=1)

    async def stop(self):
        return None


def producer_with(kafka, **producer_config):
    manager = KafkaProducerManager(
        {
            "topics": {"chat_feedback": "chat.feedback"},
            "producer": {"retries": 0, "retry_backoff_ms": 0, **producer_config},
        }
    )
    manager.producer = kafka
    manager._accepting = True
    manager._dispatcher_task = asyncio.create_task(manager._dispatch_loop())
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
async def test_waiting_feedback_is_true_once_kafka_confirmed_it():
    kafka = FakeKafka()
    manager = producer_with(kafka)
    try:
        delivered = await manager.produce_chat_feedback(
            **feedback(), wait_for_delivery=True
        )
    finally:
        await stop(manager)

    assert delivered is True
    assert [(topic, key) for topic, key, _ in kafka.sent] == [
        ("chat.feedback", REQUEST_ID)
    ]


@pytest.mark.asyncio
async def test_waiting_feedback_is_false_when_the_send_failed():
    manager = producer_with(FakeKafka("fail"))
    try:
        delivered = await manager.produce_chat_feedback(
            **feedback(), wait_for_delivery=True
        )
    finally:
        await stop(manager)

    assert delivered is False


@pytest.mark.asyncio
async def test_waiting_feedback_is_false_when_the_send_failed_and_raises():
    manager = producer_with(FakeKafka("fail"), raise_on_failure=True)
    try:
        delivered = await manager.produce_chat_feedback(
            **feedback(), wait_for_delivery=True
        )
    finally:
        await stop(manager)

    assert delivered is False


@pytest.mark.asyncio
async def test_waiting_feedback_gives_up_after_its_timeout():
    kafka = FakeKafka("hang")
    manager = producer_with(kafka)
    try:
        delivered = await asyncio.wait_for(
            manager.produce_chat_feedback(
                **feedback(), wait_for_delivery=True, timeout=0.05
            ),
            timeout=1,
        )
    finally:
        kafka.release.set()
        await stop(manager)

    assert delivered is False


@pytest.mark.asyncio
async def test_waiting_feedback_is_false_when_it_was_never_accepted():
    manager = KafkaProducerManager({"producer": {}})

    assert (
        await manager.produce_chat_feedback(**feedback(), wait_for_delivery=True)
        is False
    )


@pytest.mark.asyncio
async def test_feedback_dropped_at_shutdown_resolves_as_not_delivered():
    manager = KafkaProducerManager({"producer": {}})
    manager._accepting = True
    waiting = asyncio.create_task(
        manager.produce_chat_feedback(**feedback(), wait_for_delivery=True, timeout=5)
    )
    await asyncio.sleep(0)

    manager._drop_queued_messages("shutdown_timeout")

    assert await asyncio.wait_for(waiting, timeout=1) is False


@pytest.mark.asyncio
async def test_feedback_without_waiting_is_still_true_once_queued():
    manager = producer_with(FakeKafka("fail"))
    try:
        accepted = await manager.produce_chat_feedback(**feedback())
    finally:
        await stop(manager)

    assert accepted is True


@pytest.mark.asyncio
async def test_other_messages_are_delivered_as_before():
    kafka = FakeKafka()
    manager = producer_with(kafka)
    try:
        assert manager._enqueue_message(
            "chat_feedback", "chat.feedback", key="k", value={"a": 1}
        )
        await asyncio.wait_for(manager._delivery_queue.join(), timeout=1)
    finally:
        await stop(manager)

    assert kafka.sent == [("chat.feedback", "k", {"a": 1})]


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


def _ephemeral_texts(bot):
    return [
        call.kwargs["text"]
        for call in bot.web_client.chat_postEphemeral.await_args_list
    ]


@pytest.mark.asyncio
async def test_the_bot_does_not_thank_for_feedback_kafka_did_not_take():
    manager = producer_with(FakeKafka("fail"))
    bot = _bot(manager)
    try:
        await bot._handle_agent_interactive(_click())
    finally:
        await stop(manager)

    assert _ephemeral_texts(bot) == [CHAT_FEEDBACK_NOT_RECORDED_TEXT]
    assert "1700.1" not in bot._chat_feedback_clicks


@pytest.mark.asyncio
async def test_the_bot_thanks_once_kafka_confirmed_the_feedback():
    kafka = FakeKafka()
    manager = producer_with(kafka)
    bot = _bot(manager)
    try:
        await bot._handle_agent_interactive(_click())
    finally:
        await stop(manager)

    assert len(kafka.sent) == 1
    assert _ephemeral_texts(bot) == [CHAT_FEEDBACK_THANKS_TEXT]
    assert "1700.1" in bot._chat_feedback_clicks


@pytest.mark.asyncio
async def test_the_bot_asks_a_confirming_producer_to_wait_for_delivery():
    producer = SimpleNamespace(
        confirms_chat_feedback_delivery=True,
        produce_chat_feedback=AsyncMock(return_value=False),
    )
    bot = _bot(producer)

    await bot._handle_agent_interactive(_click())

    assert producer.produce_chat_feedback.await_args.kwargs["wait_for_delivery"]
    assert _ephemeral_texts(bot) == [CHAT_FEEDBACK_NOT_RECORDED_TEXT]
