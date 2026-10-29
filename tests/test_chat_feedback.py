"""X9a: thumbs up / down on chat answers, recorded as chat.feedback."""

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from slack.bot_real import SLACK_PRIORITY_WORK_KINDS, SlackBot
from src.llm_router_part3_pipeline import (
    ClickHouseManager,
    KafkaConsumerManager,
    KafkaProducerManager,
    build_chat_feedback_row,
)
from src.utils.schema import PlatformConfig, SlackResponseSettingsConfig

REQUEST_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "slack:T1:C1:100.0"))


class _Producer:
    def __init__(self):
        self.feedback = []

    async def produce_chat_feedback(self, **fields):
        self.feedback.append(fields)


def _bot(inference_response_factory=None, answer="four", **response_settings):
    process_query = AsyncMock()
    if inference_response_factory is not None:
        process_query = AsyncMock(
            return_value=inference_response_factory(response_text=answer)
        )
    producer = _Producer()
    bot = SlackBot(
        {
            "channels": [],
            "context": {"enabled": False},
            "response_settings": response_settings,
        },
        inference_engine=SimpleNamespace(process_query=process_query),
        services={"event_producer": producer},
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
        chat_update=AsyncMock(return_value={"ok": True}),
    )
    return bot, producer


def _mention(**fields):
    return {
        "type": "app_mention",
        "channel": "C1",
        "user": "U1",
        "text": "what is 2+2?",
        "ts": "100.0",
        "_team_id": "T1",
        **fields,
    }


def _actions(message):
    return [block for block in message.get("blocks", []) if block["type"] == "actions"]


def _click(action_id="chat_feedback_up", value=REQUEST_ID, action_ts="1700.1"):
    return {
        "type": "block_actions",
        "user": {"id": "U2"},
        "channel": {"id": "C1"},
        "trigger_id": "trigger-9",
        "container": {"message_ts": "500.0", "thread_ts": "100.0"},
        "actions": [{"action_id": action_id, "value": value, "action_ts": action_ts}],
    }


# Buttons


@pytest.mark.asyncio
async def test_the_last_part_of_an_answer_carries_the_feedback_buttons(
    inference_response_factory,
):
    answer = "\n".join(f"line {index} " + "x" * 40 for index in range(20))
    bot, _producer = _bot(
        inference_response_factory,
        answer,
        feedback_buttons=True,
        max_response_length=300,
    )

    await bot._process_message(_mention())

    first = bot.web_client.chat_update.await_args.kwargs
    later = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list]
    parts = [first, *later[1:]]
    assert len(parts) > 1
    assert all(not _actions(part) for part in parts[:-1])
    (actions,) = _actions(parts[-1])
    buttons = actions["elements"]
    assert [button["action_id"] for button in buttons] == [
        "chat_feedback_up",
        "chat_feedback_down",
    ]
    assert all(button["value"] == REQUEST_ID for button in buttons)
    assert all(button["type"] == "button" for button in buttons)


@pytest.mark.asyncio
async def test_a_thread_answer_carries_the_buttons_too(inference_response_factory):
    bot, _producer = _bot(inference_response_factory, feedback_buttons=True)

    await bot._process_message(
        {
            "type": "message",
            "channel": "C1",
            "user": "U1",
            "text": "and 3+3?",
            "ts": "101.0",
            "thread_ts": "100.0",
            "_team_id": "T1",
        }
    )

    update = bot.web_client.chat_update.await_args.kwargs
    (actions,) = _actions(update)
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "slack:T1:C1:101.0"))
    assert actions["elements"][0]["value"] == expected


@pytest.mark.asyncio
async def test_feedback_buttons_are_off_by_default(inference_response_factory):
    bot, _producer = _bot(inference_response_factory)

    await bot._process_message(_mention())

    assert not _actions(bot.web_client.chat_update.await_args.kwargs)


@pytest.mark.asyncio
async def test_slash_answers_get_no_buttons(inference_response_factory):
    bot, _producer = _bot(inference_response_factory, feedback_buttons=True)

    await bot._handle_slash_command(
        {
            "channel_id": "C1",
            "user_id": "U1",
            "team_id": "T1",
            "text": "what is 2+2?",
            "trigger_id": "13345.7384.abc",
        }
    )

    (call,) = bot.web_client.chat_postEphemeral.await_args_list
    assert "blocks" not in call.kwargs


@pytest.mark.asyncio
async def test_an_error_reply_gets_no_buttons(inference_response_factory):
    bot, _producer = _bot(feedback_buttons=True)
    bot.inference_engine.process_query = AsyncMock(
        return_value=inference_response_factory(
            response_text="", error="boom", error_type="provider_error"
        )
    )

    await bot._process_message(_mention())

    assert not _actions(bot.web_client.chat_update.await_args.kwargs)


# Clicks


@pytest.mark.asyncio
async def test_a_click_records_the_rating_the_request_and_who_clicked():
    bot, producer = _bot()

    await bot._handle_agent_interactive(_click("chat_feedback_down"))

    assert producer.feedback == [
        {
            "request_id": REQUEST_ID,
            "user_id": "U2",
            "rating": "down",
            "click_id": "1700.1",
        }
    ]
    confirmation = bot.web_client.chat_postEphemeral.await_args.kwargs
    assert confirmation["channel"] == "C1"
    assert confirmation["user"] == "U2"
    assert confirmation["thread_ts"] == "100.0"


@pytest.mark.asyncio
async def test_the_same_click_is_recorded_once():
    bot, producer = _bot()

    await bot._handle_agent_interactive(_click())
    await bot._handle_agent_interactive(_click())
    await bot._handle_agent_interactive(_click(action_ts="1700.2"))

    assert [item["click_id"] for item in producer.feedback] == ["1700.1", "1700.2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "not a valid id!", "x" * 129])
async def test_a_click_with_a_bad_request_id_is_ignored(value):
    bot, producer = _bot()

    await bot._handle_agent_interactive(_click(value=value))

    assert producer.feedback == []


@pytest.mark.asyncio
async def test_a_feedback_click_goes_to_the_priority_queue():
    bot, _producer = _bot()
    queued = []
    bot._enqueue_work = AsyncMock(side_effect=queued.append)
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    request = SimpleNamespace(envelope_id="E1", type="interactive", payload=_click())

    await bot._handle_socket_mode_request(socket_client, request)

    (item,) = queued
    assert item.kind in SLACK_PRIORITY_WORK_KINDS


# chat.feedback event


@pytest.mark.asyncio
async def test_chat_feedback_event_is_keyed_by_request_id_with_explicit_fields():
    producer = KafkaProducerManager.__new__(KafkaProducerManager)
    sent = []
    producer._enqueue_message = lambda *args, **kwargs: sent.append((args, kwargs))

    await producer.produce_chat_feedback(
        request_id=REQUEST_ID, user_id="U2", rating="up", click_id="1700.1"
    )

    ((args, kwargs),) = sent
    assert args == ("chat_feedback", "chat.feedback")
    assert kwargs["key"] == REQUEST_ID
    event = kwargs["value"]
    assert set(event) == {
        "event_type",
        "event_version",
        "emitted_at",
        "request_id",
        "user_id",
        "rating",
        "click_id",
        "source",
    }
    assert event["event_type"] == "chat.feedback"
    assert event["rating"] == "up"


@pytest.mark.asyncio
async def test_chat_feedback_rejects_an_unknown_rating():
    producer = KafkaProducerManager.__new__(KafkaProducerManager)
    producer._enqueue_message = MagicMock()

    with pytest.raises(ValueError):
        await producer.produce_chat_feedback(
            request_id=REQUEST_ID, user_id="U2", rating="meh", click_id="1"
        )
    producer._enqueue_message.assert_not_called()


# chat_feedback table


@pytest.mark.asyncio
async def test_chat_feedback_table_is_created_after_the_existing_tables():
    manager = ClickHouseManager({"database": "llm_router"})
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    statements = [call.args[1] for call in manager._run_blocking.await_args_list]
    (ddl,) = [sql for sql in statements if "llm_router.chat_feedback (" in sql]
    assert statements.index(ddl) >= 6
    assert "ENGINE = ReplacingMergeTree(version)" in ddl
    assert "INTERVAL 90 DAY" in ddl
    assert "user_id String" in ddl
    assert "request_id String" in ddl


def test_chat_feedback_row_from_the_event():
    row = build_chat_feedback_row(
        {
            "event_type": "chat.feedback",
            "emitted_at": "2026-10-01T12:00:00Z",
            "request_id": REQUEST_ID,
            "user_id": "U2",
            "rating": "down",
            "click_id": "1700.1",
            "source": "slack",
        },
        {"event_id": "chat.feedback:0:1", "kafka_topic": "chat.feedback"},
    )

    assert row["request_id"] == REQUEST_ID
    assert row["user_id"] == "U2"
    assert row["rating"] == "down"
    assert row["click_id"] == "1700.1"
    assert row["version"] > 0
    assert row["event_id"] == "chat.feedback:0:1"


@pytest.mark.asyncio
async def test_chat_feedback_is_batched_into_its_table():
    config = {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {"chat_feedback": "chat.feedback"},
        "consumer": {},
    }
    clickhouse = MagicMock()
    clickhouse.batch_insert_chat_feedback = AsyncMock()
    consumer = KafkaConsumerManager(config, clickhouse)
    message = SimpleNamespace(
        topic="chat.feedback",
        partition=0,
        offset=1,
        value={
            "event_type": "chat.feedback",
            "emitted_at": "2026-10-01T12:00:00Z",
            "request_id": REQUEST_ID,
            "user_id": "U2",
            "rating": "up",
            "click_id": "1700.1",
            "source": "slack",
        },
    )

    class _Consumer:
        def __aiter__(self):
            async def messages():
                yield message

            return messages()

    await consumer._run_consumer("chat_feedback", _Consumer())
    await consumer.flush_pending_batches()

    (rows,) = clickhouse.batch_insert_chat_feedback.await_args.args
    assert [row["request_id"] for row in rows] == [REQUEST_ID]


# Config


def test_feedback_buttons_default_off_in_the_schema():
    assert SlackResponseSettingsConfig().feedback_buttons is False


@pytest.mark.parametrize("config_name", ["config.yaml", "config.compose.yaml"])
def test_shipped_configs_turn_feedback_on_and_name_the_topic(config_name):
    path = Path(__file__).resolve().parents[1] / "config" / config_name
    raw = yaml.safe_load(path.read_text())
    config = PlatformConfig.model_validate(raw)

    assert config.slack.response_settings.feedback_buttons is True
    assert raw["kafka"]["topics"]["chat_feedback"] == "chat.feedback"
