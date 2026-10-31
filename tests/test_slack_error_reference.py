"""T1: a Slack error reply carries a short reference to its request id."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot, slack_chat_request_id


def error_bot(process_query=None, *, chat_update=True):
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=SimpleNamespace(process_query=process_query or AsyncMock()),
    )
    bot.bot_user_id = "UBOT"
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    if chat_update:
        client.chat_update = AsyncMock(return_value={"ok": True})
    bot.web_client = client
    return bot


def message(ts="100.0", text="what is 2+2?"):
    return {"type": "message", "channel": "C1", "user": "U1", "text": text, "ts": ts}


def reference_block(ts="100.0"):
    request_id = slack_chat_request_id(None, "C1", ts)
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"Reference: `{request_id[:8]}`"}],
    }


@pytest.mark.asyncio
async def test_an_error_response_carries_the_reference_and_keeps_its_text(
    inference_response_factory,
):
    failed = inference_response_factory(
        response_text="", error="boom", error_type="timeout"
    )
    plain = error_bot(AsyncMock(return_value=failed), chat_update=False)
    expected_text = str(plain.message_handler._inference_error_message("timeout"))

    await plain._process_message(message())

    reply = plain.web_client.chat_postMessage.await_args.kwargs
    assert reply["text"] == expected_text
    assert reply["blocks"] == [
        {"type": "section", "text": {"type": "mrkdwn", "text": expected_text}},
        reference_block(),
    ]


@pytest.mark.asyncio
async def test_a_failed_query_replaces_the_placeholder_with_the_reference():
    bot = error_bot(AsyncMock(side_effect=RuntimeError("boom")))

    await bot._process_message(message())

    update = bot.web_client.chat_update.await_args.kwargs
    assert update["text"] == bot._build_user_safe_error_message()
    assert update["blocks"][-1] == reference_block()
    assert "boom" not in str(update)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_update", [True, False])
async def test_an_unexpected_failure_carries_the_recomputed_reference(
    monkeypatch, chat_update
):
    bot = error_bot(chat_update=chat_update)
    monkeypatch.setattr(
        bot.message_handler,
        "handle_message",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    await bot._process_message(message(ts="200.0"))

    sent = (
        bot.web_client.chat_update.await_args.kwargs
        if chat_update
        else bot.web_client.chat_postMessage.await_args.kwargs
    )
    assert sent["text"] == bot._build_user_safe_error_message()
    assert sent["blocks"] == [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": bot._build_user_safe_error_message()},
        },
        reference_block("200.0"),
    ]


@pytest.mark.asyncio
async def test_an_error_without_a_request_id_stays_plain_text():
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.message_handler = SimpleNamespace(
        handle_message=AsyncMock(side_effect=RuntimeError("boom"))
    )

    await bot._process_message({"channel": "C1", "user": "U1", "text": "hello"})

    assert "blocks" not in bot.web_client.chat_postMessage.await_args.kwargs


@pytest.mark.asyncio
async def test_a_refused_reference_block_falls_back_to_the_plain_error():
    class BlockRejected(Exception):
        response = {"error": "invalid_blocks"}

    bot = error_bot(chat_update=False)
    bot.web_client.chat_postMessage.side_effect = [BlockRejected(), {"ok": True}]
    bot.message_handler.handle_message = AsyncMock(side_effect=RuntimeError("boom"))

    await bot._process_message(message())

    calls = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list]
    assert "blocks" in calls[0]
    assert calls[1]["text"] == bot._build_user_safe_error_message()
    assert "blocks" not in calls[1]
