"""T1 review: slash references, logged references, bootstrap rendering."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.agent_bootstrap import AgentBootstrapBuilder
from slack.bot_real import SlackBot, slack_chat_request_id
from slack.formatting import SlackNotice


def _bot(process_query=None):
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=SimpleNamespace(process_query=process_query or AsyncMock()),
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    return bot


def _message(ts="100.0"):
    return {"type": "message", "channel": "C1", "user": "U1", "text": "hi", "ts": ts}


def test_a_prefixed_request_id_takes_its_reference_after_the_prefix():
    notice = SlackNotice("failed", request_id="slash:13345224609.738474920.80")

    assert notice.reference == "13345224"
    plain = SlackNotice("failed", request_id="abcd1234-5678")
    assert plain.reference == "abcd1234"


@pytest.mark.asyncio
async def test_a_slash_error_shows_its_reference_in_the_ephemeral_reply():
    bot = _bot()
    notice = SlackNotice("❌ failed", request_id="slash:13345224609.738")

    await bot._post_command_response("C1", "U1", notice)

    reply = bot.web_client.chat_postEphemeral.await_args.kwargs
    assert reply["text"] == "❌ failed"
    assert reply["blocks"][-1] == {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "Reference: `13345224`"}],
    }


@pytest.mark.asyncio
async def test_a_slash_reply_that_is_not_an_error_stays_plain_text():
    bot = _bot()

    await bot._post_command_response("C1", "U1", "help text")

    assert bot.web_client.chat_postEphemeral.await_args.kwargs == {
        "channel": "C1",
        "user": "U1",
        "text": "help text",
    }


@pytest.mark.asyncio
async def test_a_refused_slash_reference_falls_back_to_plain_text():
    class BlockRejected(Exception):
        response = {"error": "invalid_blocks"}

    bot = _bot()
    bot.web_client.chat_postEphemeral.side_effect = [BlockRejected(), None]

    await bot._post_command_response(
        "C1", "U1", SlackNotice("❌ failed", request_id="slash:13345224609.738")
    )

    calls = [c.kwargs for c in bot.web_client.chat_postEphemeral.await_args_list]
    assert "blocks" in calls[0]
    assert calls[1] == {"channel": "C1", "user": "U1", "text": "❌ failed"}


@pytest.mark.asyncio
async def test_an_unexpected_failure_logs_the_request_id_it_shows(caplog):
    bot = _bot()
    bot.message_handler.handle_message = AsyncMock(side_effect=RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger="slack.bot_real"):
        await bot._process_message(_message())

    request_id = slack_chat_request_id(None, "C1", "100.0")
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert request_id in record.getMessage()
    assert record.exc_info is not None


@pytest.mark.asyncio
async def test_a_failed_query_logs_the_request_id_it_shows(caplog):
    bot = _bot(AsyncMock(side_effect=RuntimeError("boom")))

    with caplog.at_level(logging.ERROR, logger="slack.bot_real"):
        await bot._process_message(_message())

    request_id = slack_chat_request_id(None, "C1", "100.0")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(request_id in r.getMessage() and r.exc_info is not None for r in errors)


def test_the_agent_bootstrap_renders_an_error_reply_as_text():
    builder = AgentBootstrapBuilder({}, bot_user_id="BOT")
    notice = SlackNotice("❌ failed", request_id="abcd1234-5678")

    text = builder._flatten_blocks(notice.blocks(notice))

    assert text == "❌ failed\nReference: `abcd1234`"
