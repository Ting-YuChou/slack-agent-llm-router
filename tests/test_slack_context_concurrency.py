"""S6a: Slack history and memory search run concurrently, each fail-open."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot


def _bot(inference_response_factory, memory_manager):
    bot = SlackBot(
        {"channels": [], "context": {"enabled": True}},
        inference_engine=SimpleNamespace(
            process_query=AsyncMock(return_value=inference_response_factory())
        ),
    )
    bot.bot_user_id = "UBOT"
    bot.memory_manager = memory_manager
    return bot


def _memory(delay, results=(), error=None):
    async def search(*_args, **_kwargs):
        await asyncio.sleep(delay)
        if error is not None:
            raise error
        return list(results)

    return SimpleNamespace(
        enabled=True,
        search=AsyncMock(side_effect=search),
        build_context=lambda results, context: context,
    )


def _replies(delay, messages=(), error=None):
    async def replies(**_kwargs):
        await asyncio.sleep(delay)
        if error is not None:
            raise error
        return {"ok": True, "messages": list(messages)}

    return SimpleNamespace(conversations_replies=AsyncMock(side_effect=replies))


async def _ask(bot, client):
    await bot.message_handler._handle_query(
        "and then?", "u2", "c1", "t1", client=client, team_id="T1", message_ts="9.0"
    )
    return bot.inference_engine.process_query.await_args.args[0]


@pytest.mark.asyncio
async def test_slack_history_and_memory_search_run_concurrently(
    inference_response_factory,
):
    bot = _bot(inference_response_factory, _memory(0.2))
    client = _replies(0.2, [{"ts": "1.0", "user": "u1", "text": "earlier"}])

    started = time.perf_counter()
    request = await _ask(bot, client)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.35
    bot.memory_manager.search.assert_awaited_once()
    assert "<@u1>: earlier" in request.context


@pytest.mark.asyncio
async def test_a_failed_memory_search_keeps_the_slack_history(
    inference_response_factory,
):
    bot = _bot(inference_response_factory, _memory(0.05, error=RuntimeError("down")))
    client = _replies(0.05, [{"ts": "1.0", "user": "u1", "text": "earlier"}])

    request = await _ask(bot, client)

    assert "<@u1>: earlier" in request.context


@pytest.mark.asyncio
async def test_a_failed_slack_fetch_still_searches_memory(inference_response_factory):
    bot = _bot(inference_response_factory, _memory(0.05))
    client = _replies(0.05, error=RuntimeError("missing_scope"))

    request = await _ask(bot, client)

    bot.memory_manager.search.assert_awaited_once()
    assert request.metadata["slack_context_error"] == "missing_scope"
