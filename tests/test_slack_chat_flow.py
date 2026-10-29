"""Slack chat flow: which messages are answered, and what the user sees."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot, SlackWorkItem
from src.utils.schema import Attachment, AttachmentType


def chat_bot(process_query=None, **config):
    """A bot that answers chat queries through `process_query`."""
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}, **config},
        inference_engine=SimpleNamespace(process_query=process_query or AsyncMock()),
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "999.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    return bot


def answering(inference_response_factory, text="the answer"):
    return AsyncMock(return_value=inference_response_factory(response_text=text))


def thread_reply(ts="101.0", thread_ts="100.0", **fields):
    return {
        "type": "message",
        "channel": "C1",
        "user": "U1",
        "text": "and this?",
        "thread_ts": thread_ts,
        "ts": ts,
        **fields,
    }


def queries(bot):
    return [
        call.args[0].query
        for call in bot.inference_engine.process_query.await_args_list
    ]


# S13: follow-ups in an active thread


@pytest.mark.asyncio
async def test_file_share_in_an_active_thread_reaches_process_query(
    inference_response_factory,
):
    bot = chat_bot(answering(inference_response_factory))
    bot._mark_thread_active("C1", "100.0")
    attachment = Attachment(
        name="report.csv",
        type=AttachmentType.DOCUMENT,
        size_bytes=3,
        mime_type="text/csv",
        content=b"a,b",
    )
    bot._extract_query_attachments = AsyncMock(return_value=[attachment])
    event = thread_reply(
        subtype="file_share",
        text="what does this show?",
        files=[{"id": "F1", "name": "report.csv"}],
    )

    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=dict(event), name="file_share")
    )
    await bot._handle_message_event(event)

    request = bot.inference_engine.process_query.await_args.args[0]
    assert request.query == "what does this show?"
    assert request.attachments == [attachment]
    assert bot.web_client.chat_postMessage.await_args.kwargs["thread_ts"] == "100.0"


@pytest.mark.asyncio
async def test_thread_broadcast_follow_up_is_answered(inference_response_factory):
    bot = chat_bot(answering(inference_response_factory))
    bot._mark_thread_active("C1", "100.0")
    event = thread_reply(subtype="thread_broadcast", text="also for everyone")

    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=dict(event), name="broadcast")
    )
    await bot._handle_message_event(event)

    assert queries(bot) == ["also for everyone"]


@pytest.mark.asyncio
async def test_follow_up_sent_while_the_first_answer_is_generating_is_answered(
    inference_response_factory,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def process_query(request):
        if not started.is_set():
            started.set()
            await release.wait()
        return inference_response_factory(response_text=f"re: {request.query}")

    bot = chat_bot(AsyncMock(side_effect=process_query))
    first = asyncio.create_task(
        bot._handle_mention_event(
            {
                "type": "app_mention",
                "channel": "C1",
                "user": "U1",
                "text": "<@UBOT> first question",
                "ts": "100.0",
            }
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    await bot._handle_message_event(thread_reply(text="a follow-up"))
    release.set()
    await asyncio.wait_for(first, timeout=1)

    assert queries(bot) == ["first question", "a follow-up"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subtype", ["edited", "message_changed", "message_deleted", "channel_join"]
)
async def test_edits_and_other_subtypes_in_an_active_thread_stay_dropped(subtype):
    bot = chat_bot()
    bot._mark_thread_active("C1", "100.0")
    event = thread_reply(subtype=subtype)

    assert not await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=dict(event), name=subtype)
    )
    await bot._handle_message_event(event)

    bot.inference_engine.process_query.assert_not_awaited()
    bot.web_client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [{"bot_id": "B1"}, {"user": "UBOT"}])
@pytest.mark.parametrize("subtype", ["file_share", "thread_broadcast"])
async def test_bot_file_shares_and_broadcasts_stay_dropped(sender, subtype):
    bot = chat_bot()
    bot._mark_thread_active("C1", "100.0")
    event = thread_reply(subtype=subtype, **sender)

    assert not await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=dict(event), name=subtype)
    )
    await bot._handle_message_event(event)

    bot.inference_engine.process_query.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("subtype", ["file_share", "thread_broadcast"])
async def test_agent_threads_still_take_plain_follow_ups_only(subtype):
    bot = chat_bot()
    bot.agent_threads["C1:100.0"] = {"owner_user_id": "U1"}
    bot._mark_thread_active("C1", "100.0")
    bot._process_message = AsyncMock()
    event = thread_reply(subtype=subtype, files=[{"id": "F1"}])

    assert not await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=dict(event), name=subtype)
    )
    await bot._handle_message_event(event)

    bot._process_message.assert_not_awaited()
    bot.web_client.chat_postEphemeral.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_file_share_outside_an_active_thread_is_not_answered():
    bot = chat_bot()
    bot._process_message = AsyncMock()

    await bot._handle_message_event(thread_reply(subtype="file_share"))
    await bot._handle_message_event(
        {
            "type": "message",
            "subtype": "file_share",
            "channel": "C1",
            "user": "U1",
            "ts": "200.0",
        }
    )

    bot._process_message.assert_not_awaited()
