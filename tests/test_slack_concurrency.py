"""Slack concurrency: runtime lookups, allowlist checks and per-thread order."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot, SlackWorkItem
from src.agent_runtime import AgentRuntimeUnavailable


def chat_bot(process_query=None, runtime=None, **config):
    """A bot that answers chat queries through `process_query`."""
    services = {"agent_runtime": runtime} if runtime is not None else None
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}, **config},
        inference_engine=SimpleNamespace(process_query=process_query or AsyncMock()),
        services=services,
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "999.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    return bot


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


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# S7: an active chat thread does not wait on the agent runtime


@pytest.mark.asyncio
async def test_follow_up_reaches_process_query_when_the_runtime_is_down(
    inference_response_factory,
):
    runtime = SimpleNamespace(
        lookup_session=AsyncMock(side_effect=AgentRuntimeUnavailable("down"))
    )
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok")),
        runtime=runtime,
    )
    bot._mark_thread_active("C1", "100.0")

    await bot._handle_message_event(thread_reply(ts="101.0", text="first"))
    await bot._handle_message_event(thread_reply(ts="102.0", text="second"))

    assert queries(bot) == ["first", "second"]
    # The failure is remembered for a while instead of retried per message.
    assert runtime.lookup_session.await_count <= 1


@pytest.mark.asyncio
async def test_two_messages_in_a_chat_thread_look_the_runtime_up_at_most_once(
    inference_response_factory,
):
    runtime = SimpleNamespace(lookup_session=AsyncMock(return_value=None))
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok")),
        runtime=runtime,
    )
    bot._mark_thread_active("C1", "100.0")

    await bot._handle_message_event(thread_reply(ts="101.0", text="first"))
    await bot._handle_message_event(thread_reply(ts="102.0", text="second"))

    assert queries(bot) == ["first", "second"]
    assert runtime.lookup_session.await_count <= 1


@pytest.mark.asyncio
async def test_a_thread_the_bot_started_in_chat_never_asks_the_runtime(
    inference_response_factory,
):
    runtime = SimpleNamespace(lookup_session=AsyncMock(return_value=None))
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok")),
        runtime=runtime,
    )

    await bot._handle_mention_event(
        {
            "type": "app_mention",
            "channel": "C1",
            "user": "U1",
            "text": "<@UBOT> first question",
            "ts": "100.0",
        }
    )
    await bot._handle_message_event(thread_reply(ts="101.0", text="follow-up"))

    assert queries(bot) == ["first question", "follow-up"]
    runtime.lookup_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_lookup_is_retried_after_the_error_ttl(
    inference_response_factory,
):
    runtime = SimpleNamespace(
        lookup_session=AsyncMock(side_effect=AgentRuntimeUnavailable("down"))
    )
    bot = chat_bot(
        AsyncMock(return_value=inference_response_factory(response_text="ok")),
        runtime=runtime,
    )
    clock = FakeClock()
    bot._agent_lookup_skips._clock = clock
    bot._mark_thread_active("C1", "100.0")

    await bot._handle_message_event(thread_reply(ts="101.0", text="first"))
    clock.now += 60
    await bot._handle_message_event(thread_reply(ts="102.0", text="second"))

    assert runtime.lookup_session.await_count == 2
    assert queries(bot) == ["first", "second"]


@pytest.mark.asyncio
async def test_known_agent_thread_still_turns_a_non_owner_away_without_a_lookup():
    runtime = SimpleNamespace(lookup_session=AsyncMock(return_value=None))
    bot = chat_bot(runtime=runtime)
    bot.agent_threads["C1:100.0"] = {
        "owner_user_id": "U1",
        "channel_id": "C1",
        "thread_ts": "100.0",
    }
    bot._mark_thread_active("C1", "100.0")
    bot._process_message = AsyncMock()

    await bot._handle_message_event(thread_reply(ts="101.0", user="U2"))

    bot._process_message.assert_not_awaited()
    bot.web_client.chat_postEphemeral.assert_awaited_once()
    runtime.lookup_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_active_thread_is_still_looked_up_once_for_its_agent_owner():
    """After a restart the bot must still find an Agent session it lost."""
    runtime = SimpleNamespace(
        lookup_session=AsyncMock(
            return_value={
                "session_id": "S1",
                "run_id": None,
                "owner_user_id": "U1",
                "active": False,
            }
        )
    )
    bot = chat_bot(runtime=runtime)
    bot._mark_thread_active("C1", "100.0")
    bot._process_message = AsyncMock()

    await bot._handle_message_event(thread_reply(ts="101.0", user="U2"))

    runtime.lookup_session.assert_awaited_once()
    bot._process_message.assert_not_awaited()
    bot.web_client.chat_postEphemeral.assert_awaited_once()


@pytest.mark.asyncio
async def test_allowlist_miss_calls_conversations_info_once_per_ttl():
    bot = chat_bot(channels=["#bots"])
    bot.web_client.conversations_info = AsyncMock(
        return_value={"ok": True, "channel": {"name": "random"}}
    )
    bot.allowed_channel_names = {"bots"}
    clock = FakeClock()
    bot._channel_allowlist_misses._clock = clock

    assert not await bot._is_channel_allowed("COTHER001")
    assert not await bot._is_channel_allowed("COTHER001")
    assert bot.web_client.conversations_info.await_count == 1

    clock.now += 3600
    assert not await bot._is_channel_allowed("COTHER001")
    assert bot.web_client.conversations_info.await_count == 2


# S12: one Slack thread is answered one message at a time


async def _run_workers_until_idle(bot, release, *, picked_up):
    bot._start_work_workers()
    try:
        # Wait until both items left the queue, so without a per-thread lock
        # the second one would be running next to the first.
        for _ in range(200):
            if bot._work_queue.qsize() == 0 and bot._work_running >= picked_up:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(bot._work_queue.join(), timeout=2)
    finally:
        await bot._stop_work_workers()


@pytest.mark.asyncio
async def test_same_thread_messages_are_answered_in_order_with_history(
    inference_response_factory,
):
    release = asyncio.Event()

    async def process_query(request):
        if request.query == "first question":
            await release.wait()
        return inference_response_factory(response_text=f"re: {request.query}")

    bot = chat_bot(AsyncMock(side_effect=process_query))
    bot._mark_thread_active("C1", "100.0")
    mention = {
        "type": "app_mention",
        "channel": "C1",
        "user": "U1",
        "text": "<@UBOT> first question",
        "ts": "100.0",
    }
    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=mention, name="mention")
    )
    assert await bot._enqueue_work(
        SlackWorkItem(
            kind="event", payload=thread_reply(text="a follow-up"), name="message"
        )
    )

    await _run_workers_until_idle(bot, release, picked_up=2)

    assert queries(bot) == ["first question", "a follow-up"]
    posted = [
        call.kwargs["text"] for call in bot.web_client.chat_postMessage.await_args_list
    ]
    assert posted == ["re: first question", "re: a follow-up"]
    second_request = bot.inference_engine.process_query.await_args_list[1].args[0]
    assert "re: first question" in (second_request.context or "")
    assert not bot._thread_locks


@pytest.mark.asyncio
async def test_different_threads_are_still_answered_in_parallel(
    inference_response_factory,
):
    release = asyncio.Event()

    async def process_query(request):
        if request.query == "slow one":
            await release.wait()
        return inference_response_factory(response_text=f"re: {request.query}")

    bot = chat_bot(AsyncMock(side_effect=process_query))
    bot._mark_thread_active("C1", "100.0")
    bot._mark_thread_active("C1", "200.0")
    for ts, thread_ts, text in (
        ("101.0", "100.0", "slow one"),
        ("201.0", "200.0", "fast one"),
    ):
        assert await bot._enqueue_work(
            SlackWorkItem(
                kind="event",
                payload=thread_reply(ts=ts, thread_ts=thread_ts, text=text),
                name="message",
            )
        )

    await _run_workers_until_idle(bot, release, picked_up=2)

    posted = [
        call.kwargs["text"] for call in bot.web_client.chat_postMessage.await_args_list
    ]
    assert posted == ["re: fast one", "re: slow one"]


@pytest.mark.asyncio
async def test_agent_background_start_does_not_hold_the_thread_lock():
    bot = chat_bot()
    started = asyncio.Event()
    release = asyncio.Event()

    async def handle_event(event):
        started.set()
        await release.wait()

    bot._is_agent_request = lambda event: True
    bot._handle_event = handle_event
    assert await bot._enqueue_work(
        SlackWorkItem(
            kind="event", payload=thread_reply(text="agent go"), name="message"
        )
    )
    bot._start_work_workers()
    try:
        await asyncio.wait_for(bot._work_queue.join(), timeout=1)
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not bot._thread_locks
    finally:
        release.set()
        await bot._stop_work_workers()
        await asyncio.gather(*bot.background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_allowlist_hit_by_name_is_still_resolved():
    bot = chat_bot(channels=["#bots"])
    bot.web_client.conversations_info = AsyncMock(
        return_value={"ok": True, "channel": {"name": "bots"}}
    )
    bot.allowed_channel_names = {"bots"}

    assert await bot._is_channel_allowed("CBOTS0001")
    assert await bot._is_channel_allowed("CBOTS0001")
    assert bot.web_client.conversations_info.await_count == 1


# S6c: attachments download side by side, in their original order


@pytest.mark.asyncio
async def test_attachments_download_concurrently_and_keep_their_order():
    bot = chat_bot()
    in_flight = 0
    peak = 0

    async def build(file_payload):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            # Later files finish first, so the order must come from the input.
            await asyncio.sleep(0.01 * (10 - int(file_payload["id"][1:])))
        finally:
            in_flight -= 1
        if file_payload["id"] == "F3":
            return None
        return file_payload["id"]

    bot._build_attachment_from_slack_file = build
    files = [{"id": f"F{index}"} for index in range(6)]

    attachments = await bot._extract_query_attachments({"files": files})

    assert attachments == ["F0", "F1", "F2", "F4", "F5"]
    assert peak == 4


@pytest.mark.asyncio
async def test_a_failed_attachment_download_still_raises():
    bot = chat_bot()

    async def build(file_payload):
        if file_payload["id"] == "F1":
            raise RuntimeError("boom")
        await asyncio.sleep(0)
        return file_payload["id"]

    bot._build_attachment_from_slack_file = build

    with pytest.raises(RuntimeError, match="boom"):
        await bot._extract_query_attachments(
            {"files": [{"id": "F0"}, {"id": "F1"}, {"id": "F2"}]}
        )
