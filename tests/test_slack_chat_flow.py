"""Slack chat flow: which messages are answered, and what the user sees."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from slack.bot_real import SlackBot, SlackWorkItem, UserManager
from src.utils.schema import Attachment, AttachmentType, UserTier


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


def agent_command_bot(tmp_path):
    """A bot whose Agent runtime holds create_session until `gate` is set."""
    gate = asyncio.Event()
    accepted = {"session_id": "S1", "run_id": "R1", "status": "starting"}

    async def create_session(*_args, **_kwargs):
        await gate.wait()
        return accepted

    bot = chat_bot(state={"backend": "file", "file_path": str(tmp_path / "state.json")})
    bot.agent_runtime_client = SimpleNamespace(
        create_session=AsyncMock(side_effect=create_session)
    )
    bot.web_client.chat_update = AsyncMock()
    bot._start_agent_monitor = Mock()
    bot._extract_query_attachments = AsyncMock(return_value=[])
    bot._work_concurrency = 1
    bot._mark_thread_active("C1", "100.0")
    return bot, gate


async def deliver_while_the_agent_start_waits(bot, gate, events):
    """Queue `events`; the one worker must come free while the start waits."""
    bot._start_work_workers()
    try:
        for event in events:
            await bot._enqueue_work(
                SlackWorkItem(kind="event", payload=dict(event), name="slack_event")
            )
        await asyncio.wait_for(bot._work_queue.join(), timeout=1)
    finally:
        gate.set()
        await bot._stop_work_workers()
    while bot.background_tasks:
        await asyncio.gather(*list(bot.background_tasks))


def assert_one_agent_start(bot):
    bot.agent_runtime_client.create_session.assert_awaited_once()
    bot.inference_engine.process_query.assert_not_awaited()
    # Counted once, in the Agent bucket.
    assert len(bot.user_manager.requests_since("U1", 0, bucket="agent")) == 1
    assert bot.user_manager.requests_since("U1", 0) == []


@pytest.mark.asyncio
async def test_an_agent_command_shared_with_a_file_takes_the_agent_lane(tmp_path):
    bot, gate = agent_command_bot(tmp_path)
    event = thread_reply(
        subtype="file_share", text="!llm agent fix the tests", files=[{"id": "F1"}]
    )

    await deliver_while_the_agent_start_waits(bot, gate, [event])

    assert_one_agent_start(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["message", "app_mention"])
async def test_a_mentioned_agent_command_with_a_file_takes_one_lane_either_way(
    tmp_path, first
):
    bot, gate = agent_command_bot(tmp_path)
    # Slack sends a file_share message and an app_mention for one @bot post.
    message = thread_reply(
        subtype="file_share",
        text="<@UBOT> !llm agent fix the tests",
        files=[{"id": "F1"}],
    )
    mention = {**message, "type": "app_mention"}
    mention.pop("subtype")
    events = [message, mention] if first == "message" else [mention, message]

    await deliver_while_the_agent_start_waits(bot, gate, events)

    assert_one_agent_start(bot)


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


# S11: where errors and limits are reported, and what they say


def top_level(ts="200.0", text="what is a monad?"):
    return {"type": "message", "channel": "C1", "user": "U1", "text": text, "ts": ts}


def history(bot, thread_ts):
    context = bot.conversation_manager.get_or_create_context("U1", "C1", thread_ts)
    return context.conversation_history


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,thread_ts",
    [(top_level(ts="200.0"), "200.0"), (thread_reply(ts="101.0"), "100.0")],
    ids=["top_level", "thread_reply"],
)
async def test_an_unexpected_failure_is_reported_in_the_thread(event, thread_ts):
    bot = chat_bot()
    bot.message_handler.handle_message = AsyncMock(side_effect=RuntimeError("boom"))

    await bot._process_message(dict(event))

    post = bot.web_client.chat_postMessage.await_args.kwargs
    assert post["thread_ts"] == thread_ts
    assert post["text"] == bot._build_user_safe_error_message()


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1_000_000.0}
    monkeypatch.setattr("slack.bot_real.time.time", lambda: now["t"])
    return now


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refused,notice_thread_ts",
    [
        # Slack hides an ephemeral sent into a thread that does not exist yet:
        # a new top-level request has no replies, so its notice stays in the
        # channel.
        (top_level(ts="202.0"), None),
        (thread_reply(ts="101.0"), "100.0"),
    ],
    ids=["top_level", "thread_reply"],
)
async def test_a_burst_limit_notice_is_ephemeral_and_names_the_window(
    clock, inference_response_factory, refused, notice_thread_ts
):
    bot = chat_bot(
        answering(inference_response_factory),
        rate_limiting={"requests_per_hour": 100, "burst_requests": 2},
    )
    bot._mark_thread_active("C1", "100.0")
    for ts in ("200.0", "201.0"):
        await bot._process_message(top_level(ts=ts))
        clock["t"] += 60
    posts_before = bot.web_client.chat_postMessage.await_count

    await bot._process_message(dict(refused))

    assert bot.web_client.chat_postMessage.await_count == posts_before
    bot.web_client.chat_postEphemeral.assert_awaited_once()
    notice = bot.web_client.chat_postEphemeral.await_args.kwargs
    assert notice["channel"] == "C1"
    assert notice["user"] == "U1"
    assert notice.get("thread_ts") == notice_thread_ts
    text = notice["text"]
    assert "<@U1>" not in text
    assert "Rate Limit Exceeded" in text
    assert "last 5 minutes" in text
    assert "burst limit" in text
    # The first request leaves the 5-minute window 300 s after it was counted.
    assert "Try again in 3 minutes." in text
    assert "top of each hour" not in text
    assert bot.inference_engine.process_query.await_count == 2


@pytest.mark.asyncio
async def test_an_hourly_limit_notice_names_the_rolling_window(
    clock, inference_response_factory
):
    bot = chat_bot(
        answering(inference_response_factory),
        rate_limiting={"requests_per_hour": 2, "burst_requests": 5},
    )
    bot._mark_thread_active("C1", "100.0")
    await bot._process_message(top_level(ts="200.0"))
    clock["t"] += 1000
    await bot._process_message(top_level(ts="201.0"))
    clock["t"] += 1000

    await bot._process_message(thread_reply(ts="101.0"))

    notice = bot.web_client.chat_postEphemeral.await_args.kwargs
    assert notice["thread_ts"] == "100.0"
    text = notice["text"]
    assert "<@U1>" not in text
    assert "last 60 minutes" in text
    assert "hourly limit" in text
    # The oldest request leaves the rolling hour 1,600 s from now.
    assert "Try again in 27 minutes." in text
    assert "top of each hour" not in text


@pytest.mark.asyncio
async def test_an_agent_limit_notice_counts_agent_requests(clock):
    bot = chat_bot(rate_limiting={"requests_per_hour": 100, "burst_requests": 1})
    assert bot._within_rate_limit("U1", agent=True)
    clock["t"] += 30

    text = bot._build_rate_limit_message("U1", agent=True)

    assert "1 Agent request in the last 5 minutes" in text
    assert "Try again in 5 minutes." in text
    # The chat bucket is empty: its notice cannot name a window.
    assert "You've reached a chat request limit." in bot._build_rate_limit_message("U1")


@pytest.mark.asyncio
async def test_admission_rejection_says_busy_and_when_to_retry():
    from src.admission import AdmissionDecision, AdmissionRejectedError

    rejection = AdmissionRejectedError(
        AdmissionDecision.reject(
            status_code=429,
            error="provider_capacity_exceeded",
            reason="provider_concurrency",
            message="Provider capacity exceeded",
            retry_after_seconds=7,
        )
    )
    bot = chat_bot(AsyncMock(side_effect=rejection))

    await bot._process_message(top_level(ts="200.0"))

    post = bot.web_client.chat_postMessage.await_args.kwargs
    assert post["thread_ts"] == "200.0"
    assert "busy" in post["text"].lower()
    assert "try again in 7 seconds" in post["text"].lower()
    assert history(bot, "200.0") == []


def admission_rejection(reason, retry_after_seconds=90, **metadata):
    from src.admission import AdmissionDecision, AdmissionRejectedError

    return AdmissionRejectedError(
        AdmissionDecision.reject(
            status_code=429,
            error="rate_limited",
            reason=reason,
            message="Request rejected",
            retry_after_seconds=retry_after_seconds,
            metadata=metadata,
        )
    )


def rejection_id(rejection):
    last_reason = rejection.decision.metadata.get("last_reason")
    return (
        f"queue_timeout_on_{last_reason}" if last_reason else rejection.decision.reason
    )


async def reply_to(rejection):
    bot = chat_bot(AsyncMock(side_effect=rejection))
    await bot._process_message(top_level(ts="200.0"))
    assert history(bot, "200.0") == []
    return bot.web_client.chat_postMessage.await_args.kwargs["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection",
    [
        admission_rejection("user_request_rate_exceeded"),
        admission_rejection("identity_request_rate_exceeded"),
        admission_rejection("tier_request_rate_exceeded"),
        admission_rejection("user_token_budget_exceeded"),
        admission_rejection("tier_token_budget_exceeded"),
        # A queue wait that ended on the user's own budget.
        admission_rejection(
            "scheduler_queue_timeout", last_reason="user_token_budget_exceeded"
        ),
    ],
    ids=rejection_id,
)
async def test_an_admission_rejection_of_the_users_own_quota_says_so(rejection):
    text = await reply_to(rejection)

    assert "quota" in text
    assert "busy" not in text.lower()
    assert "Please try again in 2 minutes." in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection",
    [
        admission_rejection("global_active_requests_exceeded"),
        admission_rejection("provider_token_budget_exceeded"),
        admission_rejection("model_active_requests_exceeded"),
        admission_rejection("circuit_open"),
        admission_rejection(
            "scheduler_queue_timeout", last_reason="provider_active_requests_exceeded"
        ),
    ],
    ids=rejection_id,
)
async def test_an_admission_rejection_for_shared_capacity_says_busy(rejection):
    text = await reply_to(rejection)

    assert "busy" in text.lower()
    assert "quota" not in text
    assert "Please try again in 2 minutes." in text


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["", "   \n"])
async def test_an_empty_answer_posts_a_fallback_and_stays_out_of_history(
    answer, inference_response_factory
):
    bot = chat_bot(answering(inference_response_factory, text=answer))

    await bot._process_message(top_level(ts="200.0"))

    post = bot.web_client.chat_postMessage.await_args.kwargs
    assert post["thread_ts"] == "200.0"
    assert "empty answer" in post["text"]
    assert history(bot, "200.0") == []


@pytest.mark.asyncio
async def test_error_text_follows_the_error_type(inference_response_factory):
    async def reply_for(error_type):
        failed = inference_response_factory(
            response_text="Request processing failed",
            provider="error",
            error="inference_failed",
            error_type=error_type,
        )
        bot = chat_bot(AsyncMock(return_value=failed))
        await bot._process_message(top_level(ts="200.0"))
        assert history(bot, "200.0") == []
        return bot.web_client.chat_postMessage.await_args.kwargs["text"]

    generic = chat_bot()._build_user_safe_error_message()
    rate_limited = await reply_for("rate_limited")
    timeout = await reply_for("timeout")
    user_or_policy = await reply_for("user_or_policy")

    assert "rate-limiting" in rate_limited
    assert "took too long" in timeout
    # Any ValueError lands in user_or_policy, an internal one too: never
    # blame the request or the user's tier on that alone.
    assert "too long" not in user_or_policy and "tier" not in user_or_policy
    assert len({generic, rate_limited, timeout, user_or_policy}) == 4
    for other in ("provider_error", "internal", None):
        assert await reply_for(other) == generic


async def reply_for_failure(**fields):
    # InferenceResponse has no error_code on this branch yet; a stand-in does.
    failed = SimpleNamespace(
        response_text="Request processing failed",
        provider="error",
        error="inference_failed",
        **fields,
    )
    bot = chat_bot(AsyncMock(return_value=failed))
    await bot._process_message(top_level(ts="200.0"))
    assert history(bot, "200.0") == []
    return bot.web_client.chat_postMessage.await_args.kwargs["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code,says",
    [
        ("context_length_exceeded", "too long for the model"),
        ("prompt_too_large", "too long for the model"),
        ("model_not_entitled", "not available for your tier"),
        ("retrieval_unavailable", "document search is unavailable right now"),
        ("rate_limited", "rate-limiting"),
        ("provider_unavailable", "provider is unavailable right now"),
        ("timeout", "took too long"),
    ],
)
async def test_error_text_prefers_the_more_precise_error_code(error_code, says):
    text = await reply_for_failure(error_code=error_code, error_type="user_or_policy")

    assert says in text.lower()


@pytest.mark.asyncio
async def test_an_unknown_error_code_falls_back_to_the_error_type():
    timeout = await reply_for_failure(error_type="timeout")

    assert await reply_for_failure(error_code="new_code", error_type="timeout") == (
        timeout
    )


# S14: the tier comes from config on every request


def test_a_stored_tier_is_never_the_source():
    manager = UserManager(default_tier="free", tier_overrides={"U2": "premium"})
    manager.users["U1"] = {"tier": "enterprise"}
    manager.users["U2"] = {"tier": "enterprise"}

    assert manager.get_user_tier("U1") == UserTier.FREE
    assert manager.get_user_tier("U2") == UserTier.PREMIUM


async def ask_after_restart(tmp_path, inference_response_factory, user_tiers):
    """Restart on the saved state with `user_tiers`; the tier one query runs at."""
    bot = chat_bot(
        answering(inference_response_factory),
        state={"backend": "file", "file_path": str(tmp_path / "state.json")},
        user_tiers=user_tiers,
    )
    await bot._restore_state()
    await bot._process_message(top_level(ts="200.0"))
    request = bot.inference_engine.process_query.await_args.args[0]
    return bot, request.user_tier


@pytest.mark.asyncio
async def test_removing_a_tier_override_downgrades_a_persisted_user(
    tmp_path, inference_response_factory
):
    _, tier = await ask_after_restart(
        tmp_path, inference_response_factory, {"overrides": {"U1": "enterprise"}}
    )
    assert tier == UserTier.ENTERPRISE

    bot, tier = await ask_after_restart(
        tmp_path, inference_response_factory, {"overrides": {}}
    )

    assert tier == UserTier.FREE
    assert bot.user_manager.get_user_tier("U1") == UserTier.FREE
    # The record only follows the configured tier.
    assert bot.user_manager.users["U1"]["tier"] == "free"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before,after",
    [("premium", UserTier.FREE), ("enterprise", UserTier.PREMIUM)],
)
async def test_lowering_the_default_tier_takes_effect(
    tmp_path, inference_response_factory, before, after
):
    _, tier = await ask_after_restart(
        tmp_path, inference_response_factory, {"default": before}
    )
    assert tier == UserTier(before)

    bot, tier = await ask_after_restart(
        tmp_path, inference_response_factory, {"default": after.value}
    )

    assert tier == after
    assert bot._get_rate_limit_config_for_user("U1") == (
        bot._get_rate_limit_config_for_tier(after)
    )
