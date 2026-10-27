import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.agent_runtime import (
    AgentRuntimeClient,
    AgentRuntimeConflict,
    AgentRuntimeError,
    AgentRuntimeUnauthorized,
)
from slack.bot_real import SlackBot, SlackMessageHandler, _agent_idempotency_key


@pytest.mark.asyncio
async def test_feedback_client_uses_authenticated_endpoint(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_TOKEN", "runtime-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"status": "accepted"})

    runtime = AgentRuntimeClient({}, transport=httpx.MockTransport(handler))
    try:
        await runtime.feedback("run1", "user1", "needs_changes", "click1")
    finally:
        await runtime.close()
    assert requests[0].url.path == "/v1/runs/run1/feedback"
    assert requests[0].headers["authorization"] == "Bearer runtime-secret"
    assert json.loads(requests[0].content) == {
        "user_id": "user1",
        "verdict": "needs_changes",
        "feedback_id": "click1",
    }


@pytest.mark.asyncio
async def test_feedback_client_sends_optional_reason_and_comment(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_TOKEN", "runtime-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(202, json={"status": "accepted"})

    runtime = AgentRuntimeClient({}, transport=httpx.MockTransport(handler))
    try:
        await runtime.feedback(
            "run1",
            "user1",
            "needs_changes",
            "click1",
            reason_category="incomplete",
            comment="also update the docs",
        )
    finally:
        await runtime.close()
    assert json.loads(requests[0].content) == {
        "user_id": "user1",
        "verdict": "needs_changes",
        "feedback_id": "click1",
        "reason_category": "incomplete",
        "comment": "also update the docs",
    }


@pytest.mark.asyncio
async def test_slack_feedback_does_not_depend_on_active_run_cache_or_call_approval():
    runtime = SimpleNamespace(feedback=AsyncMock(), decide=AsyncMock())
    bot = SimpleNamespace(
        agent_runtime_client=runtime,
        web_client=SimpleNamespace(chat_postEphemeral=AsyncMock()),
    )
    payload = {
        "user": {"id": "owner"},
        "channel": {"id": "channel"},
        "actions": [
            {
                "action_id": "pi_agent_accept_result",
                "value": json.dumps({"run_id": "old-run"}),
                "action_ts": "100.1",
            }
        ],
    }
    await SlackBot._handle_agent_interactive(bot, payload)
    runtime.feedback.assert_awaited_once_with("old-run", "owner", "accepted", "100.1")
    runtime.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_feedback_reports_runtime_owner_rejection():
    runtime = SimpleNamespace(
        feedback=AsyncMock(side_effect=AgentRuntimeUnauthorized())
    )
    bot = SimpleNamespace(
        agent_runtime_client=runtime,
        web_client=SimpleNamespace(chat_postEphemeral=AsyncMock()),
    )
    await SlackBot._handle_agent_interactive(
        bot,
        {
            "user": {"id": "other"},
            "channel": {"id": "channel"},
            "actions": [
                {
                    "action_id": "pi_agent_needs_changes",
                    "value": json.dumps({"run_id": "r"}),
                    "action_ts": "101.1",
                }
            ],
        },
    )
    assert "owner" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_feedback_buttons_preserve_long_result_with_valid_slack_sections():
    answer = "中文" * 2000
    bot = SimpleNamespace(
        web_client=SimpleNamespace(chat_postMessage=AsyncMock()),
        config={"response_settings": {"max_response_length": 5000}},
        agent_config={"feedback_enabled": True},
        message_handler=SimpleNamespace(
            _format_agent_response=Mock(return_value=answer)
        ),
    )
    await SlackBot._post_agent_result(
        bot,
        {"run_id": "r", "channel_id": "c", "thread_ts": "t"},
        {"run_id": "r", "status": "completed"},
    )
    blocks = bot.web_client.chat_postMessage.await_args.kwargs["blocks"]
    sections = [block["text"]["text"] for block in blocks if block["type"] == "section"]
    assert all(len(text) <= 3000 for text in sections)
    assert "".join(sections) == answer
    assert len(blocks[-1]["elements"]) == 2


def _result_bot(feedback_enabled=False):
    return SimpleNamespace(
        web_client=SimpleNamespace(chat_postMessage=AsyncMock()),
        config={"response_settings": {"max_response_length": 2000}},
        agent_config={"feedback_enabled": feedback_enabled},
        message_handler=SimpleNamespace(
            _format_agent_response=SlackMessageHandler._format_agent_response
        ),
    )


def _posted_sections(bot):
    return [
        [
            block["text"]["text"]
            for block in call.kwargs["blocks"]
            if block["type"] == "section"
        ]
        for call in bot.web_client.chat_postMessage.await_args_list
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("feedback_enabled", [False, True])
async def test_long_final_answer_is_split_across_messages_not_cut(feedback_enabled):
    lines = [f"line {index:05d} " + "x" * 80 for index in range(500)]
    answer = "\n".join(lines)
    bot = _result_bot(feedback_enabled)

    await SlackBot._post_agent_result(
        bot,
        {"run_id": "r", "team_id": "T1", "channel_id": "c", "thread_ts": "t"},
        {"run_id": "r", "status": "completed", "answer": answer},
    )

    calls = bot.web_client.chat_postMessage.await_args_list
    assert len(calls) > 1
    assert all(call.kwargs["thread_ts"] == "t" for call in calls)
    sections = [text for message in _posted_sections(bot) for text in message]
    assert all(len(text) <= 3000 for text in sections)
    posted = "\n".join(sections)
    assert all(line in posted for line in lines)
    assert "…" not in posted
    for call in calls:
        assert len(call.kwargs["blocks"]) <= 11
        assert len(call.kwargs["text"]) <= 40000
    actions = [
        block
        for call in calls
        for block in call.kwargs["blocks"]
        if block["type"] == "actions"
    ]
    if feedback_enabled:
        assert len(actions) == 1
        assert calls[-1].kwargs["blocks"][-1]["type"] == "actions"
    else:
        assert actions == []


@pytest.mark.asyncio
async def test_split_answer_keeps_code_fences_balanced_per_block():
    code = "\n".join(f"print({index})  # " + "y" * 60 for index in range(200))
    answer = f"Here is the patch:\n```\n{code}\n```\nDone."
    bot = _result_bot()

    await SlackBot._post_agent_result(
        bot,
        {"run_id": "r", "channel_id": "c", "thread_ts": "t"},
        {"run_id": "r", "status": "completed", "answer": answer},
    )

    sections = [text for message in _posted_sections(bot) for text in message]
    assert len(sections) > 1
    assert all(text.count("```") % 2 == 0 for text in sections)
    assert any("print(199)" in text for text in sections)


@pytest.mark.asyncio
async def test_enormous_answer_is_capped_with_explicit_notice():
    bot = _result_bot()

    await SlackBot._post_agent_result(
        bot,
        {"run_id": "r", "channel_id": "c", "thread_ts": "t"},
        {"run_id": "r", "status": "completed", "answer": "z" * 500_000},
    )

    sections = [text for message in _posted_sections(bot) for text in message]
    assert all(len(text) <= 3000 for text in sections)
    assert sum(len(text) for text in sections) < 100_000
    assert "truncated" in sections[-1]


def _feedback_bot(runtime=None):
    bot = SlackBot(
        {"channels": []},
        SimpleNamespace(),
        services={"agent_runtime": runtime or SimpleNamespace()},
    )
    bot.web_client = SimpleNamespace(
        views_open=AsyncMock(),
        chat_postEphemeral=AsyncMock(),
        chat_postMessage=AsyncMock(),
    )
    return bot


def _needs_changes_click(user="owner", trigger_id="trigger-1"):
    return {
        "type": "block_actions",
        "trigger_id": trigger_id,
        "team": {"id": "T1"},
        "user": {"id": user},
        "channel": {"id": "C1"},
        "container": {"thread_ts": "100.1"},
        "actions": [
            {
                "action_id": "pi_agent_needs_changes",
                "value": json.dumps(
                    {
                        "run_id": "r1",
                        "team_id": "T1",
                        "channel_id": "C1",
                        "thread_ts": "100.1",
                    }
                ),
                "action_ts": "200.1",
            }
        ],
    }


def _feedback_submission(view, user="owner", continue_session=False):
    values = {
        "feedback_reason": {"reason": {"selected_option": {"value": "broke_tests"}}},
        "feedback_comment": {"comment": {"value": "  rerun the failing suite  "}},
        "feedback_continue": {
            "continue": {
                "selected_options": (
                    [{"value": "continue"}] if continue_session else []
                )
            }
        },
    }
    return {
        "type": "view_submission",
        "team": {"id": "T1"},
        "user": {"id": user},
        "view": {
            "callback_id": view["callback_id"],
            "private_metadata": view["private_metadata"],
            "state": {"values": values},
        },
    }


async def _open_feedback_modal(bot):
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    bot._enqueue_work = AsyncMock(return_value=True)
    await bot._handle_socket_mode_request(
        socket_client,
        SimpleNamespace(
            envelope_id="E1", type="interactive", payload=_needs_changes_click()
        ),
    )
    return bot.web_client.views_open.await_args.kwargs


def _texts(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "text" and isinstance(item, str):
                yield item
            else:
                yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["completed", "failed", "cancelled", "rejected", "timed_out", "interrupted"],
)
async def test_feedback_buttons_shown_for_every_terminal_status(status):
    bot = SimpleNamespace(
        web_client=SimpleNamespace(chat_postMessage=AsyncMock()),
        config={"response_settings": {"max_response_length": 5000}},
        agent_config={"feedback_enabled": True},
        message_handler=SimpleNamespace(
            _format_agent_response=Mock(return_value="Run timed out.")
        ),
    )
    await SlackBot._post_agent_result(
        bot,
        {"run_id": "r", "team_id": "T1", "channel_id": "c", "thread_ts": "t"},
        {"run_id": "r", "status": status},
    )
    blocks = bot.web_client.chat_postMessage.await_args.kwargs["blocks"]
    buttons = blocks[-1]["elements"]
    assert [b["action_id"] for b in buttons] == [
        "pi_agent_accept_result",
        "pi_agent_needs_changes",
    ]
    assert json.loads(buttons[1]["value"]) == {
        "run_id": "r",
        "team_id": "T1",
        "channel_id": "c",
        "thread_ts": "t",
    }
    assert all(len(text) <= 3000 for text in _texts(blocks))


@pytest.mark.asyncio
async def test_needs_changes_opens_modal_synchronously_without_work_queue():
    bot = _feedback_bot()
    opened = await _open_feedback_modal(bot)

    bot._enqueue_work.assert_not_awaited()
    assert opened["trigger_id"] == "trigger-1"
    view = opened["view"]
    assert view["type"] == "modal"
    assert view["callback_id"] == "pi_agent_feedback_submit"
    assert view["private_metadata"]
    assert len(view["private_metadata"]) <= 3000
    by_id = {block.get("block_id"): block for block in view["blocks"]}
    reasons = [o["value"] for o in by_id["feedback_reason"]["element"]["options"]]
    assert reasons == [
        "wrong_approach",
        "incomplete",
        "broke_tests",
        "scope_creep",
        "style",
        "other",
    ]
    assert by_id["feedback_comment"]["element"]["max_length"] == 2000
    assert by_id["feedback_continue"]["element"]["type"] == "checkboxes"
    assert all(len(text) <= 3000 for text in _texts(view))


@pytest.mark.asyncio
async def test_needs_changes_falls_back_to_queued_verdict_when_modal_fails():
    bot = _feedback_bot()
    bot.web_client.views_open = AsyncMock(side_effect=RuntimeError("expired"))
    await _open_feedback_modal(bot)

    item = bot._enqueue_work.await_args.args[0]
    assert item.kind == "interactive"
    assert item.payload["actions"][0]["action_id"] == "pi_agent_needs_changes"
    assert bot.agent_feedback_nonces == {}


@pytest.mark.asyncio
async def test_feedback_submission_is_queued_by_callback_id():
    bot = _feedback_bot()
    view = (await _open_feedback_modal(bot))["view"]
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    await bot._handle_socket_mode_request(
        socket_client,
        SimpleNamespace(
            envelope_id="E2", type="interactive", payload=_feedback_submission(view)
        ),
    )
    item = bot._enqueue_work.await_args.args[0]
    assert item.kind == "agent_view"


@pytest.mark.asyncio
async def test_feedback_submission_sends_category_and_comment_once():
    runtime = SimpleNamespace(feedback=AsyncMock())
    bot = _feedback_bot(runtime)
    bot.message_handler._start_agent_prompt = AsyncMock()
    view = (await _open_feedback_modal(bot))["view"]

    await bot._handle_agent_view_submission(_feedback_submission(view))
    await bot._handle_agent_view_submission(_feedback_submission(view))

    runtime.feedback.assert_awaited_once_with(
        "r1",
        "owner",
        "needs_changes",
        "200.1",
        reason_category="broke_tests",
        comment="rerun the failing suite",
    )
    bot.message_handler._start_agent_prompt.assert_not_awaited()
    assert "recorded" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_feedback_submission_from_non_owner_is_rejected():
    runtime = SimpleNamespace(feedback=AsyncMock())
    bot = _feedback_bot(runtime)
    bot.message_handler._start_agent_prompt = AsyncMock()
    view = (await _open_feedback_modal(bot))["view"]

    await bot._handle_agent_view_submission(
        _feedback_submission(view, user="intruder", continue_session=True)
    )

    runtime.feedback.assert_not_awaited()
    bot.message_handler._start_agent_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_owner_rejection_does_not_continue_session():
    runtime = SimpleNamespace(
        feedback=AsyncMock(side_effect=AgentRuntimeUnauthorized())
    )
    bot = _feedback_bot(runtime)
    bot.message_handler._start_agent_prompt = AsyncMock()
    view = (await _open_feedback_modal(bot))["view"]

    await bot._handle_agent_view_submission(
        _feedback_submission(view, continue_session=True)
    )

    bot.message_handler._start_agent_prompt.assert_not_awaited()
    assert "owner" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_checked_continue_prompts_the_session_with_the_comment():
    runtime = SimpleNamespace(feedback=AsyncMock())
    bot = _feedback_bot(runtime)
    bot.message_handler._start_agent_prompt = AsyncMock(
        return_value="🛠️ Agent run `r2` started."
    )
    view = (await _open_feedback_modal(bot))["view"]

    await bot._handle_agent_view_submission(
        _feedback_submission(view, continue_session=True)
    )

    runtime.feedback.assert_awaited_once()
    call = bot.message_handler._start_agent_prompt.await_args
    assert call.args == ("rerun the failing suite", "owner", "C1", bot.web_client)
    assert call.kwargs["team_id"] == "T1"
    assert call.kwargs["thread_ts"] == "100.1"
    posted = bot.web_client.chat_postMessage.await_args.kwargs
    assert posted["thread_ts"] == "100.1"
    assert "r2" in posted["text"]


def _patch_action_ids(blocks):
    return [
        element["action_id"]
        for block in blocks
        if block["type"] == "actions"
        for element in block["elements"]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("feedback_enabled", [False, True])
async def test_failed_run_with_retained_patch_offers_download_and_resume(
    feedback_enabled,
):
    bot = _result_bot(feedback_enabled)
    await SlackBot._post_agent_result(
        bot,
        {
            "run_id": "r",
            "session_id": "s",
            "team_id": "T1",
            "channel_id": "c",
            "thread_ts": "t",
        },
        {"run_id": "r", "status": "failed", "patch_retained": True},
    )
    blocks = bot.web_client.chat_postMessage.await_args.kwargs["blocks"]
    ids = _patch_action_ids(blocks)
    assert ids[:2] == ["pi_agent_download_patch", "pi_agent_resume_from_patch"]
    patch_block = next(
        block
        for block in blocks
        if block["type"] == "actions"
        and block["elements"][0]["action_id"] == "pi_agent_download_patch"
    )
    assert json.loads(patch_block["elements"][1]["value"]) == {
        "run_id": "r",
        "session_id": "s",
        "team_id": "T1",
        "channel_id": "c",
        "thread_ts": "t",
    }
    if feedback_enabled:
        assert blocks[-1]["elements"][0]["action_id"] == "pi_agent_accept_result"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        {"run_id": "r", "status": "failed"},
        {"run_id": "r", "status": "completed", "patch_retained": True},
    ],
)
async def test_patch_buttons_need_an_undelivered_run_with_a_retained_patch(run):
    bot = _result_bot()
    await SlackBot._post_agent_result(
        bot, {"run_id": "r", "channel_id": "c", "thread_ts": "t"}, run
    )
    blocks = bot.web_client.chat_postMessage.await_args.kwargs["blocks"]
    assert _patch_action_ids(blocks) == []


def _patch_bot(runtime, tracked=None):
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.web_client = SimpleNamespace(
        chat_postEphemeral=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "200.1"}),
        files_upload_v2=AsyncMock(),
    )
    bot._start_agent_monitor = Mock()
    bot._persist_agent_run = AsyncMock()
    if tracked:
        bot.agent_runs[tracked["run_id"]] = tracked
    return bot


def _patch_click(action_id, user="owner"):
    return {
        "user": {"id": user},
        "channel": {"id": "c"},
        "team": {"id": "T1"},
        "actions": [
            {
                "action_id": action_id,
                "action_ts": "300.1",
                "value": json.dumps(
                    {
                        "run_id": "r",
                        "session_id": "s",
                        "team_id": "T1",
                        "channel_id": "c",
                        "thread_ts": "t",
                    }
                ),
            }
        ],
    }


@pytest.mark.asyncio
async def test_download_patch_uploads_the_redacted_patch_to_the_thread():
    runtime = SimpleNamespace(
        get_run_patch=AsyncMock(
            return_value={"patch": b"diff --git a/a b/a\n", "truncated": False}
        )
    )
    bot = _patch_bot(runtime)
    await bot._handle_agent_interactive(_patch_click("pi_agent_download_patch"))
    runtime.get_run_patch.assert_awaited_once_with("r", "owner")
    upload = bot.web_client.files_upload_v2.await_args.kwargs
    assert upload["channel"] == "c"
    assert upload["thread_ts"] == "t"
    assert upload["content"] == b"diff --git a/a b/a\n"
    assert upload["filename"] == "pi-agent-r.patch"
    assert "truncated" not in upload["initial_comment"]


@pytest.mark.asyncio
async def test_download_patch_says_when_the_patch_was_truncated():
    runtime = SimpleNamespace(
        get_run_patch=AsyncMock(return_value={"patch": b"x", "truncated": True})
    )
    bot = _patch_bot(runtime)
    await bot._handle_agent_interactive(_patch_click("pi_agent_download_patch"))
    assert (
        "truncated"
        in bot.web_client.files_upload_v2.await_args.kwargs["initial_comment"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_id", ["pi_agent_download_patch", "pi_agent_resume_from_patch"]
)
async def test_patch_actions_are_owner_only(action_id):
    runtime = SimpleNamespace(get_run_patch=AsyncMock(), resume_from_patch=AsyncMock())
    tracked = {
        "run_id": "r",
        "session_id": "s",
        "team_id": "T1",
        "channel_id": "c",
        "thread_ts": "t",
        "owner_user_id": "owner",
    }
    bot = _patch_bot(runtime, tracked)
    await bot._handle_agent_interactive(_patch_click(action_id, user="intruder"))
    runtime.get_run_patch.assert_not_awaited()
    runtime.resume_from_patch.assert_not_awaited()
    bot.web_client.files_upload_v2.assert_not_awaited()
    text = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "Only the user who started this Agent run" in text


@pytest.mark.asyncio
async def test_runtime_refusal_of_a_non_owner_patch_download_is_reported():
    runtime = SimpleNamespace(
        get_run_patch=AsyncMock(
            side_effect=AgentRuntimeError(status=403, code="patch_forbidden")
        )
    )
    bot = _patch_bot(runtime)
    await bot._handle_agent_interactive(
        _patch_click("pi_agent_download_patch", user="intruder")
    )
    bot.web_client.files_upload_v2.assert_not_awaited()
    text = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "Only the user who started this Agent run" in text


@pytest.mark.asyncio
async def test_resume_from_patch_starts_and_tracks_a_new_run_with_a_stable_key():
    runtime = SimpleNamespace(
        resume_from_patch=AsyncMock(
            return_value={"session_id": "s", "run_id": "r2", "status": "starting"}
        )
    )
    bot = _patch_bot(runtime)
    await bot._handle_agent_interactive(_patch_click("pi_agent_resume_from_patch"))
    await bot._handle_agent_interactive(_patch_click("pi_agent_resume_from_patch"))
    first, second = runtime.resume_from_patch.await_args_list
    assert first.args[:2] == ("r", "owner")
    expected = _agent_idempotency_key("T1", "c", "r", "owner", "resume_from_patch")
    assert first.args[2] == expected
    assert second.args[2] == expected
    tracked = bot.agent_runs["r2"]
    assert tracked["owner_user_id"] == "owner"
    assert (tracked["channel_id"], tracked["thread_ts"]) == ("c", "t")
    assert tracked["status_ts"] == "200.1"
    posted = bot.web_client.chat_postMessage.await_args.kwargs
    assert posted["thread_ts"] == "t"
    assert "`r`" in posted["text"] and "`r2`" in posted["text"]
    bot._start_agent_monitor.assert_called_with(tracked)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected",
    [
        ("session_busy", "already has a run in progress"),
        ("patch_conflict", "no longer applies cleanly"),
        ("worktree_not_clean", "uncommitted changes"),
        ("patch_truncated", "truncated"),
    ],
)
async def test_resume_from_patch_refusals_are_explained(code, expected):
    runtime = SimpleNamespace(
        resume_from_patch=AsyncMock(
            side_effect=AgentRuntimeConflict(status=409, code=code)
        )
    )
    bot = _patch_bot(runtime)
    await bot._handle_agent_interactive(_patch_click("pi_agent_resume_from_patch"))
    bot._start_agent_monitor.assert_not_called()
    assert "r2" not in bot.agent_runs
    assert expected in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]


def test_client_reports_feedback_capture_from_cached_health():
    runtime = AgentRuntimeClient({})
    assert runtime.cached_feedback_capture() is None
    runtime._health_cache = {"status": "healthy", "telemetry": {"status": "disabled"}}
    assert runtime.cached_feedback_capture() is False
    runtime._health_cache = {"status": "healthy", "telemetry": {"depth": 0}}
    assert runtime.cached_feedback_capture() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, None])
async def test_feedback_buttons_follow_runtime_feedback_capture(capture):
    bot = _result_bot(feedback_enabled=True)
    bot.agent_runtime_client = SimpleNamespace(
        cached_feedback_capture=Mock(return_value=capture)
    )
    await SlackBot._post_agent_result(
        bot,
        {"run_id": "r", "channel_id": "c", "thread_ts": "t"},
        {"run_id": "r", "status": "completed", "answer": "done"},
    )
    blocks = bot.web_client.chat_postMessage.await_args.kwargs["blocks"]
    action_ids = [
        element["action_id"]
        for block in blocks
        if block["type"] == "actions"
        for element in block["elements"]
    ]
    if capture is False:
        assert "pi_agent_accept_result" not in action_ids
    else:
        assert action_ids == ["pi_agent_accept_result", "pi_agent_needs_changes"]


@pytest.mark.asyncio
async def test_feedback_click_explains_that_capture_is_off():
    runtime = SimpleNamespace(
        feedback=AsyncMock(
            side_effect=AgentRuntimeError(status=503, code="feedback_unavailable")
        )
    )
    bot = SimpleNamespace(
        agent_runtime_client=runtime,
        web_client=SimpleNamespace(chat_postEphemeral=AsyncMock()),
    )
    await SlackBot._handle_agent_interactive(
        bot,
        {
            "user": {"id": "owner"},
            "channel": {"id": "channel"},
            "actions": [
                {
                    "action_id": "pi_agent_accept_result",
                    "value": json.dumps({"run_id": "r"}),
                    "action_ts": "101.1",
                }
            ],
        },
    )
    text = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "telemetry" in text
    assert "try again" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        AgentRuntimeError(status=503, code="feedback_unavailable"),
        AgentRuntimeError(status=500, code="internal_error"),
    ],
)
async def test_checked_continue_starts_the_run_even_when_feedback_is_not_recorded(
    error,
):
    runtime = SimpleNamespace(feedback=AsyncMock(side_effect=error))
    bot = _feedback_bot(runtime)
    bot.message_handler._start_agent_prompt = AsyncMock(
        return_value="🛠️ Agent run `r2` started."
    )
    view = (await _open_feedback_modal(bot))["view"]

    await bot._handle_agent_view_submission(
        _feedback_submission(view, continue_session=True)
    )

    assert "not" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    call = bot.message_handler._start_agent_prompt.await_args
    assert call.args == ("rerun the failing suite", "owner", "C1", bot.web_client)
    assert call.kwargs["thread_ts"] == "100.1"
