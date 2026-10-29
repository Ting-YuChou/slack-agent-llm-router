import ast
import asyncio
import dataclasses
import inspect
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import httpx
import pytest


def test_slack_socket_mode_uses_supported_aiohttp_client_import():
    from pathlib import Path

    source = Path("slack/bot_real.py").read_text()
    assert "from slack_sdk.socket_mode.aiohttp import (" in source
    assert "SocketModeClient as AsyncSocketModeClient" in source


from src.agent_runtime import (
    AgentRuntimeConflict,
    AgentRuntimeError,
    AgentRuntimeUnauthorized,
)
from src.memory import HashEmbeddingProvider, InMemoryMemoryStore, MemoryManager
from slack.agent_bootstrap import AgentBootstrapBuilder, AgentBootstrapContext
from slack.agent_shortcut import build_agent_shortcut_modal
from slack.bot_real import (
    AGENT_ERROR_HINTS,
    AGENT_FOLLOWUP_FLAG_PATTERN,
    SLACK_BUSY_MESSAGE,
    ConversationContext,
    ConversationManager,
    FileSlackStateStore,
    RedisSlackStateStore,
    SlackBot,
    SlackMessageHandler,
    SlackWorkItem,
    UserManager,
    _agent_idempotency_key,
    _format_agent_status_line,
)
from src.utils.schema import (
    Attachment,
    AttachmentType,
    ResponseSource,
    ToolPolicy,
    UserTier,
)


def _parsed_literals(method):
    """Return the `--flags` and first-argument actions a command handler parses."""
    module = ast.parse(Path(inspect.getsourcefile(method)).read_text())
    first_line = inspect.getsourcelines(method)[1]
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method.__name__
        and first_line in (node.lineno, *(d.lineno for d in node.decorator_list))
    )
    flags, actions = set(), set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare):
            continue
        constants = {
            item.value
            for operand in (node.left, *node.comparators)
            for item in ast.walk(operand)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        flags |= {value for value in constants if value.startswith("--")}
        if isinstance(node.left, ast.Name) and node.left.id == "action":
            actions |= constants
    return flags, actions


class FailingEmbeddingProvider:
    async def embed(self, text):
        raise RuntimeError("embedding unavailable")


class TestUserManager:
    def test_get_user_tier_defaults_and_preferences(self):
        manager = UserManager()

        assert manager.get_user_tier("unknown") == UserTier.FREE
        assert manager.get_user_preferences("unknown")["response_length"] == "medium"

        manager.users["u1"] = {"tier": "premium"}
        assert manager.get_user_tier("u1") == UserTier.FREE

    def test_check_rate_limit_enforces_burst_window(self, monkeypatch):
        manager = UserManager()
        now = 1_000_000.0
        monkeypatch.setattr("slack.bot_real.time.time", lambda: now)

        config = {"requests_per_hour": 10, "burst_requests": 2}

        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is False

    def test_agent_requests_have_their_own_rate_limit_bucket(self, monkeypatch):
        manager = UserManager()
        monkeypatch.setattr("slack.bot_real.time.time", lambda: 1_000_000.0)
        config = {"requests_per_hour": 3, "burst_requests": 2}

        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is False
        assert manager.check_rate_limit("u1", config, bucket="agent") is True
        assert manager.check_rate_limit("u1", config, bucket="agent") is True
        assert manager.check_rate_limit("u1", config, bucket="agent") is False
        assert manager.check_rate_limit("u1", config) is False
        # Both buckets keep the hourly limit too, an hour's window apart.
        monkeypatch.setattr("slack.bot_real.time.time", lambda: 1_000_400.0)
        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is False
        assert manager.check_rate_limit("u1", config, bucket="agent") is True
        assert manager.check_rate_limit("u1", config, bucket="agent") is False
        # Every request that went through counts as one Slack query.
        assert len(manager.rate_limits["u1"]) == 6


class TestConversationHelpers:
    def test_conversation_context_caps_history(self):
        context = ConversationContext(
            user_id="u1",
            channel_id="c1",
            thread_ts=None,
            conversation_history=[],
            user_tier=UserTier.FREE,
            preferences={},
            last_activity=datetime.now(),
            session_id="session-1",
        )

        for index in range(25):
            context.add_message("user", f"msg-{index}")

        assert len(context.conversation_history) == 20
        assert context.conversation_history[0]["content"] == "msg-5"

    def test_conversation_manager_reuses_and_cleans_expired_sessions(self):
        manager = ConversationManager({})
        context = manager.get_or_create_context("u1", "c1")
        context.last_activity = datetime.now() - timedelta(hours=3)

        manager.cleanup_expired_sessions()

        assert manager.conversations == {}

    def test_conversation_summary_uses_last_messages(self):
        manager = ConversationManager({})
        context = manager.get_or_create_context("u1", "c1")
        context.add_message("user", "first")
        context.add_message("assistant", "second")

        summary = manager.get_conversation_summary("u1", "c1")

        assert "user: first" in summary
        assert "assistant: second" in summary

    def test_clear_contexts_removes_all_threads_for_user_channel(self):
        manager = ConversationManager({})
        manager.get_or_create_context("u1", "c1")
        manager.get_or_create_context("u1", "c1", "thread-1")
        manager.get_or_create_context("u2", "c1")

        cleared = manager.clear_contexts("u1", "c1")

        assert cleared == 2
        assert list(manager.conversations.keys()) == ["u2:c1:main"]


class TestSlackMessageHandler:
    @pytest.mark.asyncio
    async def test_agent_command_creates_thread_session_without_chat_history(self):
        agent_runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={
                    "session_id": "session-1",
                    "run_id": "run-1",
                    "status": "starting",
                }
            )
        )
        inference_engine = SimpleNamespace(process_query=AsyncMock())
        conversation_manager = ConversationManager({})
        bot = SlackBot(
            {"channels": []},
            inference_engine=inference_engine,
            services={"agent_runtime": agent_runtime},
        )
        bot.conversation_manager = conversation_manager
        bot._monitor_agent_run = AsyncMock()
        client = SimpleNamespace(
            chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
            chat_update=AsyncMock(),
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command(
            "agent fix the failing test",
            "U1",
            "C1",
            None,
            client=client,
            team_id="T1",
        )

        assert response == ""
        agent_runtime.create_session.assert_awaited_once_with(
            "T1",
            "C1",
            "100.1",
            "U1",
            "fix the failing test",
            routing_text="fix the failing test",
        )
        inference_engine.process_query.assert_not_awaited()
        assert conversation_manager.get_conversation_summary("U1", "C1") == ""
        assert bot.agent_runs["run-1"]["thread_ts"] == "100.1"

    @pytest.mark.asyncio
    async def test_agent_no_thread_context_flag_sends_only_task(self):
        agent_runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={
                    "session_id": "session-1",
                    "run_id": "run-1",
                    "status": "starting",
                }
            )
        )
        bot = SlackBot(
            {"channels": []},
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
            services={"agent_runtime": agent_runtime},
        )
        bot._monitor_agent_run = AsyncMock()
        bot.agent_bootstrap_builder = SimpleNamespace(build=AsyncMock())
        handler = SlackMessageHandler(bot)

        await handler._handle_command(
            "agent --no-thread-context fix the test",
            "U1",
            "C1",
            "100.1",
            client=SimpleNamespace(),
            team_id="T1",
            message_ts="100.2",
        )

        assert agent_runtime.create_session.await_args.args[4] == "fix the test"
        bot.agent_bootstrap_builder.build.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inline_agent_command_bootstraps_prior_thread_once(self):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
            )
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot._monitor_agent_run = AsyncMock()
        bot._is_public_agent_channel = AsyncMock(return_value=True)
        bot.agent_bootstrap_builder = SimpleNamespace(
            build=AsyncMock(
                return_value=AgentBootstrapContext(prompt="context plus current task")
            )
        )
        handler = SlackMessageHandler(bot)

        await handler._handle_command(
            "agent fix the test",
            "U1",
            "C1",
            "100.1",
            client=SimpleNamespace(),
            team_id="T1",
            message_ts="100.2",
        )

        bot.agent_bootstrap_builder.build.assert_awaited_once_with(
            ANY,
            channel_id="C1",
            thread_ts="100.1",
            owner_user_id="U1",
            task="fix the test",
            latest_ts="100.2",
        )
        assert runtime.create_session.await_args.args[4] == "context plus current task"
        assert (
            runtime.create_session.await_args.kwargs["routing_text"] == "fix the test"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "channel",
        [
            {"is_channel": True, "is_private": False, "is_ext_shared": True},
            {"is_channel": True, "is_private": False, "is_shared": True},
            {"is_channel": True, "is_private": True},
            {"is_im": True},
        ],
    )
    async def test_inline_agent_command_skips_thread_context_outside_public_channels(
        self, channel
    ):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
            )
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(return_value={"ok": True, "channel": channel})
        )
        bot._monitor_agent_run = AsyncMock()
        bot.agent_bootstrap_builder = SimpleNamespace(
            build=AsyncMock(return_value=AgentBootstrapContext(prompt="leaked"))
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command(
            "agent fix the test",
            "U1",
            "C1",
            "100.1",
            client=SimpleNamespace(),
            team_id="T1",
            message_ts="100.2",
        )

        bot.agent_bootstrap_builder.build.assert_not_awaited()
        assert runtime.create_session.await_args.args[4] == "fix the test"
        assert "without prior Slack thread context" in response

    @pytest.mark.asyncio
    async def test_agent_command_without_task_returns_usage_without_runtime_call(self):
        agent_runtime = SimpleNamespace(create_session=AsyncMock())
        inference_engine = SimpleNamespace(process_query=AsyncMock())
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=inference_engine,
            agent_runtime_client=agent_runtime,
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command("agent", "u1", "c1", None, client=None)

        assert response == "Usage: `/llm agent <task>`"
        agent_runtime.create_session.assert_not_awaited()
        inference_engine.process_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_agent_command_selects_provider_model_without_putting_flag_in_prompt(
        self,
    ):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
            )
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot._monitor_agent_run = AsyncMock()
        handler = SlackMessageHandler(bot)
        client = SimpleNamespace(
            chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
            chat_update=AsyncMock(),
        )

        response = await handler._handle_command(
            "agent --model anthropic/claude-sonnet-4-6 fix the tests",
            "U1",
            "C1",
            None,
            client=client,
            team_id="T1",
        )

        assert response == ""
        runtime.create_session.assert_awaited_once_with(
            "T1",
            "C1",
            "100.1",
            "U1",
            "fix the tests",
            model="anthropic/claude-sonnet-4-6",
            routing_text="fix the tests",
        )

    @pytest.mark.asyncio
    async def test_agent_status_stop_close_do_not_consume_quota(self):
        runtime = SimpleNamespace(
            get_run=AsyncMock(return_value={"run_id": "R1", "status": "running"}),
            cancel=AsyncMock(),
            close_session=AsyncMock(),
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot.agent_user_runs["U1"] = {
            "run_id": "R1",
            "session_id": "S1",
            "thread_ts": "1",
            "channel_id": "C1",
            "owner_user_id": "U1",
        }
        handler = SlackMessageHandler(bot)

        assert handler.command_consumes_query_quota("agent status") is False
        assert handler.command_consumes_query_quota("agent stop") is False
        assert handler.command_consumes_query_quota("agent close") is False
        assert (
            "running"
            in (
                await handler._handle_command("agent status", "U1", "C1", None, None)
            ).lower()
        )
        await handler._handle_command("agent stop", "U1", "C1", None, None)
        await handler._handle_command("agent close", "U1", "C1", None, None)

        runtime.cancel.assert_awaited_once_with("R1", "U1")
        runtime.close_session.assert_awaited_once_with("S1", "U1")

    @pytest.mark.asyncio
    async def test_agent_thread_followup_resumes_same_sidecar_session(self):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R2", "status": "starting"}
            )
        )
        inference = SimpleNamespace(process_query=AsyncMock())
        bot = SlackBot({"channels": []}, inference, services={"agent_runtime": runtime})
        bot._monitor_agent_run = AsyncMock()
        bot._track_agent_run(
            {"session_id": "S1", "run_id": "R1", "status": "starting"},
            team_id="T1",
            channel_id="C1",
            thread_ts="100.1",
            owner_user_id="U1",
        )
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            {
                "text": "also add a regression test",
                "user": "U1",
                "channel": "C1",
                "thread_ts": "100.1",
                "_team_id": "T1",
            },
            SimpleNamespace(),
        )

        assert "R2" in response
        runtime.create_session.assert_awaited_once_with(
            "T1",
            "C1",
            "100.1",
            "U1",
            "also add a regression test",
            routing_text="also add a regression test",
        )
        inference.process_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_attachment_only_agent_followup_does_not_route_attachment_summary(
        self,
    ):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R2", "status": "starting"}
            )
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot._monitor_agent_run = AsyncMock()
        bot._track_agent_run(
            {"session_id": "S1", "run_id": "R1", "status": "starting"},
            team_id="T1",
            channel_id="C1",
            thread_ts="100.1",
            owner_user_id="U1",
        )
        handler = SlackMessageHandler(bot)
        handler._build_attachment_only_query = lambda _attachments: "attachment summary"

        await handler.handle_message(
            {
                "text": "",
                "user": "U1",
                "channel": "C1",
                "thread_ts": "100.1",
                "_team_id": "T1",
                "_query_attachments": [{"name": "private.pdf"}],
            },
            SimpleNamespace(),
        )

        runtime.create_session.assert_awaited_once_with(
            "T1", "C1", "100.1", "U1", "attachment summary"
        )

    @staticmethod
    def _followup_bot(previous_mode=None):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R2", "status": "starting"}
            )
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot._monitor_agent_run = AsyncMock()
        bot._track_agent_run(
            {"session_id": "S1", "run_id": "R1", "status": "starting"},
            team_id="T1",
            channel_id="C1",
            thread_ts="100.1",
            owner_user_id="U1",
            mode=previous_mode,
        )
        return bot, runtime

    @staticmethod
    def _followup_event(text):
        return {
            "text": text,
            "user": "U1",
            "channel": "C1",
            "thread_ts": "100.1",
            "_team_id": "T1",
        }

    @pytest.mark.asyncio
    async def test_agent_thread_followup_sends_its_attachments_to_the_agent(self):
        bot, runtime = self._followup_bot()

        async def http_handler(request):
            assert request.headers["Authorization"] == "Bearer xoxb-test"
            return httpx.Response(200, text="Traceback: line 42")

        bot.agent_bootstrap_builder = AgentBootstrapBuilder(
            {},
            bot_token="xoxb-test",
            http_transport=httpx.MockTransport(http_handler),
        )
        handler = SlackMessageHandler(bot)
        event = {
            **self._followup_event("--discuss why does this fail?"),
            "files": [
                {
                    "id": "F1",
                    "name": "trace.log",
                    "mimetype": "text/plain",
                    "size": 18,
                    "url_private_download": (
                        "https://files.slack.com/files-pri/T/F1/trace.log"
                    ),
                }
            ],
        }

        response = await handler.handle_message(event, SimpleNamespace())

        assert "R2" in response
        prompt = runtime.create_session.await_args.args[4]
        assert "Traceback: line 42" in prompt
        assert '<untrusted_reference_context source="slack_message_files">' in prompt
        assert prompt.rstrip().endswith(
            '<current_task owner_user_id="U1">\nwhy does this fail?\n</current_task>'
        )
        kwargs = runtime.create_session.await_args.kwargs
        assert kwargs["routing_text"] == "why does this fail?"
        assert kwargs["task_mode"] == "discuss"

    @pytest.mark.asyncio
    async def test_agent_thread_followup_parses_leading_mode_flags(self):
        bot, runtime = self._followup_bot()
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            self._followup_event("--discuss --workflow  why does X fail?"),
            SimpleNamespace(),
        )

        runtime.create_session.assert_awaited_once_with(
            "T1",
            "C1",
            "100.1",
            "U1",
            "why does X fail?",
            routing_text="why does X fail?",
            task_mode="discuss",
            workflow_mode="structured",
        )
        assert "R2" in response
        assert "discuss" in response
        assert bot.agent_runs["R2"]["mode"] == "discuss"

    @pytest.mark.asyncio
    async def test_discuss_thread_followup_stays_discuss_without_explicit_flag(self):
        bot, runtime = self._followup_bot(previous_mode="discuss")
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            self._followup_event("now go ahead and fix it"), SimpleNamespace()
        )

        assert runtime.create_session.await_args.kwargs["task_mode"] == "discuss"
        assert runtime.create_session.await_args.args[4] == "now go ahead and fix it"
        assert "discuss" in response
        assert "--implement" in response

    @pytest.mark.asyncio
    async def test_discuss_run_in_another_thread_does_not_set_new_thread_mode(self):
        bot, runtime = self._followup_bot(previous_mode="discuss")
        handler = SlackMessageHandler(bot)

        await handler._start_agent_prompt(
            "fix it", "U1", "C1", None, team_id="T1", thread_ts="200.1"
        )

        assert "task_mode" not in runtime.create_session.await_args.kwargs

    @pytest.mark.asyncio
    async def test_discuss_thread_switches_to_implement_only_with_flag(self):
        bot, runtime = self._followup_bot(previous_mode="discuss")
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            self._followup_event("--implement fix it"), SimpleNamespace()
        )

        assert runtime.create_session.await_args.kwargs["task_mode"] == "implement"
        assert runtime.create_session.await_args.args[4] == "fix it"
        assert "implement" in response
        assert bot.agent_runs["R2"]["mode"] == "implement"

    @pytest.mark.asyncio
    async def test_agent_thread_followup_rejects_conflicting_mode_flags(self):
        bot, runtime = self._followup_bot()
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            self._followup_event("--discuss --implement fix it"), SimpleNamespace()
        )

        assert "Conflicting task modes" in response
        runtime.create_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_agent_thread_followup_keeps_non_leading_flags_in_prompt(self):
        bot, runtime = self._followup_bot()
        handler = SlackMessageHandler(bot)

        await handler.handle_message(
            self._followup_event("explain what --discuss does"), SimpleNamespace()
        )

        assert runtime.create_session.await_args.args[4] == (
            "explain what --discuss does"
        )
        assert "task_mode" not in runtime.create_session.await_args.kwargs

    @pytest.mark.asyncio
    async def test_new_agent_run_started_message_shows_effective_mode(self):
        runtime = SimpleNamespace(
            create_session=AsyncMock(
                return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
            )
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot._monitor_agent_run = AsyncMock()
        handler = SlackMessageHandler(bot)
        client = SimpleNamespace(
            chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
            chat_update=AsyncMock(),
        )

        await handler._handle_command(
            "agent --workflow fix the tests", "U1", "C1", None, client, team_id="T1"
        )

        text = client.chat_update.await_args.kwargs["text"]
        assert "Mode: *implement*" in text
        assert "structured workflow" in text
        assert bot.agent_runs["R1"]["mode"] == "implement"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (
                AgentRuntimeUnauthorized("x", status=401, code="unauthorized"),
                "credentials",
            ),
            (
                AgentRuntimeUnauthorized("x", status=401, code="approval_forbidden"),
                "Only the Agent session owner",
            ),
            (
                AgentRuntimeConflict("x", status=409, code="session_busy"),
                "already has a run in progress",
            ),
            (
                AgentRuntimeError(
                    "x", status=422, code="verification_profile_required"
                ),
                "--discuss",
            ),
        ],
    )
    async def test_agent_start_failures_branch_on_runtime_error_code(
        self, error, expected
    ):
        runtime = SimpleNamespace(create_session=AsyncMock(side_effect=error))
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        handler = SlackMessageHandler(bot)

        response = await handler._start_agent_prompt(
            "task", "U1", "C1", None, team_id="T1", thread_ts="100.1"
        )

        assert expected in response
        assert "Chat mode was not used as fallback" in response

    @pytest.mark.asyncio
    async def test_agent_control_distinguishes_bad_token_from_not_owner(self):
        runtime = SimpleNamespace(
            get_run=AsyncMock(
                side_effect=AgentRuntimeUnauthorized(
                    "x", status=401, code="unauthorized"
                )
            ),
            cancel=AsyncMock(
                side_effect=AgentRuntimeUnauthorized(
                    "x", status=401, code="approval_forbidden"
                )
            ),
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot.agent_user_runs["U1"] = {
            "run_id": "R1",
            "session_id": "S1",
            "thread_ts": "1",
            "channel_id": "C1",
            "owner_user_id": "U1",
        }
        handler = SlackMessageHandler(bot)

        status = await handler._handle_command("agent status", "U1", "C1", None, None)
        stop = await handler._handle_command("agent stop", "U1", "C1", None, None)

        assert "credentials" in status
        assert "Only the Agent session owner" in stop

    @pytest.mark.asyncio
    async def test_help_and_status_expose_agent_mode_and_runtime_health(self):
        agent_runtime = SimpleNamespace(health=AsyncMock(return_value=True))
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(),
            agent_runtime_client=agent_runtime,
            get_system_status=AsyncMock(
                return_value={
                    "healthy": True,
                    "available_models": ["gpt-5"],
                    "avg_response_time": 10,
                    "uptime": "1m",
                }
            ),
            get_user_stats=AsyncMock(return_value={}),
        )
        handler = SlackMessageHandler(bot)

        help_text = await handler._handle_help_command([], "u1", "c1", None)
        status_text = await handler._handle_status_command([], "u1", "c1", None)

        assert "/llm agent <task>" in help_text
        assert "/skill:test-gap" in help_text
        assert "Run Pi Agent" in help_text
        assert "--no-thread-context" in help_text
        assert "Agent Runtime:" in status_text
        assert "Healthy" in status_text
        agent_runtime.health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_help_hides_agent_commands_when_runtime_is_disabled(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(),
            agent_runtime_client=None,
            _format_tier_limit_summary=lambda: "Free: 10 requests/day",
        )
        handler = SlackMessageHandler(bot)

        help_text = await handler._handle_help_command([], "u1", "c1", None)

        assert "/llm agent" not in help_text
        assert "Run Pi Agent" not in help_text

    @pytest.mark.asyncio
    async def test_help_lists_every_command_subcommand_and_flag_the_bot_parses(self):
        runtime = SimpleNamespace(health_details=AsyncMock(return_value=None))
        handler = SlackMessageHandler(SimpleNamespace(agent_runtime_client=runtime))

        help_text = await handler._handle_help_command([], "u1", "c1", None)

        usages = {}
        for name, rest in re.findall(r"`/llm ([\w-]+)([^`]*)`", help_text):
            usages.setdefault(name, set()).update(
                token for token in re.split(r"[\s|\[\]]+", rest) if token
            )
        missing = []
        for name, method in handler.commands.items():
            if name not in usages:
                missing.append(name)
                continue
            flags, actions = _parsed_literals(method)
            missing.extend(
                f"{name} {token}"
                for token in sorted(flags | actions)
                if token not in usages[name]
            )
        assert missing == []
        # The derivation sees the subcommands G12 found missing from the help.
        assert {"resume", "repair-ci", "publish-repair"} <= _parsed_literals(
            handler._handle_agent_command
        )[1]
        reply_line = next(
            line for line in help_text.splitlines() if "Agent thread" in line
        )
        assert [
            flag
            for flag in re.findall(r"--[\w-]+", AGENT_FOLLOWUP_FLAG_PATTERN.pattern)
            if f"`{flag}`" not in reply_line
        ] == []

    @pytest.mark.asyncio
    async def test_help_and_models_list_agent_models_from_runtime_health(self):
        runtime = SimpleNamespace(
            health_details=AsyncMock(
                return_value={
                    "status": "healthy",
                    "models": [
                        {"ref": "acme/model-x", "configured": True},
                        {"ref": "openai/gpt-5.6-sol", "configured": False},
                    ],
                }
            )
        )
        bot = SimpleNamespace(
            agent_runtime_client=runtime,
            get_available_models=AsyncMock(return_value=[]),
        )
        handler = SlackMessageHandler(bot)

        help_text = await handler._handle_help_command([], "u1", "c1", None)
        models_text = await handler._handle_models_command([], "u1", "c1", None)

        for text in (help_text, models_text):
            assert "`acme/model-x`" in text
            assert "gpt-5.6-sol" not in text
            assert "gpt-5.6-luna" not in text
            assert "claude-sonnet-4-6" not in text

    @pytest.mark.asyncio
    async def test_help_and_models_say_agent_models_are_unknown_without_health(self):
        runtime = SimpleNamespace(
            health_details=AsyncMock(return_value=None),
            cached_configured_models=lambda: ["openai/gpt-5.6-luna"],
        )
        bot = SimpleNamespace(
            agent_runtime_client=runtime,
            get_available_models=AsyncMock(return_value=[]),
        )
        handler = SlackMessageHandler(bot)

        help_text = await handler._handle_help_command([], "u1", "c1", None)
        models_text = await handler._handle_models_command([], "u1", "c1", None)

        for text in (help_text, models_text):
            assert "Agent models are unavailable" in text
            # A stale cached allowlist is never shown as current.
            assert "gpt-5.6-luna" not in text

    @pytest.mark.asyncio
    async def test_manual_agent_command_explains_that_runtime_is_disabled(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(),
            agent_runtime_client=None,
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command(
            "agent fix the tests", "u1", "c1", None, client=None
        )

        assert response == "Agent mode is currently disabled."

    @pytest.mark.asyncio
    async def test_status_lists_configured_agent_provider_models(self):
        agent_runtime = SimpleNamespace(
            health_details=AsyncMock(
                return_value={
                    "status": "healthy",
                    "models": [
                        {"ref": "openai/gpt-5.6-luna", "configured": True},
                        {"ref": "anthropic/claude-sonnet-4-6", "configured": False},
                        {"ref": "opencode-go/deepseek-v4-pro", "configured": True},
                    ],
                }
            )
        )
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(),
            agent_runtime_client=agent_runtime,
            get_system_status=AsyncMock(
                return_value={"healthy": True, "available_models": []}
            ),
            get_user_stats=AsyncMock(return_value={}),
        )
        handler = SlackMessageHandler(bot)

        status_text = await handler._handle_status_command([], "u1", "c1", None)

        assert "openai/gpt-5.6-luna" in status_text
        assert "opencode-go/deepseek-v4-pro" in status_text
        assert "anthropic/claude-sonnet-4-6" not in status_text

    def test_agent_response_formats_code_delivery_and_respects_limit(self):
        response = SlackMessageHandler._format_agent_response(
            {
                "answer": "a" * 500,
                "changed_files": ["src/a.py"],
                "diff_stat": "1 file changed, 2 insertions(+)",
                "mcp_server": "github",
                "mcp_tool": "get_file_contents",
                "mcp_repository": "acme/widgets",
                "mcp_success": True,
                "branch": "pi-agent/20260801-r1",
                "commit": "a" * 40,
                "cherry_pick": "git cherry-pick " + "a" * 40,
            },
            max_length=360,
        )

        assert len(response) <= 360
        assert "src/a.py" in response
        assert (
            "`github` / `get_file_contents` / `acme/widgets` / *succeeded*" in response
        )
        assert "git cherry-pick" in response

        context_response = SlackMessageHandler._format_agent_response(
            {
                "answer": "docs",
                "mcp_server": "context7",
                "mcp_tool": "query_docs",
                "mcp_success": True,
            }
        )
        assert "`context7` / `query_docs` / *succeeded*" in context_response


class TestSlackAgentApprovals:
    @pytest.mark.asyncio
    async def test_agent_model_cache_refresh_uses_runtime_health(self):
        runtime = SimpleNamespace(health_details=AsyncMock(return_value={"models": []}))
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )

        await bot._refresh_agent_model_cache()

        runtime.health_details.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_shortcut_is_acked_and_opens_agent_modal(self):
        runtime = SimpleNamespace(
            cached_configured_models=lambda: [
                "openai/gpt-5.6-luna",
                "anthropic/claude-sonnet-4-6",
            ]
        )
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(
                return_value={
                    "ok": True,
                    "channel": {"is_channel": True, "is_private": False},
                }
            ),
            views_open=AsyncMock(),
        )
        socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
        request = SimpleNamespace(
            envelope_id="E-shortcut",
            type="interactive",
            payload={
                "type": "message_action",
                "callback_id": "run_pi_agent_from_thread",
                "trigger_id": "trigger-1",
                "team": {"id": "T1"},
                "user": {"id": "U1"},
                "channel": {"id": "C1"},
                "message": {"ts": "100.2", "thread_ts": "100.1"},
            },
        )

        await bot._handle_socket_mode_request(socket_client, request)

        socket_client.send_socket_mode_response.assert_awaited_once()
        bot.web_client.views_open.assert_awaited_once()
        modal = bot.web_client.views_open.await_args.kwargs["view"]
        assert modal["callback_id"] == "run_pi_agent_from_thread_submit"
        assert modal["private_metadata"]
        model_block = next(
            block for block in modal["blocks"] if block.get("block_id") == "agent_model"
        )
        assert len(model_block["element"]["options"]) == 2
        assert "initial_option" not in model_block["element"]

    @pytest.mark.asyncio
    async def test_message_shortcut_rejects_private_channels(self):
        bot = SlackBot({"channels": []}, SimpleNamespace())
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(
                return_value={
                    "ok": True,
                    "channel": {"is_channel": True, "is_private": True},
                }
            ),
            views_open=AsyncMock(),
        )

        await bot._handle_agent_shortcut(
            {
                "trigger_id": "trigger-1",
                "team": {"id": "T1"},
                "user": {"id": "U1"},
                "channel": {"id": "GPRIVATE"},
                "message": {"ts": "100.1"},
            }
        )

        bot.web_client.views_open.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_modal_submission_rechecks_public_channel_before_context_fetch(self):
        bot = SlackBot({"channels": []}, SimpleNamespace())
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(
                side_effect=[
                    {
                        "ok": True,
                        "channel": {"is_channel": True, "is_private": False},
                    },
                    {
                        "ok": True,
                        "channel": {"is_channel": True, "is_private": True},
                    },
                ]
            ),
            views_open=AsyncMock(),
        )
        bot.agent_bootstrap_builder = SimpleNamespace(build=AsyncMock())
        await bot._handle_agent_shortcut(
            {
                "trigger_id": "trigger-1",
                "team": {"id": "T1"},
                "user": {"id": "U1"},
                "channel": {"id": "C1"},
                "message": {"ts": "100.1"},
            }
        )
        nonce = bot.web_client.views_open.await_args.kwargs["view"]["private_metadata"]

        await bot._handle_agent_view_submission(
            {
                "user": {"id": "U1"},
                "view": {
                    "private_metadata": nonce,
                    "state": {
                        "values": {
                            "agent_task": {"task": {"value": "private task"}},
                            "agent_context": {
                                "include": {"selected_options": [{"value": "include"}]}
                            },
                        }
                    },
                },
            }
        )

        bot.agent_bootstrap_builder.build.assert_not_awaited()
        assert bot.user_manager.rate_limits.get("U1", []) == []

    @pytest.mark.asyncio
    async def test_view_submission_is_acked_before_agent_work_is_queued(self):
        bot = SlackBot({"channels": []}, SimpleNamespace())
        bot._enqueue_work = AsyncMock(return_value=True)
        socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
        request = SimpleNamespace(
            envelope_id="E-view",
            type="interactive",
            payload={
                "type": "view_submission",
                "view": {"callback_id": "run_pi_agent_from_thread_submit"},
                "user": {"id": "U1"},
            },
        )

        await bot._handle_socket_mode_request(socket_client, request)

        socket_client.send_socket_mode_response.assert_awaited_once()
        item = bot._enqueue_work.await_args.args[0]
        assert item.kind == "agent_view"

    @pytest.mark.asyncio
    async def test_wrong_user_cannot_consume_agent_shortcut_nonce(self):
        runtime = SimpleNamespace(create_session=AsyncMock())
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(
                return_value={
                    "ok": True,
                    "channel": {"is_channel": True, "is_private": False},
                }
            ),
            views_open=AsyncMock(),
        )
        await bot._handle_agent_shortcut(
            {
                "trigger_id": "trigger-1",
                "team": {"id": "T1"},
                "user": {"id": "U1"},
                "channel": {"id": "C1"},
                "message": {"ts": "100.1"},
            }
        )
        nonce = bot.web_client.views_open.await_args.kwargs["view"]["private_metadata"]

        await bot._handle_agent_view_submission(
            {
                "user": {"id": "U2"},
                "view": {
                    "private_metadata": nonce,
                    "state": {
                        "values": {"agent_task": {"task": {"value": "malicious task"}}}
                    },
                },
            }
        )

        runtime.create_session.assert_not_awaited()
        assert nonce in bot.agent_shortcut_nonces

    @pytest.mark.asyncio
    async def test_modal_submission_bootstraps_thread_and_starts_selected_model(self):
        runtime = SimpleNamespace(
            cached_configured_models=lambda: ["openai/gpt-5.6-luna"],
            create_session=AsyncMock(
                return_value={
                    "session_id": "session-1",
                    "run_id": "run-1",
                    "status": "starting",
                }
            ),
        )
        bot = SlackBot(
            {"channels": [], "agent_context": {"enabled": True}},
            SimpleNamespace(),
            services={"agent_runtime": runtime},
        )
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(
                return_value={
                    "ok": True,
                    "channel": {"is_channel": True, "is_private": False},
                }
            ),
            views_open=AsyncMock(),
            chat_postMessage=AsyncMock(return_value={"ts": "status-1"}),
            chat_update=AsyncMock(),
        )
        bot._monitor_agent_run = AsyncMock()
        bot.agent_bootstrap_builder = SimpleNamespace(
            build=AsyncMock(
                return_value=AgentBootstrapContext(
                    prompt="bounded bootstrap prompt",
                    context="context",
                    message_count=3,
                    resource_count=2,
                    truncated=True,
                    warnings=["canvas_body_unavailable:F1"],
                )
            )
        )
        await bot._handle_agent_shortcut(
            {
                "type": "message_action",
                "callback_id": "run_pi_agent_from_thread",
                "trigger_id": "trigger-1",
                "team": {"id": "T1"},
                "user": {"id": "U1"},
                "channel": {"id": "C1"},
                "message": {"ts": "100.2", "thread_ts": "100.1"},
            }
        )
        nonce = bot.web_client.views_open.await_args.kwargs["view"]["private_metadata"]
        payload = {
            "type": "view_submission",
            "team": {"id": "T1"},
            "user": {"id": "U1"},
            "view": {
                "callback_id": "run_pi_agent_from_thread_submit",
                "private_metadata": nonce,
                "state": {
                    "values": {
                        "agent_task": {"task": {"value": "Implement the thread todo"}},
                        "agent_model": {
                            "model": {
                                "selected_option": {"value": "openai/gpt-5.6-luna"}
                            }
                        },
                        "agent_context": {
                            "include": {"selected_options": [{"value": "include"}]}
                        },
                    }
                },
            },
        }

        await bot._handle_agent_view_submission(payload)
        # The worker only acks; bootstrap and create_session run in the background.
        await _drain_background_tasks(bot)

        bot.agent_bootstrap_builder.build.assert_awaited_once()
        assert runtime.create_session.await_args.args == (
            "T1",
            "C1",
            "100.1",
            "U1",
            "bounded bootstrap prompt",
        )
        assert runtime.create_session.await_args.kwargs == {
            "model": "openai/gpt-5.6-luna",
            "routing_text": "Implement the thread todo",
            "idempotency_key": _agent_idempotency_key(
                "T1", "C1", f"shortcut:{nonce}", "U1", "create"
            ),
        }
        status = bot.web_client.chat_update.await_args.kwargs["text"]
        assert "Loaded 3 prior Slack messages and 2 resources" in status
        assert "truncated" in status
        assert "canvas_body_unavailable:F1" in status
        assert len(bot.user_manager.rate_limits["U1"]) == 1

        await bot._handle_agent_view_submission(payload)

        runtime.create_session.assert_awaited_once()
        assert len(bot.user_manager.rate_limits["U1"]) == 1

    @pytest.mark.asyncio
    async def test_non_owner_agent_thread_message_is_rejected_before_quota(self):
        bot = SlackBot({"channels": []}, SimpleNamespace())
        bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        bot.agent_threads["C1:100.1"] = {
            "owner_user_id": "U1",
            "channel_id": "C1",
            "thread_ts": "100.1",
        }
        bot._process_message = AsyncMock()

        await bot._handle_message_event(
            {
                "type": "message",
                "channel": "C1",
                "thread_ts": "100.1",
                "ts": "100.2",
                "user": "U2",
                "text": "do something",
            }
        )

        bot._process_message.assert_not_awaited()
        bot.web_client.chat_postEphemeral.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_interactive_request_is_acked_before_approval_work_is_queued(self):
        bot = SlackBot({"channels": []}, SimpleNamespace())
        bot._enqueue_work = AsyncMock(return_value=True)
        socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
        request = SimpleNamespace(
            envelope_id="E1",
            type="interactive",
            payload={"type": "block_actions", "actions": []},
        )

        await bot._handle_socket_mode_request(socket_client, request)

        socket_client.send_socket_mode_response.assert_awaited_once()
        item = bot._enqueue_work.await_args.args[0]
        assert item.kind == "interactive"

    @pytest.mark.asyncio
    async def test_only_session_owner_can_press_approval_button(self):
        runtime = SimpleNamespace(decide=AsyncMock())
        bot = SlackBot(
            {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
        )
        bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        bot.agent_runs["R1"] = {
            "run_id": "R1",
            "session_id": "S1",
            "channel_id": "C1",
            "thread_ts": "1",
            "owner_user_id": "U1",
        }
        payload = {
            "user": {"id": "U2"},
            "channel": {"id": "C1"},
            "actions": [
                {
                    "action_id": "pi_agent_approve",
                    "value": '{"run_id":"R1","session_id":"S1","approval_id":"A1"}',
                }
            ],
        }

        await bot._handle_agent_interactive(payload)

        runtime.decide.assert_not_awaited()
        assert (
            "Only the user"
            in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
        )

    @pytest.mark.asyncio
    async def test_settings_command_updates_preferences(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
        )
        handler = SlackMessageHandler(bot)

        message = await handler._handle_settings_command(
            ["response_length", "long"],
            "u1",
            "c1",
            client=None,
        )

        assert "Updated response_length" in message
        assert bot.user_manager.get_user_preferences("u1")["response_length"] == "long"

    @pytest.mark.asyncio
    async def test_settings_command_validates_and_clears_preferred_models(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            router=SimpleNamespace(models={"gpt-5": object(), "mistral-7b": object()}),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
        )
        handler = SlackMessageHandler(bot)

        invalid_message = await handler._handle_settings_command(
            ["preferred_models", "unknown-model"],
            "u1",
            "c1",
            client=None,
        )
        cleared_message = await handler._handle_settings_command(
            ["preferred_models", "auto"],
            "u1",
            "c1",
            client=None,
        )

        assert "Unknown model" in invalid_message
        assert "Updated preferred_models" in cleared_message
        assert bot.user_manager.get_user_preferences("u1")["preferred_models"] == []

    @pytest.mark.asyncio
    async def test_handle_query_updates_history(self, inference_response_factory):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(
                process_query=AsyncMock(
                    return_value=inference_response_factory(
                        response_text="hello from model"
                    )
                )
            ),
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_query("hello", "u1", "c1", None, client=None)
        summary = bot.conversation_manager.get_conversation_summary("u1", "c1")

        assert response == "hello from model"
        assert "user: hello" in summary
        assert "assistant: hello from model" in summary

    def test_get_max_tokens_for_user_respects_tier_and_length(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
        )
        handler = SlackMessageHandler(bot)

        free_tokens = handler._get_max_tokens_for_user(
            UserTier.FREE, {"response_length": "short"}
        )
        enterprise_tokens = handler._get_max_tokens_for_user(
            UserTier.ENTERPRISE, {"response_length": "long"}
        )

        assert free_tokens == 500
        assert enterprise_tokens == 8192

    @pytest.mark.asyncio
    async def test_handle_query_passes_preference_metadata(
        self, inference_response_factory
    ):
        user_manager = UserManager()
        user_manager.update_user_preferences(
            "u1",
            {
                "technical_level": "expert",
                "preferred_models": ["gpt-5"],
                "response_length": "long",
            },
        )
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(
                    response_text="hello from model"
                )
            )
        )
        bot = SimpleNamespace(
            user_manager=user_manager,
            conversation_manager=ConversationManager({}),
            inference_engine=inference_engine,
            _build_query_metadata=SlackBot._build_query_metadata,
            _conversation_context_key=SlackBot._conversation_context_key,
            _build_response_style_instructions=SlackBot._build_response_style_instructions,
            _build_user_safe_error_message=lambda: "safe error",
        )
        bot._build_query_metadata = bot._build_query_metadata.__get__(
            bot, SimpleNamespace
        )
        bot._conversation_context_key = bot._conversation_context_key.__get__(
            bot, SimpleNamespace
        )
        bot._build_response_style_instructions = (
            bot._build_response_style_instructions.__get__(bot, SimpleNamespace)
        )
        handler = SlackMessageHandler(bot)

        await handler._handle_query("hello", "u1", "c1", None, client=None)

        request = inference_engine.process_query.await_args.args[0]
        assert request.metadata["preferred_models"] == ["gpt-5"]
        assert request.metadata["web_search_auto"] is True
        assert "advanced detail" in request.metadata["response_style_instructions"]
        assert request.session_id is not None
        assert request.conversation_id == "u1:c1:main"

    @pytest.mark.asyncio
    async def test_handle_query_injects_thread_context_before_bot_history(
        self, inference_response_factory
    ):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text="thread answer")
            )
        )
        bot = SlackBot(
            {"channels": [], "context": {"enabled": True}},
            inference_engine=inference_engine,
        )
        context = bot.conversation_manager.get_or_create_context("u1", "c1", "t1")
        context.add_message("assistant", "previous bot answer")
        client = SimpleNamespace(
            conversations_replies=AsyncMock(
                return_value={
                    "ok": True,
                    "messages": [
                        {"ts": "1.0", "user": "u2", "text": "parent decision"},
                        {"ts": "1.1", "user": "u3", "text": "prior reply"},
                        {"ts": "1.15", "subtype": "channel_join", "text": "joined"},
                        {
                            "ts": "1.16",
                            "user": "B1",
                            "bot_id": "B1",
                            "text": "bot noise",
                        },
                        {"ts": "1.2", "user": "u1", "text": "current question"},
                    ],
                }
            )
        )

        response = await bot.message_handler._handle_query(
            "what did we decide?",
            "u1",
            "c1",
            "t1",
            client=client,
            message_ts="1.2",
        )

        request = inference_engine.process_query.await_args.args[0]
        assert response == "thread answer"
        assert request.context.index("Slack thread context:") < request.context.index(
            "assistant: previous bot answer"
        )
        assert "<@u2>: parent decision" in request.context
        assert "<@u3>: prior reply" in request.context
        assert "current question" not in request.context
        assert "bot noise" not in request.context
        assert "joined" not in request.context
        assert request.metadata["slack_context_source"] == "thread"
        assert request.metadata["slack_context_message_count"] == 2
        assert request.metadata["slack_context_truncated"] is False
        client.conversations_replies.assert_awaited_once_with(
            channel="c1", ts="t1", limit=20, latest="1.2", inclusive=False
        )

    @pytest.mark.asyncio
    async def test_handle_query_uses_channel_context_for_top_level_mentions(
        self, inference_response_factory
    ):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text="channel answer")
            )
        )
        bot = SlackBot(
            {
                "channels": [],
                "context": {"enabled": True, "max_channel_messages": 3},
            },
            inference_engine=inference_engine,
        )
        client = SimpleNamespace(
            conversations_history=AsyncMock(
                return_value={
                    "ok": True,
                    "messages": [
                        {"ts": "2.2", "user": "u3", "text": "newer point"},
                        {"ts": "2.1", "user": "u2", "text": "older point"},
                        {"ts": "2.3", "user": "u1", "text": "current question"},
                    ],
                }
            ),
            conversations_replies=AsyncMock(),
        )

        await bot.message_handler._handle_query(
            "summarize above",
            "u1",
            "c1",
            "2.3",
            client=client,
            message_ts="2.3",
        )

        request = inference_engine.process_query.await_args.args[0]
        assert "Recent channel context:" in request.context
        assert request.context.index("older point") < request.context.index(
            "newer point"
        )
        assert "current question" not in request.context
        assert request.metadata["slack_context_source"] == "channel"
        assert request.metadata["slack_context_message_count"] == 2
        client.conversations_history.assert_awaited_once_with(
            channel="c1", latest="2.3", inclusive=False, limit=3
        )
        client.conversations_replies.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_query_fails_open_when_slack_context_fetch_fails(
        self, inference_response_factory
    ):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text="fallback answer")
            )
        )
        bot = SlackBot(
            {"channels": [], "context": {"enabled": True, "fail_open": True}},
            inference_engine=inference_engine,
        )
        client = SimpleNamespace(
            conversations_replies=AsyncMock(side_effect=RuntimeError("missing_scope"))
        )

        response = await bot.message_handler._handle_query(
            "what happened?",
            "u1",
            "c1",
            "t1",
            client=client,
            message_ts="1.2",
        )

        request = inference_engine.process_query.await_args.args[0]
        assert response == "fallback answer"
        assert "slack_context_error" in request.metadata
        assert request.metadata["slack_context_error"] == "missing_scope"
        assert request.metadata["slack_context_source"] == "none"
        assert request.metadata["slack_context_message_count"] == 0

    @pytest.mark.asyncio
    async def test_web_command_requires_web_search(self, inference_response_factory):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(
                    response_text="answer [1]",
                    sources=[
                        ResponseSource(
                            title="Example",
                            url="https://example.com/news",
                            snippet="snippet",
                            rank=1,
                        )
                    ],
                )
            )
        )
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=inference_engine,
            _build_query_metadata=SlackBot._build_query_metadata,
            _conversation_context_key=SlackBot._conversation_context_key,
            _build_response_style_instructions=SlackBot._build_response_style_instructions,
        )
        bot._build_query_metadata = bot._build_query_metadata.__get__(
            bot, SimpleNamespace
        )
        bot._conversation_context_key = bot._conversation_context_key.__get__(
            bot, SimpleNamespace
        )
        bot._build_response_style_instructions = (
            bot._build_response_style_instructions.__get__(bot, SimpleNamespace)
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command_or_query(
            "web latest AI news",
            "u1",
            "c1",
            None,
            client=None,
        )

        request = inference_engine.process_query.await_args.args[0]
        assert request.tool_policy == ToolPolicy.REQUIRED
        assert request.allowed_tools == ["web_search"]
        assert "Sources:" in response
        assert "https://example.com/news" in response

    @pytest.mark.asyncio
    async def test_fast_command_marks_one_request_low_latency(
        self, inference_response_factory
    ):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text="fast answer")
            )
        )
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=inference_engine,
            _build_query_metadata=SlackBot._build_query_metadata,
            _conversation_context_key=SlackBot._conversation_context_key,
            _build_response_style_instructions=SlackBot._build_response_style_instructions,
        )
        bot._build_query_metadata = bot._build_query_metadata.__get__(
            bot, SimpleNamespace
        )
        bot._conversation_context_key = bot._conversation_context_key.__get__(
            bot, SimpleNamespace
        )
        bot._build_response_style_instructions = (
            bot._build_response_style_instructions.__get__(bot, SimpleNamespace)
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command_or_query(
            "fast summarize this thread",
            "u1",
            "c1",
            None,
            client=None,
        )

        request = inference_engine.process_query.await_args.args[0]
        assert response == "fast answer"
        assert request.query == "summarize this thread"
        assert request.metadata["requires_low_latency"] is True
        assert request.metadata["latency_sla"] == "interactive"
        assert request.metadata["routing_intent_source"] == "slack_fast_command"

    @pytest.mark.asyncio
    async def test_handle_query_returns_safe_error_when_engine_returns_error_response(
        self, inference_response_factory
    ):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(
                process_query=AsyncMock(
                    return_value=inference_response_factory(
                        response_text="Error processing request: upstream failure",
                        provider="error",
                        error="upstream failure",
                    )
                )
            ),
            _build_user_safe_error_message=lambda: "safe error",
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_query("hello", "u1", "c1", None, client=None)

        assert response == "safe error"

    @pytest.mark.asyncio
    async def test_memory_commands_save_list_and_forget_user_memory(self):
        memory_manager = MemoryManager(
            {
                "enabled": True,
                "search": {"max_results": 5},
                "embedding": {"provider": "hash", "dimensions": 16},
            },
            store=InMemoryMemoryStore(),
            embedding_provider=HashEmbeddingProvider(dimensions=16),
        )
        await memory_manager.initialize()
        bot = SimpleNamespace(
            memory_manager=memory_manager,
            _memory_scope=SlackBot._memory_scope,
            _build_memory_metadata=SlackBot._build_memory_metadata,
        )
        bot._memory_scope = bot._memory_scope.__get__(bot, SimpleNamespace)
        bot._build_memory_metadata = bot._build_memory_metadata.__get__(
            bot, SimpleNamespace
        )
        handler = SlackMessageHandler(bot)

        saved = await handler._handle_command(
            "remember prefer concise answers",
            "u1",
            "c1",
            "t1",
            client=None,
            team_id="T1",
            command_surface="slash",
        )
        global_saved = await handler._handle_command(
            "remember --global prefer terse summaries",
            "u1",
            "c1",
            "t1",
            client=None,
            team_id="T1",
            command_surface="slash",
        )
        stored_memories = await memory_manager.list_memories("T1:u1", limit=10)
        channel_memory = next(
            item for item in stored_memories if item.text == "prefer concise answers"
        )
        global_memory = next(
            item for item in stored_memories if item.text == "prefer terse summaries"
        )
        listed = await handler._handle_command(
            "memories concise",
            "u1",
            "c1",
            "t1",
            client=None,
            team_id="T1",
            command_surface="slash",
        )
        memory_id = listed.split("`")[1]
        deleted = await handler._handle_command(
            f"forget {memory_id}",
            "u1",
            "c1",
            "t1",
            client=None,
            team_id="T1",
            command_surface="slash",
        )

        assert "Saved memory" in saved
        assert "Saved memory" in global_saved
        assert "prefer concise answers" in listed
        assert "Forgot memory" in deleted
        assert channel_memory.metadata["visibility"] == "channel"
        assert channel_memory.metadata["channel_id"] == "c1"
        assert global_memory.metadata["visibility"] == "global"
        assert "channel_id" not in global_memory.metadata

    @pytest.mark.asyncio
    async def test_inline_memory_commands_return_public_safe_hint(self):
        memory_manager = MemoryManager(
            {"enabled": True},
            store=InMemoryMemoryStore(),
            embedding_provider=HashEmbeddingProvider(dimensions=16),
        )
        await memory_manager.initialize()
        await memory_manager.remember(
            "T1:u1",
            "secret memory content",
            metadata={"source": "slack", "visibility": "channel", "channel_id": "c1"},
        )
        bot = SimpleNamespace(
            memory_manager=memory_manager,
            _memory_scope=SlackBot._memory_scope,
            _build_memory_metadata=SlackBot._build_memory_metadata,
        )
        bot._memory_scope = bot._memory_scope.__get__(bot, SimpleNamespace)
        bot._build_memory_metadata = bot._build_memory_metadata.__get__(
            bot, SimpleNamespace
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_command(
            "memories secret",
            "u1",
            "c1",
            "t1",
            client=None,
            team_id="T1",
        )

        assert "slash commands" in response
        assert "secret memory content" not in response

    @pytest.mark.asyncio
    async def test_handle_query_injects_memory_only_when_hits_exist(
        self, inference_response_factory
    ):
        memory_manager = MemoryManager(
            {
                "enabled": True,
                "search": {"max_results": 5},
                "embedding": {"provider": "none"},
            },
            store=InMemoryMemoryStore(),
            embedding_provider=FailingEmbeddingProvider(),
        )
        await memory_manager.initialize()
        await memory_manager.remember(
            "T1:u1",
            "Prefer Python examples for API explanations",
            metadata={"source": "slack", "visibility": "channel", "channel_id": "c1"},
        )
        await memory_manager.remember(
            "T1:u1",
            "Prefer concise answers everywhere",
            metadata={"source": "slack", "visibility": "global"},
        )
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(
                    response_text="hello from model"
                )
            )
        )
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=inference_engine,
            memory_manager=memory_manager,
            _memory_scope=SlackBot._memory_scope,
            _build_query_metadata=SlackBot._build_query_metadata,
            _conversation_context_key=SlackBot._conversation_context_key,
            _build_response_style_instructions=SlackBot._build_response_style_instructions,
            _build_user_safe_error_message=lambda: "safe error",
        )
        for method_name in (
            "_memory_scope",
            "_build_query_metadata",
            "_conversation_context_key",
            "_build_response_style_instructions",
        ):
            setattr(
                bot,
                method_name,
                getattr(bot, method_name).__get__(bot, SimpleNamespace),
            )
        handler = SlackMessageHandler(bot)

        await handler._handle_query(
            "explain concise Python API",
            "u1",
            "c1",
            None,
            client=None,
            team_id="T1",
        )
        hit_request = inference_engine.process_query.await_args.args[0]

        inference_engine.process_query.reset_mock()
        await handler._handle_query(
            "Python API concise",
            "u1",
            "c2",
            None,
            client=None,
            team_id="T1",
        )
        other_channel_request = inference_engine.process_query.await_args.args[0]

        assert "Long-term user memory:" in hit_request.context
        assert "Prefer Python examples" in hit_request.context
        assert "Prefer concise answers everywhere" in hit_request.context
        assert "memory_hit_count" not in hit_request.metadata
        assert "Long-term user memory:" in other_channel_request.context
        assert "Prefer Python examples" not in other_channel_request.context
        assert "Prefer concise answers everywhere" in other_channel_request.context

    def test_extract_team_id_and_memory_scope(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

        assert bot._extract_team_id({"team_id": "T1"}) == "T1"
        assert bot._extract_team_id({"authorizations": [{"team_id": "T2"}]}) == "T2"
        assert bot._memory_scope("T1", "U1") == "T1:U1"


class TestSlackBot:
    def test_split_response_breaks_large_message(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        text = "Paragraph one.\n\n" + ("A" * 120) + "\n\nParagraph three."

        parts = bot._split_response(text, max_length=80)

        assert len(parts) >= 2
        assert all(len(part) <= 80 for part in parts)

    @pytest.mark.asyncio
    async def test_initialize_reads_tokens_from_env_and_resolves_allowed_channels(
        self, monkeypatch
    ):
        created_clients = {}

        class DummyWebClient:
            def __init__(self, token):
                self.token = token
                self.auth_test = AsyncMock(return_value={"user_id": "B123"})
                self.conversations_list = AsyncMock(
                    return_value={
                        "channels": [{"id": "C123", "name": "general"}],
                        "response_metadata": {},
                    }
                )
                created_clients["web_client"] = self

        class DummySocketClient:
            def __init__(self, app_token, web_client):
                self.app_token = app_token
                self.web_client = web_client
                self.socket_mode_request_listeners = []
                created_clients["socket_client"] = self

            async def connect(self):
                return None

            async def disconnect(self):
                return None

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        monkeypatch.setattr("slack.bot_real.AsyncWebClient", DummyWebClient)
        monkeypatch.setattr("slack.bot_real.AsyncSocketModeClient", DummySocketClient)

        bot = SlackBot(
            {
                "bot_token_env": "SLACK_BOT_TOKEN",
                "app_token_env": "SLACK_APP_TOKEN",
                "channels": ["general"],
            },
            inference_engine=SimpleNamespace(),
        )

        await bot.initialize()

        assert created_clients["web_client"].token == "xoxb-test"
        assert created_clients["socket_client"].app_token == "xapp-test"
        assert bot.allowed_channel_ids == {"C123"}
        assert bot.is_healthy() is True

    @pytest.mark.asyncio
    async def test_initialize_disables_memory_when_optional_backend_fails(
        self, monkeypatch
    ):
        class DummyWebClient:
            def __init__(self, token):
                self.token = token
                self.auth_test = AsyncMock(return_value={"user_id": "B123"})

        class DummySocketClient:
            def __init__(self, app_token, web_client):
                self.app_token = app_token
                self.web_client = web_client
                self.socket_mode_request_listeners = []

            async def disconnect(self):
                return None

        async def fail_memory_initialize(self):
            raise RuntimeError("redis stack unavailable")

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        monkeypatch.setattr("slack.bot_real.AsyncWebClient", DummyWebClient)
        monkeypatch.setattr("slack.bot_real.AsyncSocketModeClient", DummySocketClient)
        monkeypatch.setattr(
            "slack.bot_real.MemoryManager.initialize", fail_memory_initialize
        )

        bot = SlackBot(
            {
                "bot_token_env": "SLACK_BOT_TOKEN",
                "app_token_env": "SLACK_APP_TOKEN",
                "channels": [],
                "memory": {"enabled": True},
            },
            inference_engine=SimpleNamespace(),
        )

        await bot.initialize()

        assert bot.initialized is True
        assert bot.memory_manager.enabled is False

    @pytest.mark.asyncio
    async def test_message_event_ignores_non_thread_messages(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.bot_user_id = "B1"
        bot._process_message = AsyncMock()

        await bot._handle_message_event(
            {"type": "message", "channel": "C1", "user": "U1", "text": "hello"}
        )

        bot._process_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_message_event_processes_active_thread_replies(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.bot_user_id = "B1"
        bot._process_message = AsyncMock()
        bot._mark_thread_active("C1", "1700.1")

        await bot._handle_message_event(
            {
                "type": "message",
                "channel": "C1",
                "user": "U1",
                "text": "follow up",
                "thread_ts": "1700.1",
            }
        )

        bot._process_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_message_enforces_rate_limit_for_queries(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.web_client = SimpleNamespace(
            chat_postMessage=AsyncMock(), chat_postEphemeral=AsyncMock()
        )
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: False
        bot.message_handler = SimpleNamespace(handle_message=AsyncMock())

        await bot._process_message({"channel": "C1", "user": "U1", "text": "hello"})

        bot.web_client.chat_postEphemeral.assert_awaited_once()
        bot.web_client.chat_postMessage.assert_not_awaited()
        bot.message_handler.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_message_respects_threading_preference(
        self, inference_response_factory
    ):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(
                    response_text="threadless response"
                )
            )
        )
        bot = SlackBot({"channels": []}, inference_engine=inference_engine)
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: True
        bot.user_manager.update_user_preferences("U1", {"threading": False})

        await bot._process_message(
            {
                "channel": "C1",
                "user": "U1",
                "text": "hello",
                "ts": "1700.1",
            }
        )

        assert bot.web_client.chat_postMessage.await_args.kwargs["thread_ts"] is None
        assert bot.active_threads == {}

    @pytest.mark.asyncio
    async def test_process_message_keeps_every_part_of_a_long_answer_in_thread(
        self, inference_response_factory, monkeypatch
    ):
        long_answer = " ".join(f"sentence {index}." for index in range(40))
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text=long_answer)
            )
        )
        bot = SlackBot(
            {"channels": [], "response_settings": {"max_response_length": 120}},
            inference_engine=inference_engine,
        )
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: True
        monkeypatch.setattr("slack.bot_real.asyncio.sleep", AsyncMock())

        await bot._process_message(
            {"channel": "C1", "user": "U1", "text": "hello", "ts": "1700.1"}
        )

        calls = bot.web_client.chat_postMessage.await_args_list
        assert len(calls) > 1
        assert "sentence 0." in calls[0].kwargs["text"]
        assert all(call.kwargs["thread_ts"] == "1700.1" for call in calls)
        assert not any(call.kwargs.get("reply_broadcast") for call in calls)

    @pytest.mark.asyncio
    async def test_slash_command_prefers_ephemeral_response(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.web_client = SimpleNamespace(
            chat_postEphemeral=AsyncMock(),
            chat_postMessage=AsyncMock(),
        )
        bot.message_handler = SimpleNamespace(
            _handle_command=AsyncMock(return_value="ok")
        )

        await bot._handle_slash_command(
            {
                "channel_id": "C1",
                "channel_name": "general",
                "user_id": "U1",
                "text": "help",
            }
        )

        bot.web_client.chat_postEphemeral.assert_awaited_once_with(
            channel="C1",
            user="U1",
            text="ok",
        )
        bot.web_client.chat_postMessage.assert_not_awaited()
        assert (
            bot.message_handler._handle_command.await_args.kwargs["command_surface"]
            == "slash"
        )

    @pytest.mark.asyncio
    async def test_slash_command_routes_free_form_query(
        self, inference_response_factory
    ):
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(
                    response_text="free-form answer"
                )
            )
        )
        bot = SlackBot({"channels": []}, inference_engine=inference_engine)
        bot.web_client = SimpleNamespace(
            chat_postEphemeral=AsyncMock(),
            chat_postMessage=AsyncMock(),
        )

        await bot._handle_slash_command(
            {
                "channel_id": "C1",
                "channel_name": "general",
                "user_id": "U1",
                "text": "write a Python function",
            }
        )

        request = inference_engine.process_query.await_args.args[0]
        assert request.query == "write a Python function"
        bot.web_client.chat_postEphemeral.assert_awaited_once_with(
            channel="C1",
            user="U1",
            text="free-form answer",
        )

    def test_inline_prefixed_free_form_queries_consume_quota(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

        assert bot._is_query_event({"text": "!llm write a Python function"}) is True
        assert bot._is_query_event({"text": "!llm help"}) is False
        assert bot._is_query_event({"text": "!llm agent research transformers"}) is True

    @pytest.mark.asyncio
    async def test_slash_agent_command_is_rejected_when_query_quota_is_exhausted(self):
        agent_runtime = SimpleNamespace(run=AsyncMock())
        bot = SlackBot(
            {"channels": []},
            inference_engine=SimpleNamespace(),
            services={"agent_runtime": agent_runtime},
        )
        bot.web_client = SimpleNamespace(
            chat_postEphemeral=AsyncMock(),
            chat_postMessage=AsyncMock(),
        )
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: False

        await bot._handle_slash_command(
            {
                "channel_id": "C1",
                "channel_name": "general",
                "user_id": "U1",
                "text": "agent research transformers",
            }
        )

        agent_runtime.run.assert_not_awaited()
        response = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
        assert "rate limit" in response.lower()

    @pytest.mark.asyncio
    async def test_process_message_passes_resolved_attachments_to_query(
        self, inference_response_factory
    ):
        attachment = Attachment(
            name="report.csv",
            type=AttachmentType.DOCUMENT,
            size_bytes=3,
            mime_type="text/csv",
            content=b"csv",
        )
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text="done")
            )
        )
        bot = SlackBot({"channels": []}, inference_engine=inference_engine)
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: True
        bot._extract_query_attachments = AsyncMock(return_value=[attachment])

        await bot._process_message(
            {
                "channel": "C1",
                "user": "U1",
                "text": "analyze this",
                "files": [{"id": "F1"}],
            }
        )

        request = inference_engine.process_query.await_args.args[0]
        assert request.attachments == [attachment]

    @pytest.mark.asyncio
    async def test_process_message_uses_tier_specific_rate_limits(self):
        observed_config = {}
        bot = SlackBot(
            {
                "channels": [],
                "user_tiers": {"overrides": {"U1": "premium"}},
                "rate_limiting": {
                    "requests_per_hour": 100,
                    "burst_requests": 5,
                    "by_tier": {
                        "premium": {
                            "requests_per_hour": 500,
                            "burst_requests": 20,
                        }
                    },
                },
            },
            inference_engine=SimpleNamespace(),
        )
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.message_handler = SimpleNamespace(
            handle_message=AsyncMock(return_value=None)
        )

        def capture_rate_limit(_user_id, config):
            observed_config.update(config)
            return True

        bot.user_manager.check_rate_limit = capture_rate_limit

        await bot._process_message({"channel": "C1", "user": "U1", "text": "hello"})

        assert observed_config == {"requests_per_hour": 500, "burst_requests": 20}
        assert bot.user_manager.get_user_tier("U1") == UserTier.PREMIUM

    @pytest.mark.asyncio
    async def test_premium_slack_queries_queue_ahead_of_free_ones(
        self, inference_response_factory
    ):
        from src.provider_scheduler import ProviderCapacityScheduler

        inference_engine = SimpleNamespace(
            process_query=AsyncMock(return_value=inference_response_factory())
        )
        bot = SlackBot(
            {"channels": [], "user_tiers": {"overrides": {"UP": "premium"}}},
            inference_engine=inference_engine,
        )
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: True

        sent = {}
        for user_id in ("UF", "UP"):
            await bot._process_message(
                {"channel": "C1", "user": user_id, "text": "hello", "ts": "1700.1"}
            )
            sent[user_id] = inference_engine.process_query.await_args.args[0]

        assert sent["UF"].user_tier == UserTier.FREE
        assert sent["UP"].user_tier == UserTier.PREMIUM
        scheduler = ProviderCapacityScheduler({}, None)
        assert scheduler._queue_score(sent["UP"]) < scheduler._queue_score(sent["UF"])

    @pytest.mark.asyncio
    async def test_user_stats_report_what_is_left_of_the_chat_bucket(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        config = bot._get_rate_limit_config_for_user("U1")
        assert bot.user_manager.check_rate_limit("U1", config)
        assert bot.user_manager.check_rate_limit("U1", config, bucket="agent")

        stats = await bot.get_user_stats("U1")

        assert stats["remaining_requests"] == config["requests_per_hour"] - 1

    @pytest.mark.asyncio
    async def test_attachment_only_message_uses_default_query_text(
        self, inference_response_factory
    ):
        attachment = Attachment(
            name="report.csv",
            type=AttachmentType.DOCUMENT,
            size_bytes=3,
            mime_type="text/csv",
            content=b"csv",
        )
        inference_engine = SimpleNamespace(
            process_query=AsyncMock(
                return_value=inference_response_factory(response_text="done")
            )
        )
        bot = SlackBot({"channels": []}, inference_engine=inference_engine)
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: True
        bot._extract_query_attachments = AsyncMock(return_value=[attachment])

        await bot._process_message(
            {
                "channel": "C1",
                "user": "U1",
                "text": "",
                "files": [{"id": "F1"}],
            }
        )

        request = inference_engine.process_query.await_args.args[0]
        assert request.query.startswith("Analyze the attached Slack file")
        assert "report.csv" in request.query

    @pytest.mark.asyncio
    async def test_extract_query_attachments_downloads_private_slack_files(
        self, monkeypatch
    ):
        class DummyResponse:
            def __init__(self):
                self.content = b"abc"

            def raise_for_status(self):
                return None

        class DummyHttpClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, headers=None):
                assert url == "https://files.example/report.csv"
                assert headers == {"Authorization": "Bearer xoxb-test"}
                return DummyResponse()

        monkeypatch.setattr("slack.bot_real.httpx.AsyncClient", DummyHttpClient)

        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.bot_token = "xoxb-test"

        attachments = await bot._extract_query_attachments(
            {
                "files": [
                    {
                        "id": "F1",
                        "name": "report.csv",
                        "mimetype": "text/csv",
                        "size": 3,
                        "url_private_download": "https://files.example/report.csv",
                    }
                ]
            }
        )

        assert len(attachments) == 1
        assert attachments[0].content == b"abc"
        assert attachments[0].type == AttachmentType.DOCUMENT

    @pytest.mark.asyncio
    async def test_get_available_models_filters_by_user_tier(self):
        class RouterStub:
            def __init__(self):
                self.models = {
                    "mistral-7b": SimpleNamespace(priority=3),
                    "claude-3.5-sonnet": SimpleNamespace(priority=2),
                }

            def _check_user_access(self, model_config, user_tier):
                tier_priorities = {"free": 3, "premium": 2, "enterprise": 1}
                return model_config.priority >= tier_priorities[user_tier]

            def get_model_info(self, model_name):
                configs = {
                    "mistral-7b": {
                        "provider": "vllm",
                        "capabilities": ["general"],
                        "max_tokens": 4096,
                        "cost_per_token": 0.0,
                    },
                    "gpt-5": {
                        "provider": "openai",
                        "capabilities": ["analysis"],
                        "max_tokens": 8192,
                        "cost_per_token": 0.00003,
                    },
                    "claude-3.5-sonnet": {
                        "provider": "openai",
                        "capabilities": ["analysis"],
                        "max_tokens": 8192,
                        "cost_per_token": 0.00003,
                    },
                }
                return {"config": configs[model_name]}

        healthy_provider = SimpleNamespace(
            get_health_status=lambda: {"status": "healthy"}
        )
        inference_engine = SimpleNamespace(
            providers={"vllm": healthy_provider, "openai": healthy_provider}
        )
        bot = SlackBot(
            {
                "channels": [],
                "user_tiers": {"overrides": {"UP": "premium"}},
            },
            inference_engine=inference_engine,
            router=RouterStub(),
        )

        free_models = await bot.get_available_models("UF")
        premium_models = await bot.get_available_models("UP")

        assert [model["name"] for model in free_models] == ["mistral-7b"]
        assert [model["name"] for model in premium_models] == [
            "mistral-7b",
            "claude-3.5-sonnet",
        ]

    @pytest.mark.asyncio
    async def test_process_message_hides_internal_exception_details(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: True
        bot.message_handler = SimpleNamespace(
            handle_message=AsyncMock(side_effect=RuntimeError("boom"))
        )

        await bot._process_message({"channel": "C1", "user": "U1", "text": "hello"})

        message_text = bot.web_client.chat_postMessage.await_args.kwargs["text"]
        assert "boom" not in message_text
        assert "couldn't process that request" in message_text


class TestSlackStateStores:
    @pytest.mark.asyncio
    async def test_file_state_store_round_trips_snapshot(self, tmp_path):
        store = FileSlackStateStore({"file_path": str(tmp_path / "slack_state.json")})
        snapshot = {
            "schema_version": 1,
            "users": {"u1": {"tier": "premium"}},
            "rate_limits": {"u1": [1.0, 2.0]},
            "conversations": {},
            "active_threads": {"C1:1": datetime.now().isoformat()},
        }

        await store.save_state(snapshot)

        assert await store.load_state() == snapshot

    @pytest.mark.asyncio
    async def test_file_state_store_writes_owner_only_and_fsyncs_before_rename(
        self, tmp_path, monkeypatch
    ):
        state_file = tmp_path / "state" / "slack_state.json"
        state_file.parent.mkdir()
        state_file.write_text("{}", encoding="utf-8")
        state_file.chmod(0o644)
        events = []
        real_fsync = os.fsync
        real_replace = os.replace

        def fsync(fd):
            events.append("fsync")
            real_fsync(fd)

        def replace(src, dst, **kwargs):
            events.append(("replace", os.stat(src).st_mode & 0o777))
            real_replace(src, dst, **kwargs)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)
        previous_umask = os.umask(0o022)
        try:
            store = FileSlackStateStore({"file_path": str(state_file)})
            await store.save_state({"schema_version": 1, "users": {}})
        finally:
            os.umask(previous_umask)

        assert state_file.stat().st_mode & 0o777 == 0o600
        assert events[0] == "fsync"
        assert ("replace", 0o600) in events
        assert events.index("fsync") < events.index(("replace", 0o600))
        assert list(state_file.parent.iterdir()) == [state_file]
        assert await store.load_state() == {"schema_version": 1, "users": {}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "redis_config,connection",
        [
            (
                {"url": "redis://cache:6379/2"},
                {"url": "redis://cache:6379/2"},
            ),
            (
                {"host": "cache", "port": 6380, "db": 2},
                {"host": "cache", "port": 6380, "db": 2},
            ),
        ],
        ids=["url", "host"],
    )
    async def test_redis_slack_store_authenticates_with_password_env(
        self, monkeypatch, redis_config, connection
    ):
        monkeypatch.setenv("REDIS_PASSWORD", "s3cret")
        created = []

        class CapturingRedis:
            def __init__(self, **kwargs):
                created.append(kwargs)

            @classmethod
            def from_url(cls, url, **kwargs):
                return cls(url=url, **kwargs)

            async def ping(self):
                return True

        monkeypatch.setattr("redis.asyncio.Redis", CapturingRedis)
        store = RedisSlackStateStore(
            {"redis": {**redis_config, "password_env": "REDIS_PASSWORD"}}
        )

        await store.initialize()

        assert created == [{**connection, "password": "s3cret"}]

    @pytest.mark.asyncio
    async def test_redis_slack_store_expires_session_state_but_keeps_user_settings(
        self,
    ):
        class ExpiringRedis:
            def __init__(self):
                self.values = {}
                self.expiry = {}
                self.sets = {}

            async def get(self, key):
                return self.values.get(key)

            async def set(self, key, value, ex=None):
                self.values[key] = value
                self.expiry[key] = ex

            async def delete(self, key):
                self.values.pop(key, None)

            async def sadd(self, key, member):
                self.sets.setdefault(key, set()).add(member)

            async def srem(self, key, member):
                self.sets.get(key, set()).discard(member)

            async def smembers(self, key):
                return set(self.sets.get(key, set()))

        store = RedisSlackStateStore({"redis": {"key_prefix": "p"}})
        store.client = ExpiringRedis()
        now = datetime.now().isoformat()

        await store.persist_user_state("U1", {"tier": "premium"}, [time.time()])
        await store.persist_conversation_state("U1:C1:T1", {"last_activity": now})
        await store.persist_active_thread_state("C1:T1", now)
        await store.persist_agent_run_state("R1", {"run_id": "R1"})

        session = int(ConversationManager({}).session_timeout.total_seconds())
        month = 30 * 24 * 3600
        assert store.client.expiry == {
            # The only copy of a user's /llm settings: it never expires.
            "p:user:U1": None,
            # Counted requests older than the hourly window are dropped anyway.
            "p:rate_limit:U1": 3600,
            "p:conversation:U1:C1:T1": session,
            "p:active_thread:C1:T1": session,
            "p:agent_run:R1": month,
        }

    @pytest.mark.asyncio
    async def test_redis_state_store_round_trips_snapshot(self):
        store = RedisSlackStateStore(
            {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "url": os.environ.get("TEST_REDIS_URL"),
                    "db": 0,
                    "key_prefix": "test",
                }
            }
        )
        snapshot = {
            "schema_version": 1,
            "users": {"u1": {"tier": "premium"}},
            "rate_limits": {},
            "conversations": {},
            "active_threads": {},
            "agent_runs": {},
            "posted_ci_jobs": [],
        }

        await store.initialize()
        await store.save_state(snapshot)

        assert await store.load_state() == snapshot

    @pytest.mark.asyncio
    async def test_redis_state_store_preserves_other_users_across_clients(self):
        # Rate limits merge with what is stored, so share no keys between runs.
        prefix = f"test-multi-user-{uuid.uuid4().hex}"
        now = datetime.now().timestamp()
        store_one = RedisSlackStateStore(
            {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "url": os.environ.get("TEST_REDIS_URL"),
                    "db": 0,
                    "key_prefix": prefix,
                }
            }
        )
        store_two = RedisSlackStateStore(
            {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "url": os.environ.get("TEST_REDIS_URL"),
                    "db": 0,
                    "key_prefix": prefix,
                }
            }
        )

        await store_one.initialize()
        await store_two.initialize()

        await store_one.persist_user_state(
            "u1",
            {"tier": "premium", "updated_at": datetime.now().isoformat()},
            [now],
        )
        await store_two.persist_user_state(
            "u2",
            {"tier": "enterprise", "updated_at": datetime.now().isoformat()},
            [now + 1.0],
        )

        loaded = await store_one.load_state()
        # User keys never expire: leave nothing behind in the shared Redis.
        for user_id in ("u1", "u2"):
            await store_one.delete_user_state(user_id)
        await store_one.shutdown()
        await store_two.shutdown()

        assert loaded["users"]["u1"]["tier"] == "premium"
        assert loaded["users"]["u2"]["tier"] == "enterprise"
        assert loaded["rate_limits"]["u1"] == [now]
        assert loaded["rate_limits"]["u2"] == [now + 1.0]

    @pytest.mark.asyncio
    async def test_redis_state_store_rejects_stale_conversation_overwrite(self):
        prefix = "test-stale-conversation"
        store = RedisSlackStateStore(
            {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "url": os.environ.get("TEST_REDIS_URL"),
                    "db": 0,
                    "key_prefix": prefix,
                }
            }
        )
        await store.initialize()

        newer = datetime.now()
        older = newer - timedelta(minutes=5)

        await store.persist_conversation_state(
            "u1:C1:T1",
            {
                "user_id": "u1",
                "channel_id": "C1",
                "thread_ts": "T1",
                "conversation_history": [{"role": "user", "content": "new"}],
                "user_tier": "free",
                "preferences": {},
                "last_activity": newer.isoformat(),
                "session_id": "s1",
            },
        )
        await store.persist_conversation_state(
            "u1:C1:T1",
            {
                "user_id": "u1",
                "channel_id": "C1",
                "thread_ts": "T1",
                "conversation_history": [{"role": "user", "content": "old"}],
                "user_tier": "free",
                "preferences": {},
                "last_activity": older.isoformat(),
                "session_id": "s1",
            },
        )

        loaded = await store.load_state()

        history = loaded["conversations"]["u1:C1:T1"]["conversation_history"]
        assert history == [{"role": "user", "content": "new"}]

    @pytest.mark.asyncio
    async def test_bot_restores_state_from_file_backend(self, monkeypatch, tmp_path):
        class DummyWebClient:
            def __init__(self, token):
                self.token = token
                self.auth_test = AsyncMock(return_value={"user_id": "B123"})

        class DummySocketClient:
            def __init__(self, app_token, web_client):
                self.app_token = app_token
                self.web_client = web_client
                self.socket_mode_request_listeners = []

            async def connect(self):
                return None

            async def disconnect(self):
                return None

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
        monkeypatch.setattr("slack.bot_real.AsyncWebClient", DummyWebClient)
        monkeypatch.setattr("slack.bot_real.AsyncSocketModeClient", DummySocketClient)

        config = {
            "bot_token_env": "SLACK_BOT_TOKEN",
            "app_token_env": "SLACK_APP_TOKEN",
            "state": {
                "backend": "file",
                "file_path": str(tmp_path / "slack_state.json"),
            },
        }

        first_bot = SlackBot(config, inference_engine=SimpleNamespace())
        await first_bot.initialize()
        first_bot.user_manager.users["u1"] = {
            "tier": "premium",
            "preferences": {"response_length": "long"},
        }
        first_bot.user_manager.rate_limits["u1"] = [1.0]
        context = first_bot.conversation_manager.get_or_create_context("u1", "C1", "T1")
        context.add_message("user", "hello")
        first_bot._mark_thread_active("C1", "T1")
        await first_bot._persist_state()

        second_bot = SlackBot(config, inference_engine=SimpleNamespace())
        await second_bot.initialize()

        assert second_bot.user_manager.get_user_tier("u1") == UserTier.FREE
        assert second_bot.user_manager.rate_limits["u1"] == [1.0]
        restored = second_bot.conversation_manager.get_conversation_summary(
            "u1", "C1", "T1"
        )
        assert "user: hello" in restored
        assert second_bot._is_active_thread("C1", "T1") is True


@pytest.mark.asyncio
async def test_slack_work_queue_bounds_running_and_queued_and_replies_busy():
    bot = SlackBot(
        {
            "channels": [],
            "work_queue": {
                "capacity": 1,
                "concurrency": 1,
                "overload_reply_timeout_seconds": 1,
            },
        },
        inference_engine=SimpleNamespace(),
    )
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_event(event):
        started.set()
        await release.wait()

    bot._handle_event = blocked_event
    bot._start_work_workers()

    def request(index):
        return SimpleNamespace(
            envelope_id=f"e-{index}",
            type="events_api",
            payload={
                "event": {
                    "type": "app_mention",
                    "channel": "C1",
                    "user": "U1",
                    "ts": str(index),
                }
            },
        )

    await bot._handle_socket_mode_request(socket_client, request(1))
    await started.wait()
    await bot._handle_socket_mode_request(socket_client, request(2))
    await bot._handle_socket_mode_request(socket_client, request(3))

    assert bot._work_running == 1
    assert bot._work_queue.qsize() == 1
    bot.web_client.chat_postMessage.assert_awaited_once()
    assert "busy" in bot.web_client.chat_postMessage.await_args.kwargs["text"].lower()

    release.set()
    await bot._stop_work_workers()


@pytest.mark.asyncio
async def test_slack_worker_shutdown_cancellation_still_cancels_workers_and_drains_queue():
    bot = SlackBot(
        {"channels": [], "work_queue": {"capacity": 2, "concurrency": 1}},
        inference_engine=SimpleNamespace(),
    )
    started = asyncio.Event()

    async def blocked_event(_event):
        started.set()
        await asyncio.Event().wait()

    bot._handle_event = blocked_event
    bot._start_work_workers()
    item = lambda index: SlackWorkItem(
        kind="event", name=f"event-{index}", payload={"index": index}
    )
    await bot._enqueue_work(item(1))
    await started.wait()
    await bot._enqueue_work(item(2))

    stop_task = asyncio.create_task(bot._stop_work_workers())
    await asyncio.sleep(0)
    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert bot._work_workers == []
    assert bot._work_queue.empty()
    assert bot._work_running == 0


def _busy_bot(situation, channels=()):
    """A bot that cannot take work: its one queue slot is full, or it is draining."""
    bot = SlackBot(
        {
            "channels": list(channels),
            "work_queue": {"capacity": 1, "overload_reply_timeout_seconds": 1},
        },
        inference_engine=SimpleNamespace(),
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(), chat_postEphemeral=AsyncMock()
    )
    if situation == "queue_full":
        bot._work_queue.put_nowait(
            SlackWorkItem(kind="event", name="filler", payload={})
        )
    else:
        bot._accepting_work = False
    return bot


def _message(**fields):
    return {"type": "message", "channel": "C1", **fields}


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", ["queue_full", "draining"])
@pytest.mark.parametrize(
    "kind,payload,posts,ephemerals",
    [
        # The bot's own busy reply lands in the thread it answered: a loop.
        (
            "event",
            _message(thread_ts="100.0", ts="101.0", bot_id="B1", text="busy"),
            0,
            0,
        ),
        ("event", _message(thread_ts="100.0", ts="101.0", user="UBOT"), 0, 0),
        ("event", _message(thread_ts="100.0", ts="101.0", subtype="edited"), 0, 0),
        # Human chatter the bot would never have answered.
        ("event", _message(user="U1", ts="200.0", text="lunch?"), 0, 0),
        ("event", _message(user="U1", thread_ts="900.0", ts="901.0"), 0, 0),
        ("event", {"type": "reaction_added", "user": "U1"}, 0, 0),
        # Someone waiting on the bot.
        (
            "event",
            {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "300.0"},
            1,
            0,
        ),
        ("event", _message(user="U1", thread_ts="100.0", ts="102.0"), 1, 0),
        ("event", _message(user="U1", thread_ts="400.0", ts="401.0"), 1, 0),
        (
            "slash",
            {"channel_id": "C1", "user_id": "U1", "command": "/llm", "text": "hi"},
            0,
            1,
        ),
    ],
    ids=[
        "bot_message",
        "own_message",
        "subtype_message",
        "top_level_chatter",
        "untracked_thread_reply",
        "other_event",
        "app_mention",
        "active_thread_reply",
        "agent_thread_reply",
        "slash_command",
    ],
)
async def test_slack_busy_reply_answers_only_people_waiting_on_the_bot(
    situation, kind, payload, posts, ephemerals
):
    bot = _busy_bot(situation)
    last_activity = datetime.now() - timedelta(minutes=5)
    bot.active_threads["C1:100.0"] = last_activity
    bot.agent_threads["C1:400.0"] = {"owner_user_id": "U1"}

    accepted = await bot._enqueue_work(
        SlackWorkItem(kind=kind, payload=dict(payload), name="test")
    )

    assert accepted is False
    assert bot.web_client.chat_postMessage.await_count == posts
    assert bot.web_client.chat_postEphemeral.await_count == ephemerals
    for call in (
        bot.web_client.chat_postMessage.await_args_list
        + bot.web_client.chat_postEphemeral.await_args_list
    ):
        assert call.kwargs["text"] == SLACK_BUSY_MESSAGE
        assert call.kwargs["channel"] == "C1"
    # Checking the thread is read-only: a busy reply keeps no thread alive.
    assert bot.active_threads == {"C1:100.0": last_activity}


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", ["queue_full", "draining"])
@pytest.mark.parametrize("first", ["message", "app_mention"])
async def test_slack_busy_reply_answers_a_mention_in_a_thread_once(situation, first):
    bot = _busy_bot(situation)
    bot.active_threads["C1:100.0"] = datetime.now()
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())

    async def deliver(event_type, ts):
        # Slack sends a message and an app_mention event for one @bot post.
        event = _message(user="U1", thread_ts="100.0", ts=ts, text="<@UBOT> hi")
        event["type"] = event_type
        await bot._handle_socket_mode_request(
            socket_client,
            SimpleNamespace(
                envelope_id=f"E-{event_type}-{ts}",
                type="events_api",
                payload={"event": event},
            ),
        )

    second = "app_mention" if first == "message" else "message"
    await deliver(first, "101.0")
    await deliver(second, "101.0")

    bot.web_client.chat_postMessage.assert_awaited_once()
    assert bot.web_client.chat_postMessage.await_args.kwargs["thread_ts"] == "100.0"

    # Another post in the same thread still gets its own reply.
    await deliver(first, "102.0")
    await deliver(second, "102.0")
    assert bot.web_client.chat_postMessage.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("situation", ["queue_full", "draining"])
async def test_slack_busy_reply_stays_out_of_channels_the_bot_is_not_enabled_in(
    situation,
):
    bot = _busy_bot(situation, channels=["CALLOWED1", "#bots"])
    bot.web_client.conversations_list = AsyncMock(return_value={"channels": []})
    bot.web_client.conversations_info = AsyncMock(
        return_value={"channel": {"name": "random"}}
    )
    await bot._resolve_allowed_channels()

    for channel, ts in (("COTHER001", "300.0"), ("CALLOWED1", "301.0")):
        await bot._enqueue_work(
            SlackWorkItem(
                kind="event",
                payload={
                    "type": "app_mention",
                    "channel": channel,
                    "user": "U1",
                    "ts": ts,
                },
                name="mention",
            )
        )

    bot.web_client.chat_postMessage.assert_awaited_once()
    assert bot.web_client.chat_postMessage.await_args.kwargs["channel"] == "CALLOWED1"


@pytest.mark.asyncio
async def test_slack_bot_and_own_messages_are_acked_but_never_queued():
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    events = [
        _message(thread_ts="100.0", ts="101.0", bot_id="B1"),
        _message(thread_ts="100.0", ts="102.0", user="UBOT"),
        _message(thread_ts="100.0", ts="103.0", subtype="message_changed"),
    ]

    for index, event in enumerate(events):
        await bot._handle_socket_mode_request(
            socket_client,
            SimpleNamespace(
                envelope_id=f"E-{index}",
                type="events_api",
                payload={"event": event},
            ),
        )

    assert socket_client.send_socket_mode_response.await_count == len(events)
    assert bot._work_queue.empty()
    bot.web_client.chat_postMessage.assert_not_awaited()

    human = _message(user="U1", thread_ts="100.0", ts="104.0")
    assert await bot._enqueue_work(
        SlackWorkItem(kind="event", payload=human, name="human")
    )
    assert bot._work_queue.qsize() == 1


@pytest.mark.asyncio
async def test_slack_start_propagates_terminal_socket_failure_and_disconnects_once():
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    bot.socket_client = SimpleNamespace(
        connect=AsyncMock(side_effect=RuntimeError("socket failed")),
        disconnect=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="socket failed"):
        await bot.start()

    await bot._disconnect_socket_once()
    bot.socket_client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,method",
    [("repair-ci", "repair_ci"), ("publish-repair", "retry_publication")],
)
async def test_ci_commands_preserve_the_configured_slack_identity(action, method):
    request = AsyncMock(
        return_value={
            "run_id": "ci-run",
            "session_id": "ci-session",
            "status": "awaiting_approval",
        }
    )
    runtime = SimpleNamespace(**{method: request})
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(),
        services={"agent_runtime": runtime},
    )
    bot._monitor_agent_run = AsyncMock()
    handler = SlackMessageHandler(bot)
    response = await handler._handle_agent_command(
        [action, "target"],
        "owner",
        "channel",
        SimpleNamespace(),
        team_id="team",
        thread_ts="1.0",
    )
    request.assert_awaited_once_with(
        "target", "owner", team_id="team", channel_id="channel"
    )
    assert "ci-run" in response
    if bot.background_tasks:
        await asyncio.gather(*list(bot.background_tasks))


@pytest.mark.asyncio
async def test_structured_slack_flags_select_workflow_and_resume_tracks_same_run():
    runtime = SimpleNamespace(
        resume=AsyncMock(
            return_value={"run_id": "R", "session_id": "S", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(),
        services={"agent_runtime": runtime},
    )
    bot._monitor_agent_run = AsyncMock()
    handler = SlackMessageHandler(bot)
    handler._start_agent_prompt = AsyncMock(return_value="started")
    response = await handler._handle_agent_command(
        ["--workflow", "fix", "counter"],
        "U",
        "C",
        SimpleNamespace(),
        team_id="T",
        thread_ts="1",
    )
    assert response == "started"
    assert (
        handler._start_agent_prompt.await_args.kwargs["workflow_mode"] == "structured"
    )
    assert handler._start_agent_prompt.await_args.args[0] == "fix counter"
    response = await handler._handle_agent_command(
        ["resume", "R"], "U", "C", SimpleNamespace(), team_id="T", thread_ts="1"
    )
    runtime.resume.assert_awaited_once_with("R", "U")
    assert "R" in response
    if bot.background_tasks:
        await asyncio.gather(*list(bot.background_tasks))


@pytest.mark.asyncio
async def test_slack_discussion_flag_is_explicit_and_not_inherited():
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(),
        services={"agent_runtime": SimpleNamespace()},
    )
    handler = SlackMessageHandler(bot)
    handler._start_agent_prompt = AsyncMock(return_value="started")
    for flags, expected in [
        (["--discuss", "--workflow"], "discuss"),
        (["--implement"], "implement"),
        ([], None),
    ]:
        response = await handler._handle_agent_command(
            [*flags, "explain", "project"],
            "U",
            "C",
            SimpleNamespace(),
            team_id="T",
            thread_ts="1",
        )
        assert response == "started"
        assert (
            handler._start_agent_prompt.await_args.kwargs.get("task_mode") == expected
        )
    handler._start_agent_prompt.reset_mock()
    response = await handler._handle_agent_command(
        ["--discuss", "--implement", "project"],
        "U",
        "C",
        SimpleNamespace(),
        team_id="T",
        thread_ts="1",
    )
    assert "conflict" in response.lower()
    handler._start_agent_prompt.assert_not_awaited()


def test_discussion_response_shows_no_verification_or_delivery():
    response = SlackMessageHandler._format_agent_response(
        {
            "answer": "Project explanation",
            "task_mode": "discuss",
            "verification_status": "not_applicable",
            "grader_status": "not_applicable",
            "branch": "pi-agent/context-only",
        }
    )
    assert "Read-only discussion" in response
    assert "not_applicable" in response
    assert "*Delivery:*" not in response


def _section_texts(blocks):
    return [block["text"]["text"] for block in blocks if block["type"] == "section"]


@pytest.mark.parametrize("detail_length", [0, 120, 2_950, 3_000, 4_096])
def test_agent_approval_blocks_stay_within_slack_section_limit(detail_length):
    detail = "x" * detail_length
    blocks = SlackBot._agent_approval_blocks(
        {
            "title": "Approve write to src/app.py",
            "detail": detail,
            "expires_at": "2026-10-18T12:05:00.000Z",
        }
    )
    sections = _section_texts(blocks)
    assert all(len(text) <= 3000 for text in sections)
    assert sections[0] == "*Approve write to src/app.py*"
    detail_text = sections[1]
    if detail_length + 6 > 3000:
        shown = detail_text.split("```")[1]
        assert (
            f"_Truncated: showing {len(shown)}/{detail_length} chars._" in detail_text
        )
    else:
        assert "Truncated" not in detail_text
        if detail_length:
            assert detail_text == f"```{detail}```"
    footer = blocks[-1]["elements"][0]["text"]
    expected_ts = int(datetime.fromisoformat("2026-10-18T12:05:00+00:00").timestamp())
    assert f"<!date^{expected_ts}^{{time}}|2026-10-18T12:05:00.000Z>" in footer
    assert "5 minutes" not in footer


def test_agent_approval_blocks_neutralize_fences_and_missing_expiry():
    blocks = SlackBot._agent_approval_blocks(
        {"title": "Run command", "detail": "echo ```break``` out"}
    )
    detail_text = _section_texts(blocks)[1]
    assert detail_text.count("```") == 2
    assert "expires soon" in blocks[-1]["elements"][0]["text"]


CI_TARGET = {
    "repository": "acme/app",
    "pr": 42,
    "head": "abc123",
    "branch": "feature",
    "workflow": "Tests",
    "run": 9001,
    "attempt": 2,
    "failed_jobs": ["pytest (3.12)", "lint"],
    "diagnostic": "secret-looking log text",
}


def test_ci_repair_approval_card_names_the_ci_job_it_repairs():
    blocks = SlackBot._agent_approval_blocks(
        {
            "title": "Allow workspace edit: src/app.py?",
            "detail": "src/app.py\n\n- a\n+ b",
        },
        ci=CI_TARGET,
    )
    sections = _section_texts(blocks)
    assert sections[0] == "*Allow workspace edit: src/app.py?*"
    ci_line = sections[1]
    assert "*Tests*" in ci_line
    assert "<https://github.com/acme/app/actions/runs/9001|run 9001>" in ci_line
    assert "attempt 2" in ci_line
    assert "<https://github.com/acme/app/pull/42|PR #42>" in ci_line
    assert "`pytest (3.12)`, `lint`" in ci_line
    assert "secret-looking" not in "".join(sections)
    assert sections[2].startswith("```src/app.py")
    assert all(len(text) <= 3000 for text in sections)


def test_ci_repair_card_escapes_and_bounds_what_github_reported():
    blocks = SlackBot._agent_approval_blocks(
        {"title": "Allow workspace write: <!channel>&x?", "detail": "x"},
        ci={
            "repository": "acme/app",
            "workflow": "<!here> CI",
            "run": "not-a-number",
            "pr": None,
            "failed_jobs": ["`odd` <job>"] + [f"job-{i}" for i in range(40)],
        },
    )
    sections = _section_texts(blocks)
    assert sections[0] == "*Allow workspace write: &lt;!channel&gt;&amp;x?*"
    ci_line = sections[1]
    assert "<!here>" not in ci_line and "&lt;!here&gt; CI" in ci_line
    assert "actions/runs" not in ci_line and "pull/" not in ci_line
    assert "'odd' &lt;job&gt;" in ci_line
    assert "…and 31 more" in ci_line
    assert len(ci_line) <= 3000


def test_ci_repair_card_says_when_github_named_no_failed_job():
    blocks = SlackBot._agent_approval_blocks(
        {"title": "Allow shell command: make test?", "detail": "make test"},
        ci={"repository": "acme/app", "workflow": "CI", "run": 7},
    )
    assert "Failed jobs: _not reported_" in _section_texts(blocks)[1]


@pytest.mark.asyncio
async def test_monitor_posts_oversized_approval_card_with_valid_blocks():
    runtime = SimpleNamespace(
        wait_for_update=AsyncMock(
            side_effect=[
                {
                    "run_id": "r",
                    "session_id": "s",
                    "status": "awaiting_approval",
                    "events": [
                        {
                            "type": "approval",
                            "approval_id": "a",
                            "title": "Approve shell command",
                            "detail": "rm -rf build && " * 300,
                            "expires_at": "2026-10-18T12:05:00Z",
                        }
                    ],
                },
                {"run_id": "r", "session_id": "s", "status": "rejected"},
            ]
        )
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.agent_monitor_poll_seconds = 0

    await bot._monitor_agent_run({"run_id": "r", "channel_id": "c", "thread_ts": "t"})

    blocks = bot.web_client.chat_postMessage.await_args_list[0].kwargs["blocks"]
    assert all(len(text) <= 3000 for text in _section_texts(blocks))
    assert "Truncated: showing" in _section_texts(blocks)[1]
    assert blocks[-1]["type"] == "actions"


@pytest.mark.parametrize(
    "status,error,expected",
    [
        ("completed", None, "*completed*"),
        (
            "failed",
            {"code": "commit_failed", "message": "Changes remain"},
            "*failed* — commit_failed: Changes remain",
        ),
        (
            "timed_out",
            {"code": "run_timeout", "message": "Agent prompt exceeded its deadline"},
            "*timed_out* — run_timeout",
        ),
        ("timed_out", None, "*timed_out* — the run exceeded its deadline"),
        ("cancelled", None, "*cancelled*"),
        ("rejected", None, "*rejected*"),
        ("interrupted", None, "*interrupted*"),
    ],
)
def test_agent_response_leads_with_terminal_status(status, error, expected):
    result = {"run_id": "r1", "status": status, "answer": ""}
    if error:
        result["error"] = error
    response = SlackMessageHandler._format_agent_response(result, max_length=2000)
    first_line = response.splitlines()[0]
    assert response.strip()
    assert "`r1`" in first_line
    assert expected in first_line


def test_agent_response_status_line_survives_truncation():
    response = SlackMessageHandler._format_agent_response(
        {
            "run_id": "r1",
            "status": "failed",
            "error": {"code": "container_start_failed", "message": "boom"},
            "answer": "a" * 5000,
            "changed_files": ["src/a.py"],
        },
        max_length=400,
    )
    assert len(response) <= 400
    assert response.startswith(
        "⚠️ Agent run `r1` *failed* — container_start_failed: boom"
    )
    assert "src/a.py" in response


def test_agent_response_is_never_empty_and_tolerates_partial_findings():
    assert SlackMessageHandler._format_agent_response({"answer": ""}).strip()
    response = SlackMessageHandler._format_agent_response(
        {
            "answer": "done",
            "grader_findings": [
                {"path": "src/a.py", "reason": "missing test"},
                {"line": 3},
                "not-a-finding",
            ],
        }
    )
    assert "• `src/a.py` missing test" in response
    assert "• `unknown:3`" in response


def _delivery_approval(files=3, excerpt="+print('hi')\n", stat=" src/a.py | 2 +-"):
    return {
        "type": "approval",
        "kind": "delivery",
        "approval_id": "a",
        "title": "Deliver changes",
        "expires_at": "2026-10-18T12:30:00.000Z",
        "delivery": {
            "files": [f"src/file_{index}.py" for index in range(files)],
            "stat": stat,
            "verification_status": "passed",
            "grader_status": "approved",
            "excerpt": excerpt,
            "delivery_digest": "sha256:abc",
        },
    }


def _block_texts(blocks):
    texts = _section_texts(blocks)
    for block in blocks:
        if block["type"] == "context":
            texts.extend(element["text"] for element in block["elements"])
    return texts


def test_delivery_card_lists_files_stat_and_check_results():
    blocks = SlackBot._agent_delivery_blocks(_delivery_approval())
    text = "\n".join(_block_texts(blocks))
    assert "*Deliver 3 changed files?*" in text
    for index in range(3):
        assert f"`src/file_{index}.py`" in text
    assert "src/a.py | 2 +-" in text
    assert "print('hi')" in text
    assert "Verification: *passed*" in text
    assert "Grader: *approved*" in text
    assert "<!date^" in text


@pytest.mark.parametrize(
    "files,excerpt_length", [(1, 0), (40, 0), (5, 5_000), (60, 9_000)]
)
def test_delivery_card_text_stays_within_3000_chars(files, excerpt_length):
    approval = _delivery_approval(
        files=files, excerpt="```" + "x" * excerpt_length, stat="s" * 2_000
    )
    texts = _block_texts(SlackBot._agent_delivery_blocks(approval))
    assert sum(len(text) for text in texts) <= 3000
    assert all(text.count("```") % 2 == 0 for text in texts)
    if files > 15:
        assert f"…and {files - 15} more." in "\n".join(texts)


@pytest.mark.parametrize("files", [12, 15, 40])
def test_delivery_card_bounds_a_list_of_long_file_names(files):
    approval = _delivery_approval(files=files, stat="s" * 500)
    approval["delivery"]["files"] = [
        f"src/{index:02d}/" + "n" * 300 for index in range(files)
    ]
    blocks = SlackBot._agent_delivery_blocks(approval)
    texts = _block_texts(blocks)
    assert sum(len(text) for text in texts) <= 3000
    files_text = _section_texts(blocks)[1]
    shown = files_text.count("• `")
    assert 0 < shown < files
    assert files_text.endswith(f"_…and {files - shown} more._")
    assert "s" * 100 in "\n".join(texts)


@pytest.mark.asyncio
async def test_monitor_posts_delivery_card_with_deliver_and_discard_buttons():
    runtime = SimpleNamespace(
        wait_for_update=AsyncMock(
            side_effect=[
                {
                    "run_id": "r",
                    "session_id": "s",
                    "status": "awaiting_approval",
                    "phase": "awaiting_delivery",
                    "events": [_delivery_approval()],
                },
                {"run_id": "r", "session_id": "s", "status": "completed"},
            ]
        )
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.agent_monitor_poll_seconds = 0

    await bot._monitor_agent_run({"run_id": "r", "channel_id": "c", "thread_ts": "t"})

    kwargs = bot.web_client.chat_postMessage.await_args_list[0].kwargs
    actions = kwargs["blocks"][-1]
    assert actions["type"] == "actions"
    assert [element["text"]["text"] for element in actions["elements"]] == [
        "Deliver",
        "Discard",
    ]
    assert [element["action_id"] for element in actions["elements"]] == [
        "pi_agent_approve",
        "pi_agent_reject",
    ]
    assert "*Deliver 3 changed files?*" in _section_texts(kwargs["blocks"])[0]


def test_delivery_expired_status_line_suggests_the_next_step():
    response = SlackMessageHandler._format_agent_response(
        {
            "run_id": "r1",
            "status": "failed",
            "error": {
                "code": "delivery_expired",
                "message": "Delivery approval expired",
            },
            "answer": "",
        },
        max_length=2000,
    )
    lines = response.splitlines()
    assert "*failed* — delivery_expired" in lines[0]
    assert lines[1].startswith("Next step: ")
    assert "approve it in time" in lines[1]


@pytest.mark.parametrize(
    "error",
    [
        {"code": "scope_violation"},
        {"code": "diff_limit_exceeded"},
        {"code": "agent_tool_limit"},
        {"code": "agent_turn_limit"},
        {"code": "verification_failed", "cause_code": "grader_timeout"},
        {"code": "verification_failed", "cause_code": "insufficient_evidence"},
        {"code": "quality_gate_unavailable", "cause_code": "docker_unavailable"},
        {"code": "delivery_expired"},
        {"code": "delivery_rejected"},
        {"code": "archive_failed"},
        {"code": "session_integrity_failed"},
    ],
)
def test_agent_status_line_appends_next_step_for_known_codes(error):
    key = error.get("cause_code") or error["code"]
    line = _format_agent_status_line(
        {"run_id": "r1", "status": "failed", "error": {**error, "message": "m"}}
    )
    assert line.startswith(f"⚠️ Agent run `r1` *failed* — {error['code']}: m")
    assert line.endswith(f"\nNext step: {AGENT_ERROR_HINTS[key]}")


def _runtime_cause_codes():
    import re
    from pathlib import Path

    source = Path("agent-runtime/src/error-taxonomy.ts").read_text()
    union = re.search(r"export type CauseCode =(.*?);", source, re.S)
    assert union, "CauseCode union not found in error-taxonomy.ts"
    codes = re.findall(r'"([a-z_]+)"', union.group(1))
    assert len(codes) >= 16
    return codes


@pytest.mark.parametrize("cause", _runtime_cause_codes())
def test_every_runtime_cause_code_has_a_next_step_hint(cause):
    line = _format_agent_status_line(
        {
            "run_id": "r1",
            "status": "failed",
            "error": {"code": "verification_failed", "cause_code": cause},
        }
    )
    assert cause in AGENT_ERROR_HINTS
    assert line.endswith(f"\nNext step: {AGENT_ERROR_HINTS[cause]}")


def test_agent_status_line_prefers_cause_code_hint_over_code_hint():
    line = _format_agent_status_line(
        {
            "status": "failed",
            "error": {"code": "agent_tool_limit", "cause_code": "docker_unavailable"},
        }
    )
    assert AGENT_ERROR_HINTS["docker_unavailable"] in line
    assert AGENT_ERROR_HINTS["agent_tool_limit"] not in line


def test_agent_status_line_keeps_generic_text_for_unknown_codes():
    line = _format_agent_status_line(
        {
            "run_id": "r1",
            "status": "failed",
            "error": {
                "code": "something_new",
                "cause_code": "also_new",
                "message": "x",
            },
        }
    )
    assert line == "⚠️ Agent run `r1` *failed* — something_new: x"
    assert (
        _format_agent_status_line({"status": "cancelled"})
        == "⚠️ Agent run *cancelled* — the run was cancelled"
    )


def test_agent_status_line_mentions_a_retained_patch_without_a_path():
    line = _format_agent_status_line(
        {
            "status": "failed",
            "error": {"code": "agent_aborted"},
            "patch_retained": True,
        }
    )
    assert line.endswith("\nThe undelivered patch was kept on the runtime host.")
    assert "/" not in line.split("\n", 1)[1]


@pytest.mark.asyncio
async def test_agent_start_sends_a_key_derived_from_the_triggering_message():
    agent_runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "s", "run_id": "r", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(process_query=AsyncMock()),
        services={"agent_runtime": agent_runtime},
    )
    bot._monitor_agent_run = AsyncMock()
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
        chat_update=AsyncMock(),
    )
    handler = SlackMessageHandler(bot)

    for message_ts in ("200.5", "200.5", "200.6"):
        await handler._handle_agent_command(
            ["fix", "it"],
            "U1",
            "C1",
            client,
            team_id="T1",
            thread_ts="100.1",
            message_ts=message_ts,
        )

    keys = [
        call.kwargs.get("idempotency_key")
        for call in agent_runtime.create_session.await_args_list
    ]
    assert keys[0] and keys[0] == keys[1], "a redelivered message reuses its key"
    assert keys[2] and keys[2] != keys[0]
    assert "200.5" not in keys[0], "the key does not echo Slack identifiers"


@pytest.mark.asyncio
async def test_agent_start_message_describes_delivery_approval():
    agent_runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "s", "run_id": "r", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(process_query=AsyncMock()),
        services={"agent_runtime": agent_runtime},
    )
    bot._monitor_agent_run = AsyncMock()
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
        chat_update=AsyncMock(),
    )
    handler = SlackMessageHandler(bot)

    await handler._handle_agent_command(
        ["fix", "it"], "U1", "C1", client, team_id="T1", message_ts="100.1"
    )

    started = client.chat_update.await_args.kwargs["text"]
    # Delivery mode runs worktree edits without a prompt; the owner approves the diff.
    assert "edits and risky shell commands require approval" not in started
    assert "Edits in this worktree run automatically" in started
    assert "approves the final diff" in started


def _agent_control_bot(method):
    accepted = {"run_id": "R", "session_id": "S", "status": "starting"}
    if method in {"repair_ci", "retry_publication"}:
        accepted["status"] = "awaiting_approval"
    runtime = SimpleNamespace(**{method: AsyncMock(return_value=accepted)})
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot._start_agent_monitor = lambda tracked: None
    bot.agent_user_runs["U1"] = {
        "run_id": "R",
        "session_id": "S",
        "channel_id": "C1",
        "thread_ts": "100.1",
        "owner_user_id": "U1",
    }
    return bot, getattr(runtime, method)


AGENT_CONTROL_COMMANDS = [
    (["stop"], "cancel"),
    (["resume", "R"], "resume"),
    (["repair-ci", "7"], "repair_ci"),
    (["publish-repair", "R"], "retry_publication"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("args,method", AGENT_CONTROL_COMMANDS)
async def test_agent_control_keys_are_derived_from_the_triggering_message(args, method):
    bot, request = _agent_control_bot(method)
    handler = SlackMessageHandler(bot)

    for message_ts in ("200.5", "200.5", "200.6"):
        await handler._handle_agent_command(
            list(args),
            "U1",
            "C1",
            SimpleNamespace(),
            team_id="T1",
            thread_ts="100.1",
            message_ts=message_ts,
        )

    keys = [call.kwargs.get("idempotency_key") for call in request.await_args_list]
    assert keys[0] == _agent_idempotency_key("T1", "C1", "200.5", "U1", method)
    assert keys[1] == keys[0], "a redelivered message reuses its key"
    assert keys[2] == _agent_idempotency_key("T1", "C1", "200.6", "U1", method)


@pytest.mark.asyncio
@pytest.mark.parametrize("args,method", AGENT_CONTROL_COMMANDS)
async def test_slash_agent_control_keys_are_derived_from_the_command_trigger(
    args, method
):
    bot, request = _agent_control_bot(method)
    bot.web_client = SimpleNamespace(
        chat_postEphemeral=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "300.1"}),
    )

    for trigger_id in ("trigger-1", "trigger-1", "trigger-2"):
        await bot._handle_slash_command(
            {
                "text": " ".join(["agent", *args]),
                "user_id": "U1",
                "channel_id": "C1",
                "team_id": "T1",
                "trigger_id": trigger_id,
            }
        )

    keys = [call.kwargs.get("idempotency_key") for call in request.await_args_list]
    assert keys[0] and keys[1] == keys[0], "a redelivered command reuses its key"
    assert keys[2] and keys[2] != keys[0]
    assert "trigger-1" not in keys[0], "the key does not echo Slack identifiers"


async def _drain_background_tasks(bot):
    while bot.background_tasks:
        await asyncio.gather(*list(bot.background_tasks))


def _public_channel_client(**methods):
    return SimpleNamespace(
        conversations_info=AsyncMock(
            return_value={
                "ok": True,
                "channel": {"is_channel": True, "is_private": False},
            }
        ),
        views_open=AsyncMock(),
        **methods,
    )


async def _open_agent_shortcut(bot, trigger_id="trigger-1"):
    await bot._handle_agent_shortcut(
        {
            "trigger_id": trigger_id,
            "team": {"id": "T1"},
            "user": {"id": "U1"},
            "channel": {"id": "C1"},
            "message": {"ts": "100.2", "thread_ts": "100.1"},
        }
    )
    return bot.web_client.views_open.await_args.kwargs["view"]["private_metadata"]


def _shortcut_submission(nonce, task, **values):
    return {
        "type": "view_submission",
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "view": {
            "callback_id": "run_pi_agent_from_thread_submit",
            "private_metadata": nonce,
            "state": {"values": {"agent_task": {"task": {"value": task}}, **values}},
        },
    }


@pytest.mark.asyncio
async def test_approval_clicks_are_not_blocked_by_slow_agent_starts():
    bot = SlackBot(
        {"channels": [], "work_queue": {"capacity": 8, "concurrency": 2}},
        inference_engine=SimpleNamespace(),
    )
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    release = asyncio.Event()
    workers_busy = asyncio.Event()
    started = []

    async def slow_agent_start(event):
        # Stands in for a thread follow-up whose create_session waits on a
        # worktree and a container.
        started.append(event)
        if len(started) == 2:
            workers_busy.set()
        await release.wait()

    decided = asyncio.Event()

    async def decide(payload):
        decided.set()

    bot._handle_event = slow_agent_start
    bot._handle_agent_interactive = decide
    bot._start_work_workers()
    for index in range(3):
        await bot._handle_socket_mode_request(
            socket_client,
            SimpleNamespace(
                envelope_id=f"E-{index}",
                type="events_api",
                payload={"event": {"type": "message", "ts": str(index)}},
            ),
        )
    await asyncio.wait_for(workers_busy.wait(), timeout=1)

    await bot._handle_socket_mode_request(
        socket_client,
        SimpleNamespace(
            envelope_id="E-approve",
            type="interactive",
            payload={
                "type": "block_actions",
                "user": {"id": "U1"},
                "channel": {"id": "C1"},
                "actions": [
                    {
                        "action_id": "pi_agent_approve",
                        "value": '{"run_id":"R1","approval_id":"A1"}',
                    }
                ],
            },
        ),
    )

    try:
        await asyncio.wait_for(decided.wait(), timeout=1)
    finally:
        release.set()
        await bot._stop_work_workers()


@pytest.mark.asyncio
async def test_slash_agent_start_acks_then_creates_in_the_background():
    release = asyncio.Event()

    async def slow_create_session(*args, **kwargs):
        await release.wait()
        return {"session_id": "S1", "run_id": "R1", "status": "starting"}

    runtime = SimpleNamespace(create_session=AsyncMock(side_effect=slow_create_session))
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot._monitor_agent_run = AsyncMock()
    bot.web_client = SimpleNamespace(
        chat_postEphemeral=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
        chat_update=AsyncMock(),
    )

    await asyncio.wait_for(
        bot._handle_slash_command(
            {
                "text": "agent fix the flaky test",
                "user_id": "U1",
                "channel_id": "C1",
                "team_id": "T1",
            }
        ),
        timeout=1,
    )

    ack = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "starting" in ack.lower()
    assert "R1" not in bot.agent_runs

    release.set()
    await _drain_background_tasks(bot)

    runtime.create_session.assert_awaited_once()
    assert runtime.create_session.await_args.args[:5] == (
        "T1",
        "C1",
        "100.1",
        "U1",
        "fix the flaky test",
    )
    assert bot.agent_runs["R1"]["owner_user_id"] == "U1"
    assert "R1" in bot.web_client.chat_update.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_modal_agent_start_acks_then_bootstraps_in_the_background():
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": [], "agent_context": {"enabled": True}},
        SimpleNamespace(),
        services={"agent_runtime": runtime},
    )
    bot._monitor_agent_run = AsyncMock()
    bot.web_client = _public_channel_client(
        chat_postMessage=AsyncMock(return_value={"ts": "status-1"}),
        chat_update=AsyncMock(),
    )
    release = asyncio.Event()

    async def slow_build(*args, **kwargs):
        await release.wait()
        return AgentBootstrapContext(prompt="bootstrap prompt", message_count=1)

    bot.agent_bootstrap_builder = SimpleNamespace(
        build=AsyncMock(side_effect=slow_build)
    )
    nonce = await _open_agent_shortcut(bot)

    await asyncio.wait_for(
        bot._handle_agent_view_submission(
            _shortcut_submission(
                nonce,
                "Implement the todo",
                agent_context={"include": {"selected_options": [{"value": "include"}]}},
            )
        ),
        timeout=1,
    )

    assert "loading" in bot.web_client.chat_postMessage.await_args.kwargs["text"]
    runtime.create_session.assert_not_awaited()

    release.set()
    await _drain_background_tasks(bot)

    runtime.create_session.assert_awaited_once()
    assert runtime.create_session.await_args.args[4] == "bootstrap prompt"
    assert "R1" in bot.web_client.chat_update.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_modal_agent_start_marks_the_selected_message_in_context():
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": [], "agent_context": {"enabled": True}},
        SimpleNamespace(),
        services={"agent_runtime": runtime},
    )
    bot._monitor_agent_run = AsyncMock()
    bot.web_client = _public_channel_client(
        chat_postMessage=AsyncMock(return_value={"ts": "status-1"}),
        chat_update=AsyncMock(),
    )
    bot.agent_bootstrap_builder = SimpleNamespace(
        build=AsyncMock(
            return_value=AgentBootstrapContext(prompt="bootstrap prompt", partial=True)
        )
    )
    nonce = await _open_agent_shortcut(bot)

    await bot._handle_agent_view_submission(
        _shortcut_submission(
            nonce,
            "Implement the todo",
            agent_context={"include": {"selected_options": [{"value": "include"}]}},
        )
    )
    await _drain_background_tasks(bot)

    build = bot.agent_bootstrap_builder.build.await_args
    assert build.kwargs["thread_ts"] == "100.1"
    assert build.kwargs["selected_ts"] == "100.2"
    assert runtime.create_session.await_args.args[4] == "bootstrap prompt"
    status = bot.web_client.chat_update.await_args.kwargs["text"]
    assert "partial" in status.lower()


def _watch_until(bot, finished, run_id="R1"):
    """Keep the monitor of `run_id` watching until `finished` is set."""

    async def monitor(tracked):
        if tracked["run_id"] == run_id:
            await finished.wait()
            tracked["result_posted"] = True

    bot._monitor_agent_run = monitor


@pytest.mark.asyncio
async def test_per_user_agent_cap_names_the_active_run():
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S2", "run_id": "R2", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    finished = asyncio.Event()
    _watch_until(bot, finished)
    bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    active = bot._track_agent_run(
        {"session_id": "S1", "run_id": "R1", "status": "running"},
        team_id="T1",
        channel_id="C1",
        thread_ts="100.1",
        owner_user_id="U1",
    )
    bot._start_agent_monitor(active)
    handler = bot.message_handler
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "300.1"}),
        chat_update=AsyncMock(),
    )

    response = await handler._handle_agent_command(
        ["fix", "another", "bug"], "U1", "C2", client, team_id="T1", thread_ts="200.1"
    )
    assert "R1" in response
    runtime.create_session.assert_not_awaited()
    client.chat_postMessage.assert_not_awaited()

    await bot._handle_slash_command(
        {"text": "agent fix it", "user_id": "U1", "channel_id": "C2", "team_id": "T1"}
    )
    assert "R1" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert not [
        task for task in bot.background_tasks if task.get_name() == "pi_agent_start"
    ]
    runtime.create_session.assert_not_awaited()

    # A follow-up in the active run's own thread is not a second agent.
    await handler._handle_agent_command(
        ["add", "a", "test"], "U1", "C1", client, team_id="T1", thread_ts="100.1"
    )
    assert runtime.create_session.await_count == 1
    # Another user is not capped by U1's run.
    await handler._handle_agent_command(
        ["fix"], "U2", "C2", client, team_id="T1", thread_ts="200.1"
    )
    assert runtime.create_session.await_count == 2

    # Once R1 finishes the slot is free again.
    finished.set()
    await _drain_background_tasks(bot)
    assert active["result_posted"] is True
    await handler._handle_agent_command(
        ["fix", "another", "bug"], "U1", "C2", client, team_id="T1", thread_ts="200.2"
    )
    assert runtime.create_session.await_count == 3
    await _drain_background_tasks(bot)


@pytest.mark.asyncio
async def test_per_user_agent_cap_ignores_a_run_nobody_is_watching():
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S2", "run_id": "R2", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot._monitor_agent_run = AsyncMock()
    # Its monitor gave up without posting a result, e.g. on a runtime error.
    bot._track_agent_run(
        {"session_id": "S1", "run_id": "R1", "status": "running"},
        team_id="T1",
        channel_id="C1",
        thread_ts="100.1",
        owner_user_id="U1",
    )

    response = await bot.message_handler._handle_agent_command(
        ["fix"], "U1", "C2", SimpleNamespace(), team_id="T1", thread_ts="200.1"
    )

    assert "R2" in response
    await _drain_background_tasks(bot)


@pytest.mark.asyncio
async def test_per_user_agent_cap_is_configurable():
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S2", "run_id": "R2", "status": "starting"}
        )
    )
    finished = asyncio.Event()
    bots = []
    for limit, started in [(1, False), (2, True)]:
        bot = SlackBot(
            {"channels": []},
            SimpleNamespace(),
            services={"agent_runtime": runtime},
            agent_config={"max_active_runs_per_user": limit},
        )
        bots.append(bot)
        _watch_until(bot, finished)
        bot._start_agent_monitor(
            bot._track_agent_run(
                {"session_id": "S1", "run_id": "R1", "status": "running"},
                team_id="T1",
                channel_id="C1",
                thread_ts="100.1",
                owner_user_id="U1",
            )
        )

        response = await bot.message_handler._handle_agent_command(
            ["fix"], "U1", "C2", SimpleNamespace(), team_id="T1", thread_ts="200.1"
        )

        assert ("R2" in response) is started
        assert ("R1" in response) is not started
    runtime.create_session.assert_awaited_once()
    finished.set()
    for bot in bots:
        await _drain_background_tasks(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["message", "app_mention"])
async def test_message_and_mention_for_one_ts_are_handled_once_for_the_owner(first):
    second = "app_mention" if first == "message" else "message"
    bot = SlackBot({"channels": []}, SimpleNamespace())
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    bot.agent_threads["C1:100.1"] = {
        "owner_user_id": "U1",
        "channel_id": "C1",
        "thread_ts": "100.1",
    }
    bot._mark_thread_active("C1", "100.1")
    bot._process_message = AsyncMock()

    def event(event_type, ts, user):
        return {
            "type": event_type,
            "channel": "C1",
            "thread_ts": "100.1",
            "ts": ts,
            "user": user,
            "text": "<@UBOT> add a regression test",
        }

    await bot._handle_event(event(first, "100.2", "U1"))
    await bot._handle_event(event(second, "100.2", "U1"))

    bot._process_message.assert_awaited_once()
    assert bot._process_message.await_args.args[0]["text"] == "add a regression test"

    await bot._handle_event(event(first, "100.3", "U2"))
    await bot._handle_event(event(second, "100.3", "U2"))

    bot._process_message.assert_awaited_once()
    bot.web_client.chat_postEphemeral.assert_awaited_once()
    assert (
        "Only the user" in bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    )


@pytest.mark.asyncio
async def test_restored_session_without_a_run_never_cancels_none():
    runtime = SimpleNamespace(
        lookup_session=AsyncMock(
            return_value={
                "session_id": "S1",
                "run_id": None,
                "owner_user_id": "U1",
                "active": False,
                "task_mode": "implement",
            }
        ),
        get_run=AsyncMock(),
        cancel=AsyncMock(),
        close_session=AsyncMock(),
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.state_store = SimpleNamespace(
        persist_agent_run_state=AsyncMock(), delete_agent_run_state=AsyncMock()
    )
    bot._uses_granular_redis_state = lambda: True
    await bot._restore_agent_thread(
        {"channel": "C1", "thread_ts": "100.1", "_team_id": "T1"}
    )
    handler = bot.message_handler

    for action in ("status", "stop"):
        for thread_ts in ("100.1", None):
            response = await handler._handle_agent_command(
                [action],
                "U1",
                "C1",
                SimpleNamespace(),
                team_id="T1",
                thread_ts=thread_ts,
            )
            assert "no active run" in response.lower()
            assert "S1" in response

    runtime.get_run.assert_not_awaited()
    runtime.cancel.assert_not_awaited()

    response = await handler._handle_agent_command(
        ["close"], "U1", "C1", SimpleNamespace(), team_id="T1", thread_ts="100.1"
    )
    assert "closed" in response
    runtime.close_session.assert_awaited_once_with("S1", "U1")
    bot.state_store.delete_agent_run_state.assert_not_awaited()
    assert not bot.agent_threads


@pytest.mark.asyncio
async def test_expired_shortcut_form_tells_the_user_to_run_it_again():
    runtime = SimpleNamespace(create_session=AsyncMock())
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.web_client = _public_channel_client(chat_postEphemeral=AsyncMock())
    nonce = await _open_agent_shortcut(bot)
    bot.agent_shortcut_nonces[nonce] = dataclasses.replace(
        bot.agent_shortcut_nonces[nonce], expires_at=time.time() - 1
    )
    # Opening another form meanwhile must not erase the expired one's notice.
    await _open_agent_shortcut(bot, trigger_id="trigger-2")

    await bot._handle_agent_view_submission(
        _shortcut_submission(nonce, "A long, carefully written task")
    )

    runtime.create_session.assert_not_awaited()
    notice = bot.web_client.chat_postEphemeral.await_args.kwargs
    assert notice["channel"] == "C1"
    assert notice["user"] == "U1"
    assert "expired" in notice["text"].lower()
    assert "run the shortcut again" in notice["text"].lower()
    assert "A long, carefully written task" in notice["text"]
    assert nonce not in bot.agent_shortcut_nonces


def test_agent_shortcut_modal_offers_discuss_and_workflow():
    modal = build_agent_shortcut_modal("nonce", [])
    blocks = {block.get("block_id"): block for block in modal["blocks"]}

    mode = blocks["agent_mode"]["element"]
    assert [option["value"] for option in mode["options"]] == ["implement", "discuss"]
    assert mode["initial_option"]["value"] == "implement"
    workflow = blocks["agent_workflow"]["element"]
    assert [option["value"] for option in workflow["options"]] == ["structured"]
    assert "initial_options" not in workflow


@pytest.mark.asyncio
async def test_modal_mode_choices_reach_create_session():
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot._monitor_agent_run = AsyncMock()
    bot.web_client = _public_channel_client(
        chat_postMessage=AsyncMock(return_value={"ts": "status-1"}),
        chat_update=AsyncMock(),
    )
    nonce = await _open_agent_shortcut(bot)

    await bot._handle_agent_view_submission(
        _shortcut_submission(
            nonce,
            "Explain the router",
            agent_mode={"mode": {"selected_option": {"value": "discuss"}}},
            agent_workflow={
                "workflow": {"selected_options": [{"value": "structured"}]}
            },
        )
    )
    await _drain_background_tasks(bot)

    kwargs = runtime.create_session.await_args.kwargs
    assert kwargs["task_mode"] == "discuss"
    assert kwargs["workflow_mode"] == "structured"


@pytest.mark.asyncio
async def test_shortcut_in_a_discuss_thread_preselects_discuss():
    bot = SlackBot({"channels": []}, SimpleNamespace())
    bot.web_client = _public_channel_client()
    bot._track_agent_run(
        {"session_id": "S1", "run_id": "R1"},
        team_id="T1",
        channel_id="C1",
        thread_ts="100.1",
        owner_user_id="U1",
        mode="discuss",
    )

    await _open_agent_shortcut(bot)

    modal = bot.web_client.views_open.await_args.kwargs["view"]
    mode = next(b for b in modal["blocks"] if b.get("block_id") == "agent_mode")
    assert mode["element"]["initial_option"]["value"] == "discuss"
