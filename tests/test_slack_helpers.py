import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from src.memory import HashEmbeddingProvider, InMemoryMemoryStore, MemoryManager
from slack.agent_bootstrap import AgentBootstrapContext
from slack.bot_real import (
    ConversationContext,
    ConversationManager,
    FileSlackStateStore,
    RedisSlackStateStore,
    SlackBot,
    SlackMessageHandler,
    SlackWorkItem,
    UserManager,
)
from src.utils.schema import (
    Attachment,
    AttachmentType,
    ResponseSource,
    ToolPolicy,
    UserTier,
)


class FailingEmbeddingProvider:
    async def embed(self, text):
        raise RuntimeError("embedding unavailable")


class TestUserManager:
    def test_get_user_tier_defaults_and_preferences(self):
        manager = UserManager()

        assert manager.get_user_tier("unknown") == UserTier.FREE
        assert manager.get_user_preferences("unknown")["response_length"] == "medium"

        manager.users["u1"] = {"tier": "premium"}
        assert manager.get_user_tier("u1") == UserTier.PREMIUM

    def test_check_rate_limit_enforces_burst_window(self, monkeypatch):
        manager = UserManager()
        now = 1_000_000.0
        monkeypatch.setattr("slack.bot_real.time.time", lambda: now)

        config = {"requests_per_hour": 10, "burst_requests": 2}

        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is False


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
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.user_manager.check_rate_limit = lambda *_args, **_kwargs: False
        bot.message_handler = SimpleNamespace(handle_message=AsyncMock())

        await bot._process_message({"channel": "C1", "user": "U1", "text": "hello"})

        bot.web_client.chat_postMessage.assert_awaited_once()
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
    async def test_redis_state_store_round_trips_snapshot(self):
        store = RedisSlackStateStore(
            {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
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
        }

        await store.initialize()
        await store.save_state(snapshot)

        assert await store.load_state() == snapshot

    @pytest.mark.asyncio
    async def test_redis_state_store_preserves_other_users_across_clients(self):
        prefix = "test-multi-user"
        now = datetime.now().timestamp()
        store_one = RedisSlackStateStore(
            {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
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

        assert second_bot.user_manager.get_user_tier("u1") == UserTier.PREMIUM
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
