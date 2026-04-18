from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import (
    ConversationContext,
    ConversationManager,
    FileSlackStateStore,
    RedisSlackStateStore,
    SlackBot,
    SlackMessageHandler,
    UserManager,
)
from src.utils.schema import UserTier


class FakeRouter:
    def __init__(self):
        self.models = {
            "gpt-5": SimpleNamespace(
                provider="openai",
                capabilities=["general", "analysis", "reasoning", "coding"],
                max_tokens=128000,
                cost_per_token=0.000015,
                priority=2,
            ),
            "mistral-7b": SimpleNamespace(
                provider="vllm",
                capabilities=["general", "coding"],
                max_tokens=8192,
                cost_per_token=0.0,
                priority=3,
            ),
        }
        self.classifier = SimpleNamespace(
            classify_query=lambda _text: (SimpleNamespace(value="analysis"), 0.9)
        )

    def is_healthy(self):
        return True

    def _check_user_access(self, model_config, user_tier):
        tier_priorities = {"free": 3, "premium": 2, "enterprise": 1}
        user_priority = tier_priorities[user_tier.value]
        return model_config.priority >= user_priority

    def get_model_info(self, model_name):
        stats = {
            "gpt-5": {"success_rate": 0.99, "avg_latency": 420},
            "mistral-7b": {"success_rate": 0.96, "avg_latency": 140},
        }
        return {"stats": stats[model_name]}


class FakeInferenceEngine:
    def __init__(self, response=None):
        self.router = FakeRouter()
        self.process_query = AsyncMock(return_value=response) if response else AsyncMock()
        self.inference_stats = {
            "gpt-5": {"total_requests": 4, "total_time": 2.0},
            "mistral-7b": {"total_requests": 2, "total_time": 0.4},
        }

    def is_healthy(self):
        return True

    def get_health_status(self):
        return {
            "providers": {
                "openai": {"status": "healthy"},
                "vllm": {"status": "healthy"},
            }
        }


class FakePipeline:
    def is_healthy(self):
        return True

    async def get_query_analytics(self, user_id=None, hours=24):
        return {
            "total_queries": 5 if user_id else 12,
            "total_tokens": 240,
            "total_cost": 1.75,
            "avg_latency": 321.0,
            "success_rate": 99.0,
            "model_breakdown": {
                "gpt-5": {"queries": 3, "cost": 1.25},
                "mistral-7b": {"queries": 2, "cost": 0.5},
            },
            "query_type_breakdown": {"general": 2, "analysis": 3},
        }


class TestUserManager:
    def test_get_user_tier_defaults_and_preferences(self):
        manager = UserManager()

        assert manager.get_user_tier("unknown") == UserTier.FREE
        assert manager.get_user_preferences("unknown")["response_length"] == "medium"

        manager.set_user_tier("u1", UserTier.PREMIUM)
        assert manager.get_user_tier("u1") == UserTier.PREMIUM

    def test_check_rate_limit_enforces_burst_window(self, monkeypatch):
        manager = UserManager()
        now = 1_000_000.0
        monkeypatch.setattr("slack.bot_real.time.time", lambda: now)

        config = {"requests_per_hour": 10, "burst_requests": 2}

        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is True
        assert manager.check_rate_limit("u1", config) is False

    def test_record_query_event_persists_history(self, tmp_path, inference_response_factory):
        store = FileSlackStateStore(str(tmp_path / "slack-state.json"))
        manager = UserManager(store=store)

        manager.record_query_event(
            "u1",
            UserTier.PREMIUM,
            query_type="analysis",
            response=inference_response_factory(model_name="gpt-5", cost_usd=0.25),
            success=True,
            latency_ms=120,
        )

        restored = UserManager(store=FileSlackStateStore(str(tmp_path / "slack-state.json")))
        history = restored.get_query_history("u1", hours=24)

        assert len(history) == 1
        assert history[0]["model_name"] == "gpt-5"
        assert history[0]["query_type"] == "analysis"
        assert history[0]["success"] is True


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

    def test_file_store_persists_user_preferences_and_conversations(self, tmp_path):
        state_file = tmp_path / "slack-state.json"
        store = FileSlackStateStore(str(state_file))
        user_manager = UserManager(store=store)
        conversation_manager = ConversationManager({}, store=store)

        user_manager.set_user_tier("u1", UserTier.PREMIUM)
        user_manager.update_user_preferences("u1", {"response_length": "long"})
        context = conversation_manager.get_or_create_context("u1", "c1")
        context.user_tier = UserTier.PREMIUM
        context.preferences = user_manager.get_user_preferences("u1")
        conversation_manager.save_context(context)
        context.add_message("user", "persist me")

        restored_store = FileSlackStateStore(str(state_file))
        restored_user_manager = UserManager(store=restored_store)
        restored_conversation_manager = ConversationManager({}, store=restored_store)

        assert restored_user_manager.get_user_tier("u1") == UserTier.PREMIUM
        assert (
            restored_user_manager.get_user_preferences("u1")["response_length"]
            == "long"
        )
        summary = restored_conversation_manager.get_conversation_summary("u1", "c1")
        assert "user: persist me" in summary

    def test_redis_store_persists_user_preferences_and_conversations(self, tmp_path):
        redis_config = {"host": "localhost", "port": 6379, "db": 19}
        key_prefix = f"slack-state-{tmp_path.name}"
        store = RedisSlackStateStore(redis_config, key_prefix=key_prefix)
        user_manager = UserManager(store=store)
        conversation_manager = ConversationManager({}, store=store)

        user_manager.set_user_tier("u1", UserTier.PREMIUM)
        user_manager.update_user_preferences("u1", {"response_length": "long"})
        context = conversation_manager.get_or_create_context("u1", "c1", "thread-1")
        context.add_message("user", "persist me in redis")

        restored_store = RedisSlackStateStore(redis_config, key_prefix=key_prefix)
        restored_user_manager = UserManager(store=restored_store)
        restored_conversation_manager = ConversationManager({}, store=restored_store)

        assert restored_user_manager.get_user_tier("u1") == UserTier.PREMIUM
        assert (
            restored_user_manager.get_user_preferences("u1")["response_length"]
            == "long"
        )
        summary = restored_conversation_manager.get_conversation_summary(
            "u1", "c1", "thread-1"
        )
        assert "user: persist me in redis" in summary

    def test_active_thread_context_requires_matching_thread(self):
        manager = ConversationManager({})
        manager.get_or_create_context("u1", "c1", "thread-1")

        assert manager.has_active_thread_context("c1", "thread-1") is True
        assert manager.has_active_thread_context("c1", "thread-2") is False


class TestSlackMessageHandler:
    @pytest.mark.asyncio
    async def test_help_command_matches_threaded_usage(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
        )
        handler = SlackMessageHandler(bot)

        message = await handler._handle_help_command([], "u1", "c1", client=None)

        assert "Mention me in a channel" in message
        assert "Just type your question normally" not in message

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
    async def test_handle_query_updates_history(self, inference_response_factory):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=FakeInferenceEngine(
                response=inference_response_factory(response_text="hello from model")
            ),
        )
        handler = SlackMessageHandler(bot)

        response = await handler._handle_query("hello", "u1", "c1", None, client=None)
        summary = bot.conversation_manager.get_conversation_summary("u1", "c1")
        history = bot.user_manager.get_query_history("u1", hours=24)

        assert response == "hello from model"
        assert "user: hello" in summary
        assert "assistant: hello from model" in summary
        assert len(history) == 1
        assert history[0]["query_type"] == "analysis"

    @pytest.mark.asyncio
    async def test_handle_message_uses_root_ts_for_top_level_thread_context(
        self, inference_response_factory
    ):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=FakeInferenceEngine(
                response=inference_response_factory(response_text="thread-ready")
            ),
        )
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            {"text": "hello", "user": "u1", "channel": "c1", "ts": "123.456"},
            client=None,
        )

        assert response == "thread-ready"
        assert bot.conversation_manager.has_active_thread_context("c1", "123.456")

    @pytest.mark.asyncio
    async def test_app_mention_recognizes_bare_help_command(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
        )
        handler = SlackMessageHandler(bot)

        response = await handler.handle_message(
            {
                "type": "app_mention",
                "text": "help",
                "user": "u1",
                "channel": "c1",
                "ts": "123.456",
            },
            client=None,
        )

        assert "LLM Router Bot Help" in response
        bot.inference_engine.process_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_command_clears_current_thread_context(self):
        bot = SimpleNamespace(
            user_manager=UserManager(),
            conversation_manager=ConversationManager({}),
            inference_engine=SimpleNamespace(process_query=AsyncMock()),
        )
        handler = SlackMessageHandler(bot)
        context = bot.conversation_manager.get_or_create_context("u1", "c1", "thread-1")
        context.add_message("user", "hello")

        response = await handler._handle_command(
            "clear", "u1", "c1", client=None, thread_ts="thread-1"
        )

        assert response == "🧹 Cleared this thread's conversation history."
        assert bot.conversation_manager.has_active_thread_context("c1", "thread-1") is False

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
        assert enterprise_tokens == 16000


class TestSlackBot:
    def test_split_response_breaks_large_message(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        text = "Paragraph one.\n\n" + ("A" * 120) + "\n\nParagraph three."

        parts = bot._split_response(text, max_length=80)

        assert len(parts) >= 2
        assert all(len(part) <= 80 for part in parts)

    def test_file_backed_state_store_is_selected(self, tmp_path):
        state_file = tmp_path / "slack-state.json"
        bot = SlackBot(
            {
                "channels": [],
                "state_backend": "file",
                "state_file": str(state_file),
            },
            inference_engine=SimpleNamespace(),
        )

        bot.user_manager.update_user_preferences("u1", {"response_length": "long"})

        restored_bot = SlackBot(
            {
                "channels": [],
                "state_backend": "file",
                "state_file": str(state_file),
            },
            inference_engine=SimpleNamespace(),
        )

        assert (
            restored_bot.user_manager.get_user_preferences("u1")["response_length"]
            == "long"
        )

    def test_redis_backed_state_store_is_selected(self, tmp_path):
        key_prefix = f"slack-state-{tmp_path.name}"
        bot = SlackBot(
            {
                "channels": [],
                "state_backend": "redis",
                "state_key_prefix": key_prefix,
                "redis": {"host": "localhost", "port": 6379, "db": 20},
            },
            inference_engine=SimpleNamespace(),
        )

        bot.user_manager.update_user_preferences("u1", {"response_length": "long"})

        restored_bot = SlackBot(
            {
                "channels": [],
                "state_backend": "redis",
                "state_key_prefix": key_prefix,
                "redis": {"host": "localhost", "port": 6379, "db": 20},
            },
            inference_engine=SimpleNamespace(),
        )

        assert (
            restored_bot.user_manager.get_user_preferences("u1")["response_length"]
            == "long"
        )

    @pytest.mark.asyncio
    async def test_message_event_ignores_top_level_channel_message(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.bot_user_id = "B123"
        bot._process_message = AsyncMock()
        bot._send_rate_limit_message = AsyncMock()

        await bot._handle_message_event(
            {"user": "U123", "channel": "C123", "text": "hello", "ts": "111.222"}
        )

        bot._process_message.assert_not_awaited()
        bot._send_rate_limit_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_message_event_accepts_active_thread_reply(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.bot_user_id = "B123"
        bot._process_message = AsyncMock()
        bot._send_rate_limit_message = AsyncMock()

        context = bot.conversation_manager.get_or_create_context(
            "U123", "C123", "111.222"
        )
        bot.conversation_manager.save_context(context)

        await bot._handle_message_event(
            {
                "user": "U123",
                "channel": "C123",
                "text": "follow up",
                "thread_ts": "111.222",
                "parent_user_id": "U456",
            }
        )

        bot._process_message.assert_awaited_once()
        bot._send_rate_limit_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowed_channel_accepts_configured_channel_name(self):
        bot = SlackBot({"channels": ["general"]}, inference_engine=SimpleNamespace())
        bot.web_client = SimpleNamespace(
            conversations_info=AsyncMock(return_value={"channel": {"name": "general"}})
        )

        allowed = await bot._is_allowed_channel({"channel": "C123"})

        assert allowed is True
        bot.web_client.conversations_info.assert_awaited_once_with(channel="C123")

    @pytest.mark.asyncio
    async def test_empty_app_mention_maps_to_help(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.bot_user_id = "B123"
        bot._process_message = AsyncMock()

        event = {"channel": "C123", "user": "U123", "text": "<@B123>", "ts": "111.222"}

        await bot._handle_mention_event(event)

        assert event["text"] == "help"
        bot._process_message.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_slash_command_uses_ephemeral_response(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.message_handler = SimpleNamespace(_handle_command=AsyncMock(return_value="ok"))
        bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())

        await bot._handle_slash_command(
            {"text": "help", "user_id": "U123", "channel_id": "C123"}
        )

        bot.message_handler._handle_command.assert_awaited_once()
        bot.web_client.chat_postEphemeral.assert_awaited_once_with(
            channel="C123", user="U123", text="ok"
        )

    @pytest.mark.asyncio
    async def test_process_message_keeps_thread_reply_and_does_not_mutate_channel(self):
        bot = SlackBot(
            {"channels": [], "response_settings": {"typing_indicator": True}},
            inference_engine=SimpleNamespace(),
        )
        bot.web_client = SimpleNamespace(
            chat_postMessage=AsyncMock(),
            conversations_setTopic=AsyncMock(),
        )
        bot.message_handler = SimpleNamespace(handle_message=AsyncMock(return_value="hello"))

        await bot._process_message(
            {"type": "app_mention", "channel": "C123", "user": "U123", "text": "hello", "ts": "111.222"}
        )

        bot.web_client.conversations_setTopic.assert_not_awaited()
        bot.web_client.chat_postMessage.assert_awaited_once_with(
            channel="C123", text="hello", thread_ts="111.222"
        )

    @pytest.mark.asyncio
    async def test_process_message_sanitizes_internal_errors(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
        bot.message_handler = SimpleNamespace(
            handle_message=AsyncMock(side_effect=RuntimeError("secret provider failure"))
        )

        await bot._process_message(
            {"type": "app_mention", "channel": "C123", "user": "U123", "text": "hello", "ts": "111.222"}
        )

        bot.web_client.chat_postMessage.assert_awaited_once()
        _, kwargs = bot.web_client.chat_postMessage.await_args
        assert kwargs["thread_ts"] == "111.222"
        assert "secret provider failure" not in kwargs["text"]

    @pytest.mark.asyncio
    async def test_get_system_status_uses_live_services(self):
        bot = SlackBot(
            {"channels": [], "rate_limiting": {"requests_per_hour": 100}},
            inference_engine=FakeInferenceEngine(),
            services={"pipeline": FakePipeline()},
        )
        bot.user_manager.set_user_tier("u1", UserTier.PREMIUM)

        status = await bot.get_system_status(user_id="u1")

        assert status["healthy"] is True
        assert status["avg_response_time"] == 321.0
        assert "gpt-5" in status["available_models"]
        assert status["services"]["pipeline"] is True

    @pytest.mark.asyncio
    async def test_get_available_models_reads_router_and_provider_health(self):
        inference_engine = FakeInferenceEngine()
        bot = SlackBot({"channels": []}, inference_engine=inference_engine)
        bot.user_manager.set_user_tier("u1", UserTier.PREMIUM)

        models = await bot.get_available_models("u1")

        assert [model["name"] for model in models] == ["mistral-7b", "gpt-5"]
        assert models[0]["avg_latency_ms"] == 140
        assert models[1]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_get_user_analytics_aggregates_recorded_history(self, inference_response_factory):
        bot = SlackBot({"channels": []}, inference_engine=FakeInferenceEngine())
        bot.user_manager.record_query_event(
            "u1",
            UserTier.PREMIUM,
            query_type="analysis",
            response=inference_response_factory(
                model_name="gpt-5",
                total_tokens=30,
                cost_usd=0.5,
                cached=True,
                latency_ms=200,
            ),
            success=True,
            latency_ms=200,
        )
        bot.user_manager.record_query_event(
            "u1",
            UserTier.PREMIUM,
            query_type="general",
            response=inference_response_factory(
                model_name="mistral-7b",
                provider="vllm",
                total_tokens=10,
                cost_usd=0.0,
                cached=False,
                latency_ms=100,
            ),
            success=False,
            latency_ms=100,
        )

        analytics = await bot.get_user_analytics("u1")

        assert analytics["total_queries"] == 2
        assert analytics["total_tokens"] == 40
        assert analytics["total_cost"] == 0.5
        assert analytics["success_rate"] == 50.0
        assert analytics["cache_hit_rate"] == 50.0
        assert analytics["model_breakdown"]["gpt-5"]["queries"] == 1
        assert analytics["query_type_breakdown"]["analysis"] == 1
