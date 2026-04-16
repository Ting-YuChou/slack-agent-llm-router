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
from src.utils.schema import Attachment, AttachmentType, UserTier


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
    async def test_slash_command_prefers_ephemeral_response(self):
        bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
        bot.web_client = SimpleNamespace(
            chat_postEphemeral=AsyncMock(),
            chat_postMessage=AsyncMock(),
        )
        bot.message_handler = SimpleNamespace(_handle_command=AsyncMock(return_value="ok"))

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
            {"redis": {"host": "localhost", "port": 6379, "db": 0, "key_prefix": "test"}}
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
            {"redis": {"host": "localhost", "port": 6379, "db": 0, "key_prefix": prefix}}
        )
        store_two = RedisSlackStateStore(
            {"redis": {"host": "localhost", "port": 6379, "db": 0, "key_prefix": prefix}}
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
            {"redis": {"host": "localhost", "port": 6379, "db": 0, "key_prefix": prefix}}
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
