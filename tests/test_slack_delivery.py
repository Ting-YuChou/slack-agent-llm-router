"""Slack answer delivery: progress placeholder, splitting, formatting."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from slack.bot_real import SLACK_THINKING_TEXT, SlackBot


def delivery_bot(process_query=None, *, chat_update=True, **config):
    """A chat bot whose client can update messages unless told otherwise."""
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}, **config},
        inference_engine=SimpleNamespace(process_query=process_query or AsyncMock()),
    )
    bot.bot_user_id = "UBOT"
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
    )
    if chat_update:
        client.chat_update = AsyncMock(return_value={"ok": True})
    bot.web_client = client
    return bot


def top_level(ts="100.0", text="what is 2+2?", **fields):
    return {
        "type": "message",
        "channel": "C1",
        "user": "U1",
        "text": text,
        "ts": ts,
        **fields,
    }


def answering(inference_response_factory, text="the answer", **fields):
    return AsyncMock(
        return_value=inference_response_factory(response_text=text, **fields)
    )


# S5: progress placeholder


@pytest.mark.asyncio
async def test_placeholder_is_posted_in_the_answer_thread_before_the_model_runs(
    inference_response_factory,
):
    seen_before_model = []

    async def process_query(_request):
        seen_before_model.extend(bot.web_client.chat_postMessage.await_args_list)
        return inference_response_factory(response_text="four")

    bot = delivery_bot(AsyncMock(side_effect=process_query))

    await bot._process_message(top_level())

    assert len(seen_before_model) == 1
    placeholder = seen_before_model[0].kwargs
    assert placeholder["text"] == SLACK_THINKING_TEXT
    assert placeholder["channel"] == "C1"
    assert placeholder["thread_ts"] == "100.0"


@pytest.mark.asyncio
async def test_answer_replaces_the_placeholder_and_later_parts_follow(
    inference_response_factory,
):
    answer = "\n".join(f"line {index} " + "x" * 40 for index in range(20))
    bot = delivery_bot(
        answering(inference_response_factory, answer),
        response_settings={"max_response_length": 300},
    )

    await bot._process_message(top_level())

    update = bot.web_client.chat_update.await_args.kwargs
    assert update["channel"] == "C1"
    assert update["ts"] == "500.0"
    assert update["text"].startswith("line 0 ")
    posts = bot.web_client.chat_postMessage.await_args_list
    assert posts[0].kwargs["text"] == SLACK_THINKING_TEXT
    rest = [call.kwargs for call in posts[1:]]
    assert rest, "the later parts are posted as new messages"
    assert all(post["thread_ts"] == "100.0" for post in rest)
    assert "line 19 " in rest[-1]["text"]
    assert bot.web_client.chat_update.await_count == 1


@pytest.mark.asyncio
async def test_short_answer_only_updates_the_placeholder(inference_response_factory):
    bot = delivery_bot(answering(inference_response_factory, "four"))

    await bot._process_message(top_level())

    assert bot.web_client.chat_update.await_args.kwargs["text"] == "four"
    assert bot.web_client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_failure_turns_the_placeholder_into_the_error(monkeypatch):
    bot = delivery_bot()
    monkeypatch.setattr(
        bot.message_handler,
        "handle_message",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    await bot._process_message(top_level())

    update = bot.web_client.chat_update.await_args.kwargs
    assert update["ts"] == "500.0"
    assert update["text"] == bot._build_user_safe_error_message()
    assert "boom" not in update["text"]
    assert bot.web_client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_failed_update_falls_back_to_posting_the_answer(
    inference_response_factory,
):
    bot = delivery_bot(answering(inference_response_factory, "four"))
    bot.web_client.chat_update.side_effect = RuntimeError("message_not_found")

    await bot._process_message(top_level())

    posts = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list]
    assert [post["text"] for post in posts] == [SLACK_THINKING_TEXT, "four"]
    assert posts[1]["thread_ts"] == "100.0"


@pytest.mark.asyncio
async def test_typing_indicator_off_posts_no_placeholder(inference_response_factory):
    bot = delivery_bot(
        answering(inference_response_factory, "four"),
        response_settings={"typing_indicator": False},
    )

    await bot._process_message(top_level())

    bot.web_client.chat_update.assert_not_awaited()
    posts = bot.web_client.chat_postMessage.await_args_list
    assert [call.kwargs["text"] for call in posts] == ["four"]


@pytest.mark.asyncio
async def test_client_without_chat_update_posts_no_placeholder(
    inference_response_factory,
):
    bot = delivery_bot(answering(inference_response_factory, "four"), chat_update=False)

    await bot._process_message(top_level())

    posts = bot.web_client.chat_postMessage.await_args_list
    assert [call.kwargs["text"] for call in posts] == ["four"]


@pytest.mark.asyncio
async def test_commands_and_agent_requests_post_no_placeholder(monkeypatch):
    bot = delivery_bot()
    monkeypatch.setattr(
        bot.message_handler, "handle_message", AsyncMock(return_value="help text")
    )

    await bot._process_message(top_level(text="!llm help"))
    await bot._process_message(top_level(ts="101.0", text="!llm agent fix the bug"))

    texts = [
        call.kwargs["text"] for call in bot.web_client.chat_postMessage.await_args_list
    ]
    assert SLACK_THINKING_TEXT not in texts
    bot.web_client.chat_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_slash_command_ack_says_it_is_working():
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    bot._enqueue_work = AsyncMock()
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    request = SimpleNamespace(
        envelope_id="E1",
        type="slash_commands",
        payload={"text": "explain monads", "user_id": "U1", "channel_id": "C1"},
    )

    await bot._handle_socket_mode_request(socket_client, request)

    ack = socket_client.send_socket_mode_response.await_args.args[0]
    assert ack.envelope_id == "E1"
    assert ack.payload == {"response_type": "ephemeral", "text": "Working on it…"}
    bot._enqueue_work.assert_awaited_once()


@pytest.mark.asyncio
async def test_slash_command_ack_stays_empty_for_commands_and_when_disabled():
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    for config, text in (
        ({"channels": []}, "help"),
        ({"channels": [], "response_settings": {"typing_indicator": False}}, "hi"),
    ):
        bot = SlackBot(config, inference_engine=SimpleNamespace())
        bot._enqueue_work = AsyncMock()
        request = SimpleNamespace(
            envelope_id="E1",
            type="slash_commands",
            payload={"text": text, "user_id": "U1", "channel_id": "C1"},
        )

        await bot._handle_socket_mode_request(socket_client, request)

        ack = socket_client.send_socket_mode_response.await_args.args[0]
        assert getattr(ack, "payload", None) is None


@pytest.mark.asyncio
async def test_a_failure_after_the_first_part_keeps_that_part(
    inference_response_factory,
):
    answer = "\n".join(f"line {index} " + "x" * 40 for index in range(20))
    bot = delivery_bot(
        answering(inference_response_factory, answer),
        response_settings={"max_response_length": 300},
    )
    bot.web_client.chat_postMessage.side_effect = [
        {"ok": True, "ts": "500.0"},
        RuntimeError("channel_not_found"),
        {"ok": True, "ts": "502.0"},
    ]

    await bot._process_message(top_level())

    assert bot.web_client.chat_update.await_count == 1
    assert bot.web_client.chat_update.await_args.kwargs["text"].startswith("line 0 ")
    last = bot.web_client.chat_postMessage.await_args.kwargs
    assert last["text"] == bot._build_user_safe_error_message()


# S9: fence-aware splitting and Slack 429 retries


def fenced_answer(lines=60):
    code = "\n".join(
        f"    result_{index} = compute({index})  # step" for index in range(lines)
    )
    return f"Here is the code:\n\n```python\n{code}\n```\n\nThat is all."


@pytest.mark.parametrize("limit", [200, 500, 2000])
@pytest.mark.parametrize(
    "text",
    [
        fenced_answer(),
        fenced_answer(30) + "\n\n" + fenced_answer(30),
        "Intro.\n\n```\n" + "x" * 3000 + "\n```\nDone.",
    ],
    ids=["one-fence", "two-fences", "long-line-in-fence"],
)
def test_every_part_keeps_code_fences_balanced_and_fits(text, limit):
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

    parts = bot._split_response(text, limit)

    assert len(parts) > 1
    assert all(len(part) <= limit for part in parts)
    assert all(part.count("```") % 2 == 0 for part in parts)
    for line in text.splitlines():
        if line and line != "```" and len(line) < limit // 2:
            assert any(line in part for part in parts), line


def test_split_does_not_leave_tiny_fragments():
    paragraph = " ".join(f"word{index}" for index in range(500))
    text = f"Intro line.\n\n{paragraph}\n\nBye."
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

    parts = bot._split_response(text, 2000)

    assert len(parts) == 2
    assert parts[0].startswith("Intro line.")
    assert parts[-1].endswith("Bye.")
    assert all(len(part) >= 500 for part in parts)
    assert all(not part.startswith(" ") for part in parts)


@pytest.mark.asyncio
async def test_answer_parts_are_posted_without_a_fixed_sleep(
    inference_response_factory, monkeypatch
):
    sleep = AsyncMock()
    monkeypatch.setattr("slack.bot_real.asyncio.sleep", sleep)
    answer = "\n".join(f"line {index} " + "x" * 40 for index in range(20))
    bot = delivery_bot(
        answering(inference_response_factory, answer),
        chat_update=False,
        response_settings={"max_response_length": 300},
    )

    await bot._process_message(top_level())

    assert bot.web_client.chat_postMessage.await_count > 1
    sleep.assert_not_awaited()


def test_configs_allow_long_slack_answers():
    root = Path(__file__).resolve().parents[1]
    for name in ("config.yaml", "config.compose.yaml", "config.demo.yaml"):
        raw = yaml.safe_load((root / "config" / name).read_text(encoding="utf-8"))
        settings = raw["slack"]["response_settings"]
        assert settings["max_response_length"] == 3900, name


@pytest.mark.asyncio
async def test_initialize_adds_a_rate_limit_retry_handler(monkeypatch):
    from slack import bot_real

    created = {}

    class DummyWebClient:
        def __init__(self, token):
            self.token = token
            self.retry_handlers = ["connection-handler"]
            self.auth_test = AsyncMock(return_value={"user_id": "B123"})
            created["web_client"] = self

    class DummySocketClient:
        def __init__(self, app_token, web_client):
            self.socket_mode_request_listeners = []

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
            "channels": [],
        },
        inference_engine=SimpleNamespace(),
    )

    await bot.initialize()

    handlers = created["web_client"].retry_handlers
    assert handlers[0] == "connection-handler"
    assert len(handlers) == 2
    assert isinstance(handlers[1], bot_real.AsyncRateLimitErrorRetryHandler)
    assert handlers[1].max_retry_count == 2


def _slack_sdk_modules():
    return [
        name
        for name in sys.modules
        if name == "slack_sdk" or name.startswith("slack_sdk.")
    ]


@pytest.mark.asyncio
async def test_real_client_retries_a_429_and_posts_once(monkeypatch):
    """With the real slack_sdk, a 429 then a 200 sends the message once."""
    from slack import bot_real

    web = pytest.importorskip("aiohttp.web")
    stubs = {name: sys.modules.pop(name) for name in _slack_sdk_modules()}
    try:
        try:
            async_client = importlib.import_module("slack_sdk.web.async_client")
            retry = importlib.import_module(
                "slack_sdk.http_retry.builtin_async_handlers"
            )
        except ImportError:
            pytest.skip("slack_sdk is not installed")
        monkeypatch.setattr(retry, "random", SimpleNamespace(random=lambda: 0.0))
        monkeypatch.setattr(
            bot_real,
            "AsyncRateLimitErrorRetryHandler",
            retry.AsyncRateLimitErrorRetryHandler,
        )
        received = []

        async def post_message(request):
            received.append(await request.json())
            if len(received) == 1:
                return web.json_response(
                    {"ok": False, "error": "ratelimited"},
                    status=429,
                    headers={"Retry-After": "0"},
                )
            return web.json_response({"ok": True, "channel": "C1", "ts": "1.0"})

        app = web.Application()
        app.router.add_post("/chat.postMessage", post_message)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            client = async_client.AsyncWebClient(
                token="xoxb-test", base_url=f"http://127.0.0.1:{port}/"
            )
            bot_real._add_rate_limit_retry(client)

            response = await client.chat_postMessage(channel="C1", text="hello")
        finally:
            await runner.cleanup()
    finally:
        for name in _slack_sdk_modules():
            del sys.modules[name]
        sys.modules.update(stubs)

    assert response["ok"] is True
    assert len(received) == 2
    assert received[0]["text"] == received[1]["text"] == "hello"
