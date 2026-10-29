"""Slack answer delivery: progress placeholder, splitting, formatting."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from slack.bot_real import (
    SLACK_EMPTY_ANSWER_MESSAGE,
    SLACK_THINKING_TEXT,
    SlackBot,
)


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


class FakeSlackApiError(Exception):
    """Shaped like slack_sdk's SlackApiError: the API error is on `response`."""

    def __init__(self, error):
        super().__init__(f"The request to the Slack API failed. ({error})")
        self.response = {"ok": False, "error": error}


@pytest.mark.asyncio
async def test_rejected_blocks_update_is_retried_as_text(inference_response_factory):
    bot = delivery_bot(answering(inference_response_factory, "**four**"))
    bot.web_client.chat_update.side_effect = [
        FakeSlackApiError("invalid_blocks"),
        {"ok": True},
    ]

    await bot._process_message(top_level())

    first, second = [call.kwargs for call in bot.web_client.chat_update.await_args_list]
    assert "blocks" in first
    assert second == {"channel": "C1", "ts": "500.0", "text": "**four**"}
    assert bot.web_client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_update_failing_for_other_reasons_is_not_retried_as_text(
    inference_response_factory,
):
    # A timeout or 5xx may have been applied: resending as text would
    # overwrite the formatted answer. Post the answer instead.
    bot = delivery_bot(answering(inference_response_factory, "**four**"))
    bot.web_client.chat_update.side_effect = [TimeoutError("read timed out")]

    await bot._process_message(top_level())

    assert bot.web_client.chat_update.await_count == 1
    posts = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list]
    assert [post["text"] for post in posts] == [SLACK_THINKING_TEXT, "**four**"]


@pytest.mark.asyncio
async def test_rejected_blocks_post_is_retried_as_text(inference_response_factory):
    bot = delivery_bot(answering(inference_response_factory, "four"), chat_update=False)
    bot.web_client.chat_postMessage.side_effect = [
        FakeSlackApiError("invalid_blocks"),
        {"ok": True, "ts": "501.0"},
    ]

    await bot._process_message(top_level())

    first, second = [
        call.kwargs for call in bot.web_client.chat_postMessage.await_args_list
    ]
    assert "blocks" in first
    assert second == {"channel": "C1", "thread_ts": "100.0", "text": "four"}


@pytest.mark.asyncio
async def test_other_post_failures_are_not_resent(inference_response_factory):
    bot = delivery_bot(answering(inference_response_factory, "four"), chat_update=False)
    bot.web_client.chat_postMessage.side_effect = [
        FakeSlackApiError("channel_not_found"),
        {"ok": True, "ts": "501.0"},
    ]

    await bot._process_message(top_level())

    texts = [
        call.kwargs["text"] for call in bot.web_client.chat_postMessage.await_args_list
    ]
    assert texts == ["four", bot._build_user_safe_error_message()]


@pytest.mark.asyncio
async def test_empty_answer_is_posted_when_the_placeholder_cannot_be_updated(
    monkeypatch,
):
    bot = delivery_bot()
    monkeypatch.setattr(
        bot.message_handler, "handle_message", AsyncMock(return_value=None)
    )
    bot.web_client.chat_update.side_effect = RuntimeError("message_not_found")

    await bot._process_message(top_level())

    posts = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list]
    assert [post["text"] for post in posts] == [
        SLACK_THINKING_TEXT,
        SLACK_EMPTY_ANSWER_MESSAGE,
    ]
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
@pytest.mark.parametrize(
    "text, shows_progress",
    [
        ("!llm web latest python release", True),
        ("!llm fast what is 2+2", True),
        ("!llm web", False),
        ("!llm fast", False),
    ],
)
async def test_web_and_fast_commands_show_progress(monkeypatch, text, shows_progress):
    bot = delivery_bot()
    seen_before_handler = []

    async def handle(_event, _client):
        seen_before_handler.extend(bot.web_client.chat_postMessage.await_args_list)
        return "answer"

    monkeypatch.setattr(
        bot.message_handler, "handle_message", AsyncMock(side_effect=handle)
    )

    await bot._process_message(top_level(text=text))

    texts = [call.kwargs["text"] for call in seen_before_handler]
    assert texts == ([SLACK_THINKING_TEXT] if shows_progress else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["web latest python release", "fast what is 2+2"])
async def test_slash_web_and_fast_acks_say_they_are_working(text):
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    bot._enqueue_work = AsyncMock()
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    request = SimpleNamespace(
        envelope_id="E1",
        type="slash_commands",
        payload={"text": text, "user_id": "U1", "channel_id": "C1"},
    )

    await bot._handle_socket_mode_request(socket_client, request)

    ack = socket_client.send_socket_mode_response.await_args.args[0]
    assert ack.payload == {"response_type": "ephemeral", "text": "Working on it…"}


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


def _fence_inner(part):
    """A part of an all-code answer, without its opening and closing fence."""
    lines = part.split("\n")
    assert lines[0].startswith("```") and lines[-1] == "```", part
    return "\n".join(lines[1:-1])


@pytest.mark.parametrize("limit", [120, 300, 1000])
def test_code_lines_and_blank_lines_survive_a_split(limit):
    code = "\n".join(
        "" if index % 7 == 3 else f"    value_{index} = f({index})  # note"
        for index in range(120)
    )
    text = f"```python\n{code}\n```"
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

    parts = bot._split_response(text, limit)

    assert len(parts) > 1
    assert "\n".join(_fence_inner(part) for part in parts) == code


@pytest.mark.parametrize("limit", [80, 200, 700])
def test_no_character_inside_a_code_fence_is_dropped(limit):
    # Long lines with spaces and no newline near the limit: the splitter
    # must not break on (and drop) a space inside the code.
    code = "\n".join(
        " ".join(f"tok{index}_{word}" for word in range(40)) for index in range(6)
    )
    text = f"```\n{code}\n```"
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

    parts = bot._split_response(text, limit)

    assert len(parts) > 1
    assert all(len(part) <= limit for part in parts)
    joined = "".join(_fence_inner(part) for part in parts)
    assert joined.replace("\n", "") == code.replace("\n", "")


def test_a_break_before_a_closing_fence_leaves_no_empty_code_block():
    text = "```\n" + "x" * 48 + "\n```\n" + "word" * 30
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

    parts = bot._split_response(text, 60)

    assert all(len(part) <= 60 for part in parts)
    assert all(part.count("```") % 2 == 0 for part in parts)
    assert parts[0] == "```\n" + "x" * 48 + "\n```"
    assert all(not part.startswith("```\n```") for part in parts)
    assert "".join(parts[1:]) == "word" * 30


def test_a_hard_cut_never_splits_a_run_of_backticks():
    text = "y" * 47 + "```code```" + "z" * 80
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())

    parts = bot._split_response(text, 60)

    assert all(len(part) <= 60 for part in parts)
    assert all(part.count("```") % 2 == 0 for part in parts)
    assert not any(part.endswith("``") and not part.endswith("```") for part in parts)
    assert parts[-1].endswith("z" * 10)


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


# S10: Markdown blocks and the answer footer


def _blocks_of_type(message, block_type):
    return [block for block in message.get("blocks", []) if block["type"] == block_type]


@pytest.mark.asyncio
async def test_answer_is_a_markdown_block_with_a_text_fallback(
    inference_response_factory,
):
    answer = "**Bold** and a [link](https://example.com)\n\n| a | b |\n|---|---|"
    bot = delivery_bot(answering(inference_response_factory, answer), chat_update=False)

    await bot._process_message(top_level())

    message = bot.web_client.chat_postMessage.await_args.kwargs
    assert message["text"] == answer
    assert message["blocks"][0] == {"type": "markdown", "text": answer}


@pytest.mark.asyncio
async def test_last_part_carries_a_footer_with_model_latency_and_cache(
    inference_response_factory,
):
    answer = "\n".join(f"line {index} " + "x" * 40 for index in range(20))
    bot = delivery_bot(
        answering(
            inference_response_factory,
            answer,
            model_name="claude-sonnet-4-6",
            latency_ms=1234,
            cached=True,
            cost_usd=0.0004,
        ),
        response_settings={"max_response_length": 300},
    )

    await bot._process_message(top_level())

    first = bot.web_client.chat_update.await_args.kwargs
    later = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list][
        1:
    ]
    parts = [first, *later]
    assert all(part["blocks"][0]["type"] == "markdown" for part in parts)
    assert all(part["blocks"][0]["text"] == part["text"] for part in parts)
    assert all(not _blocks_of_type(part, "context") for part in parts[:-1])
    (footer,) = _blocks_of_type(parts[-1], "context")
    footer_text = footer["elements"][0]["text"]
    assert footer_text == "claude-sonnet-4-6 · 1.2 s · cached"
    assert "$" not in footer_text
    assert all("claude-sonnet-4-6" not in part["text"] for part in parts)


@pytest.mark.asyncio
async def test_footer_says_nothing_about_cache_for_a_fresh_answer(
    inference_response_factory,
):
    bot = delivery_bot(answering(inference_response_factory, "four"))

    await bot._process_message(top_level())

    update = bot.web_client.chat_update.await_args.kwargs
    assert update["text"] == "four"
    (footer,) = _blocks_of_type(update, "context")
    assert footer["elements"][0]["text"] == "gpt-5 · 0.1 s"


@pytest.mark.asyncio
async def test_footer_can_be_turned_off(inference_response_factory):
    bot = delivery_bot(
        answering(inference_response_factory, "four"),
        response_settings={"answer_footer": False},
    )

    await bot._process_message(top_level())

    update = bot.web_client.chat_update.await_args.kwargs
    assert update["blocks"] == [{"type": "markdown", "text": "four"}]


@pytest.mark.asyncio
async def test_command_replies_and_errors_stay_plain_text(
    inference_response_factory, monkeypatch
):
    bot = delivery_bot(
        AsyncMock(
            return_value=inference_response_factory(
                response_text="", error="boom", error_type="provider_error"
            )
        ),
        chat_update=False,
    )

    await bot._process_message(top_level())
    error_reply = bot.web_client.chat_postMessage.await_args.kwargs
    monkeypatch.setattr(
        bot.message_handler, "handle_message", AsyncMock(return_value="help text")
    )
    await bot._process_message(top_level(ts="101.0", text="!llm help"))
    help_reply = bot.web_client.chat_postMessage.await_args.kwargs

    assert "blocks" not in error_reply
    assert help_reply == {"channel": "C1", "text": "help text", "thread_ts": "101.0"}


@pytest.mark.asyncio
async def test_notes_and_sources_stay_on_their_own_lines_in_markdown(
    inference_response_factory,
):
    from src.utils.schema import ResponseSource

    sources = [
        ResponseSource(title="News", url="https://example.com/n", rank=1),
        ResponseSource(title="Docs", url="https://example.com/d", rank=2),
    ]
    model_text = "First line of the answer\nsecond line of the answer"
    bot = delivery_bot(
        answering(
            inference_response_factory,
            model_text,
            warnings=["response_truncated", "web_search_unavailable"],
            sources=sources,
        ),
        chat_update=False,
    )

    await bot._process_message(top_level())

    message = bot.web_client.chat_postMessage.await_args.kwargs
    text, markdown = message["text"], message["blocks"][0]["text"]
    # The plain text is unchanged ...
    assert "checked._\n_Note: the answer was cut off._" in text
    assert "Sources:\n[1] News - https://example.com/n\n[2] Docs" in text
    # ... while Markdown, which joins single newlines, gets paragraph breaks.
    assert "checked._\n\n_Note: the answer was cut off._" in markdown
    assert "Sources:\n\n[1] News - https://example.com/n\n\n[2] Docs" in markdown
    # The model's own text is left alone.
    assert markdown.startswith(model_text + "\n\n_Note:")


def test_schema_accepts_the_footer_setting():
    from src.utils.schema import SlackResponseSettingsConfig

    assert SlackResponseSettingsConfig().answer_footer is True
    assert SlackResponseSettingsConfig(answer_footer=False).answer_footer is False


# S6b: Agent requests download their own files


def agent_thread_bot():
    bot = delivery_bot()
    bot.agent_threads[bot._thread_key("C1", "100.0")] = {
        "channel_id": "C1",
        "thread_ts": "100.0",
        "owner_user_id": "U1",
    }
    bot._extract_query_attachments = AsyncMock(return_value=[])
    bot.agent_bootstrap_builder = SimpleNamespace(
        build_message_files=AsyncMock(
            return_value=SimpleNamespace(
                prompt="task with files",
                resource_count=1,
                truncated=False,
                partial=False,
                warnings=[],
            )
        )
    )
    bot.message_handler._start_agent_prompt = AsyncMock(return_value="started")
    return bot


def agent_followup(text):
    return {
        "type": "message",
        "subtype": "file_share",
        "channel": "C1",
        "user": "U1",
        "text": text,
        "thread_ts": "100.0",
        "ts": "101.0",
        "files": [{"id": "F1", "name": "trace.log"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["look at this log", ""], ids=["text", "files-only"])
async def test_agent_thread_files_are_not_downloaded_by_the_chat_path(text):
    bot = agent_thread_bot()

    await bot._process_message(agent_followup(text))

    bot._extract_query_attachments.assert_not_awaited()
    build = bot.agent_bootstrap_builder.build_message_files
    build.assert_awaited_once()
    assert build.await_args.kwargs["files"] == [{"id": "F1", "name": "trace.log"}]
    start = bot.message_handler._start_agent_prompt
    start.assert_awaited_once()
    assert start.await_args.args[0] == "task with files"
    if not text:
        assert "trace.log" in build.await_args.kwargs["task"]


@pytest.mark.asyncio
async def test_agent_command_with_files_is_not_downloaded_by_the_chat_path(
    monkeypatch,
):
    bot = delivery_bot()
    bot._extract_query_attachments = AsyncMock(return_value=[])
    handle_command = AsyncMock(return_value="agent started")
    monkeypatch.setattr(bot.message_handler, "_handle_command", handle_command)

    await bot._process_message(
        top_level(
            text="!llm agent fix the failing test",
            subtype="file_share",
            files=[{"id": "F1", "name": "trace.log"}],
        )
    )

    bot._extract_query_attachments.assert_not_awaited()
    handle_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_files_are_still_downloaded(inference_response_factory):
    bot = delivery_bot(answering(inference_response_factory, "it is a log"))
    bot._extract_query_attachments = AsyncMock(return_value=[])

    await bot._process_message(
        top_level(
            text="what is this?",
            subtype="file_share",
            files=[{"id": "F1", "name": "trace.log"}],
        )
    )

    bot._extract_query_attachments.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_or_agent_is_decided_once_per_message(inference_response_factory):
    # The thread becomes an Agent thread (the runtime accepted `agent <task>`)
    # while this follow-up is already on the chat path.
    bot = agent_thread_bot()
    del bot.agent_threads[bot._thread_key("C1", "100.0")]
    bot.inference_engine.process_query = answering(
        inference_response_factory, "it is a stack trace"
    )

    async def agent_run_starts(_event):
        bot.agent_threads[bot._thread_key("C1", "100.0")] = {
            "channel_id": "C1",
            "thread_ts": "100.0",
            "owner_user_id": "U1",
        }
        return []

    bot._extract_query_attachments = AsyncMock(side_effect=agent_run_starts)

    await bot._process_message(agent_followup("what is this?"))

    bot.message_handler._start_agent_prompt.assert_not_awaited()
    bot.agent_bootstrap_builder.build_message_files.assert_not_awaited()
    update = bot.web_client.chat_update.await_args.kwargs
    assert update["text"] == "it is a stack trace"


@pytest.mark.asyncio
@pytest.mark.parametrize("update_fails", [False, True])
async def test_empty_chat_reply_never_claims_an_agent_handoff(
    monkeypatch, update_fails
):
    # The placeholder is only posted on the chat lane, so a thread that
    # turned into an Agent thread meanwhile is no sign of a handoff.
    bot = delivery_bot()
    if update_fails:
        bot.web_client.chat_update.side_effect = RuntimeError("message_not_found")

    async def agent_thread_appears(_event, _client):
        bot.agent_threads[bot._thread_key("C1", "100.0")] = {
            "channel_id": "C1",
            "thread_ts": "100.0",
            "owner_user_id": "U1",
        }
        return None

    monkeypatch.setattr(
        bot.message_handler,
        "handle_message",
        AsyncMock(side_effect=agent_thread_appears),
    )

    await bot._process_message(top_level())

    update = bot.web_client.chat_update.await_args.kwargs
    assert update["text"] == SLACK_EMPTY_ANSWER_MESSAGE
    posts = [call.kwargs for call in bot.web_client.chat_postMessage.await_args_list]
    expected = [SLACK_THINKING_TEXT]
    if update_fails:
        expected.append(SLACK_EMPTY_ANSWER_MESSAGE)
    assert [post["text"] for post in posts] == expected
    assert posts[-1]["thread_ts"] == "100.0"
