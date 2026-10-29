"""Slack answer delivery: progress placeholder, splitting, formatting."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
