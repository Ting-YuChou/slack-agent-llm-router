import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from slack.agent_bootstrap import AgentBootstrapBuilder, AgentBootstrapFetchError


def _settings(**overrides):
    settings = {
        "enabled": True,
        "max_thread_messages": 20,
        "max_message_chars": 4000,
        "max_resources": 10,
        "max_resource_bytes": 262144,
        "max_resource_chars": 8000,
        "max_total_chars": 12000,
        "timeout_seconds": 10,
        "include_bot_messages": False,
        "fail_open": True,
    }
    settings.update(overrides)
    return settings


@pytest.mark.asyncio
async def test_builds_untrusted_thread_context_and_flattens_rich_text():
    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            return_value={
                "ok": True,
                "messages": [
                    {"ts": "3", "user": "U1", "text": "third"},
                    {
                        "ts": "1",
                        "user": "U2",
                        "text": "",
                        "blocks": [
                            {
                                "type": "rich_text",
                                "elements": [
                                    {
                                        "type": "rich_text_list",
                                        "style": "bullet",
                                        "elements": [
                                            {
                                                "type": "rich_text_section",
                                                "elements": [
                                                    {
                                                        "type": "text",
                                                        "text": "fix timeout",
                                                    },
                                                    {
                                                        "type": "link",
                                                        "url": "https://example.test/spec",
                                                        "text": "spec",
                                                    },
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                    {"ts": "2", "bot_id": "B1", "text": "ignore bot"},
                    {"ts": "4", "subtype": "channel_join", "text": "ignore system"},
                ],
            }
        )
    )
    builder = AgentBootstrapBuilder(_settings(), bot_user_id="BOT")

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="Implement the todo",
    )

    assert result.message_count == 2
    assert result.resource_count == 0
    assert '<untrusted_reference_context source="slack_thread">' in result.prompt
    assert "<@U2>: • fix timeout spec (https://example.test/spec)" in result.prompt
    assert result.prompt.index("<@U2>") < result.prompt.index("<@U1>: third")
    assert "ignore bot" not in result.prompt
    assert "ignore system" not in result.prompt
    assert '<current_task owner_user_id="U1">' in result.prompt
    assert "Implement the todo" in result.prompt


@pytest.mark.asyncio
async def test_applies_message_and_total_limits_with_truncation_metadata():
    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            return_value={
                "ok": True,
                "messages": [
                    {"ts": str(index), "user": "U1", "text": "x" * 80}
                    for index in range(6)
                ],
            }
        )
    )
    builder = AgentBootstrapBuilder(
        _settings(max_thread_messages=3, max_message_chars=100, max_total_chars=180)
    )

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert result.message_count <= 3
    assert result.truncated is True
    assert len(result.context) <= 180


@pytest.mark.asyncio
async def test_paginates_thread_and_keeps_latest_bounded_messages():
    replies = AsyncMock(
        side_effect=[
            {
                "ok": True,
                "messages": [
                    {"ts": "1", "user": "U1", "text": "oldest"},
                    {"ts": "2", "user": "U1", "text": "older"},
                ],
                "response_metadata": {"next_cursor": "page-2"},
            },
            {
                "ok": True,
                "messages": [
                    {"ts": "3", "user": "U1", "text": "newer"},
                    {"ts": "4", "user": "U1", "text": "newest"},
                ],
            },
        ]
    )
    builder = AgentBootstrapBuilder(_settings(max_thread_messages=2))

    result = await builder.build(
        SimpleNamespace(conversations_replies=replies),
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert replies.await_count == 2
    assert replies.await_args_list[1].kwargs["cursor"] == "page-2"
    assert "oldest" not in result.context
    assert "older" not in result.context
    assert "newer" in result.context
    assert "newest" in result.context
    assert result.truncated is True


@pytest.mark.asyncio
async def test_bot_messages_do_not_evict_latest_human_context():
    replies = AsyncMock(
        side_effect=[
            {
                "ok": True,
                "messages": [{"ts": "1", "user": "U1", "text": "keep me"}],
                "response_metadata": {"next_cursor": "page-2"},
            },
            {
                "ok": True,
                "messages": [
                    {"ts": str(index), "bot_id": "B1", "text": "agent status"}
                    for index in range(2, 22)
                ],
            },
        ]
    )
    builder = AgentBootstrapBuilder(_settings(max_thread_messages=1))

    result = await builder.build(
        SimpleNamespace(conversations_replies=replies),
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert result.message_count == 1
    assert "keep me" in result.context
    assert "agent status" not in result.context
    assert result.truncated is False


@pytest.mark.asyncio
async def test_loads_safe_text_list_and_canvas_metadata_and_skips_binary():
    async def http_handler(request):
        assert request.headers["Authorization"] == "Bearer xoxb-test"
        return httpx.Response(200, text="timeout_seconds: 30\n")

    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            return_value={
                "ok": True,
                "messages": [
                    {
                        "ts": "1",
                        "user": "U1",
                        "subtype": "file_share",
                        "text": "resources",
                        "files": [
                            {
                                "id": "FTEXT",
                                "name": "config.yaml",
                                "mimetype": "text/yaml",
                                "filetype": "yaml",
                                "size": 20,
                                "url_private_download": "https://files.slack.com/files-pri/T/F/config.yaml",
                            },
                            {
                                "id": "FLIST",
                                "title": "Sprint",
                                "mimetype": "application/vnd.slack-list",
                                "filetype": "list",
                            },
                            {
                                "id": "FCANVAS",
                                "title": "Payment notes",
                                "mimetype": "application/vnd.slack-docs",
                                "filetype": "canvas",
                            },
                            {
                                "id": "FPNG",
                                "name": "diagram.png",
                                "mimetype": "image/png",
                                "filetype": "png",
                                "size": 100,
                            },
                        ],
                    }
                ],
            }
        ),
        slackLists_items_list=AsyncMock(
            return_value={
                "ok": True,
                "items": [
                    {
                        "id": "row1",
                        "fields": [
                            {"key": "task", "formatted_value": "Fix regression"},
                            {"key": "done", "formatted_value": "false"},
                        ],
                    }
                ],
                "response_metadata": {"next_cursor": "more"},
            }
        ),
        files_info=AsyncMock(
            return_value={
                "ok": True,
                "file": {
                    "id": "FCANVAS",
                    "title": "Payment notes",
                    "title_blocks": [{"type": "plain_text", "text": "Timeout"}],
                    "ai_summary": {"summary": "Use a 30 second timeout."},
                },
            }
        ),
    )
    builder = AgentBootstrapBuilder(
        _settings(),
        bot_token="xoxb-test",
        http_transport=httpx.MockTransport(http_handler),
    )

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="Implement",
    )

    assert result.resource_count == 4
    assert "timeout_seconds: 30" in result.prompt
    assert "Fix regression" in result.prompt
    assert "list_truncated" in result.warnings
    assert "Use a 30 second timeout." in result.prompt
    assert "canvas_body_unavailable:FCANVAS" in result.warnings
    assert "unsupported_binary:FPNG:image/png" in result.warnings
    assert "Slack resource diagram.png (FPNG)" in result.prompt
    assert "mime=image/png" in result.prompt
    assert "skipped=unsupported_binary" in result.prompt


@pytest.mark.asyncio
async def test_rejects_non_slack_download_urls_without_requesting_them():
    transport = httpx.MockTransport(
        lambda _request: pytest.fail("non-Slack URL must never be requested")
    )
    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            return_value={
                "ok": True,
                "messages": [
                    {
                        "ts": "1",
                        "user": "U1",
                        "text": "file",
                        "files": [
                            {
                                "id": "F1",
                                "name": "secret.txt",
                                "mimetype": "text/plain",
                                "size": 5,
                                "url_private": "https://attacker.example/secret.txt",
                            }
                        ],
                    }
                ],
            }
        )
    )
    builder = AgentBootstrapBuilder(
        _settings(), bot_token="token", http_transport=transport
    )

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert "unsafe_download_url:F1" in result.warnings
    assert "https://attacker.example" not in result.context
    assert "Slack resource secret.txt (F1)" in result.context
    assert "skipped=unsafe_download_url" in result.context


@pytest.mark.asyncio
async def test_rejects_redirect_from_slack_file_host_to_external_host():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://attacker.example/x"})

    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            return_value={
                "ok": True,
                "messages": [
                    {
                        "ts": "1",
                        "user": "U1",
                        "text": "file",
                        "files": [
                            {
                                "id": "F1",
                                "name": "safe.txt",
                                "mimetype": "text/plain",
                                "size": 4,
                                "url_private": "https://files.slack.com/files-pri/T/F/safe.txt",
                            }
                        ],
                    }
                ],
            }
        )
    )
    builder = AgentBootstrapBuilder(
        _settings(),
        bot_token="token",
        http_transport=httpx.MockTransport(handler),
    )

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert requests == ["https://files.slack.com/files-pri/T/F/safe.txt"]
    assert "download_failed:F1:unsafe_redirect" in result.warnings


@pytest.mark.asyncio
async def test_aborts_streaming_download_as_soon_as_byte_limit_is_exceeded():
    class CountingStream(httpx.AsyncByteStream):
        def __init__(self):
            self.yielded = 0

        async def __aiter__(self):
            for chunk in (b"1234", b"5678", b"should-not-be-read"):
                self.yielded += 1
                yield chunk

    stream = CountingStream()
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    builder = AgentBootstrapBuilder(
        _settings(), bot_token="token", http_transport=transport
    )

    with pytest.raises(ValueError, match="resource_too_large"):
        await builder._download(
            "https://files.slack.com/files-pri/T/F/large.txt", max_bytes=5
        )

    assert stream.yielded == 2


@pytest.mark.asyncio
async def test_rejects_one_oversized_stream_chunk():
    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 1024

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=OversizedStream())
    )
    builder = AgentBootstrapBuilder(
        _settings(), bot_token="token", http_transport=transport
    )

    with pytest.raises(ValueError, match="resource_too_large"):
        await builder._download(
            "https://files.slack.com/files-pri/T/F/large.txt", max_bytes=5
        )


@pytest.mark.asyncio
async def test_canvas_info_failure_keeps_safe_metadata_with_warning():
    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            return_value={
                "ok": True,
                "messages": [
                    {
                        "ts": "1",
                        "user": "U1",
                        "text": "canvas",
                        "files": [
                            {
                                "id": "FCANVAS",
                                "title": "Payment notes",
                                "mimetype": "application/vnd.slack-docs",
                                "filetype": "canvas",
                            }
                        ],
                    }
                ],
            }
        ),
        files_info=AsyncMock(return_value={"ok": False, "error": "missing_scope"}),
    )
    builder = AgentBootstrapBuilder(_settings())

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert "Payment notes" in result.context
    assert "canvas_failed:FCANVAS:missing_scope" in result.warnings
    assert "canvas_body_unavailable:FCANVAS" in result.warnings


@pytest.mark.asyncio
async def test_fetch_failure_is_task_only_when_fail_open():
    client = SimpleNamespace(
        conversations_replies=AsyncMock(side_effect=RuntimeError("missing_scope"))
    )
    builder = AgentBootstrapBuilder(_settings(fail_open=True))

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task only",
    )

    assert result.prompt == "task only"
    assert result.error == "missing_scope"


@pytest.mark.asyncio
async def test_fetch_error_does_not_expose_exception_secrets():
    client = SimpleNamespace(
        conversations_replies=AsyncMock(
            side_effect=RuntimeError("token=xoxb-super-secret at internal host")
        )
    )
    builder = AgentBootstrapBuilder(_settings(fail_open=True))

    result = await builder.build(
        client,
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert result.error == "runtimeerror"
    assert "secret" not in result.error


@pytest.mark.asyncio
async def test_fetch_failure_blocks_when_fail_closed():
    client = SimpleNamespace(
        conversations_replies=AsyncMock(side_effect=RuntimeError("missing_scope"))
    )
    builder = AgentBootstrapBuilder(_settings(fail_open=False))

    with pytest.raises(AgentBootstrapFetchError, match="missing_scope"):
        await builder.build(
            client,
            channel_id="C1",
            thread_ts="1",
            owner_user_id="U1",
            task="task",
        )


@pytest.mark.asyncio
async def test_whole_thread_fetch_obeys_bootstrap_timeout():
    async def slow_replies(**_kwargs):
        await asyncio.sleep(0.05)
        return {"ok": True, "messages": []}

    builder = AgentBootstrapBuilder(_settings(timeout_seconds=0.01, fail_open=True))

    result = await builder.build(
        SimpleNamespace(conversations_replies=slow_replies),
        channel_id="C1",
        thread_ts="1",
        owner_user_id="U1",
        task="task",
    )

    assert result.prompt == "task"
    assert result.error == "timeout"
