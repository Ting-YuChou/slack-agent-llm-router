import logging
from types import SimpleNamespace

import httpx
import pytest

from slack.bot_real import SlackBot
from slack.files import (
    download_slack_file,
    is_allowed_slack_file_url,
    is_slack_hosted_file,
)

SLACK_URL = "https://files.slack.com/files-pri/T1-F1/download/report.csv"


class CountingStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.yielded = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk


def _bot(monkeypatch, handler, **attachment_settings):
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client_with_mock_transport(*args, **kwargs):
        return real_client(*args, **{**kwargs, "transport": transport})

    monkeypatch.setattr("slack.bot_real.httpx.AsyncClient", client_with_mock_transport)
    bot = SlackBot({"channels": []}, inference_engine=SimpleNamespace())
    bot.bot_token = "xoxb-test"
    bot.attachment_settings.update(attachment_settings)
    return bot


def _file(**overrides):
    payload = {
        "id": "F1",
        "name": "report.csv",
        "mimetype": "text/csv",
        "size": 3,
        "url_private_download": SLACK_URL,
    }
    payload.update(overrides)
    return payload


def _recording_handler(*responses):
    requests = []
    queue = list(responses)

    def handler(request):
        requests.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return requests, handler


@pytest.mark.parametrize(
    "url",
    [
        "https://files.slack.com/files-pri/T1-F1/report.csv",
        "https://FILES.SLACK.COM/files-pri/T1-F1/report.csv",
    ],
)
def test_slack_file_hosts_are_allowed(url):
    assert is_allowed_slack_file_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://files.slack.com/files-pri/T1-F1/report.csv",
        "https://files.example/report.csv",
        "https://files.slack.com.attacker.example/report.csv",
        "https://files.slack.com@attacker.example/report.csv",
        "https://drive.google.com/file/d/1",
        "",
    ],
)
def test_other_hosts_and_schemes_are_not_allowed(url):
    assert not is_allowed_slack_file_url(url)


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_external": True},
        {"mode": "external"},
        {"mode": "tombstone"},
        {"mode": "hidden_by_limit"},
    ],
)
def test_external_and_withheld_files_are_not_slack_hosted(overrides):
    assert not is_slack_hosted_file(_file(**overrides))


def test_ordinary_slack_files_are_slack_hosted():
    assert is_slack_hosted_file(_file(mode="hosted", is_external=False))
    assert is_slack_hosted_file(_file())


@pytest.mark.asyncio
async def test_download_stops_reading_once_the_byte_cap_is_passed():
    stream = CountingStream([b"1234", b"5678", b"never-read"])
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="resource_too_large"):
            await download_slack_file(
                client, SLACK_URL, bot_token="xoxb-test", max_bytes=5
            )

    assert stream.yielded == 2


@pytest.mark.asyncio
async def test_download_does_not_follow_redirects_even_if_the_client_would():
    requests, handler = _recording_handler(
        httpx.Response(302, headers={"location": "https://attacker.example/x"})
    )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        with pytest.raises(ValueError, match="unsafe_redirect"):
            await download_slack_file(
                client, SLACK_URL, bot_token="xoxb-test", max_bytes=100
            )

    assert [str(request.url) for request in requests] == [SLACK_URL]


@pytest.mark.asyncio
async def test_bot_never_requests_a_non_slack_host(monkeypatch):
    requests, handler = _recording_handler(httpx.Response(200, content=b"abc"))
    bot = _bot(monkeypatch, handler)

    attachments = await bot._extract_query_attachments(
        {
            "files": [
                _file(url_private_download="https://files.example/report.csv"),
                _file(
                    id="F2",
                    url_private_download="http://files.slack.com/files-pri/T/F2/a",
                ),
            ]
        }
    )

    assert requests == []
    assert all(attachment.content is None for attachment in attachments)


@pytest.mark.asyncio
async def test_bot_skips_a_file_without_size_whose_body_passes_the_cap(monkeypatch):
    stream = CountingStream([b"x" * 4, b"x" * 4, b"x" * 4, b"never-read"])
    bot = _bot(
        monkeypatch,
        lambda _request: httpx.Response(
            200, headers={"content-type": "text/csv"}, stream=stream
        ),
        max_file_size_bytes=10,
    )

    attachments = await bot._extract_query_attachments({"files": [_file(size=0)]})

    assert attachments == []
    assert stream.yielded == 3


@pytest.mark.asyncio
async def test_bot_does_not_follow_a_redirect_off_slack(monkeypatch, caplog):
    requests, handler = _recording_handler(
        httpx.Response(302, headers={"location": "https://attacker.example/x"})
    )
    bot = _bot(monkeypatch, handler)

    with caplog.at_level(logging.WARNING, logger="slack.bot_real"):
        attachments = await bot._extract_query_attachments({"files": [_file()]})

    assert [str(request.url) for request in requests] == [SLACK_URL]
    assert attachments[0].content is None
    assert "attacker.example" not in caplog.text


@pytest.mark.asyncio
async def test_bot_follows_a_redirect_that_stays_on_slack_file_hosts(monkeypatch):
    target = "https://files.slack.com/files-pri/T1-F1/report.csv?origin=redirect"
    requests, handler = _recording_handler(
        httpx.Response(302, headers={"location": target}),
        httpx.Response(200, headers={"content-type": "text/csv"}, content=b"abc"),
    )
    bot = _bot(monkeypatch, handler)

    attachments = await bot._extract_query_attachments({"files": [_file()]})

    assert [str(request.url) for request in requests] == [SLACK_URL, target]
    assert requests[1].headers["Authorization"] == "Bearer xoxb-test"
    assert attachments[0].content == b"abc"


@pytest.mark.asyncio
async def test_bot_download_failure_logs_the_file_id_but_not_the_url(
    monkeypatch, caplog
):
    _requests, handler = _recording_handler(httpx.Response(500))
    bot = _bot(monkeypatch, handler)

    with caplog.at_level(logging.WARNING, logger="slack.bot_real"):
        attachments = await bot._extract_query_attachments({"files": [_file()]})

    assert attachments[0].content is None
    assert "F1" in caplog.text
    assert "files.slack.com" not in caplog.text
    assert "xoxb-test" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"is_external": True},
        {"mode": "external"},
        {"mode": "tombstone"},
        {"mode": "hidden_by_limit"},
    ],
)
async def test_bot_skips_external_and_withheld_files(monkeypatch, overrides):
    requests, handler = _recording_handler(httpx.Response(200, content=b"abc"))
    bot = _bot(monkeypatch, handler)

    attachments = await bot._extract_query_attachments({"files": [_file(**overrides)]})

    assert attachments == []
    assert requests == []


@pytest.mark.asyncio
async def test_bot_rejects_html_when_the_file_is_not_html(monkeypatch):
    _requests, handler = _recording_handler(
        httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html>Sign in to Slack</html>",
        )
    )
    bot = _bot(monkeypatch, handler)

    attachments = await bot._extract_query_attachments({"files": [_file()]})

    assert attachments[0].content is None


@pytest.mark.asyncio
async def test_bot_accepts_html_for_a_declared_html_file(monkeypatch):
    _requests, handler = _recording_handler(
        httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<p>hi</p>",
        )
    )
    bot = _bot(monkeypatch, handler)

    attachments = await bot._extract_query_attachments(
        {"files": [_file(name="page.html", mimetype="text/html", filetype="html")]}
    )

    assert attachments[0].content == b"<p>hi</p>"


@pytest.mark.asyncio
async def test_bot_shares_one_http_client_and_closes_it_on_shutdown(monkeypatch):
    _requests, handler = _recording_handler(
        httpx.Response(200, headers={"content-type": "text/csv"}, content=b"abc")
    )
    bot = _bot(monkeypatch, handler)

    await bot._extract_query_attachments({"files": [_file()]})
    client = bot._attachment_http_client
    await bot._extract_query_attachments({"files": [_file(id="F2")]})

    assert client is not None
    assert bot._attachment_http_client is client
    assert client.is_closed is False

    await bot.shutdown()

    assert client.is_closed is True
    assert bot._attachment_http_client is None
