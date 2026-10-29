"""Round-2 review fix (X2): Slack file names stay out of event query_text."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import main
from slack.bot_real import SlackBot
from src.llm_router_part3_pipeline import (
    build_inference_completed_event,
    build_request_raw_event,
)
from src.utils.schema import Attachment, AttachmentType

from test_main import (  # noqa: F401
    DummyEventProducer,
    _admission_platform,
    patched_platform_deps,
)


@pytest.mark.asyncio
async def test_a_slack_file_only_dm_keeps_the_file_name_out_of_events(
    tmp_path, patched_platform_deps
):
    platform = _admission_platform(tmp_path)
    producer = DummyEventProducer({})
    platform.services["event_producer"] = producer
    inference = platform.services["inference"]
    bot = SlackBot(
        {"channels": [], "context": {"enabled": False}},
        inference_engine=inference,
        services={**platform.services, "chat_ingress": main.ChatIngress(platform)},
    )
    bot.bot_user_id = "UBOT"
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "500.0"}),
        chat_postEphemeral=AsyncMock(),
        chat_update=AsyncMock(return_value={"ok": True}),
    )
    bot._extract_query_attachments = AsyncMock(
        return_value=[
            Attachment(
                name="salary-review-jane.pdf",
                type=AttachmentType.DOCUMENT,
                size_bytes=11,
                mime_type="application/pdf",
                content=b"file detail",
            )
        ]
    )

    await bot._process_message(
        {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "user": "U1",
            "text": "",
            "ts": "100.0",
            "files": [{"id": "F1"}],
        }
    )

    (request,) = producer.request_events
    # The model still gets the file's name to work with.
    assert "salary-review-jane" in request.query
    raw = build_request_raw_event(request)
    completed = build_inference_completed_event(request, inference.response)
    assert "salary-review-jane" not in str(raw)
    assert "salary-review-jane" not in str(completed)
    assert raw["query_text"].startswith("Analyze the attached Slack file")


def test_an_api_query_text_keeps_its_shape():
    from src.utils.schema import QueryRequest

    request = QueryRequest(
        query="Summarize notes.txt",
        user_id="u1",
        attachments=[
            Attachment(
                name="notes.txt",
                type=AttachmentType.DOCUMENT,
                size_bytes=1,
                mime_type="text/plain",
                content=b"x",
            )
        ],
    )

    assert build_request_raw_event(request)["query_text"] == "Summarize notes.txt"
