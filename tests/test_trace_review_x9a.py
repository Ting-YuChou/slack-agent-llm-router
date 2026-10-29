"""Round-1 review fixes for X9a: no thanks for feedback that was not recorded."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import SlackBot

REQUEST_ID = "3f0c7c6e-6c55-5a43-9d0a-0d6f8f6f2b11"


def _feedback_bot(producer):
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(),
        services={"event_producer": producer} if producer is not None else {},
    )
    bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    return bot


def _click(action_ts="1700.1"):
    return {
        "type": "block_actions",
        "user": {"id": "U2"},
        "channel": {"id": "C1"},
        "actions": [
            {
                "action_id": "chat_feedback_up",
                "value": REQUEST_ID,
                "action_ts": action_ts,
            }
        ],
    }


def _ephemeral_texts(bot):
    return [
        call.kwargs["text"]
        for call in bot.web_client.chat_postEphemeral.await_args_list
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "producer",
    [
        None,
        SimpleNamespace(produce_chat_feedback=AsyncMock(return_value=False)),
        SimpleNamespace(produce_chat_feedback=AsyncMock(side_effect=RuntimeError())),
    ],
    ids=["no-producer", "not-enqueued", "raised"],
)
async def test_unrecorded_feedback_is_not_thanked(producer):
    bot = _feedback_bot(producer)

    await bot._handle_agent_interactive(_click())

    assert not any("Thanks" in text for text in _ephemeral_texts(bot))


@pytest.mark.asyncio
async def test_a_click_that_failed_can_be_recorded_on_retry():
    producer = SimpleNamespace(
        produce_chat_feedback=AsyncMock(side_effect=[False, True])
    )
    bot = _feedback_bot(producer)

    await bot._handle_agent_interactive(_click())
    await bot._handle_agent_interactive(_click())
    await bot._handle_agent_interactive(_click())

    assert producer.produce_chat_feedback.await_count == 2
    assert sum("Thanks" in text for text in _ephemeral_texts(bot)) == 1
