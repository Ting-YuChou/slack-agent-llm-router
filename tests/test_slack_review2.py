"""Round-2 review: feedback replies across shutdown; slash references."""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import (
    CHAT_FEEDBACK_THANKS_TEXT,
    CHAT_FEEDBACK_UNCONFIRMED_TEXT,
    SlackBot,
)
from slack.formatting import SlackNotice

REQUEST_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "slack:T1:C1:100.0"))
TRIGGER_ID = "13345224609.738474920.8088930838d88f008e0"


def _bot(produce):
    producer = SimpleNamespace(
        confirms_chat_feedback_delivery=True,
        produce_chat_feedback=AsyncMock(side_effect=produce),
    )
    bot = SlackBot(
        {"channels": []},
        inference_engine=SimpleNamespace(),
        services={"event_producer": producer},
    )
    bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    return bot


def _click():
    return {
        "type": "block_actions",
        "user": {"id": "U2"},
        "channel": {"id": "C1"},
        "actions": [
            {
                "action_id": "chat_feedback_up",
                "value": REQUEST_ID,
                "action_ts": "1700.1",
            }
        ],
    }


def _texts(bot):
    return [c.kwargs["text"] for c in bot.web_client.chat_postEphemeral.await_args_list]


@pytest.mark.asyncio
async def test_shutdown_lets_an_in_flight_feedback_confirmation_finish():
    async def produce(**fields):
        await asyncio.sleep(0.1)
        return True

    bot = _bot(produce)
    await bot._handle_agent_interactive(_click())

    await asyncio.wait_for(bot.shutdown(), timeout=2)

    assert _texts(bot) == [CHAT_FEEDBACK_THANKS_TEXT]


@pytest.mark.asyncio
async def test_a_feedback_confirmation_cut_off_by_shutdown_still_gets_a_reply():
    async def produce(**fields):
        await asyncio.Event().wait()

    bot = _bot(produce)
    bot.chat_feedback_shutdown_wait_seconds = 0.05
    await bot._handle_agent_interactive(_click())

    await asyncio.wait_for(bot.shutdown(), timeout=2)

    assert _texts(bot) == [CHAT_FEEDBACK_UNCONFIRMED_TEXT]


def test_a_slash_reference_comes_from_the_trigger_ids_last_segment():
    notice = SlackNotice("failed", request_id=f"slash:{TRIGGER_ID}")

    assert notice.reference == "80889308"
