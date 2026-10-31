"""S17d review: the help text only advertises features the router has."""

from types import SimpleNamespace

import pytest

from slack.bot_real import SlackMessageHandler


@pytest.mark.asyncio
async def test_help_does_not_advertise_context_compression():
    handler = SlackMessageHandler(
        SimpleNamespace(
            agent_runtime_client=None, _format_tier_limit_summary=lambda: ""
        )
    )

    help_text = await handler._handle_help_command([], "U1", "C1", None)

    assert "*Features:*" in help_text
    assert "compression" not in help_text.lower()
