import importlib.util
import logging
import sys
from pathlib import Path

import pytest

BOT_MODULE = Path(__file__).resolve().parents[1] / "slack" / "bot.py"


def load_bot_module(monkeypatch):
    """Import a fresh copy of `slack/bot.py` while `slack.bot_real` fails."""
    import slack  # noqa: F401  (the package the stub imports from)

    monkeypatch.setitem(sys.modules, "slack.bot_real", None)
    spec = importlib.util.spec_from_file_location("slack._bot_stub_test", BOT_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_stub_reports_unhealthy_and_logs_the_import_error(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        module = load_bot_module(monkeypatch)

    assert "import of slack.bot_real halted" in caplog.text
    assert any(record.exc_info for record in caplog.records)

    bot = module.SlackBot({}, services={})
    await bot.initialize()
    await bot.start()
    assert bot.is_healthy() is False
    assert "slack.bot_real" in bot.import_error
    await bot.shutdown()


def test_real_bot_is_exported_when_import_succeeds():
    from slack.bot import SlackBot
    from slack.bot_real import SlackBot as RealSlackBot

    assert SlackBot is RealSlackBot
