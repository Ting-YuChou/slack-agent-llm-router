import logging

try:
    from .bot_real import SlackBot  # noqa: F401
except Exception as exc:
    # Without the real bot every Slack feature is gone, so say so loudly and
    # report unhealthy instead of standing in as a working bot.
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    logging.getLogger(__name__).exception(
        "Slack bot is unavailable; slack.bot_real failed to import: %s",
        _IMPORT_ERROR,
    )

    class SlackBot:
        import_error = _IMPORT_ERROR

        def __init__(self, *args, **kwargs):
            self.running = False

        async def initialize(self):
            return None

        async def start(self):
            self.running = True

        async def shutdown(self):
            self.running = False

        def is_healthy(self):
            return False
