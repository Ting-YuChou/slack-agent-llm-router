try:
    from .bot_real import SlackBot  # noqa: F401
except Exception:

    class SlackBot:
        def __init__(self, *args, **kwargs):
            self.running = False

        async def initialize(self):
            return None

        async def start(self):
            self.running = True

        async def shutdown(self):
            self.running = False

        def is_healthy(self):
            return True
