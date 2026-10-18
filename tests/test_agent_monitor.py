"""Durable Slack-side tracking of Pi coding-agent runs (audit G3)."""

import json
from types import SimpleNamespace

import pytest

from slack.bot_real import RedisSlackStateStore, SlackBot


class FakeRedis:
    """Minimal in-memory stand-in for the redis.asyncio client."""

    def __init__(self):
        self.values = {}
        self.sets = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value):
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def srem(self, key, member):
        self.sets.get(key, set()).discard(member)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))


def file_bot(tmp_path, **services):
    return SlackBot(
        {
            "channels": [],
            "state": {"backend": "file", "file_path": str(tmp_path / "state.json")},
        },
        SimpleNamespace(),
        services=services,
    )


def track(bot, run_id="R1", **overrides):
    accepted = {"session_id": overrides.pop("session_id", "S1"), "run_id": run_id}
    kwargs = {
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "1.0",
        "owner_user_id": "U1",
        "mode": "implement",
    }
    kwargs.update(overrides)
    return bot._track_agent_run(accepted, **kwargs)


EXPECTED_RECORD = {
    "run_id": "R1",
    "session_id": "S1",
    "team_id": "T1",
    "channel_id": "C1",
    "thread_ts": "1.0",
    "owner_user_id": "U1",
    "mode": "implement",
    "posted_approval_ids": ["A1"],
    "result_posted": False,
}


@pytest.mark.asyncio
async def test_tracked_agent_run_round_trips_through_file_state(tmp_path):
    first = file_bot(tmp_path)
    tracked = track(first)
    tracked["posted_approval_ids"].append("A1")
    await first._persist_agent_run(tracked)

    second = file_bot(tmp_path)
    await second._restore_state()

    assert second.agent_runs["R1"] == EXPECTED_RECORD
    assert second._is_agent_thread("C1", "1.0")
    assert second._agent_run_for("U1", "C1", "1.0")["run_id"] == "R1"


@pytest.mark.asyncio
async def test_redis_agent_run_records_maintain_an_index_set():
    store = RedisSlackStateStore({"redis": {"key_prefix": "p"}})
    store.client = FakeRedis()

    await store.persist_agent_run_state("R1", EXPECTED_RECORD)

    assert store.client.sets["p:agent_runs"] == {"R1"}
    assert json.loads(store.client.values["p:agent_run:R1"]) == EXPECTED_RECORD
    assert (await store.load_state())["agent_runs"] == {"R1": EXPECTED_RECORD}

    await store.delete_agent_run_state("R1")

    assert store.client.sets["p:agent_runs"] == set()
    assert "p:agent_run:R1" not in store.client.values
    assert (await store.load_state())["agent_runs"] == {}


@pytest.mark.asyncio
async def test_bot_writes_agent_runs_granularly_to_redis():
    bot = SlackBot({"channels": []}, SimpleNamespace())
    store = RedisSlackStateStore({"redis": {"key_prefix": "p"}})
    store.client = FakeRedis()
    bot.state_store = store
    tracked = track(bot)
    tracked["posted_approval_ids"].append("A1")

    await bot._persist_agent_run(tracked)
    restored = SlackBot({"channels": []}, SimpleNamespace())
    restored.state_store = store
    await restored._restore_state()

    assert restored.agent_runs["R1"] == EXPECTED_RECORD

    await bot._forget_agent_run(tracked)

    assert store.client.sets["p:agent_runs"] == set()
    assert "R1" not in bot.agent_runs


@pytest.mark.asyncio
async def test_new_run_in_a_thread_prunes_finished_runs_from_state(tmp_path):
    bot = file_bot(tmp_path)
    old = track(bot)
    old["result_posted"] = True
    await bot._persist_agent_run(old)

    await bot._persist_agent_run(track(bot, run_id="R2"))
    restored = file_bot(tmp_path)
    await restored._restore_state()

    assert set(restored.agent_runs) == {"R2"}
