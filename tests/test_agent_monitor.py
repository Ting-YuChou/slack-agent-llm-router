"""Durable Slack-side tracking of Pi coding-agent runs (audit G3)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack.bot_real import RedisSlackStateStore, SlackBot
from src.agent_runtime import AgentRuntimeError


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


def scripted_runtime(script):
    """Return a fake runtime whose wait_for_update replays per-run outcomes."""
    queues = {run_id: list(outcomes) for run_id, outcomes in script.items()}
    calls = []

    async def wait_for_update(run_id, **kwargs):
        calls.append(run_id)
        outcome = (
            queues[run_id].pop(0) if len(queues[run_id]) > 1 else queues[run_id][0]
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return SimpleNamespace(wait_for_update=wait_for_update, calls=calls)


def run_state(run_id, status, **extra):
    return {"run_id": run_id, "session_id": "S1", "status": status, **extra}


def approval_state(run_id, approval_id):
    return run_state(
        run_id,
        "awaiting_approval",
        events=[{"type": "approval", "approval_id": approval_id, "title": "Edit"}],
    )


async def drain(bot):
    while bot.background_tasks:
        await asyncio.gather(*list(bot.background_tasks))


def posted_texts(bot):
    return [
        call.kwargs["text"] for call in bot.web_client.chat_postMessage.await_args_list
    ]


@pytest.mark.asyncio
async def test_startup_reconcile_resumes_monitors_and_posts_missed_results(tmp_path):
    first = file_bot(tmp_path)
    for run_id, thread_ts in [("OPEN", "1"), ("DONE", "2"), ("GONE", "3")]:
        await first._persist_agent_run(track(first, run_id, thread_ts=thread_ts))
    posted = track(first, "POSTED", thread_ts="4")
    posted["result_posted"] = True
    await first._persist_agent_run(posted)

    runtime = scripted_runtime(
        {
            "OPEN": [run_state("OPEN", "running"), run_state("OPEN", "completed")],
            "DONE": [run_state("DONE", "failed")],
            "GONE": [AgentRuntimeError("missing", status=404, code="not_found")],
        }
    )
    second = file_bot(tmp_path, agent_runtime=runtime)
    second.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    second.agent_monitor_poll_seconds = 0
    await second._restore_state()

    second._reconcile_agent_runs()
    await drain(second)

    assert sorted(runtime.calls) == ["DONE", "GONE", "OPEN"]
    assert second.web_client.chat_postMessage.await_count == 2
    assert "GONE" not in second.agent_runs
    assert second.agent_runs["OPEN"]["result_posted"] is True
    assert second.agent_runs["DONE"]["result_posted"] is True

    third = file_bot(tmp_path, agent_runtime=runtime)
    third.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    await third._restore_state()
    third._reconcile_agent_runs()
    await drain(third)

    third.web_client.chat_postMessage.assert_not_awaited()
    assert set(third.agent_runs) == {"OPEN", "DONE", "POSTED"}


@pytest.mark.asyncio
async def test_monitor_posts_each_approval_once_and_keeps_watching(tmp_path):
    runtime = scripted_runtime(
        {
            "R1": [
                approval_state("R1", "A1"),
                approval_state("R1", "A1"),
                approval_state("R1", "A2"),
                run_state("R1", "completed", answer="done"),
            ]
        }
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.agent_monitor_poll_seconds = 0
    tracked = track(bot)

    await bot._monitor_agent_run(tracked)

    texts = posted_texts(bot)
    assert [text.startswith("Approval required") for text in texts] == [
        True,
        True,
        False,
    ]
    assert tracked["posted_approval_ids"] == ["A1", "A2"]
    restored = file_bot(tmp_path)
    await restored._restore_state()
    assert restored.agent_runs["R1"]["posted_approval_ids"] == ["A1", "A2"]
    assert restored.agent_runs["R1"]["result_posted"] is True


@pytest.mark.asyncio
async def test_restart_does_not_repost_an_approval_card_already_posted(tmp_path):
    runtime = scripted_runtime(
        {"R1": [approval_state("R1", "A1"), run_state("R1", "rejected")]}
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.agent_monitor_poll_seconds = 0
    tracked = track(bot)
    tracked["posted_approval_ids"].append("A1")

    await bot._monitor_agent_run(tracked)

    assert len(posted_texts(bot)) == 1
    assert not posted_texts(bot)[0].startswith("Approval required")


@pytest.mark.asyncio
async def test_one_run_never_gets_two_monitors(tmp_path):
    release = asyncio.Event()
    calls = []

    async def wait_for_update(run_id, **kwargs):
        calls.append(run_id)
        await release.wait()
        return run_state(run_id, "completed")

    bot = file_bot(
        tmp_path, agent_runtime=SimpleNamespace(wait_for_update=wait_for_update)
    )
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    tracked = track(bot)

    bot._start_agent_monitor(tracked)
    bot._start_agent_monitor(tracked)
    bot._reconcile_agent_runs()
    await asyncio.sleep(0)
    bot._start_agent_monitor(tracked)
    release.set()
    await drain(bot)

    assert calls == ["R1"]
    bot.web_client.chat_postMessage.assert_awaited_once()


@pytest.mark.asyncio
async def test_restored_agent_thread_with_active_run_starts_a_monitor(tmp_path):
    runtime = scripted_runtime({"R9": [run_state("R9", "completed")]})
    runtime.lookup_session = AsyncMock(
        return_value={
            "session_id": "S9",
            "run_id": "R9",
            "owner_user_id": "U1",
            "active": True,
        }
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())

    await bot._restore_agent_thread(
        {"channel": "C1", "thread_ts": "9.0", "team": "T1", "user": "U1"}
    )
    await drain(bot)

    assert runtime.calls == ["R9"]
    assert bot.agent_runs["R9"]["result_posted"] is True


@pytest.mark.asyncio
async def test_restored_agent_thread_without_active_run_does_not_monitor(tmp_path):
    runtime = scripted_runtime({})
    runtime.lookup_session = AsyncMock(
        return_value={
            "session_id": "S9",
            "run_id": "R9",
            "owner_user_id": "U1",
            "active": False,
        }
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())

    await bot._restore_agent_thread(
        {"channel": "C1", "thread_ts": "9.0", "team": "T1", "user": "U1"}
    )
    await drain(bot)

    assert runtime.calls == []
    assert bot._is_agent_thread("C1", "9.0")
