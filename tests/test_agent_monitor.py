"""Durable Slack-side tracking of Pi coding-agent runs (audit G3)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from slack.bot_real import RedisSlackStateStore, SlackBot, SlackMessageHandler
from src.agent_runtime import AgentRuntimeError, AgentRuntimeUnauthorized


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


def approval_click(user_id="U1", action_id="pi_agent_approve"):
    return {
        "user": {"id": user_id},
        "team": {"id": "T1"},
        "channel": {"id": "C1"},
        "container": {"channel_id": "C1", "message_ts": "5.0", "thread_ts": "1.0"},
        "message": {"ts": "5.0", "thread_ts": "1.0"},
        "actions": [
            {
                "action_id": action_id,
                "value": '{"run_id":"R1","session_id":"S1","approval_id":"A1"}',
            }
        ],
    }


def click_bot(tmp_path, decide):
    bot = file_bot(tmp_path, agent_runtime=SimpleNamespace(decide=decide))
    bot.web_client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    bot._start_agent_monitor = Mock()
    return bot


@pytest.mark.asyncio
async def test_approval_click_for_untracked_run_rebuilds_tracking(tmp_path):
    bot = click_bot(tmp_path, AsyncMock())

    await bot._handle_agent_interactive(approval_click())

    bot.agent_runtime_client.decide.assert_awaited_once_with(
        "R1", "A1", "U1", "approve"
    )
    tracked = bot.agent_runs["R1"]
    assert {
        key: tracked[key]
        for key in ("session_id", "team_id", "channel_id", "thread_ts")
    } == {"session_id": "S1", "team_id": "T1", "channel_id": "C1", "thread_ts": "1.0"}
    assert tracked["owner_user_id"] == "U1"
    assert tracked["posted_approval_ids"] == ["A1"]
    bot._start_agent_monitor.assert_called_once_with(tracked)
    bot.web_client.chat_postEphemeral.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_click_for_untracked_run_explains_ownership(tmp_path):
    bot = click_bot(tmp_path, AsyncMock(side_effect=AgentRuntimeUnauthorized("no")))

    await bot._handle_agent_interactive(approval_click(user_id="U2"))

    text = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "Only the Agent session owner" in text
    assert "R1" not in bot.agent_runs
    bot._start_agent_monitor.assert_not_called()


@pytest.mark.asyncio
async def test_bad_runtime_token_on_click_is_not_blamed_on_ownership(tmp_path):
    bot = click_bot(
        tmp_path,
        AsyncMock(
            side_effect=AgentRuntimeUnauthorized("no", status=401, code="unauthorized")
        ),
    )

    await bot._handle_agent_interactive(approval_click())

    text = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "credentials" in text
    assert "owner" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked_before", [False, True])
async def test_failed_decision_is_reported_ephemerally(tmp_path, tracked_before):
    bot = click_bot(
        tmp_path, AsyncMock(side_effect=AgentRuntimeError("boom", status=500))
    )
    if tracked_before:
        track(bot)

    await bot._handle_agent_interactive(approval_click())

    text = bot.web_client.chat_postEphemeral.await_args.kwargs["text"]
    assert "could not be recorded" in text
    bot._start_agent_monitor.assert_not_called()


def ci_job(run_id):
    return {
        "run_id": run_id,
        "session_id": f"S-{run_id}",
        "status": "awaiting_approval",
        "team_id": "T1",
        "channel_id": "C1",
        "owner_user_id": "U1",
    }


def inbox_bot(tmp_path, jobs, **runtime_methods):
    runtime = SimpleNamespace(
        ci_repairs=AsyncMock(return_value=jobs), **runtime_methods
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(
            side_effect=lambda **kw: {"ts": f"ts-{kw.get('client_msg_id', 'root')}"}
        )
    )
    bot._start_agent_monitor = Mock()
    return bot


@pytest.mark.asyncio
async def test_ci_inbox_does_not_repost_jobs_after_restart(tmp_path):
    first = inbox_bot(tmp_path, [ci_job("J1")])
    await first._poll_ci_inbox_once()
    await first._forget_agent_run(first.agent_runs["J1"])

    second = inbox_bot(tmp_path, [ci_job("J1")])
    await second._restore_state()
    await second._poll_ci_inbox_once()

    assert first.web_client.chat_postMessage.await_count == 1
    second.web_client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_ci_inbox_failure_on_one_job_does_not_block_others(tmp_path):
    bot = inbox_bot(tmp_path, [ci_job("J1"), ci_job("J2")])
    results = iter([RuntimeError("slack down"), {"ts": "ts-J2"}, {"ts": "ts-J1"}])

    def post(**kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    bot.web_client.chat_postMessage.side_effect = post

    await bot._poll_ci_inbox_once()
    assert set(bot.agent_runs) == {"J2"}

    await bot._poll_ci_inbox_once()
    assert set(bot.agent_runs) == {"J1", "J2"}
    assert bot.agent_runs["J1"]["thread_ts"] == "ts-J1"


@pytest.mark.asyncio
async def test_repair_ci_command_claims_its_job_before_the_inbox(tmp_path):
    accepted = {**ci_job("J1"), "status": "awaiting_approval"}
    bot = inbox_bot(
        tmp_path, [ci_job("J1")], repair_ci=AsyncMock(return_value=accepted)
    )
    handler = SlackMessageHandler(bot)

    response = await handler._handle_agent_command(
        ["repair-ci", "7"], "U1", "C1", bot.web_client, team_id="T1"
    )
    await bot._poll_ci_inbox_once()

    assert "J1" in response
    assert bot.web_client.chat_postMessage.await_count == 1
    bot._start_agent_monitor.assert_called_once()


@pytest.mark.asyncio
async def test_repair_ci_command_skips_posting_when_inbox_won_the_race(tmp_path):
    accepted = {**ci_job("J1"), "status": "awaiting_approval"}
    bot = inbox_bot(
        tmp_path, [ci_job("J1")], repair_ci=AsyncMock(return_value=accepted)
    )
    handler = SlackMessageHandler(bot)

    await bot._poll_ci_inbox_once()
    response = await handler._handle_agent_command(
        ["repair-ci", "7"], "U1", "C1", bot.web_client, team_id="T1"
    )

    assert "J1" in response
    assert bot.web_client.chat_postMessage.await_count == 1
    bot._start_agent_monitor.assert_called_once()


@pytest.mark.asyncio
async def test_ci_job_claims_are_shared_through_redis():
    store = RedisSlackStateStore({"redis": {"key_prefix": "p"}})
    store.client = FakeRedis()
    first = SlackBot({"channels": []}, SimpleNamespace())
    first.state_store = store

    assert await first._claim_ci_job("J1") is True
    assert await first._claim_ci_job("J1") is False
    assert store.client.sets["p:posted_ci_jobs"] == {"J1"}

    second = SlackBot({"channels": []}, SimpleNamespace())
    second.state_store = store
    await second._restore_state()
    assert await second._claim_ci_job("J1") is False


UNVERIFIED = "⚠️ 未驗證（沒有 verification profile）"


def test_unverified_result_is_flagged_in_the_response():
    text = SlackMessageHandler._format_agent_response(
        {
            "status": "completed",
            "answer": "Done.",
            "verification_status": "unverified",
            "grader_status": "skipped",
            "commit": "abc123",
        }
    )

    assert f"*Verification:* {UNVERIFIED}" in text
    assert "*Verification:* unverified" not in text


def test_unverified_flag_survives_a_long_answer():
    text = SlackMessageHandler._format_agent_response(
        {
            "status": "completed",
            "answer": "x" * 5000,
            "verification_status": "unverified",
        },
        max_length=500,
    )

    assert UNVERIFIED in text
    assert len(text) <= 500


def test_passed_verification_is_shown_unchanged():
    text = SlackMessageHandler._format_agent_response(
        {"status": "completed", "verification_status": "passed"}
    )

    assert "*Verification:* passed" in text
    assert UNVERIFIED not in text
