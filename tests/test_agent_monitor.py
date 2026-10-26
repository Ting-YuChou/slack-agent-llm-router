"""Durable Slack-side tracking of Pi coding-agent runs (audit G3)."""

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from slack.agent_bootstrap import AgentBootstrapContext
from slack.agent_feedback import AgentFeedbackNonce
from slack.agent_shortcut import AgentShortcutNonce, AgentShortcutSubmission
from slack.bot_real import RedisSlackStateStore, SlackBot, SlackMessageHandler
from src.agent_runtime import (
    AgentRuntimeClient,
    AgentRuntimeConflict,
    AgentRuntimeError,
    AgentRuntimeUnauthorized,
    AgentRuntimeUnavailable,
)


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


def degraded_bot(tmp_path, runtime, *, deadline=900.0):
    """A bot whose monitor backoff runs on a fake clock; returns its sleeps."""
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.agent_monitor_poll_seconds = 0
    bot.agent_run_deadline_seconds = deadline
    now = [0.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    bot.agent_monitor_sleep = sleep
    bot.agent_monitor_clock = lambda: now[0]
    bot.agent_monitor_random = lambda: 1.0
    return bot, sleeps


def delay_notices(bot):
    return [text for text in posted_texts(bot) if "status is delayed" in text]


@pytest.mark.asyncio
async def test_monitor_reattaches_after_the_client_gives_up_and_posts_one_notice(
    tmp_path,
):
    down = AgentRuntimeError("down", status=503, code="runtime_unhealthy")
    runtime = scripted_runtime(
        {
            "R1": [
                down,
                AgentRuntimeUnavailable("unreachable"),
                down,
                run_state("R1", "completed", answer="done"),
            ]
        }
    )
    bot, sleeps = degraded_bot(tmp_path, runtime)
    tracked = track(bot)

    await bot._monitor_agent_run(tracked)

    assert runtime.calls == ["R1"] * 4
    assert len(delay_notices(bot)) == 1
    assert not any("could not be retrieved" in text for text in posted_texts(bot))
    assert tracked["result_posted"] is True
    # Backoff between reattaches grows, with full jitter (random() == 1 here).
    assert len(sleeps) == 3
    assert sleeps == sorted(sleeps)
    assert sleeps[0] > 0


@pytest.mark.asyncio
async def test_monitor_stops_reattaching_at_the_run_deadline(tmp_path):
    runtime = scripted_runtime({"R1": [AgentRuntimeError("down", status=502)]})
    bot, sleeps = degraded_bot(tmp_path, runtime, deadline=600.0)
    tracked = track(bot)

    await bot._monitor_agent_run(tracked)

    texts = posted_texts(bot)
    assert len(delay_notices(bot)) == 1
    assert sum("could not be retrieved" in text for text in texts) == 1
    assert len(runtime.calls) > 2
    assert sum(sleeps) <= 600
    # The run is not forgotten: a restart reconciles it again.
    assert bot.agent_runs["R1"]["result_posted"] is False


@pytest.mark.asyncio
async def test_monitor_does_not_reattach_after_a_non_transient_error(tmp_path):
    runtime = scripted_runtime(
        {
            "R1": [
                AgentRuntimeError("down", status=503),
                AgentRuntimeUnauthorized("bad token", status=401, code="unauthorized"),
                run_state("R1", "completed"),
            ]
        }
    )
    bot, sleeps = degraded_bot(tmp_path, runtime)

    await bot._monitor_agent_run(track(bot))

    assert runtime.calls == ["R1", "R1"]
    assert len(delay_notices(bot)) == 1
    assert sum("could not be retrieved" in t for t in posted_texts(bot)) == 1


@pytest.mark.asyncio
async def test_an_approval_wait_pushes_the_monitor_deadline_out(tmp_path):
    down = AgentRuntimeError("down", status=503)
    runtime = scripted_runtime(
        {
            "R1": [
                down,
                down,
                approval_state("R1", "A1"),
                down,
                down,
                run_state("R1", "completed"),
            ]
        }
    )
    bot, sleeps = degraded_bot(tmp_path, runtime, deadline=60.0)
    bot.agent_monitor_reattach_seconds = 20.0
    bot.agent_monitor_reattach_max_seconds = 20.0
    tracked = track(bot)

    await bot._monitor_agent_run(tracked)

    # 80 seconds of backoff in all, but never more than 60 without contact.
    assert sum(sleeps) == 80
    assert tracked["result_posted"] is True
    assert len(delay_notices(bot)) == 1


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


def card_click(action_id="pi_agent_approve", user_id="U1"):
    payload = approval_click(user_id=user_id, action_id=action_id)
    payload["message"]["blocks"] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Edit*"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "Expires"}]},
        {"type": "actions", "elements": []},
    ]
    return payload


def card_bot(tmp_path, decide):
    bot = click_bot(tmp_path, decide)
    bot.web_client.chat_update = AsyncMock()
    return bot


def updated_card(bot):
    kwargs = bot.web_client.chat_update.await_args.kwargs
    assert (kwargs["channel"], kwargs["ts"]) == ("C1", "5.0")
    assert all(block["type"] != "actions" for block in kwargs["blocks"])
    assert kwargs["blocks"][0]["text"]["text"] == "*Edit*"
    return kwargs["blocks"][-1]["elements"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_id,verdict",
    [("pi_agent_approve", "Approved"), ("pi_agent_reject", "Rejected")],
)
async def test_decided_card_replaces_its_buttons_with_the_decision(
    tmp_path, action_id, verdict
):
    bot = card_bot(tmp_path, AsyncMock())
    tracked = track(bot)
    tracked["approval_cards"] = {"A1": "5.0"}

    await bot._handle_agent_interactive(card_click(action_id))

    text = updated_card(bot)
    assert text.startswith(f"{verdict} by <@U1> at <!date^")
    assert "Expires" not in json.dumps(bot.web_client.chat_update.await_args.kwargs)
    assert tracked["approval_cards"] == {}


@pytest.mark.asyncio
async def test_card_for_an_expired_approval_says_expired(tmp_path):
    bot = card_bot(
        tmp_path,
        AsyncMock(side_effect=AgentRuntimeConflict("gone", status=409)),
    )
    track(bot)["approval_cards"] = {"A1": "5.0"}

    await bot._handle_agent_interactive(card_click())

    assert updated_card(bot).startswith("Expired")
    bot.web_client.chat_postEphemeral.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_decision_keeps_the_card_buttons(tmp_path):
    bot = card_bot(tmp_path, AsyncMock(side_effect=AgentRuntimeError("x", status=500)))
    tracked = track(bot)
    tracked["approval_cards"] = {"A1": "5.0"}

    await bot._handle_agent_interactive(card_click())

    bot.web_client.chat_update.assert_not_awaited()
    assert tracked["approval_cards"] == {"A1": "5.0"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "later_events,verdict",
    [
        ([], "Expired"),
        (
            [
                {
                    "type": "approval_decision",
                    "decision": "approved",
                    "approval_id": "A1",
                }
            ],
            "Approved at <!date^",
        ),
        (
            [
                {
                    "type": "approval_decision",
                    "decision": "rejected",
                    "approval_id": "A1",
                }
            ],
            "Rejected at <!date^",
        ),
    ],
)
async def test_monitor_closes_cards_the_run_moved_past(tmp_path, later_events, verdict):
    approval = {"type": "approval", "approval_id": "A1", "title": "Edit"}
    runtime = scripted_runtime(
        {
            "R1": [
                approval_state("R1", "A1"),
                run_state("R1", "failed", events=[approval, *later_events]),
            ]
        }
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "5.0"}),
        chat_update=AsyncMock(),
    )
    bot.agent_monitor_poll_seconds = 0
    tracked = track(bot)

    await bot._monitor_agent_run(tracked)

    kwargs = bot.web_client.chat_update.await_args.kwargs
    assert (kwargs["channel"], kwargs["ts"]) == ("C1", "5.0")
    assert kwargs["blocks"][0]["text"]["text"] == "*Edit*"
    assert all(block["type"] != "actions" for block in kwargs["blocks"])
    assert kwargs["blocks"][-1]["elements"][0]["text"].startswith(verdict)
    bot.web_client.chat_update.assert_awaited_once()
    assert tracked["approval_cards"] == {}


@pytest.mark.asyncio
async def test_open_approval_card_survives_a_restart(tmp_path):
    first = file_bot(tmp_path)
    tracked = track(first)
    tracked["approval_cards"] = {"A1": "5.0"}
    await first._persist_agent_run(tracked)

    second = file_bot(tmp_path)
    await second._restore_state()

    assert second.agent_runs["R1"]["approval_cards"] == {"A1": "5.0"}


def restore_bot(tmp_path, session, run=None):
    runtime = scripted_runtime({})
    runtime.lookup_session = AsyncMock(return_value=session)
    runtime.get_run = AsyncMock(return_value=run or {})
    runtime.create_session = AsyncMock(
        return_value={"session_id": "S9", "run_id": "R10", "status": "starting"}
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot._start_agent_monitor = Mock()
    bot._monitor_agent_run = AsyncMock()
    return bot, runtime


RESTORE_EVENT = {"channel": "C1", "thread_ts": "9.0", "team": "T1", "user": "U1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session,run,expected",
    [
        # The lookup names the mode.
        ({"task_mode": "discuss"}, None, "discuss"),
        ({"task_mode": "implement"}, None, "implement"),
        # The lookup names a run; the run names the mode.
        ({"run_id": "R9", "active": True}, {"task_mode": "discuss"}, "discuss"),
        ({"run_id": "R9", "active": False}, {"task_mode": "discuss"}, "discuss"),
        # Runs created before task_mode existed were implement runs.
        ({"run_id": "R9", "active": True}, {"run_id": "R9"}, "implement"),
        # Nothing names the mode: stay read-only rather than guess implement.
        ({}, None, "discuss"),
    ],
)
async def test_restored_agent_thread_keeps_its_task_mode(
    tmp_path, session, run, expected
):
    bot, runtime = restore_bot(
        tmp_path, {"session_id": "S9", "owner_user_id": "U1", **session}, run
    )

    await bot._restore_agent_thread(RESTORE_EVENT)

    assert bot.agent_threads[bot._thread_key("C1", "9.0")]["mode"] == expected
    if session.get("run_id"):
        runtime.get_run.assert_awaited_once_with("R9")
        assert bot.agent_runs["R9"]["mode"] == expected


@pytest.mark.asyncio
async def test_restored_run_lookup_failure_keeps_the_thread_read_only(tmp_path):
    bot, runtime = restore_bot(
        tmp_path, {"session_id": "S9", "owner_user_id": "U1", "run_id": "R9"}
    )
    runtime.get_run.side_effect = AgentRuntimeError("down", status=503)

    await bot._restore_agent_thread(RESTORE_EVENT)

    assert bot.agent_runs["R9"]["mode"] == "discuss"


@pytest.mark.asyncio
async def test_followup_in_a_restored_discuss_thread_stays_discuss(tmp_path):
    bot, runtime = restore_bot(
        tmp_path,
        {"session_id": "S9", "owner_user_id": "U1", "run_id": "R9", "active": False},
        {"task_mode": "discuss"},
    )
    await bot._restore_agent_thread(RESTORE_EVENT)

    response = await SlackMessageHandler(bot).handle_message(
        {
            "text": "now go ahead and fix it",
            "user": "U1",
            "channel": "C1",
            "thread_ts": "9.0",
            "_team_id": "T1",
        },
        SimpleNamespace(),
    )

    assert runtime.create_session.await_args.kwargs["task_mode"] == "discuss"
    assert "--implement" in response


def progress_state(run_id, status, tools, turns, cost, phase=None):
    return run_state(
        run_id,
        status,
        phase=phase or status,
        tool_count=tools,
        turn_count=turns,
        estimated_cost_usd=cost,
    )


@pytest.mark.asyncio
async def test_progress_updates_the_root_status_message_at_most_every_ten_seconds(
    tmp_path,
):
    now = [0.0]
    refreshes = []

    async def wait_for_update(run_id, on_progress=None, **kwargs):
        refreshes.append(kwargs.get("refresh_seconds"))
        for at, tools in [(0, 1), (4, 2), (9.9, 3), (10, 4), (15, 5), (25, 6)]:
            now[0] = at
            await on_progress(
                progress_state(run_id, "running", tools, 2, 0.0123, "implementing")
            )
        return progress_state(run_id, "completed", 6, 3, 0.02)

    bot = file_bot(
        tmp_path, agent_runtime=SimpleNamespace(wait_for_update=wait_for_update)
    )
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(), chat_update=AsyncMock()
    )
    bot.agent_progress_clock = lambda: now[0]
    tracked = track(bot)
    tracked["status_ts"] = "1.0"
    tracked["status_text"] = "🛠️ Pi coding agent started run `R1`."

    await bot._monitor_agent_run(tracked)

    # Turns and cost come only from full run reads, refreshed as often as shown.
    assert refreshes == [bot.agent_progress_interval_seconds]
    updates = [call.kwargs for call in bot.web_client.chat_update.await_args_list]
    assert [(update["channel"], update["ts"]) for update in updates] == [
        ("C1", "1.0")
    ] * 4
    texts = [update["text"] for update in updates]
    assert all(
        text.startswith("🛠️ Pi coding agent started run `R1`.\n") for text in texts
    )
    for text, tools in zip(texts[:3], [1, 4, 6]):
        assert "Phase: *implementing*" in text
        assert f"Tools: {tools}" in text
        assert "Turns: 2" in text
        assert "Cost: $0.012" in text
    # The run's end always reaches the status message, throttle or not.
    assert "Phase: *completed*" in texts[3]
    assert "Tools: 6" in texts[3] and "Turns: 3" in texts[3]
    assert "Cost: $0.020" in texts[3]
    assert "*completed*" in posted_texts(bot)[-1]


@pytest.mark.asyncio
async def test_progress_without_a_status_message_posts_nothing(tmp_path):
    async def wait_for_update(run_id, on_progress=None, **kwargs):
        await on_progress(progress_state(run_id, "running", 1, 1, 0.01))
        return run_state(run_id, "completed")

    bot = file_bot(
        tmp_path, agent_runtime=SimpleNamespace(wait_for_update=wait_for_update)
    )
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(), chat_update=AsyncMock()
    )

    await bot._monitor_agent_run(track(bot))

    bot.web_client.chat_update.assert_not_awaited()
    bot.web_client.chat_postMessage.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_progress_update_does_not_stop_the_monitor(tmp_path):
    async def wait_for_update(run_id, on_progress=None, **kwargs):
        await on_progress(progress_state(run_id, "running", 1, 1, 0.01))
        return run_state(run_id, "completed")

    bot = file_bot(
        tmp_path, agent_runtime=SimpleNamespace(wait_for_update=wait_for_update)
    )
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(),
        chat_update=AsyncMock(side_effect=RuntimeError("slack down")),
    )
    tracked = track(bot)
    tracked["status_ts"] = "1.0"

    await bot._monitor_agent_run(tracked)

    assert tracked["result_posted"] is True


@pytest.mark.asyncio
async def test_started_message_becomes_the_durable_status_message(tmp_path):
    runtime = SimpleNamespace(
        create_session=AsyncMock(
            return_value={"session_id": "S1", "run_id": "R1", "status": "starting"}
        )
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot._monitor_agent_run = AsyncMock()
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "100.1"}),
        chat_update=AsyncMock(),
    )

    await SlackMessageHandler(bot)._handle_command(
        "agent fix the tests", "U1", "C1", None, client, team_id="T1"
    )

    started = client.chat_update.await_args.kwargs["text"]
    assert bot.agent_runs["R1"]["status_ts"] == "100.1"
    assert bot.agent_runs["R1"]["status_text"] == started
    # The monitor persists the run first; it is mocked out here.
    await bot._persist_agent_run(bot.agent_runs["R1"])
    restored = file_bot(tmp_path)
    await restored._restore_state()
    assert restored.agent_runs["R1"]["status_ts"] == "100.1"
    assert restored.agent_runs["R1"]["status_text"] == started


def thread_start_bot(tmp_path):
    """A bot whose runtime accepts starts and resumes of run R1 in thread 100.1."""
    accepted = {"session_id": "S1", "run_id": "R1", "status": "starting"}
    runtime = SimpleNamespace(
        create_session=AsyncMock(return_value=accepted),
        resume=AsyncMock(return_value=accepted),
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "300.1"}),
        chat_update=AsyncMock(),
        chat_postEphemeral=AsyncMock(),
    )
    bot._start_agent_monitor = Mock()
    return bot


async def assert_durable_status_message(tmp_path, bot, status_ts):
    tracked = bot.agent_runs["R1"]
    assert tracked["status_ts"] == status_ts
    assert "`R1`" in tracked["status_text"]
    bot._start_agent_monitor.assert_called_once_with(tracked)
    # The monitor persists the run first; it is mocked out here.
    await bot._persist_agent_run(tracked)
    restored = file_bot(tmp_path)
    await restored._restore_state()
    assert restored.agent_runs["R1"]["status_ts"] == status_ts
    assert restored.agent_runs["R1"]["status_text"] == tracked["status_text"]
    return restored


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "also add a regression test",
        "!llm agent fix the tests",
        "/llm agent fix the tests",
        "!llm agent resume R1",
    ],
)
async def test_a_run_started_inside_a_thread_posts_its_own_status_message(
    tmp_path, text
):
    bot = thread_start_bot(tmp_path)
    if not text.startswith(("!llm", "/llm")):
        track(bot, "R0", thread_ts="100.1")["result_posted"] = True

    await bot._process_message(
        {
            "text": text,
            "user": "U1",
            "channel": "C1",
            "thread_ts": "100.1",
            "ts": "200.1",
            "_team_id": "T1",
        }
    )

    # The status message is the run's only reply; nothing else is posted.
    post = bot.web_client.chat_postMessage.await_args
    bot.web_client.chat_postMessage.assert_awaited_once()
    assert (post.kwargs["channel"], post.kwargs["thread_ts"]) == ("C1", "100.1")
    assert bot.agent_runs["R1"]["status_text"] == post.kwargs["text"]
    if "agent fix" in text:
        # A private channel's context stays out, and the status message says so.
        assert "without prior Slack thread context" in post.kwargs["text"]
    restored = await assert_durable_status_message(tmp_path, bot, "300.1")
    # Handling the message still counted against the owner's query quota.
    assert len(restored.user_manager.rate_limits["U1"]) == 1


@pytest.mark.asyncio
async def test_a_needs_changes_followup_posts_its_own_status_message(tmp_path):
    bot = thread_start_bot(tmp_path)
    bot.agent_runtime_client.feedback = AsyncMock()
    bot.agent_feedback_nonces["N1"] = AgentFeedbackNonce(
        run_id="R0",
        team_id="T1",
        channel_id="C1",
        thread_ts="100.1",
        owner_user_id="U1",
        feedback_id="click-1",
        expires_at=time.time() + 60,
    )

    await bot._handle_agent_feedback_submission(
        {
            "user": {"id": "U1"},
            "view": {
                "private_metadata": "N1",
                "state": {
                    "values": {
                        "feedback_comment": {"comment": {"value": "rerun the suite"}},
                        "feedback_continue": {
                            "continue": {"selected_options": [{"value": "continue"}]}
                        },
                    }
                },
            },
        }
    )

    post = bot.web_client.chat_postMessage.await_args
    bot.web_client.chat_postMessage.assert_awaited_once()
    assert (post.kwargs["channel"], post.kwargs["thread_ts"]) == ("C1", "100.1")
    assert bot.agent_runs["R1"]["status_text"] == post.kwargs["text"]
    await assert_durable_status_message(tmp_path, bot, "300.1")


@pytest.mark.asyncio
async def test_a_shortcut_run_keeps_its_loading_message_as_the_status_message(
    tmp_path,
):
    bot = thread_start_bot(tmp_path)
    bot.agent_bootstrap_builder = SimpleNamespace(
        build=AsyncMock(
            return_value=AgentBootstrapContext(
                prompt="bootstrap prompt", message_count=3, resource_count=2
            )
        )
    )
    nonce = AgentShortcutNonce(
        team_id="T1",
        channel_id="C1",
        thread_ts="100.1",
        selected_message_ts="100.2",
        owner_user_id="U1",
        expires_at=time.time() + 60,
    )
    submission = AgentShortcutSubmission(
        task="fix the tests", model=None, include_context=True
    )

    await bot._bootstrap_agent_submission(nonce, "N1", submission, "U1", "250.1")

    # The shortcut's own "loading" message becomes the status message.
    bot.web_client.chat_postMessage.assert_not_awaited()
    update = bot.web_client.chat_update.await_args
    assert (update.kwargs["channel"], update.kwargs["ts"]) == ("C1", "250.1")
    assert bot.agent_runs["R1"]["status_text"] == update.kwargs["text"]
    assert "Loaded 3 prior Slack messages" in update.kwargs["text"]
    await assert_durable_status_message(tmp_path, bot, "250.1")


@pytest.mark.asyncio
async def test_a_thread_started_run_has_exactly_one_status_message(tmp_path):
    async def wait_for_update(run_id, on_progress=None, **kwargs):
        await on_progress(progress_state(run_id, "running", 1, 1, 0.01))
        await on_progress(progress_state(run_id, "running", 2, 1, 0.01))
        return progress_state(run_id, "completed", 2, 2, 0.02)

    bot = thread_start_bot(tmp_path)
    bot.agent_runtime_client.wait_for_update = wait_for_update
    del bot._start_agent_monitor
    track(bot, "R0", thread_ts="100.1")["result_posted"] = True

    response = await SlackMessageHandler(bot).handle_message(
        {
            "text": "also add a regression test",
            "user": "U1",
            "channel": "C1",
            "thread_ts": "100.1",
            "ts": "200.1",
            "_team_id": "T1",
        },
        bot.web_client,
    )
    await drain(bot)

    assert response == ""
    status, result = posted_texts(bot)
    assert "`R1`" in status
    assert "*completed*" in result
    updates = [call.kwargs for call in bot.web_client.chat_update.await_args_list]
    assert updates and all(update["ts"] == "300.1" for update in updates)
    assert "Phase: *completed*" in updates[-1]["text"]


@pytest.mark.asyncio
async def test_a_replayed_start_keeps_the_runs_status_message(tmp_path):
    bot = thread_start_bot(tmp_path)
    track(bot, "R0", thread_ts="100.1")["result_posted"] = True
    followup = {
        "text": "also add a regression test",
        "user": "U1",
        "channel": "C1",
        "thread_ts": "100.1",
        "ts": "200.1",
        "_team_id": "T1",
    }

    # The runtime answers a redelivered request's key with the same run.
    first = await SlackMessageHandler(bot).handle_message(followup, bot.web_client)
    second = await SlackMessageHandler(bot).handle_message(followup, bot.web_client)

    assert (first, second) == ("", "")
    bot.web_client.chat_postMessage.assert_awaited_once()
    assert bot.agent_runs["R1"]["status_ts"] == "300.1"


def epoch(iso):
    return datetime.fromisoformat(iso).timestamp()


@pytest.mark.asyncio
async def test_progress_shows_the_time_elapsed_since_the_run_started(tmp_path):
    created_at = "2026-10-26T03:00:00.000Z"
    now = [epoch("2026-10-26T03:02:05+00:00")]

    async def wait_for_update(run_id, on_progress=None, **kwargs):
        await on_progress(
            progress_state(run_id, "running", 1, 1, 0.01) | {"created_at": created_at}
        )
        now[0] = epoch("2026-10-26T04:02:30+00:00")
        return run_state(run_id, "completed", created_at=created_at)

    bot = file_bot(
        tmp_path, agent_runtime=SimpleNamespace(wait_for_update=wait_for_update)
    )
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(), chat_update=AsyncMock()
    )
    bot.agent_progress_clock = lambda: now[0]
    tracked = track(bot)
    tracked["status_ts"] = "1.0"

    await bot._monitor_agent_run(tracked)

    texts = [call.kwargs["text"] for call in bot.web_client.chat_update.await_args_list]
    assert "Elapsed: 2m 05s" in texts[0]
    assert "Elapsed: 1h 02m" in texts[-1]


@pytest.mark.asyncio
async def test_progress_times_a_run_without_timestamps_from_when_it_was_tracked(
    tmp_path,
):
    now = [1000.0]

    async def wait_for_update(run_id, on_progress=None, **kwargs):
        now[0] = 1045.0
        await on_progress(progress_state(run_id, "running", 1, 1, 0.01))
        return run_state(run_id, "completed")

    bot = file_bot(
        tmp_path, agent_runtime=SimpleNamespace(wait_for_update=wait_for_update)
    )
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(), chat_update=AsyncMock()
    )
    bot.agent_progress_clock = lambda: now[0]
    tracked = track(bot)
    tracked["status_ts"] = "1.0"

    await bot._monitor_agent_run(tracked)

    texts = [call.kwargs["text"] for call in bot.web_client.chat_update.await_args_list]
    assert "Elapsed: 45s" in texts[0]


def delivery_state(run_id, approval_id):
    return run_state(
        run_id,
        "awaiting_approval",
        events=[
            {
                "type": "approval",
                "kind": "delivery",
                "approval_id": approval_id,
                "delivery": {"files": ["a.py"], "stat": "1 file changed"},
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [approval_state, delivery_state])
async def test_approval_and_result_posts_mention_the_owner(tmp_path, pending):
    runtime = scripted_runtime(
        {"R1": [pending("R1", "A1"), run_state("R1", "completed", answer="done")]}
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(chat_postMessage=AsyncMock())
    bot.agent_monitor_poll_seconds = 0

    await bot._monitor_agent_run(track(bot))

    card, result = [
        call.kwargs for call in bot.web_client.chat_postMessage.await_args_list
    ]
    assert card["blocks"][0]["text"]["text"].startswith("<@U1> ")
    assert "<@U1>" in card["text"]
    assert result["text"].startswith("<@U1> ✅ Agent run `R1` *completed*")
    assert result["blocks"][0]["text"]["text"].startswith(
        "<@U1> ✅ Agent run `R1` *completed*"
    )


@pytest.mark.asyncio
async def test_status_message_shows_a_run_waiting_for_approval(tmp_path):
    waiting = delivery_state("R1", "A1")
    waiting.update(phase="awaiting_delivery", tool_count=3, turn_count=2)
    runtime = scripted_runtime({"R1": [waiting, run_state("R1", "rejected")]})
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot.web_client = SimpleNamespace(
        chat_postMessage=AsyncMock(), chat_update=AsyncMock()
    )
    bot.agent_monitor_poll_seconds = 0
    tracked = track(bot)
    tracked["status_ts"] = "1.0"

    await bot._monitor_agent_run(tracked)

    texts = [call.kwargs["text"] for call in bot.web_client.chat_update.await_args_list]
    assert "Phase: *awaiting_delivery* · Tools: 3 · Turns: 2" in texts[0]
    assert "Phase: *rejected*" in texts[-1]


# The runtime's answer when a redelivered Slack request carries its first key
# but a different body (for example a new thread context).
REUSED_KEY = AgentRuntimeError(
    "Agent runtime rejected the request with status 422",
    status=422,
    code="idempotency_key_reused",
    server_message="Idempotency-Key was already used for a different request",
)
ACTIVE_SESSION = {
    "session_id": "S1",
    "run_id": "R2",
    "owner_user_id": "U1",
    "active": True,
}


def redelivery_bot(tmp_path, session):
    """A bot whose runtime already started a run for the request it receives."""
    runtime = SimpleNamespace(
        create_session=AsyncMock(side_effect=REUSED_KEY),
        lookup_session=AsyncMock(return_value=session),
        get_run=AsyncMock(return_value=run_state("R2", "running", task_mode="discuss")),
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot._start_agent_monitor = Mock()
    return bot, runtime


def redeliver_followup(bot):
    return SlackMessageHandler(bot).handle_message(
        {
            "text": "also add a regression test",
            "user": "U1",
            "channel": "C1",
            "thread_ts": "100.1",
            "ts": "200.6",
            "_team_id": "T1",
        },
        SimpleNamespace(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["start", "followup"])
async def test_a_redelivered_request_adopts_the_run_its_first_delivery_started(
    tmp_path, delivery
):
    bot, runtime = redelivery_bot(tmp_path, ACTIVE_SESSION)
    if delivery == "start":
        response = await SlackMessageHandler(bot)._start_agent_prompt(
            "fix it",
            "U1",
            "C1",
            None,
            team_id="T1",
            thread_ts="100.1",
            request_id="200.5",
        )
    else:
        track(bot, "R1", thread_ts="100.1")["result_posted"] = True
        response = await redeliver_followup(bot)

    runtime.create_session.assert_awaited_once()
    runtime.lookup_session.assert_awaited_once_with("T1", "C1", "100.1")
    tracked = bot.agent_runs["R2"]
    assert (tracked["session_id"], tracked["thread_ts"]) == ("S1", "100.1")
    assert (tracked["owner_user_id"], tracked["mode"]) == ("U1", "discuss")
    assert bot.agent_threads["C1:100.1"] is tracked
    bot._start_agent_monitor.assert_called_once_with(tracked)
    assert "`R2` is already running" in response
    assert "could not start" not in response


@pytest.mark.asyncio
async def test_a_redelivery_keeps_the_slack_state_of_a_run_already_followed(
    tmp_path,
):
    bot, _runtime = redelivery_bot(tmp_path, ACTIVE_SESSION)
    tracked = track(bot, "R2", thread_ts="100.1", session_id="S1")
    tracked.update(
        status_ts="100.1", status_text="🛠️ started", posted_approval_ids=["A1"]
    )

    response = await redeliver_followup(bot)

    assert bot.agent_runs["R2"] is tracked
    assert tracked["status_ts"] == "100.1"
    assert tracked["posted_approval_ids"] == ["A1"]
    bot._start_agent_monitor.assert_called_once_with(tracked)
    assert "`R2` is already running" in response


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["missing", "unavailable"])
async def test_a_redelivery_the_runtime_cannot_match_is_not_a_generic_failure(
    tmp_path, lookup
):
    bot, runtime = redelivery_bot(tmp_path, None)
    if lookup == "unavailable":
        runtime.lookup_session.side_effect = AgentRuntimeUnavailable("down")

    response = await SlackMessageHandler(bot)._start_agent_prompt(
        "fix it", "U1", "C1", None, team_id="T1", thread_ts="100.1", request_id="200.5"
    )

    assert "already received this request" in response
    assert "no second run was started" in response
    assert "could not start" not in response
    assert not bot.agent_runs
    bot._start_agent_monitor.assert_not_called()


@pytest.mark.asyncio
async def test_a_start_whose_answer_timed_out_ends_as_one_monitored_run(
    tmp_path, monkeypatch
):
    runs = {}
    keys = []

    def runtime_server(request):
        """The runtime's idempotency contract: one run per key and body."""
        key = request.headers["idempotency-key"]
        keys.append(key)
        if key not in runs:
            runs[key] = {"session_id": "S1", "run_id": "R1", "status": "starting"}
            # The run was accepted, but the answer never reached the bot.
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(202, json=runs[key])

    async def no_sleep(_seconds):
        return None

    monkeypatch.setenv("AGENT_RUNTIME_TOKEN", "runtime-token")
    runtime = AgentRuntimeClient(
        {"base_url": "http://agent.test:3001"},
        transport=httpx.MockTransport(runtime_server),
        sleep=no_sleep,
    )
    bot = file_bot(tmp_path, agent_runtime=runtime)
    bot._start_agent_monitor = Mock()
    try:
        response = await SlackMessageHandler(bot)._start_agent_prompt(
            "fix it",
            "U1",
            "C1",
            None,
            team_id="T1",
            thread_ts="100.1",
            request_id="200.5",
        )
    finally:
        await runtime.close()

    assert len(runs) == 1, "the retry started no second run"
    assert len(keys) == 2 and keys[0] == keys[1]
    assert set(bot.agent_runs) == {"R1"}
    bot._start_agent_monitor.assert_called_once_with(bot.agent_runs["R1"])
    assert "`R1` started" in response
