from types import SimpleNamespace
from unittest.mock import AsyncMock
import json
import time

import pytest

from slack.agent_tree import AgentTreeNonce, build_agent_tree_modal
from slack.bot_real import SlackBot, SlackMessageHandler


def tree_payload(count=3):
    turns = []
    parent = None
    for number in range(1, count + 1):
        run_id = f"R{number}"
        turns.append(
            {
                "number": number,
                "run_id": run_id,
                "parent_run_id": parent,
                "task_preview": f"task {number}",
                "status": "completed" if number not in {2, 7} else "failed",
                "commit": f"c{number}" if number not in {2, 7} else None,
                "active": number == count,
                "forkable": number not in {2, 7},
            }
        )
        parent = run_id
    return {
        "session_id": "S1",
        "active_leaf_id": "leaf",
        "turns": turns,
        "lineage": {},
        "truncated": False,
    }


def test_tree_modal_limits_display_and_allows_only_completed_checkpoints():
    payload = tree_payload(25)
    modal = build_agent_tree_modal("nonce-1", payload)

    assert modal["callback_id"] == "pi_agent_tree_fork_submit"
    rendered = str(modal["blocks"])
    assert "Turn 6" in rendered
    assert "Turn 5" not in rendered
    assert "● *Turn 25*" in rendered
    select = next(
        block for block in modal["blocks"] if block.get("block_id") == "fork_turn"
    )
    values = [option["value"] for option in select["element"]["options"]]
    assert "R2" not in values
    assert len(values) == 19


@pytest.mark.asyncio
async def test_history_tree_stats_and_fork_do_not_consume_quota_but_compact_does():
    handler = SlackMessageHandler(SimpleNamespace())

    for command in ("agent history", "agent tree", "agent stats", "agent fork 1"):
        assert handler.command_consumes_query_quota(command) is False
    assert handler.command_consumes_query_quota("agent compact") is True
    assert handler.command_consumes_query_quota("agent compact Keep tests") is True


@pytest.mark.asyncio
async def test_history_posts_a_tree_button_and_compact_creates_a_maintenance_run():
    runtime = SimpleNamespace(
        get_tree=AsyncMock(return_value=tree_payload()),
        compact=AsyncMock(
            return_value={"session_id": "S1", "run_id": "RC", "status": "starting"}
        ),
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.agent_user_runs["U1"] = {
        "run_id": "R3",
        "session_id": "S1",
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "100.1",
        "owner_user_id": "U1",
    }
    bot._monitor_agent_run = AsyncMock()
    client = SimpleNamespace(chat_postEphemeral=AsyncMock())
    handler = SlackMessageHandler(bot)

    history = await handler._handle_command(
        "agent history", "U1", "C1", "100.1", client, team_id="T1"
    )
    compact = await handler._handle_command(
        "agent compact Preserve decisions",
        "U1",
        "C1",
        "100.1",
        client,
        team_id="T1",
    )

    assert history == ""
    blocks = client.chat_postEphemeral.await_args.kwargs["blocks"]
    button = blocks[-1]["elements"][0]
    assert button["action_id"] == "pi_agent_view_tree"
    assert "Turn 3" in str(blocks)
    assert "RC" in compact
    runtime.compact.assert_awaited_once_with("S1", "U1", "Preserve decisions")


@pytest.mark.asyncio
async def test_fork_creates_a_new_root_and_tracks_an_idle_child_session():
    runtime = SimpleNamespace(
        get_tree=AsyncMock(return_value=tree_payload()),
        fork_session=AsyncMock(
            return_value={
                "session_id": "S2",
                "branch": "pi-agent/child",
                "baseline_commit": "c1",
                "parent_session_id": "S1",
                "fork_source_run_id": "R1",
                "model_ref": "openai/gpt-5.6-luna",
            }
        ),
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.agent_user_runs["U1"] = {
        "run_id": "R3",
        "session_id": "S1",
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "100.1",
        "owner_user_id": "U1",
    }
    bot._is_public_agent_channel = AsyncMock(return_value=True)
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(return_value={"ts": "200.1"}),
        chat_update=AsyncMock(),
    )
    handler = SlackMessageHandler(bot)

    response = await handler._handle_command(
        "agent fork 1", "U1", "C1", "100.1", client, team_id="T1"
    )

    assert response == ""
    runtime.fork_session.assert_awaited_once_with(
        "S1",
        "U1",
        "R1",
        team_id="T1",
        channel_id="C1",
        thread_ts="200.1",
    )
    assert bot.agent_threads["C1:200.1"]["session_id"] == "S2"
    assert bot.agent_threads["C1:200.1"]["run_id"] is None
    assert "Please reply in this thread" in client.chat_update.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_view_tree_button_opens_owner_bound_modal_and_submission_is_acked():
    runtime = SimpleNamespace(get_tree=AsyncMock(return_value=tree_payload()))
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.agent_threads["C1:100.1"] = {
        "run_id": "R3",
        "session_id": "S1",
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "100.1",
        "owner_user_id": "U1",
    }
    bot.web_client = SimpleNamespace(views_open=AsyncMock())
    payload = {
        "type": "block_actions",
        "trigger_id": "trigger-1",
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "actions": [
            {
                "action_id": "pi_agent_view_tree",
                "value": json.dumps(
                    {
                        "session_id": "S1",
                        "channel_id": "C1",
                        "thread_ts": "100.1",
                    }
                ),
            }
        ],
    }

    await bot._handle_agent_interactive(payload)

    bot.web_client.views_open.assert_awaited_once()
    modal = bot.web_client.views_open.await_args.kwargs["view"]
    nonce = modal["private_metadata"]
    assert modal["callback_id"] == "pi_agent_tree_fork_submit"
    assert bot.agent_tree_nonces[nonce].owner_user_id == "U1"
    assert bot.agent_tree_nonces[nonce].session_id == "S1"

    bot._enqueue_work = AsyncMock(return_value=True)
    socket_client = SimpleNamespace(send_socket_mode_response=AsyncMock())
    request = SimpleNamespace(
        envelope_id="E-tree",
        type="interactive",
        payload={
            "type": "view_submission",
            "view": {"callback_id": "pi_agent_tree_fork_submit"},
            "user": {"id": "U1"},
        },
    )
    await bot._handle_socket_mode_request(socket_client, request)

    socket_client.send_socket_mode_response.assert_awaited_once()
    assert bot._enqueue_work.await_args.args[0].kind == "agent_tree_fork"


@pytest.mark.asyncio
async def test_tree_fork_nonce_is_owner_only_one_time_and_rechecks_public_channel():
    runtime = SimpleNamespace(
        get_tree=AsyncMock(return_value=tree_payload()),
        fork_session=AsyncMock(
            return_value={
                "session_id": "S2",
                "branch": "pi-agent/child",
                "baseline_commit": "c1",
                "parent_session_id": "S1",
                "fork_source_run_id": "R1",
                "model_ref": "openai/gpt-5.6-luna",
            }
        ),
    )
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.agent_threads["C1:100.1"] = {
        "run_id": "R3",
        "session_id": "S1",
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "100.1",
        "owner_user_id": "U1",
    }
    bot._is_public_agent_channel = AsyncMock(return_value=True)
    bot.web_client = SimpleNamespace(
        views_open=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "200.1"}),
        chat_update=AsyncMock(),
    )
    await bot._handle_agent_interactive(
        {
            "trigger_id": "trigger-1",
            "team": {"id": "T1"},
            "user": {"id": "U1"},
            "channel": {"id": "C1"},
            "actions": [
                {
                    "action_id": "pi_agent_view_tree",
                    "value": '{"session_id":"S1","channel_id":"C1","thread_ts":"100.1"}',
                }
            ],
        }
    )
    nonce = bot.web_client.views_open.await_args.kwargs["view"]["private_metadata"]
    view = {
        "private_metadata": nonce,
        "state": {
            "values": {
                "fork_turn": {"source_run": {"selected_option": {"value": "R1"}}}
            }
        },
    }

    await bot._handle_agent_tree_fork_submission({"user": {"id": "U2"}, "view": view})
    assert nonce in bot.agent_tree_nonces
    runtime.fork_session.assert_not_awaited()

    await bot._handle_agent_tree_fork_submission({"user": {"id": "U1"}, "view": view})
    assert nonce not in bot.agent_tree_nonces
    runtime.fork_session.assert_awaited_once()
    bot._is_public_agent_channel.assert_awaited_once_with("C1")

    await bot._handle_agent_tree_fork_submission({"user": {"id": "U1"}, "view": view})
    runtime.fork_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_tree_fork_nonce_expires_without_calling_runtime():
    runtime = SimpleNamespace(fork_session=AsyncMock())
    bot = SlackBot(
        {"channels": []}, SimpleNamespace(), services={"agent_runtime": runtime}
    )
    bot.agent_tree_nonces["expired"] = AgentTreeNonce(
        team_id="T1",
        channel_id="C1",
        thread_ts="100.1",
        session_id="S1",
        owner_user_id="U1",
        expires_at=time.time() - 1,
    )

    await bot._handle_agent_tree_fork_submission(
        {
            "user": {"id": "U1"},
            "view": {
                "private_metadata": "expired",
                "state": {
                    "values": {
                        "fork_turn": {
                            "source_run": {"selected_option": {"value": "R1"}}
                        }
                    }
                },
            },
        }
    )

    assert "expired" not in bot.agent_tree_nonces
    runtime.fork_session.assert_not_awaited()
