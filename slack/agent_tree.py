"""Pure Block Kit rendering for sanitized Pi session trees."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class AgentTreeNonce:
    team_id: str
    channel_id: str
    thread_ts: str
    session_id: str
    owner_user_id: str
    expires_at: float


def build_agent_tree_modal(nonce: str, tree: Dict[str, Any]) -> Dict[str, Any]:
    """Render at most 20 sanitized turns and forkable checkpoints."""
    turns = [turn for turn in tree.get("turns", []) if isinstance(turn, dict)][-20:]
    by_run = {str(turn.get("run_id")): turn for turn in turns if turn.get("run_id")}
    depths: Dict[str, int] = {}

    def depth(turn: Dict[str, Any]) -> int:
        run_id = str(turn.get("run_id") or "")
        if run_id in depths:
            return depths[run_id]
        parent_id = str(turn.get("parent_run_id") or "")
        parent = by_run.get(parent_id)
        value = min(4, depth(parent) + 1) if parent and parent is not turn else 0
        depths[run_id] = value
        return value

    lines: List[str] = []
    for turn in turns:
        marker = "●" if turn.get("active") else "○"
        indent = "　" * depth(turn)
        number = turn.get("number", "?")
        status = str(turn.get("status") or "unknown")
        commit = str(turn.get("commit") or "")[:8]
        preview = str(turn.get("task_preview") or "(task unavailable)")[:160]
        suffix = f" · `{commit}`" if commit else ""
        lines.append(
            f"{indent}{marker} *Turn {number}* · {status}{suffix}\n"
            f"{indent}　{preview}"
        )

    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(lines) or "No completed Agent turns yet.",
            },
        }
    ]
    forkable = [turn for turn in turns if turn.get("forkable") is True]
    if forkable:
        options = [
            {
                "text": {
                    "type": "plain_text",
                    "text": (
                        f"Turn {turn.get('number', '?')}: "
                        f"{str(turn.get('task_preview') or '')[:55]}"
                    )[:75],
                },
                "value": str(turn["run_id"])[:150],
            }
            for turn in forkable
        ]
        blocks.append(
            {
                "type": "input",
                "block_id": "fork_turn",
                "label": {"type": "plain_text", "text": "Fork after turn"},
                "element": {
                    "type": "static_select",
                    "action_id": "source_run",
                    "options": options,
                },
            }
        )
    return {
        "type": "modal",
        "callback_id": "pi_agent_tree_fork_submit",
        "private_metadata": nonce,
        "title": {"type": "plain_text", "text": "Pi Agent Tree"},
        **(
            {
                "submit": {"type": "plain_text", "text": "Fork"},
                "close": {"type": "plain_text", "text": "Cancel"},
            }
            if forkable
            else {"close": {"type": "plain_text", "text": "Close"}}
        ),
        "blocks": blocks,
    }


def parse_tree_fork_submission(view: Dict[str, Any]) -> str:
    option = (
        view.get("state", {})
        .get("values", {})
        .get("fork_turn", {})
        .get("source_run", {})
        .get("selected_option")
        or {}
    )
    return str(option.get("value") or "").strip()
