"""Pure Slack Block Kit helpers for starting a Pi Agent from a message."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentShortcutNonce:
    team_id: str
    channel_id: str
    thread_ts: str
    selected_message_ts: str
    owner_user_id: str
    expires_at: float


@dataclass(frozen=True)
class AgentShortcutSubmission:
    task: str
    model: Optional[str]
    include_context: bool
    # None when the form had no such field, so the runtime default applies.
    task_mode: Optional[str] = None
    workflow_mode: Optional[str] = None


def build_agent_shortcut_modal(
    nonce: str, model_refs: List[str], initial_task_mode: str = "implement"
) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = [
        {
            "type": "input",
            "block_id": "agent_task",
            "label": {"type": "plain_text", "text": "Coding task"},
            "element": {
                "type": "plain_text_input",
                "action_id": "task",
                "multiline": True,
                "placeholder": {
                    "type": "plain_text",
                    "text": "Describe what Pi should implement in the repository",
                },
            },
        }
    ]
    if model_refs:
        options = [
            {
                "text": {"type": "plain_text", "text": model_ref[:75]},
                "value": model_ref[:150],
            }
            for model_ref in model_refs[:100]
        ]
        blocks.append(
            {
                "type": "input",
                "block_id": "agent_model",
                "optional": True,
                "label": {"type": "plain_text", "text": "Agent model"},
                "element": {
                    "type": "static_select",
                    "action_id": "model",
                    "options": options,
                },
            }
        )
    implement_option = {
        "text": {
            "type": "plain_text",
            "text": "Implement: edit and commit (edits need approval)",
        },
        "value": "implement",
    }
    discuss_option = {
        "text": {"type": "plain_text", "text": "Discuss: read-only, no edits"},
        "value": "discuss",
    }
    workflow_option = {
        "text": {
            "type": "mrkdwn",
            "text": "Structured workflow: investigate in planned waves first",
        },
        "value": "structured",
    }
    include_option = {
        "text": {"type": "mrkdwn", "text": "Include this Slack thread"},
        "value": "include",
    }
    blocks.extend(
        [
            {
                "type": "input",
                "block_id": "agent_mode",
                "label": {"type": "plain_text", "text": "Mode"},
                "element": {
                    "type": "radio_buttons",
                    "action_id": "mode",
                    "options": [implement_option, discuss_option],
                    "initial_option": (
                        discuss_option
                        if initial_task_mode == "discuss"
                        else implement_option
                    ),
                },
            },
            {
                "type": "input",
                "block_id": "agent_workflow",
                "optional": True,
                "label": {"type": "plain_text", "text": "Workflow"},
                "element": {
                    "type": "checkboxes",
                    "action_id": "workflow",
                    "options": [workflow_option],
                },
            },
            {
                "type": "input",
                "block_id": "agent_context",
                "optional": True,
                "label": {"type": "plain_text", "text": "Bootstrap context"},
                "element": {
                    "type": "checkboxes",
                    "action_id": "include",
                    "options": [include_option],
                    "initial_options": [include_option],
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "Selected thread content is read-only reference data and "
                            "will be sent to the selected model provider. Agent tool "
                            "approvals and workspace policy still apply. When Jev "
                            "routing is enabled, only your task text is also sent "
                            "to OpenRouter for model selection."
                        ),
                    }
                ],
            },
        ]
    )
    return {
        "type": "modal",
        "callback_id": "run_pi_agent_from_thread_submit",
        "private_metadata": nonce,
        "title": {"type": "plain_text", "text": "Run Pi Agent"},
        "submit": {"type": "plain_text", "text": "Start"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": blocks,
    }


def parse_agent_shortcut_submission(view: Dict[str, Any]) -> AgentShortcutSubmission:
    values = view.get("state", {}).get("values", {}) or {}
    task = str(values.get("agent_task", {}).get("task", {}).get("value") or "").strip()
    selected_model = (
        values.get("agent_model", {}).get("model", {}).get("selected_option") or {}
    )
    include_options = (
        values.get("agent_context", {}).get("include", {}).get("selected_options", [])
        or []
    )
    selected_mode = (
        values.get("agent_mode", {}).get("mode", {}).get("selected_option") or {}
    ).get("value")
    workflow_options = (
        values.get("agent_workflow", {}).get("workflow", {}).get("selected_options", [])
        or []
    )
    return AgentShortcutSubmission(
        task=task,
        model=selected_model.get("value") or None,
        include_context=any(
            option.get("value") == "include"
            for option in include_options
            if isinstance(option, dict)
        ),
        task_mode=(
            selected_mode if selected_mode in {"implement", "discuss"} else None
        ),
        workflow_mode=(
            "structured"
            if any(
                option.get("value") == "structured"
                for option in workflow_options
                if isinstance(option, dict)
            )
            else None
        ),
    )
