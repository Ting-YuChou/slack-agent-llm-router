"""Pure Slack Block Kit helpers for owner feedback on a finished Agent run."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

AGENT_FEEDBACK_CALLBACK_ID = "pi_agent_feedback_submit"
AGENT_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "rejected", "timed_out", "interrupted"}
)
FEEDBACK_COMMENT_MAX_CHARS = 2000
FEEDBACK_REASONS = [
    ("wrong_approach", "Wrong approach"),
    ("incomplete", "Incomplete"),
    ("broke_tests", "Broke tests"),
    ("scope_creep", "Scope creep"),
    ("style", "Style"),
    ("other", "Other"),
]


@dataclass(frozen=True)
class AgentFeedbackNonce:
    run_id: str
    team_id: str
    channel_id: str
    thread_ts: str
    owner_user_id: str
    feedback_id: str
    expires_at: float


@dataclass(frozen=True)
class AgentFeedbackSubmission:
    reason_category: Optional[str]
    comment: str
    continue_session: bool


def build_agent_feedback_modal(nonce: str) -> Dict[str, Any]:
    continue_option = {
        "text": {
            "type": "mrkdwn",
            "text": "Continue this session with this comment",
        },
        "value": "continue",
    }
    return {
        "type": "modal",
        "callback_id": AGENT_FEEDBACK_CALLBACK_ID,
        "private_metadata": nonce,
        "title": {"type": "plain_text", "text": "Needs changes"},
        "submit": {"type": "plain_text", "text": "Send"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "feedback_reason",
                "label": {"type": "plain_text", "text": "Reason"},
                "element": {
                    "type": "static_select",
                    "action_id": "reason",
                    "options": [
                        {"text": {"type": "plain_text", "text": label}, "value": value}
                        for value, label in FEEDBACK_REASONS
                    ],
                },
            },
            {
                "type": "input",
                "block_id": "feedback_comment",
                "optional": True,
                "label": {"type": "plain_text", "text": "What should change?"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "comment",
                    "multiline": True,
                    "max_length": FEEDBACK_COMMENT_MAX_CHARS,
                },
            },
            {
                "type": "input",
                "block_id": "feedback_continue",
                "optional": True,
                "label": {"type": "plain_text", "text": "Follow-up"},
                "element": {
                    "type": "checkboxes",
                    "action_id": "continue",
                    "options": [continue_option],
                },
            },
        ],
    }


def parse_agent_feedback_submission(view: Dict[str, Any]) -> AgentFeedbackSubmission:
    values = view.get("state", {}).get("values", {}) or {}
    reason = (
        values.get("feedback_reason", {}).get("reason", {}).get("selected_option") or {}
    ).get("value")
    comment = str(
        values.get("feedback_comment", {}).get("comment", {}).get("value") or ""
    ).strip()
    selected = (
        values.get("feedback_continue", {}).get("continue", {}).get("selected_options")
        or []
    )
    return AgentFeedbackSubmission(
        reason_category=(
            reason if reason in {value for value, _ in FEEDBACK_REASONS} else None
        ),
        comment=comment[:FEEDBACK_COMMENT_MAX_CHARS],
        continue_session=any(option.get("value") == "continue" for option in selected),
    )
