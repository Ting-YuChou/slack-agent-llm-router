"""Pi Agent runs from Slack: texts, helpers and SlackBot's agent mixin."""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from dataclasses import dataclass

from src.agent_runtime import (
    AgentRuntimeConflict,
    AgentRuntimeError,
    AgentRuntimeUnauthorized,
    is_transient_error,
)
from slack.agent_bootstrap import (
    AgentBootstrapContext,
    AgentBootstrapFetchError,
)
from slack.agent_feedback import (
    AGENT_FEEDBACK_CALLBACK_ID,
    AgentFeedbackNonce,
    build_agent_feedback_modal,
    parse_agent_feedback_submission,
)
from slack.agent_shortcut import (
    AgentShortcutNonce,
    AgentShortcutSubmission,
    build_agent_shortcut_modal,
    parse_agent_shortcut_submission,
)
from slack.formatting import (
    CHAT_FEEDBACK_ACTIONS,
    _escape_mrkdwn,
    _split_fenced_text,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


POSTED_CI_JOBS_LIMIT = 1000
AGENT_LOOKUP_ERROR_TTL_SECONDS = 10.0
UNVERIFIED_LABEL = "⚠️ 未驗證（沒有 verification profile）"
AGENT_TASK_ECHO_LIMIT = 2500
AGENT_SHORTCUT_NONCE_TTL_SECONDS = 300
# How long an expired form is remembered, so its submission gets a notice.
AGENT_SHORTCUT_NONCE_GRACE_SECONDS = 3600
SLACK_SECTION_TEXT_LIMIT = 3000
AGENT_DELIVERY_CARD_LIMIT = 3000
AGENT_DELIVERY_MAX_FILES = 15
AGENT_DELIVERY_FILES_TEXT_LIMIT = AGENT_DELIVERY_CARD_LIMIT // 2
# Jev acts on a route label only at this confidence; the discuss hint uses the same.
AGENT_DISCUSS_HINT_PROBABILITY = 0.9
GITHUB_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
AGENT_TERMINAL_STATUSES = {
    "completed",
    "rejected",
    "cancelled",
    "failed",
    "timed_out",
    "interrupted",
}


def _agent_idempotency_key(
    team_id: Optional[str], channel_id: str, request_id: str, user_id: str, op: str
) -> str:
    """Stable per Slack request, so a redelivered event cannot start a second run."""
    raw = "\n".join([team_id or "", channel_id, request_id, user_id, op])
    return "slack-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


# Next step shown after a failed run's reason, keyed by error code or cause_code.
AGENT_ERROR_HINTS = {
    "scope_violation": (
        "Ask for the change again within the profile's allowed paths, "
        "or widen allowedPaths if those files belong in scope."
    ),
    "diff_limit_exceeded": (
        "Split the request into smaller changes, each within the diff limits."
    ),
    "session_integrity_failed": (
        "The saved agent session changed after it was stored, so it was not "
        "continued; start a new thread to begin a fresh session."
    ),
    "disk_quota_exceeded": (
        "The worktree outgrew its disk quota; avoid generating large files or "
        "dependencies in the repository, then retry."
    ),
    "agent_tool_limit": (
        "Narrow the request or split it so the agent needs fewer tool calls."
    ),
    "agent_turn_limit": (
        "Narrow the request or split it so the agent can finish in fewer turns."
    ),
    "grader_timeout": (
        "The grader timed out; retry the run, and report it if it keeps happening."
    ),
    "insufficient_evidence": (
        "The grader could not confirm the change; add tests or point to how "
        "the behaviour can be checked, then retry."
    ),
    "docker_unavailable": (
        "Docker is unavailable on the runtime host; ask an operator to start it, "
        "then retry."
    ),
    "snapshot_too_large": (
        "The change was too large to snapshot for verification; split it into "
        "smaller changes or drop large generated files, then retry."
    ),
    "candidate_changed": (
        "The worktree changed while it was being verified; retry the run, "
        "and report it if it keeps happening."
    ),
    "candidate_artifact_missing": (
        "The verified change could not be found; retry the run, and ask an "
        "operator to check the runtime's artifact storage if it repeats."
    ),
    "verification_stop_unconfirmed": (
        "The verification container could not be confirmed stopped; ask an "
        "operator to check the runtime host before retrying."
    ),
    "grader_did_not_submit": (
        "The grader finished without a verdict; retry the run, and report it "
        "if it keeps happening."
    ),
    "grader_invalid_submission": (
        "The grader's verdict was malformed and rejected; retry the run, and "
        "report it if it keeps happening."
    ),
    "grader_execution_failed": (
        "The grader could not run (often a model or budget limit); retry later, "
        "or ask an operator to check the model gateway."
    ),
    "investigation_timeout": (
        "The investigation ran out of time; narrow the question or point to the "
        "relevant files, then retry."
    ),
    "investigation_missing_structured_result": (
        "The investigation ended without a structured result; retry the run, "
        "and report it if it keeps happening."
    ),
    "investigation_invalid_submission": (
        "The investigation's result was malformed and rejected; retry the run, "
        "and report it if it keeps happening."
    ),
    "investigation_execution_failed": (
        "The investigation could not run (often a model or budget limit); retry "
        "later, or ask an operator to check the model gateway."
    ),
    "investigation_reported_unknown": (
        "The investigation could not find the cause; add the error, logs or "
        "steps to reproduce, then retry."
    ),
    "investigation_stop_unconfirmed": (
        "The investigation container could not be confirmed stopped; ask an "
        "operator to check the runtime host before retrying."
    ),
    "verification_timeout": (
        "Verification ran out of time; narrow the change or ask an operator to "
        "raise the slow check's timeout, then retry."
    ),
    "verification_incomplete": (
        "A verification check never reported a result; retry the run, and ask "
        "an operator to check the verification image if it repeats."
    ),
    "delivery_expired": (
        "The delivery approval expired; rerun the request and approve it in time."
    ),
    "delivery_rejected": (
        "Delivery was rejected; rerun with updated instructions if you still "
        "want the change."
    ),
    "archive_failed": (
        "The change could not be archived, so it stays in the isolated worktree; "
        "ask an operator to recover it before retrying."
    ),
}


def _format_agent_status_line(result: Dict[str, Any]) -> str:
    """Lead every terminal agent result with an explicit outcome."""
    status = str(result.get("status") or "")
    if status not in AGENT_TERMINAL_STATUSES:
        return ""
    run_id = result.get("run_id")
    label = f"Agent run `{run_id}`" if run_id else "Agent run"
    if status == "completed":
        return f"✅ {label} *completed*"
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    code = str(error.get("code") or "").strip()
    message = str(error.get("message") or "").strip()
    reason = ": ".join(part for part in (code, message) if part)
    if not reason:
        reason = {
            "rejected": "the requested action was rejected or the approval expired",
            "cancelled": "the run was cancelled",
            "timed_out": "the run exceeded its deadline",
            "interrupted": "the runtime stopped before the run finished",
        }.get(status, "no error detail was reported")
    line = f"⚠️ {label} *{status}* — {reason}"
    cause = str(error.get("cause_code") or "").strip()
    hint = AGENT_ERROR_HINTS.get(cause) or AGENT_ERROR_HINTS.get(code)
    if hint:
        line = f"{line}\nNext step: {hint}"
    if result.get("patch_retained") is True:
        line = f"{line}\nThe undelivered patch was kept on the runtime host."
    return line


def _agent_run_started_at(run: Dict[str, Any]) -> Optional[float]:
    """The run's start in epoch seconds, from the runtime's `created_at`."""
    value = run.get("created_at")
    if not isinstance(value, str) or not value:
        return None
    try:
        started = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started.timestamp()


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


async def _show_agent_status(
    client: Any,
    tracked: Dict[str, Any],
    text: str,
    status_ts: Optional[str] = None,
) -> bool:
    """Make a Slack message the run's status message, which its monitor keeps current.

    Rewrites the message at `status_ts` when given, otherwise posts a reply in
    the run's thread. Returns False when Slack did not take the message.
    """
    try:
        if status_ts:
            if not hasattr(client, "chat_update"):
                return False
            await client.chat_update(
                channel=tracked["channel_id"], ts=status_ts, text=text
            )
        else:
            if not hasattr(client, "chat_postMessage"):
                return False
            posted = await client.chat_postMessage(
                channel=tracked["channel_id"], thread_ts=tracked["thread_ts"], text=text
            )
            status_ts = posted.get("ts") if hasattr(posted, "get") else None
    except Exception:
        logger.warning("Agent status message could not be posted", exc_info=True)
        return False
    if not isinstance(status_ts, str) or not status_ts:
        return False
    tracked["status_ts"] = status_ts
    tracked["status_text"] = text
    return True


def _int_field(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ci_repair_line(target: Dict[str, Any]) -> str:
    """Name the CI workflow, run, PR and failed jobs a repair approval is for."""
    workflow = _escape_mrkdwn(
        str(target.get("workflow") or "CI")[:100].replace("*", "")
    )
    repository = str(target.get("repository") or "")
    linkable = bool(GITHUB_REPOSITORY_PATTERN.match(repository))
    line = f"🔧 Repairing CI workflow *{workflow}*"
    run = _int_field(target.get("run"))
    if run is not None:
        line += (
            f" <https://github.com/{repository}/actions/runs/{run}|run {run}>"
            if linkable
            else f" run {run}"
        )
    attempt = _int_field(target.get("attempt"))
    if attempt is not None:
        line += f" (attempt {attempt})"
    pr = _int_field(target.get("pr"))
    if pr is not None:
        line += (
            f" on <https://github.com/{repository}/pull/{pr}|PR #{pr}>"
            if linkable
            else f" on PR #{pr}"
        )
    jobs = [
        f"`{_escape_mrkdwn(str(job)[:100].replace('`', chr(39)))}`"
        for job in target.get("failed_jobs") or []
    ]
    if not jobs:
        return f"{line}\nFailed jobs: _not reported_"
    # Keep room for the "and N more" marker so the section stays within its limit.
    budget = SLACK_SECTION_TEXT_LIMIT - len(line) - 40
    shown: List[str] = []
    for job in jobs[:10]:
        if sum(len(item) + 2 for item in shown) + len(job) > budget:
            break
        shown.append(job)
    listed = ", ".join(shown)
    if len(jobs) > len(shown):
        listed += f" _…and {len(jobs) - len(shown)} more_"
    return f"{line}\nFailed jobs: {listed}"


def _format_grader_finding(finding: Dict[str, Any]) -> str:
    location = str(finding.get("path") or "unknown")
    if finding.get("line") is not None:
        location = f"{location}:{finding.get('line')}"
    return f"• `{location}` {finding.get('reason') or ''}".rstrip()


AGENT_RUNTIME_CREDENTIALS_TEXT = (
    "Agent mode is misconfigured: the runtime rejected the bot's credentials."
)
AGENT_STATUS_DELAYED_TEXT = (
    "⏳ Agent status is delayed: the Agent runtime is not answering. "
    "I'll keep checking until the run's deadline."
)
AGENT_START_ERROR_TEXT = {
    "session_busy": "This Agent session already has a run in progress; wait for it or use `agent stop`.",
    "verification_profile_required": "Implement runs need a trusted verification profile here; use `--discuss` for a read-only run.",
    "invalid_model": "That Agent model is not allowlisted.",
    "provider_not_configured": "That Agent model's provider is not configured.",
}


AGENT_PATCH_OWNER_TEXT = "Only the user who started this Agent run can use its patch."
AGENT_PATCH_RESUME_ERROR_TEXT = {
    "session_busy": AGENT_START_ERROR_TEXT["session_busy"],
    "patch_conflict": (
        "The kept patch no longer applies cleanly to this session's worktree; "
        "download it or start a new task."
    ),
    "worktree_not_clean": (
        "This session's worktree has uncommitted changes, so the kept patch "
        "was not applied."
    ),
    "patch_truncated": (
        "The kept patch was truncated when it was archived, so it cannot be "
        "resumed; download it instead."
    ),
}


AGENT_RESULT_BLOCK_CHARS = 3000
AGENT_RESULT_BLOCKS_PER_MESSAGE = 10
AGENT_RESULT_MAX_CHARS = 60000


def _split_agent_result(text: str, limit: int = AGENT_RESULT_BLOCK_CHARS) -> List[str]:
    """Split a result into Slack section-sized chunks without dropping content.

    Chunks break on a newline when one is near the limit, and a code fence
    left open at a break is closed and reopened so each block renders alone.
    """
    if len(text) > AGENT_RESULT_MAX_CHARS:
        omitted = len(text) - AGENT_RESULT_MAX_CHARS
        text = (
            text[:AGENT_RESULT_MAX_CHARS].rstrip()
            + f"\n\n_…answer truncated: {omitted:,} more characters were not posted._"
        )
    return _split_fenced_text(text, limit)


def _agent_task_echo(task: str) -> str:
    """Quote a submitted task back so a refused form does not lose it."""
    task = task.replace("```", "`\u200b``")
    if len(task) > AGENT_TASK_ECHO_LIMIT:
        task = task[:AGENT_TASK_ECHO_LIMIT] + "…"
    return f"\nYour task was:\n```{task}```"


def _is_complete_agent_run_record(record: Any) -> bool:
    """Whether a persisted run record names its run, session, thread and owner."""
    return isinstance(record, dict) and all(
        record.get(field)
        for field in (
            "run_id",
            "session_id",
            "channel_id",
            "thread_ts",
            "owner_user_id",
        )
    )


def _is_agent_command_text(command_text: str) -> bool:
    """Whether an `/llm` or `!llm` payload is an `agent` command."""
    parts = command_text.split(maxsplit=1)
    return bool(parts) and parts[0].lower() == "agent"


def _agent_auth_error_text(exc: AgentRuntimeError, owner_text: str) -> str:
    """Tell a bad bot token (`unauthorized`) apart from a non-owner (`*_forbidden`)."""
    if exc.code == "unauthorized":
        return AGENT_RUNTIME_CREDENTIALS_TEXT
    return owner_text


AGENT_FEEDBACK_RETRY_TEXT = "Result feedback could not be recorded. Please try again."
AGENT_FEEDBACK_UNAVAILABLE_TEXT = (
    "Result feedback was not recorded: this Agent runtime stores feedback "
    "through telemetry, which is turned off (PI_AGENT_TELEMETRY_ENABLED)."
)


def _agent_feedback_error_text(exc: AgentRuntimeError) -> str:
    """Retrying cannot help when the runtime has no feedback capture at all."""
    if exc.code == "feedback_unavailable":
        return AGENT_FEEDBACK_UNAVAILABLE_TEXT
    return AGENT_FEEDBACK_RETRY_TEXT


@dataclass(frozen=True)
class AgentFollowupFlags:
    text: str
    task_mode: Optional[str] = None
    workflow_mode: Optional[str] = None
    consumed: bool = False
    error: Optional[str] = None


class SlackBotAgentMixin:
    """SlackBot's Pi Agent runs."""

    async def _monitor_ci_inbox(self):
        while self.running:
            await self._poll_ci_inbox_once()
            await asyncio.sleep(5)

    async def _poll_ci_inbox_once(self):
        runtime = self.agent_runtime_client
        if runtime is None:
            return
        try:
            jobs = await runtime.ci_repairs()
        except AgentRuntimeError:
            logger.warning("CI repair inbox unavailable")
            return
        for job in jobs:
            try:
                await self._post_ci_inbox_job(job)
            except Exception:
                logger.warning(
                    "CI repair Slack notification failed for %s", job.get("run_id")
                )

    async def _post_ci_inbox_job(self, job: Dict[str, Any]):
        run_id = job["run_id"]
        if run_id in self.agent_runs:
            await self._claim_ci_job(run_id)
            return
        if not await self._claim_ci_job(run_id):
            return
        try:
            message = await self.web_client.chat_postMessage(
                channel=job["channel_id"],
                client_msg_id=run_id,
                text=f"CI repair `{run_id}` requires the repository owner's review.",
            )
        except Exception:
            await self._release_ci_job(run_id)
            raise
        tracked = self._track_agent_run(
            job,
            team_id=job["team_id"],
            channel_id=job["channel_id"],
            thread_ts=message["ts"],
            owner_user_id=job["owner_user_id"],
            mode="ci_repair",
        )
        self._start_agent_monitor(tracked)

    async def _claim_ci_job(self, run_id: str) -> bool:
        """Claim a CI repair job for posting; False when it was already claimed."""
        if run_id in self.posted_ci_jobs:
            return False
        # Claim synchronously so the inbox and /llm agent repair-ci cannot
        # both pass this check before either one awaits.
        self.posted_ci_jobs[run_id] = None
        evicted = []
        while len(self.posted_ci_jobs) > POSTED_CI_JOBS_LIMIT:
            oldest = next(iter(self.posted_ci_jobs))
            self.posted_ci_jobs.pop(oldest)
            evicted.append(oldest)
        if self._uses_granular_redis_state():
            await self.state_store.add_posted_ci_job(run_id)
            for oldest in evicted:
                await self.state_store.remove_posted_ci_job(oldest)
        else:
            await self._persist_state()
        return True

    async def _release_ci_job(self, run_id: str):
        self.posted_ci_jobs.pop(run_id, None)
        if self._uses_granular_redis_state():
            await self.state_store.remove_posted_ci_job(run_id)
        else:
            await self._persist_state()

    async def _reject_non_owner_agent_message(self, event: Dict[str, Any]) -> bool:
        """Turn away anyone but the owner in an Agent thread; True if turned away."""
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts")
        if not thread_ts:
            return False
        await self._look_up_agent_thread(event)
        tracked = self.agent_threads.get(self._thread_key(channel_id, thread_ts))
        if not tracked or tracked.get("owner_user_id") == event.get("user"):
            return False
        if self._claim_message(event) and hasattr(
            self.web_client, "chat_postEphemeral"
        ):
            await self.web_client.chat_postEphemeral(
                channel=channel_id,
                user=event.get("user"),
                text=(
                    "Only the user who started this Agent session can send "
                    "coding prompts in this thread."
                ),
            )
        return True

    async def _restored_agent_mode(self, session: Dict[str, Any]) -> str:
        """The task mode of a thread known only to the runtime.

        A follow-up inherits discuss from its thread, so a mode nobody can
        name restores as discuss: the owner upgrades with `--implement`
        instead of a read-only thread silently gaining write access.
        """
        mode = session.get("task_mode")
        if mode in {"discuss", "implement"}:
            return str(mode)
        runtime = self.agent_runtime_client
        run_id = session.get("run_id")
        if run_id and runtime is not None and hasattr(runtime, "get_run"):
            try:
                run = await runtime.get_run(run_id)
            except AgentRuntimeError:
                run = None
            if isinstance(run, dict):
                # Runs created before task_mode existed were implement runs.
                mode = run.get("task_mode") or "implement"
                if mode in {"discuss", "implement"}:
                    return str(mode)
        return "discuss"

    async def _look_up_agent_thread(self, event: Dict[str, Any]) -> None:
        """Ask the runtime about a thread not known locally as an Agent thread.

        A thread already known to be a chat thread, or whose lookup failed in
        the last few seconds, is not asked about again (S7): a follow-up must
        not wait on a runtime that is slow or down. A failed lookup counts as
        "not an Agent thread".
        """
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts")
        if not channel_id or not thread_ts:
            return
        if self._is_agent_thread(channel_id, thread_ts):
            return
        key = self._thread_key(channel_id, thread_ts)
        if key in self._agent_lookup_skips:
            return
        await self._restore_agent_thread(event)
        if self._is_agent_thread(channel_id, thread_ts):
            return
        if key not in self._agent_lookup_skips:
            # The runtime has no session for the thread: a chat thread.
            self._agent_lookup_skips[key] = "chat"

    def _remember_chat_thread(
        self, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> None:
        """Record a thread the bot started in chat, so no lookup is needed."""
        if channel_id and thread_ts:
            self._agent_lookup_skips[self._thread_key(channel_id, thread_ts)] = "chat"

    async def _restore_agent_thread(self, event: Dict[str, Any]):
        await self._adopt_agent_session(
            self._extract_team_id(event) or "unknown-team",
            event.get("channel"),
            event.get("thread_ts"),
        )

    async def _adopt_agent_session(
        self, team_id: str, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Track the runtime's open session for a thread and watch its active run.

        Returns the tracked record, or None when the runtime has no open session
        for the thread or cannot be asked. A run this bot already follows keeps
        its Slack-side state.
        """
        runtime = self.agent_runtime_client
        if runtime is None or not hasattr(runtime, "lookup_session"):
            return None
        try:
            session = await runtime.lookup_session(team_id, channel_id, thread_ts)
        except AgentRuntimeError:
            if channel_id and thread_ts:
                self._agent_lookup_skips.set(
                    self._thread_key(channel_id, thread_ts),
                    "error",
                    ttl_seconds=AGENT_LOOKUP_ERROR_TTL_SECONDS,
                )
            return None
        if not session:
            return None
        known = self.agent_runs.get(session.get("run_id") or "")
        if known is not None:
            if session.get("active"):
                self._start_agent_monitor(known)
            return known
        mode = await self._restored_agent_mode(session)
        if session.get("run_id"):
            tracked = self._track_agent_run(
                session,
                team_id=team_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
                owner_user_id=session["owner_user_id"],
                mode=mode,
            )
            if session.get("active"):
                self._start_agent_monitor(tracked)
            else:
                # Without Slack-side state we cannot tell whether the result
                # was posted; never repost it on a later restart.
                tracked["result_posted"] = True
            return tracked
        tracked = {
            "session_id": session["session_id"],
            "run_id": None,
            "team_id": team_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "owner_user_id": session["owner_user_id"],
            "mode": mode,
        }
        self._set_agent_thread(channel_id, thread_ts, tracked)
        self._remember_agent_session(tracked)
        self._mark_thread_active(channel_id, thread_ts)
        return tracked

    def _is_agent_thread(
        self, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> bool:
        if not channel_id or not thread_ts:
            return False
        return self._thread_key(channel_id, thread_ts) in self.agent_threads

    def _track_agent_run(
        self,
        accepted: Dict[str, Any],
        *,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        owner_user_id: str,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        previous = self.agent_runs.get(accepted["run_id"]) or {}
        tracked = {
            "session_id": accepted["session_id"],
            "run_id": accepted["run_id"],
            "team_id": team_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "owner_user_id": owner_user_id,
            "mode": mode,
            "posted_approval_ids": list(previous.get("posted_approval_ids", [])),
            "result_posted": False,
            # Times the progress line when the runtime reports no created_at.
            "tracked_at": self.agent_progress_clock(),
        }
        if previous.get("approval_cards"):
            tracked["approval_cards"] = dict(previous["approval_cards"])
        for field in ("status_ts", "status_text"):
            if previous.get(field):
                tracked[field] = previous[field]
        self._register_agent_run(tracked)
        return tracked

    def _set_agent_thread(
        self, channel_id: str, thread_ts: str, tracked: Dict[str, Any]
    ) -> None:
        """Track a thread as an Agent thread; it is no longer a chat thread."""
        key = self._thread_key(channel_id, thread_ts)
        self.agent_threads[key] = tracked
        self._agent_lookup_skips.pop(key, None)

    def _register_agent_run(self, tracked: Dict[str, Any]) -> None:
        channel_id = tracked["channel_id"]
        thread_ts = tracked["thread_ts"]
        self._set_agent_thread(channel_id, thread_ts, tracked)
        self.agent_runs[tracked["run_id"]] = tracked
        self._remember_agent_session(tracked)
        self._mark_thread_active(channel_id, thread_ts)

    def _agent_sessions_of(self, user_id: str) -> List[Dict[str, Any]]:
        """The owner's tracked sessions, one per thread, the latest last."""
        sessions = self.agent_user_runs.get(user_id)
        if isinstance(sessions, dict):
            # One record per owner, the shape this map held before it kept them all.
            sessions = self.agent_user_runs[user_id] = [sessions]
        return sessions or []

    def _remember_agent_session(self, tracked: Dict[str, Any]) -> None:
        """Make a run its owner's latest session, replacing its thread's older one."""
        thread = (tracked.get("channel_id"), tracked.get("thread_ts"))
        self.agent_user_runs[tracked["owner_user_id"]] = [
            other
            for other in self._agent_sessions_of(tracked["owner_user_id"])
            if (other.get("channel_id"), other.get("thread_ts")) != thread
        ] + [tracked]

    @staticmethod
    def _agent_run_record(tracked: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "run_id": tracked["run_id"],
            "session_id": tracked.get("session_id"),
            "team_id": tracked.get("team_id"),
            "channel_id": tracked.get("channel_id"),
            "thread_ts": tracked.get("thread_ts"),
            "owner_user_id": tracked.get("owner_user_id"),
            "mode": tracked.get("mode"),
            "posted_approval_ids": list(tracked.get("posted_approval_ids", [])),
            "result_posted": bool(tracked.get("result_posted", False)),
        }
        # The root message the monitor rewrites with progress, when there is one.
        for field in ("status_ts", "status_text"):
            if isinstance(tracked.get(field), str) and tracked[field]:
                record[field] = tracked[field]
        # Message ts of each approval card still showing buttons, by approval_id.
        cards = tracked.get("approval_cards")
        if isinstance(cards, dict) and cards:
            record["approval_cards"] = {
                str(key): str(value) for key, value in cards.items()
            }
        return record

    async def _persist_agent_run(self, tracked: Dict[str, Any]):
        """Persist one tracked run and drop finished runs it superseded."""
        superseded = [
            run_id
            for run_id, other in self.agent_runs.items()
            if run_id != tracked["run_id"]
            and other.get("result_posted")
            and other.get("channel_id") == tracked["channel_id"]
            and other.get("thread_ts") == tracked["thread_ts"]
        ]
        for run_id in superseded:
            self.agent_runs.pop(run_id, None)

        if self._uses_granular_redis_state():
            await self.state_store.persist_agent_run_state(
                tracked["run_id"], self._agent_run_record(tracked)
            )
            for run_id in superseded:
                await self.state_store.delete_agent_run_state(run_id)
            return

        await self._persist_state()

    async def _forget_agent_run(self, tracked: Dict[str, Any]):
        """Stop tracking a run in memory and in the configured state backend."""
        self._forget_agent_session(tracked)
        if self._uses_granular_redis_state():
            if tracked.get("run_id"):
                await self.state_store.delete_agent_run_state(tracked["run_id"])
            return

        await self._persist_state()

    async def _agent_run_home(
        self, accepted: Dict[str, Any], owner_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """A Slack record placing a run: its own, or one of its session's.

        A session lives in one thread. The runtime keeps no Slack thread for a
        run, so tracked state answers, then the shared Redis state another bot
        process may have written; a record found there is tracked from now on,
        so its posted approvals are not posted again.
        """
        known = self.agent_runs.get(accepted["run_id"])
        if known is None:
            known = next(
                (
                    tracked
                    for tracked in reversed(self._agent_sessions_of(owner_user_id))
                    if tracked.get("session_id") == accepted.get("session_id")
                ),
                None,
            )
        if known is None and self._uses_granular_redis_state():
            record = await self.state_store.load_agent_run_state(accepted["run_id"])
            if _is_complete_agent_run_record(record):
                known = self._agent_run_record(record)
                self._register_agent_run(known)
        return known

    def _agent_run_for(
        self,
        user_id: str,
        channel_id: Optional[str],
        thread_ts: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if channel_id and thread_ts:
            tracked = self.agent_threads.get(self._thread_key(channel_id, thread_ts))
            if tracked and tracked.get("owner_user_id") == user_id:
                return tracked
        sessions = self._agent_sessions_of(user_id)
        return sessions[-1] if sessions else None

    def _agent_cap_message(
        self,
        user_id: str,
        channel_id: Optional[str],
        thread_ts: Optional[str],
    ) -> Optional[str]:
        """Why the owner cannot start another agent now, or None if they can.

        A run counts while its monitor is watching it: a monitor that gave up
        must not lock its owner out. A follow-up in a thread reuses that
        thread's session, so runs and starts in the same thread never count.
        CI repair runs are opened by the CI inbox, not by the owner, and do
        not count either.
        """
        this_thread = (
            self._thread_key(channel_id, thread_ts)
            if channel_id and thread_ts
            else None
        )
        active = [
            tracked
            for tracked in self.agent_runs.values()
            if tracked.get("owner_user_id") == user_id
            and not tracked.get("result_posted")
            and self._is_agent_run_monitored(tracked.get("run_id"))
            and tracked.get("mode") != "ci_repair"
            and (
                this_thread is None
                or self._thread_key(
                    tracked.get("channel_id", ""), tracked.get("thread_ts", "")
                )
                != this_thread
            )
        ]
        pending = [
            key
            for key in self.agent_pending_starts.get(user_id, {}).values()
            if key is None or key != this_thread
        ]
        if len(active) + len(pending) < self.agent_max_active_runs_per_user:
            return None
        if active:
            run = active[0]
            return (
                f"You already have an active Agent run `{run['run_id']}` in "
                f"<#{run['channel_id']}>. Wait for it to finish, or stop it with "
                "`/llm agent stop`, then start the new task."
            )
        return (
            "You already have an Agent run starting. Wait for it to open its "
            "thread, then start the new task."
        )

    def _is_agent_run_monitored(self, run_id: Optional[str]) -> bool:
        task = self.agent_monitor_tasks.get(run_id) if run_id else None
        return task is not None and not task.done()

    def _reserve_agent_start(
        self, user_id: str, channel_id: Optional[str], thread_ts: Optional[str]
    ) -> str:
        token = uuid.uuid4().hex
        self.agent_pending_starts.setdefault(user_id, {})[token] = (
            self._thread_key(channel_id, thread_ts)
            if channel_id and thread_ts
            else None
        )
        return token

    def _release_agent_start(self, user_id: str, token: str) -> None:
        pending = self.agent_pending_starts.get(user_id)
        if pending is None:
            return
        pending.pop(token, None)
        if not pending:
            self.agent_pending_starts.pop(user_id, None)

    def _forget_agent_session(self, tracked: Dict[str, Any]):
        self.agent_runs.pop(tracked.get("run_id"), None)
        self.agent_threads.pop(
            self._thread_key(
                tracked.get("channel_id", ""), tracked.get("thread_ts", "")
            ),
            None,
        )
        owner_user_id = tracked.get("owner_user_id") or ""
        sessions = [
            other
            for other in self._agent_sessions_of(owner_user_id)
            if other is not tracked
        ]
        if sessions:
            self.agent_user_runs[owner_user_id] = sessions
        else:
            self.agent_user_runs.pop(owner_user_id, None)

    @staticmethod
    def _agent_approval_blocks(
        approval: Dict[str, Any], ci: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Build approval card blocks that each stay within Slack's section limit.

        `ci` is a CI repair's target; the card then names the job it repairs.
        """
        # Titles name agent-chosen paths and commands, which must not ping or link.
        title = _escape_mrkdwn(str(approval.get("title") or "Approval required")[:200])
        # Keep embedded fences from closing the code block early.
        detail = str(approval.get("detail") or "").replace("```", "`\u200b``")
        fence = "```"
        text = f"{fence}{detail}{fence}" if detail else "_No detail provided._"
        if len(text) > SLACK_SECTION_TEXT_LIMIT:
            marker_reserve = len(
                f"\n_Truncated: showing {len(detail)}/{len(detail)} chars._"
            )
            shown = SLACK_SECTION_TEXT_LIMIT - 2 * len(fence) - marker_reserve
            text = (
                f"{fence}{detail[:shown]}{fence}\n"
                f"_Truncated: showing {shown}/{len(detail)} chars._"
            )

        expiry = "This request expires soon."
        expires_at = approval.get("expires_at")
        if isinstance(expires_at, str) and expires_at:
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                expiry = (
                    f"This request expires at <!date^{int(parsed.timestamp())}"
                    f"^{{time}}|{expires_at}>."
                )
            except ValueError:
                pass

        repairing = (
            [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": _ci_repair_line(ci)},
                }
            ]
            if isinstance(ci, dict)
            else []
        )
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
            *repairing,
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Only the session owner can decide. {expiry}",
                    }
                ],
            },
        ]

    @staticmethod
    def _agent_delivery_blocks(approval: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the Deliver/Discard card; all of its text stays within 3000 chars."""
        delivery = approval.get("delivery")
        delivery = delivery if isinstance(delivery, dict) else {}
        files = [str(item) for item in delivery.get("files") or []]
        # The list takes at most half the card so the stat and diff still fit.
        file_lines: List[str] = []
        used = 0
        for index, name in enumerate(files[:AGENT_DELIVERY_MAX_FILES]):
            line = f"• `{name[:200]}`"
            rest = len(files) - index - 1
            marker = len(f"\n_…and {len(files)} more._") if rest else 0
            if used + len(line) + 1 + marker > AGENT_DELIVERY_FILES_TEXT_LIMIT:
                break
            file_lines.append(line)
            used += len(line) + 1
        if len(files) > len(file_lines):
            file_lines.append(f"_…and {len(files) - len(file_lines)} more._")
        title = f"*Deliver {len(files)} changed file{'s' if len(files) != 1 else ''}?*"
        files_text = "\n".join(file_lines) or "_No file list provided._"
        checks = (
            f"Verification: *{delivery.get('verification_status') or 'unknown'}*"
            f" · Grader: *{delivery.get('grader_status') or 'unknown'}*"
        )

        expiry = "This request expires soon."
        expires_at = approval.get("expires_at")
        if isinstance(expires_at, str) and expires_at:
            try:
                parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                expiry = (
                    f"Expires at <!date^{int(parsed.timestamp())}"
                    f"^{{date_short_pretty}} {{time}}|{expires_at}>; "
                    "unanswered changes are discarded."
                )
            except ValueError:
                pass
        footer = f"Only the session owner can deliver. {expiry}"

        fence = "```"
        # Spend whatever the fixed parts leave on the stat and the diff excerpt.
        budget = AGENT_DELIVERY_CARD_LIMIT - sum(
            len(part) for part in (title, files_text, checks, footer)
        )
        extras = []
        for raw in (delivery.get("stat"), delivery.get("excerpt")):
            text = str(raw or "").replace(fence, "`\u200b``")
            if not text or budget <= 2 * len(fence) + 20:
                continue
            room = budget - 2 * len(fence)
            if len(text) > room:
                text = text[: room - 2] + "\n…"
            extras.append(f"{fence}{text}{fence}")
            budget -= len(extras[-1])

        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": title}},
            {"type": "section", "text": {"type": "mrkdwn", "text": files_text}},
            *(
                {"type": "section", "text": {"type": "mrkdwn", "text": text}}
                for text in extras
            ),
            {"type": "section", "text": {"type": "mrkdwn", "text": checks}},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": footer}],
            },
        ]

    def _reconcile_agent_runs(self) -> None:
        """Resume monitoring every restored run whose result was never posted."""
        if self.agent_runtime_client is None:
            return
        for tracked in list(self.agent_runs.values()):
            if tracked.get("run_id") and not tracked.get("result_posted"):
                self._start_agent_monitor(tracked)

    def _start_agent_monitor(self, tracked: Dict[str, Any]) -> asyncio.Task:
        """Spawn a monitor for a run unless one is already running."""
        run_id = tracked["run_id"]
        task = self.agent_monitor_tasks.get(run_id)
        if task is not None and not task.done():
            return task
        task = self._spawn_background_task(
            self._monitor_agent_run(tracked), f"pi_agent_monitor:{run_id}"
        )
        self.agent_monitor_tasks[run_id] = task

        def release(done: asyncio.Task) -> None:
            if self.agent_monitor_tasks.get(run_id) is done:
                self.agent_monitor_tasks.pop(run_id, None)

        task.add_done_callback(release)
        return task

    async def _monitor_agent_run(self, tracked: Dict[str, Any]):
        run_id = tracked["run_id"]
        if run_id in self.agent_monitoring:
            return
        self.agent_monitoring.add(run_id)
        try:
            await self._watch_agent_run(tracked)
        finally:
            self.agent_monitoring.discard(run_id)

    async def _watch_agent_run(self, tracked: Dict[str, Any]):
        """Post each new approval card and the final result exactly once."""
        runtime = self.agent_runtime_client
        if runtime is None or self.web_client is None:
            return
        await self._persist_agent_run(tracked)
        # The runtime pauses a run's deadline while it waits for approval, so
        # each approval restarts the window the monitor keeps reattaching in.
        deadline = self.agent_monitor_clock() + self.agent_run_deadline_seconds
        failures = 0
        delay_posted = False
        while True:
            try:
                run = await runtime.wait_for_update(
                    tracked["run_id"],
                    on_progress=lambda update: self._update_agent_progress(
                        tracked, update
                    ),
                    # The status message is rewritten at most this often anyway.
                    refresh_seconds=self.agent_progress_interval_seconds,
                )
            except AgentRuntimeError as exc:
                if exc.status == 404:
                    logger.info(
                        "Agent run %s is gone; forgetting it", tracked["run_id"]
                    )
                    await self._forget_agent_run(tracked)
                    return
                remaining = deadline - self.agent_monitor_clock()
                if is_transient_error(exc) and remaining > 0:
                    logger.warning(
                        "Agent runtime unreachable for run %s; reattaching: %s",
                        tracked["run_id"],
                        exc,
                    )
                    if not delay_posted:
                        delay_posted = True
                        await self.web_client.chat_postMessage(
                            channel=tracked["channel_id"],
                            thread_ts=tracked["thread_ts"],
                            text=AGENT_STATUS_DELAYED_TEXT,
                        )
                    ceiling = min(
                        self.agent_monitor_reattach_max_seconds,
                        self.agent_monitor_reattach_seconds * 2**failures,
                    )
                    failures += 1
                    await self.agent_monitor_sleep(
                        min(remaining, self.agent_monitor_random() * ceiling)
                    )
                    continue
                logger.warning(
                    "Agent run monitor failed for %s: %s", tracked["run_id"], exc
                )
                await self.web_client.chat_postMessage(
                    channel=tracked["channel_id"],
                    thread_ts=tracked["thread_ts"],
                    text="Agent status could not be retrieved; Chat mode was not used as fallback.",
                )
                return
            failures = 0
            await self._close_agent_approval_cards(tracked, run)
            if run.get("status") == "awaiting_approval":
                deadline = self.agent_monitor_clock() + self.agent_run_deadline_seconds
                await self._update_agent_progress(tracked, run)
                await self._post_agent_approval(tracked, run)
                await asyncio.sleep(self.agent_monitor_poll_seconds)
                continue
            if not tracked.get("result_posted"):
                await self._update_agent_progress(tracked, run, force=True)
                await self._post_agent_result(tracked, run)
                tracked["result_posted"] = True
                await self._persist_agent_run(tracked)
            return

    @staticmethod
    def _agent_progress_line(run: Dict[str, Any], elapsed: float) -> str:
        try:
            cost = float(run.get("estimated_cost_usd") or 0)
        except (TypeError, ValueError):
            cost = 0.0
        line = (
            f"Phase: *{run.get('phase') or run.get('status') or 'unknown'}*"
            f" · Tools: {int(run.get('tool_count') or 0)}"
            f" · Turns: {int(run.get('turn_count') or 0)}"
            f" · Cost: ${cost:.3f}"
            f" · Elapsed: {_format_elapsed(elapsed)}"
        )
        if run.get("last_tool"):
            line += f" · Last tool: `{str(run['last_tool'])[:80]}`"
        return line

    @staticmethod
    def _ci_repair_target(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The CI run a repair job answers to; only CI repair runs carry one."""
        target = run.get("target")
        return target if isinstance(target, dict) else None

    @staticmethod
    def _agent_discuss_hint(run: Dict[str, Any]) -> Optional[str]:
        """Suggest --discuss when Jev is confident an implement run is read-only.

        Only a hint: the run keeps its task mode, permissions and approvals.
        """
        routing = run.get("routing")
        if run.get("task_mode") != "implement" or not isinstance(routing, dict):
            return None
        probability = routing.get("probability")
        if (
            # Shadow routing is observed, never acted on.
            routing.get("source") != "jev"
            or routing.get("label") != "read_only"
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or probability < AGENT_DISCUSS_HINT_PROBABILITY
        ):
            return None
        return (
            f"💡 This request looks read-only (route `read_only`, p={probability:.2f})."
            " `--discuss` may fit better: it uses read-only tools, with no"
            " verification or commit. This run continues as an implement run."
        )

    async def _update_agent_progress(
        self, tracked: Dict[str, Any], run: Dict[str, Any], *, force: bool = False
    ) -> None:
        """Rewrite the run's status message, at most once per progress interval."""
        status_ts = tracked.get("status_ts")
        if not status_ts or not hasattr(self.web_client, "chat_update"):
            return
        now = self.agent_progress_clock()
        last = tracked.get("progress_updated_at")
        if (
            not force
            and last is not None
            and now - last < self.agent_progress_interval_seconds
        ):
            return
        header = (
            tracked.get("status_text")
            or f"🛠️ Pi coding agent run `{run.get('run_id')}`."
        )
        started = _agent_run_started_at(run)
        if started is None:
            # A run restored after a bot restart is timed from its first update.
            started = tracked.setdefault("tracked_at", now)
        text = f"{header}\n{self._agent_progress_line(run, now - started)}"
        hint = self._agent_discuss_hint(run)
        if hint:
            text = f"{text}\n{hint}"
        if text == tracked.get("progress_text"):
            return
        try:
            await self.web_client.chat_update(
                channel=tracked["channel_id"], ts=status_ts, text=text
            )
        except Exception:
            logger.warning("Agent status message could not be updated", exc_info=True)
            return
        tracked["progress_updated_at"] = now
        tracked["progress_text"] = text

    async def _post_agent_approval(
        self, tracked: Dict[str, Any], run: Dict[str, Any]
    ) -> None:
        approval = next(
            (
                event
                for event in reversed(run.get("events", []))
                if event.get("type") == "approval"
            ),
            None,
        )
        posted = tracked.setdefault("posted_approval_ids", [])
        if not approval or approval.get("approval_id") in posted:
            return
        value = json.dumps(
            {
                "run_id": run["run_id"],
                "session_id": run["session_id"],
                "approval_id": approval["approval_id"],
            },
            separators=(",", ":"),
        )
        kind = approval.get("kind")
        labels = {
            "publish": ("Publish repair Draft PR", "Keep local result"),
            "delivery": ("Deliver", "Discard"),
        }.get(str(kind), ("Approve", "Reject"))
        card = (
            self._agent_delivery_blocks(approval)
            if kind == "delivery"
            else self._agent_approval_blocks(approval, self._ci_repair_target(run))
        )
        text = (
            "Ready to deliver: review the changes"
            if kind == "delivery"
            else "Approval required: "
            + _escape_mrkdwn(str(approval.get("title") or "Agent action"))
        )
        owner = tracked.get("owner_user_id")
        if owner:
            # Notify the owner: a slash-command thread does not include them.
            first = card[0]
            card[0] = {
                **first,
                "text": {
                    **first["text"],
                    "text": f"<@{owner}> {first['text']['text']}",
                },
            }
            text = f"{text} (<@{owner}>)"
        response = await self.web_client.chat_postMessage(
            channel=tracked["channel_id"],
            thread_ts=tracked["thread_ts"],
            text=text,
            blocks=[
                *card,
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "pi_agent_approve",
                            "text": {"type": "plain_text", "text": labels[0]},
                            "style": "primary",
                            "value": value,
                        },
                        {
                            "type": "button",
                            "action_id": "pi_agent_reject",
                            "text": {"type": "plain_text", "text": labels[1]},
                            "style": "danger",
                            "value": value,
                        },
                    ],
                },
            ],
        )
        posted.append(approval["approval_id"])
        card_ts = response.get("ts") if hasattr(response, "get") else None
        if isinstance(card_ts, str) and card_ts:
            tracked.setdefault("approval_cards", {})[approval["approval_id"]] = card_ts
        await self._persist_agent_run(tracked)

    @staticmethod
    def _closed_agent_card_blocks(
        blocks: List[Dict[str, Any]], outcome: str
    ) -> List[Dict[str, Any]]:
        """Keep a card's sections; its buttons and expiry footer give way to the outcome."""
        kept = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "section"
        ]
        return [
            *kept,
            {"type": "context", "elements": [{"type": "mrkdwn", "text": outcome}]},
        ]

    @staticmethod
    def _agent_card_outcome(verdict: str, user_id: Optional[str] = None) -> str:
        now = int(time.time())
        when = f"<!date^{now}^{{date_short_pretty}} {{time}}|{now}>"
        by = f" by <@{user_id}>" if user_id else ""
        return f"{verdict}{by} at {when}"

    async def _update_agent_card(
        self,
        channel_id: str,
        ts: str,
        blocks: List[Dict[str, Any]],
        outcome: str,
    ) -> None:
        if not hasattr(self.web_client, "chat_update"):
            return
        try:
            await self.web_client.chat_update(
                channel=channel_id,
                ts=ts,
                text=outcome,
                blocks=self._closed_agent_card_blocks(blocks, outcome),
            )
        except Exception:
            logger.warning("Agent approval card could not be updated", exc_info=True)

    async def _close_agent_approval_cards(
        self, tracked: Dict[str, Any], run: Dict[str, Any]
    ) -> None:
        """Replace the buttons of every card the run has moved past."""
        cards = tracked.get("approval_cards")
        if not isinstance(cards, dict) or not cards:
            return
        events = [event for event in run.get("events") or [] if isinstance(event, dict)]
        pending = None
        if run.get("status") == "awaiting_approval":
            pending = next(
                (
                    event.get("approval_id")
                    for event in reversed(events)
                    if event.get("type") == "approval"
                ),
                None,
            )
        closed = False
        for approval_id, ts in list(cards.items()):
            if approval_id == pending:
                continue
            approval = next(
                (
                    event
                    for event in events
                    if event.get("type") == "approval"
                    and event.get("approval_id") == approval_id
                ),
                {},
            )
            decision = next(
                (
                    event.get("decision")
                    for event in events
                    if event.get("type") == "approval_decision"
                    and event.get("approval_id") == approval_id
                ),
                None,
            )
            verdict = {"approved": "Approved", "rejected": "Rejected"}.get(
                str(decision)
            )
            outcome = self._agent_card_outcome(verdict) if verdict else "Expired"
            blocks = (
                self._agent_delivery_blocks(approval)
                if approval.get("kind") == "delivery"
                else self._agent_approval_blocks(approval, self._ci_repair_target(run))
                if approval
                else []
            )
            cards.pop(approval_id, None)
            closed = True
            await self._update_agent_card(tracked["channel_id"], ts, blocks, outcome)
        if closed:
            await self._persist_agent_run(tracked)

    async def _post_agent_result(
        self, tracked: Dict[str, Any], run: Dict[str, Any]
    ) -> None:
        # Final answers are posted whole: split into section-sized blocks
        # across as many thread messages as needed instead of cut at a limit.
        formatted = self.message_handler._format_agent_response(run)
        if tracked.get("owner_user_id"):
            formatted = f"<@{tracked['owner_user_id']}> {formatted}"
        chunks = _split_agent_result(formatted) or [formatted]
        messages = [
            chunks[offset : offset + AGENT_RESULT_BLOCKS_PER_MESSAGE]
            for offset in range(0, len(chunks), AGENT_RESULT_BLOCKS_PER_MESSAGE)
        ]
        actions: List[Dict[str, Any]] = []
        # Hide the buttons only when the runtime says it cannot store feedback.
        cached_capture = getattr(
            getattr(self, "agent_runtime_client", None), "cached_feedback_capture", None
        )
        feedback_capture = cached_capture() if callable(cached_capture) else None
        if (
            run.get("status") in AGENT_TERMINAL_STATUSES
            and self.agent_config.get("feedback_enabled", False)
            and feedback_capture is not False
        ):
            value = json.dumps(
                {
                    "run_id": run["run_id"],
                    "team_id": tracked.get("team_id"),
                    "channel_id": tracked["channel_id"],
                    "thread_ts": tracked["thread_ts"],
                }
            )
            actions = [
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": action_id,
                            "text": {"type": "plain_text", "text": label},
                            "value": value,
                        }
                        for action_id, label in [
                            ("pi_agent_accept_result", "Accept result"),
                            ("pi_agent_needs_changes", "Needs changes"),
                        ]
                    ],
                },
            ]
        if (
            run.get("status") in AGENT_TERMINAL_STATUSES
            and run.get("status") != "completed"
            and run.get("patch_retained") is True
        ):
            patch_value = json.dumps(
                {
                    "run_id": run["run_id"],
                    "session_id": tracked.get("session_id") or run.get("session_id"),
                    "team_id": tracked.get("team_id"),
                    "channel_id": tracked["channel_id"],
                    "thread_ts": tracked["thread_ts"],
                }
            )
            actions.insert(
                0,
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": action_id,
                            "text": {"type": "plain_text", "text": label},
                            "value": patch_value,
                        }
                        for action_id, label in [
                            ("pi_agent_download_patch", "Download patch"),
                            ("pi_agent_resume_from_patch", "Resume from patch"),
                        ]
                    ],
                },
            )
        for index, message_chunks in enumerate(messages):
            blocks: List[Dict[str, Any]] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": chunk}}
                for chunk in message_chunks
            ]
            if index == len(messages) - 1:
                blocks.extend(actions)
            await self.web_client.chat_postMessage(
                channel=tracked["channel_id"],
                thread_ts=tracked["thread_ts"],
                text="\n".join(message_chunks),
                blocks=blocks,
            )

    async def _open_agent_feedback_modal(self, payload: Dict[str, Any]) -> bool:
        """Open the Needs-changes modal; False lets the click record a bare verdict."""
        actions = payload.get("actions") or []
        if not actions or actions[0].get("action_id") != "pi_agent_needs_changes":
            return False
        action = actions[0]
        trigger_id = payload.get("trigger_id")
        user_id = payload.get("user", {}).get("id")
        click_id = str(action.get("action_ts") or "")
        try:
            value = json.loads(action.get("value", "{}"))
            run_id = value["run_id"]
        except (KeyError, TypeError, ValueError):
            return False
        team = payload.get("team")
        team_id = value.get("team_id") or (
            team.get("id") if isinstance(team, dict) else None
        )
        channel_id = value.get("channel_id") or payload.get("channel", {}).get("id")
        thread_ts = value.get("thread_ts") or (payload.get("container") or {}).get(
            "thread_ts"
        )
        if not all((trigger_id, user_id, click_id, channel_id, thread_ts)):
            return False

        now = time.time()
        self.agent_feedback_nonces = {
            key: nonce
            for key, nonce in self.agent_feedback_nonces.items()
            if nonce.expires_at > now
        }
        nonce = uuid.uuid4().hex
        self.agent_feedback_nonces[nonce] = AgentFeedbackNonce(
            run_id=run_id,
            team_id=team_id or "unknown-team",
            channel_id=channel_id,
            thread_ts=thread_ts,
            owner_user_id=user_id,
            feedback_id=click_id,
            expires_at=now + 1800,
        )
        try:
            await self.web_client.views_open(
                trigger_id=trigger_id, view=build_agent_feedback_modal(nonce)
            )
        except Exception:
            logger.warning("Agent feedback modal could not be opened", exc_info=True)
            self.agent_feedback_nonces.pop(nonce, None)
            return False
        return True

    async def _handle_agent_feedback_submission(self, payload: Dict[str, Any]) -> None:
        """Consume a one-shot feedback nonce, record it, and optionally follow up."""
        view = payload.get("view", {}) or {}
        nonce_value = str(view.get("private_metadata") or "")
        nonce = self.agent_feedback_nonces.get(nonce_value)
        user_id = payload.get("user", {}).get("id")
        if not nonce:
            return
        if nonce.expires_at <= time.time():
            self.agent_feedback_nonces.pop(nonce_value, None)
            return
        if nonce.owner_user_id != user_id:
            return
        self.agent_feedback_nonces.pop(nonce_value, None)

        submission = parse_agent_feedback_submission(view)
        try:
            await self.agent_runtime_client.feedback(
                nonce.run_id,
                user_id,
                "needs_changes",
                nonce.feedback_id,
                reason_category=submission.reason_category,
                comment=submission.comment or None,
            )
            message = "Result feedback recorded."
        except AgentRuntimeUnauthorized as exc:
            await self.web_client.chat_postEphemeral(
                channel=nonce.channel_id,
                user=user_id,
                text=_agent_auth_error_text(
                    exc, "Only the Agent run owner can submit result feedback."
                ),
            )
            return
        except AgentRuntimeError as exc:
            # The owner's requested follow-up does not depend on the label
            # being stored, so a lost label must not also lose the next run.
            message = _agent_feedback_error_text(exc)
        await self.web_client.chat_postEphemeral(
            channel=nonce.channel_id, user=user_id, text=message
        )
        if not (submission.continue_session and submission.comment):
            return

        if not self._within_rate_limit(user_id, agent=True):
            await self._post_command_response(
                nonce.channel_id,
                user_id,
                self._build_rate_limit_message(user_id, agent=True),
            )
            return
        response = await self.message_handler._start_agent_prompt(
            submission.comment,
            user_id,
            nonce.channel_id,
            self.web_client,
            team_id=nonce.team_id,
            thread_ts=nonce.thread_ts,
            routing_text=submission.comment,
            request_id=f"feedback:{nonce.feedback_id}",
        )
        if response:
            await self.web_client.chat_postMessage(
                channel=nonce.channel_id, thread_ts=nonce.thread_ts, text=response
            )
        await self._persist_message_state(user_id=user_id)

    async def _handle_agent_shortcut(self, payload: Dict[str, Any]) -> None:
        """Open the Agent bootstrap modal while the Slack trigger is still valid."""
        trigger_id = payload.get("trigger_id")
        team_id = payload.get("team", {}).get("id") or self._extract_team_id(payload)
        user_id = payload.get("user", {}).get("id")
        channel_id = payload.get("channel", {}).get("id")
        message = payload.get("message", {}) or {}
        selected_ts = message.get("ts")
        thread_ts = message.get("thread_ts") or selected_ts
        if not all((trigger_id, team_id, user_id, channel_id, selected_ts, thread_ts)):
            return
        if not await self._is_channel_allowed(channel_id):
            return
        if not await self._is_public_agent_channel(channel_id):
            return

        now = time.time()
        self.agent_shortcut_nonces = {
            key: value
            for key, value in self.agent_shortcut_nonces.items()
            if value.expires_at + AGENT_SHORTCUT_NONCE_GRACE_SECONDS > now
        }
        nonce = uuid.uuid4().hex
        self.agent_shortcut_nonces[nonce] = AgentShortcutNonce(
            team_id=team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            selected_message_ts=selected_ts,
            owner_user_id=user_id,
            expires_at=now + AGENT_SHORTCUT_NONCE_TTL_SECONDS,
        )

        runtime = self.agent_runtime_client
        cached_models = getattr(runtime, "cached_configured_models", None)
        model_refs = list(cached_models()) if callable(cached_models) else []
        # Never preselect an upgrade of a read-only discussion thread.
        thread = self.agent_threads.get(self._thread_key(channel_id, thread_ts)) or {}
        initial_task_mode = (
            "discuss"
            if thread.get("owner_user_id") == user_id
            and thread.get("mode") == "discuss"
            else "implement"
        )
        await self.web_client.views_open(
            trigger_id=trigger_id,
            view=build_agent_shortcut_modal(nonce, model_refs, initial_task_mode),
        )

    async def _is_public_agent_channel(self, channel_id: str) -> bool:
        """Fail closed unless Slack confirms a non-shared public channel."""
        method = getattr(self.web_client, "conversations_info", None)
        if not callable(method):
            logger.warning("Cannot verify public channel for Agent shortcut")
            return False
        try:
            response = await method(channel=channel_id)
            if not response.get("ok", True):
                return False
            channel = response.get("channel", {}) or {}
            return bool(channel.get("is_channel")) and not any(
                channel.get(flag)
                for flag in (
                    "is_private",
                    "is_im",
                    "is_mpim",
                    "is_shared",
                    "is_ext_shared",
                )
            )
        except Exception:
            logger.warning(
                "Failed to verify public channel for Agent shortcut", exc_info=True
            )
            return False

    async def _refresh_agent_model_cache(self) -> None:
        runtime = self.agent_runtime_client
        health_details = getattr(runtime, "health_details", None)
        if not callable(health_details):
            return
        try:
            await health_details()
        except Exception:
            logger.warning("Agent model health cache refresh failed", exc_info=True)

    async def _refresh_agent_model_cache_periodically(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            if self.running:
                await self._refresh_agent_model_cache()

    async def _handle_agent_view_submission(self, payload: Dict[str, Any]) -> None:
        """Consume a one-shot modal nonce and start one quota-counted Agent prompt."""
        view = payload.get("view", {}) or {}
        if view.get("callback_id") == AGENT_FEEDBACK_CALLBACK_ID:
            await self._handle_agent_feedback_submission(payload)
            return
        nonce_value = str(view.get("private_metadata") or "")
        nonce = self.agent_shortcut_nonces.get(nonce_value)
        user_id = payload.get("user", {}).get("id")
        if not nonce:
            return
        if nonce.expires_at <= time.time():
            self.agent_shortcut_nonces.pop(nonce_value, None)
            if nonce.owner_user_id == user_id:
                task = parse_agent_shortcut_submission(view).task
                await self.web_client.chat_postEphemeral(
                    channel=nonce.channel_id,
                    user=user_id,
                    text=(
                        "This Agent form expired before it was submitted. "
                        "Run the shortcut again."
                        + (_agent_task_echo(task) if task else "")
                    ),
                )
            return
        if nonce.owner_user_id != user_id:
            return
        self.agent_shortcut_nonces.pop(nonce_value, None)

        submission = parse_agent_shortcut_submission(view)
        if not submission.task:
            return
        if not await self._is_public_agent_channel(nonce.channel_id):
            return
        cap_message = self._agent_cap_message(
            user_id, nonce.channel_id, nonce.thread_ts
        )
        if cap_message:
            await self.web_client.chat_postEphemeral(
                channel=nonce.channel_id,
                user=user_id,
                text=cap_message + _agent_task_echo(submission.task),
            )
            return

        if not self._within_rate_limit(user_id, agent=True):
            await self._post_command_response(
                nonce.channel_id,
                user_id,
                self._build_rate_limit_message(user_id, agent=True),
            )
            return

        status = await self.web_client.chat_postMessage(
            channel=nonce.channel_id,
            thread_ts=nonce.thread_ts,
            text="🛠️ Pi coding agent is loading the Slack thread snapshot…",
        )
        # Bootstrap and create_session can take minutes; never hold a worker.
        self._spawn_background_task(
            self._start_agent_from_submission(
                nonce, nonce_value, submission, user_id, status.get("ts")
            ),
            "pi_agent_shortcut_start",
        )

    async def _start_agent_from_submission(
        self,
        nonce: AgentShortcutNonce,
        nonce_value: str,
        submission: AgentShortcutSubmission,
        user_id: str,
        status_ts: Optional[str],
    ) -> None:
        try:
            await self._bootstrap_agent_submission(
                nonce, nonce_value, submission, user_id, status_ts
            )
        except Exception:
            logger.exception("Agent shortcut start failed")
            if status_ts:
                await self.web_client.chat_update(
                    channel=nonce.channel_id,
                    ts=status_ts,
                    text="Agent mode could not start this task. Chat mode was not used as fallback.",
                )
        finally:
            await self._persist_message_state(user_id=user_id)

    async def _bootstrap_agent_submission(
        self,
        nonce: AgentShortcutNonce,
        nonce_value: str,
        submission: AgentShortcutSubmission,
        user_id: str,
        status_ts: Optional[str],
    ) -> None:
        bootstrap = AgentBootstrapContext(prompt=submission.task)
        if submission.include_context:
            try:
                bootstrap = await self.agent_bootstrap_builder.build(
                    self.web_client,
                    channel_id=nonce.channel_id,
                    thread_ts=nonce.thread_ts,
                    owner_user_id=user_id,
                    task=submission.task,
                    selected_ts=nonce.selected_message_ts,
                )
            except AgentBootstrapFetchError as exc:
                if status_ts:
                    await self.web_client.chat_update(
                        channel=nonce.channel_id,
                        ts=status_ts,
                        text=(
                            "Agent mode did not start because Slack thread context "
                            f"could not be loaded: `{exc}`."
                        ),
                    )
                return

        details = []
        if submission.include_context:
            details.append(
                f"Loaded {bootstrap.message_count} prior Slack messages and "
                f"{bootstrap.resource_count} resources."
            )
            if bootstrap.truncated:
                details.append("Context truncated to configured limits.")
            if bootstrap.partial:
                details.append(
                    "Context is partial: the Slack fetch time budget ran out."
                )
            if bootstrap.error:
                details.append(
                    f"Started without prior Slack context: `{bootstrap.error}`."
                )
            if bootstrap.warnings:
                details.append(
                    "Skipped or partial resources: `"
                    + "`, `".join(bootstrap.warnings[:10])
                    + "`."
                )
            details.append("Loaded context was sent to the selected model provider.")
        # The loading message becomes the run's status message once it starts.
        response = await self.message_handler._start_agent_prompt(
            bootstrap.prompt,
            user_id,
            nonce.channel_id,
            self.web_client,
            team_id=nonce.team_id,
            thread_ts=nonce.thread_ts,
            model=submission.model,
            routing_text=submission.task,
            request_id=f"shortcut:{nonce_value}",
            task_mode=submission.task_mode,
            workflow_mode=submission.workflow_mode,
            status_ts=status_ts,
            status_note="\n".join(details),
        )
        if response and status_ts:
            await self.web_client.chat_update(
                channel=nonce.channel_id,
                ts=status_ts,
                text="\n".join(part for part in (response, *details) if part),
            )

    async def _handle_agent_interactive(self, payload: Dict[str, Any]):
        actions = payload.get("actions") or []
        if not actions:
            return
        action = actions[0]
        if action.get("action_id") in CHAT_FEEDBACK_ACTIONS:
            await self._handle_chat_feedback(payload, action)
            return
        if action.get("action_id") in {
            "pi_agent_download_patch",
            "pi_agent_resume_from_patch",
        }:
            await self._handle_agent_patch_action(payload, action)
            return
        if action.get("action_id") in {
            "pi_agent_accept_result",
            "pi_agent_needs_changes",
        }:
            user_id = payload.get("user", {}).get("id")
            channel_id = payload.get("channel", {}).get("id")
            try:
                value = json.loads(action.get("value", "{}"))
                verdict = (
                    "accepted"
                    if action["action_id"] == "pi_agent_accept_result"
                    else "needs_changes"
                )
                click_id = str(
                    action.get("action_ts") or payload.get("trigger_id") or ""
                )
                if not user_id or not click_id:
                    return
                await self.agent_runtime_client.feedback(
                    value["run_id"], user_id, verdict, click_id
                )
                message = "Result feedback recorded."
            except AgentRuntimeUnauthorized as exc:
                message = _agent_auth_error_text(
                    exc, "Only the Agent run owner can submit result feedback."
                )
            except AgentRuntimeError as exc:
                message = _agent_feedback_error_text(exc)
            except (KeyError, TypeError, ValueError):
                message = AGENT_FEEDBACK_RETRY_TEXT
            await self.web_client.chat_postEphemeral(
                channel=channel_id, user=user_id, text=message
            )
            return
        if action.get("action_id") not in {"pi_agent_approve", "pi_agent_reject"}:
            return
        user_id = payload.get("user", {}).get("id")
        channel_id = payload.get("channel", {}).get("id")
        try:
            value = json.loads(action.get("value", "{}"))
            run_id = value["run_id"]
            approval_id = value["approval_id"]
        except (KeyError, TypeError, ValueError):
            return
        tracked = self.agent_runs.get(run_id)
        if tracked and tracked.get("owner_user_id") != user_id:
            if hasattr(self.web_client, "chat_postEphemeral"):
                await self.web_client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="Only the user who started this Agent session can approve it.",
                )
            return
        decision = "approve" if action["action_id"] == "pi_agent_approve" else "reject"
        message = payload.get("message") or {}
        card_ts = message.get("ts") or (payload.get("container") or {}).get(
            "message_ts"
        )
        card_blocks = message.get("blocks") or []
        # Claim the card first so the monitor does not close it without the decider.
        cards = tracked.get("approval_cards") if tracked else None
        claimed = cards.pop(approval_id, None) if isinstance(cards, dict) else None
        try:
            await self.agent_runtime_client.decide(
                run_id, approval_id, user_id, decision
            )
        except AgentRuntimeError as exc:
            if isinstance(exc, AgentRuntimeConflict):
                if channel_id and card_ts:
                    await self._update_agent_card(
                        channel_id, card_ts, card_blocks, "Expired"
                    )
            elif claimed is not None and isinstance(cards, dict):
                cards[approval_id] = claimed
            if isinstance(exc, AgentRuntimeUnauthorized):
                text = _agent_auth_error_text(
                    exc, "Only the Agent session owner can approve this action."
                )
            elif isinstance(exc, AgentRuntimeConflict):
                text = "This approval has expired or was already used."
            else:
                logger.warning("Agent decision failed for %s", run_id)
                text = "Agent decision could not be recorded. Please try again."
            await self.web_client.chat_postEphemeral(
                channel=channel_id, user=user_id, text=text
            )
            return
        if channel_id and card_ts:
            await self._update_agent_card(
                channel_id,
                card_ts,
                card_blocks,
                self._agent_card_outcome(
                    "Approved" if decision == "approve" else "Rejected", user_id
                ),
            )
        if not tracked:
            # The bot lost this run (e.g. restarted without state); the runtime
            # accepted the owner's decision, so resume tracking from the card.
            container = payload.get("container") or {}
            message = payload.get("message") or {}
            thread_ts = (
                container.get("thread_ts")
                or message.get("thread_ts")
                or container.get("message_ts")
                or message.get("ts")
            )
            if not value.get("session_id") or not channel_id or not thread_ts:
                return
            team = payload.get("team")
            team_id = team.get("id") if isinstance(team, dict) else None
            tracked = self._track_agent_run(
                value,
                team_id=team_id or self._extract_team_id(payload) or "unknown-team",
                channel_id=channel_id,
                thread_ts=thread_ts,
                owner_user_id=user_id,
            )
        if approval_id not in tracked.setdefault("posted_approval_ids", []):
            tracked["posted_approval_ids"].append(approval_id)
        self._start_agent_monitor(tracked)

    async def _handle_agent_patch_action(
        self, payload: Dict[str, Any], action: Dict[str, Any]
    ) -> None:
        """Download a failed run's kept patch, or start a new run from it (owner only)."""
        user_id = payload.get("user", {}).get("id")
        try:
            value = json.loads(action.get("value", "{}"))
            run_id = value["run_id"]
        except (KeyError, TypeError, ValueError):
            return
        tracked = self.agent_runs.get(run_id) or {}
        channel_id = (
            value.get("channel_id")
            or tracked.get("channel_id")
            or payload.get("channel", {}).get("id")
        )
        thread_ts = value.get("thread_ts") or tracked.get("thread_ts")
        team = payload.get("team")
        team_id = (
            value.get("team_id")
            or tracked.get("team_id")
            or (team.get("id") if isinstance(team, dict) else None)
        )
        if not user_id or not channel_id or not thread_ts:
            return

        async def tell(text: str) -> None:
            await self.web_client.chat_postEphemeral(
                channel=channel_id, user=user_id, text=text
            )

        if tracked and tracked.get("owner_user_id") != user_id:
            await tell(AGENT_PATCH_OWNER_TEXT)
            return
        runtime = self.agent_runtime_client
        if action["action_id"] == "pi_agent_download_patch":
            try:
                patch = await runtime.get_run_patch(run_id, user_id)
            except AgentRuntimeError as exc:
                await tell(self._agent_patch_error_text(exc))
                return
            comment = f"Kept patch of Agent run `{run_id}` (secrets redacted)."
            if patch.get("truncated"):
                comment += " It was truncated to the download size limit."
            try:
                await self.web_client.files_upload_v2(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    content=patch["patch"],
                    filename=f"pi-agent-{run_id}.patch",
                    title=f"Agent run {run_id} patch",
                    initial_comment=comment,
                )
            except Exception:
                logger.warning("Agent patch upload failed for %s", run_id)
                await tell(
                    "The patch could not be uploaded to Slack. Please try again."
                )
            return
        try:
            accepted = await runtime.resume_from_patch(
                run_id,
                user_id,
                # One resume per run and owner, however often the button is clicked.
                _agent_idempotency_key(
                    team_id, channel_id, run_id, user_id, "resume_from_patch"
                ),
            )
        except AgentRuntimeError as exc:
            await tell(self._agent_patch_error_text(exc))
            return
        resumed = self._track_agent_run(
            accepted,
            team_id=team_id or "unknown-team",
            channel_id=channel_id,
            thread_ts=thread_ts,
            owner_user_id=user_id,
            mode=tracked.get("mode"),
        )
        text = (
            f"🛠️ Pi coding agent resumed from the kept patch of run `{run_id}` "
            f"as run `{accepted['run_id']}`."
        )
        await _show_agent_status(self.web_client, resumed, text)
        self._start_agent_monitor(resumed)

    @staticmethod
    def _agent_patch_error_text(exc: AgentRuntimeError) -> str:
        if isinstance(exc, AgentRuntimeUnauthorized) or exc.status == 403:
            return _agent_auth_error_text(exc, AGENT_PATCH_OWNER_TEXT)
        if exc.status == 404:
            return "No kept patch is available for this Agent run any more."
        known = AGENT_PATCH_RESUME_ERROR_TEXT.get(exc.code or "")
        if known:
            return known
        if isinstance(exc, AgentRuntimeConflict):
            return "This Agent run cannot be resumed from its patch right now."
        logger.warning("Agent patch action failed (code=%s)", exc.code)
        return "The Agent patch action failed. Please try again."
