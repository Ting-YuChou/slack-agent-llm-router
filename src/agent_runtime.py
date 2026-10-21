"""Authenticated async client for the local Pi coding-agent orchestrator."""

import asyncio
import json
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx


class AgentRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str = "",
        *,
        status: Optional[int] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code


class AgentRuntimeTimeout(AgentRuntimeError):
    pass


class AgentRuntimeUnavailable(AgentRuntimeError):
    pass


class AgentRuntimeBusy(AgentRuntimeError):
    pass


class AgentRuntimeConflict(AgentRuntimeError):
    pass


class AgentRuntimeUnauthorized(AgentRuntimeError):
    pass


class AgentRuntimeInvalidResponse(AgentRuntimeError):
    pass


class AgentRuntimeClient:
    """Call the loopback-only asynchronous Pi coding-agent API."""

    TERMINAL_STATES = {
        "completed",
        "rejected",
        "cancelled",
        "failed",
        "timed_out",
        "interrupted",
    }
    MAX_RETRY_BACKOFF_SECONDS = 30.0
    # Events after which the cheap event log no longer describes the run.
    RUN_CHANGE_EVENT_TYPES = {"status", "phase", "approval", "approval_decision"}

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        base_url = str(config.get("base_url", "http://127.0.0.1:3001")).rstrip("/")
        connect_timeout = float(config.get("connect_timeout_seconds", 2))
        request_timeout = float(config.get("request_timeout_seconds", 910))
        # Status, lookup and decision calls answer immediately; never let one
        # hold a Slack handler for the long run-start timeout.
        control_timeout = float(config.get("control_timeout_seconds", 10))
        token_env = str(config.get("token_env", "AGENT_RUNTIME_TOKEN"))
        self._token = os.getenv(token_env, "")
        self._health_cache: Optional[Dict[str, Any]] = None
        self._monotonic = time.monotonic
        self._health_timeout_seconds = min(connect_timeout, 2.0)
        self._control_timeout = httpx.Timeout(control_timeout, connect=connect_timeout)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
            transport=transport,
        )

    async def health_details(self) -> Optional[Dict[str, Any]]:
        try:
            response = await asyncio.wait_for(
                self._client.get("/health"), timeout=self._health_timeout_seconds
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            if isinstance(payload, dict):
                self._health_cache = payload
                return payload
            return None
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError):
            return None

    def cached_configured_models(self) -> List[str]:
        """Return the last healthy model allowlist without blocking a Slack trigger."""
        models = (self._health_cache or {}).get("models", [])
        return [
            str(model["ref"])
            for model in models
            if isinstance(model, dict) and model.get("configured") and model.get("ref")
        ]

    async def health(self) -> bool:
        payload = await self.health_details()
        return payload is not None and payload.get("status") == "healthy"

    async def create_session(
        self,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        prompt: str,
        *,
        model: Optional[str] = None,
        workflow_mode: Optional[str] = None,
        task_mode: Optional[str] = None,
        routing_text: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "team_id": team_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "user_id": user_id,
            "prompt": prompt,
        }
        if task_mode is not None:
            payload["task_mode"] = task_mode
        if workflow_mode is not None:
            payload["workflow_mode"] = workflow_mode
        if model:
            payload["model"] = model
        if routing_text is not None:
            payload["routing_text"] = routing_text
        result = await self._request(
            "POST",
            "/v1/sessions",
            json=payload,
            expected_status=202,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
        )
        return self._validate_accepted(result)

    async def ci_repairs(self) -> List[Dict[str, Any]]:
        result = await self._request("GET", "/v1/ci-repairs", control=True)
        if not isinstance(result, list):
            raise AgentRuntimeInvalidResponse("Invalid CI repair list")
        return result

    async def repair_ci(
        self,
        target: str,
        user_id: str,
        *,
        team_id: str,
        channel_id: str,
        workflow_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = await self._request(
            "POST",
            "/v1/ci-repairs",
            json={
                **(
                    {"workflow_mode": workflow_mode}
                    if workflow_mode is not None
                    else {}
                ),
                "target": target,
                "user_id": user_id,
                "team_id": team_id,
                "channel_id": channel_id,
            },
            expected_status=202,
        )
        return self._validate_accepted(result, expected_status="awaiting_approval")

    async def retry_publication(
        self, run_id: str, user_id: str, *, team_id: str, channel_id: str
    ) -> Dict[str, Any]:
        result = await self._request(
            "POST",
            f"/v1/ci-repairs/{run_id}/publication",
            json={"user_id": user_id, "team_id": team_id, "channel_id": channel_id},
            expected_status=202,
        )
        return self._validate_accepted(result, expected_status="awaiting_approval")

    async def lookup_session(
        self, team_id: str, channel_id: str, thread_ts: str
    ) -> Optional[Dict[str, Any]]:
        response = await self._send(
            "GET",
            "/v1/sessions/lookup",
            params={
                "team_id": team_id,
                "channel_id": channel_id,
                "thread_ts": thread_ts,
            },
            timeout=self._control_timeout,
        )
        if response.status_code != 200:
            self._raise_for_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned invalid lookup data"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("found"), bool):
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned invalid lookup data"
            )
        session = payload.get("session")
        return session if payload["found"] and isinstance(session, dict) else None

    async def prompt(
        self,
        session_id: str,
        prompt: str,
        user_id: str,
        *,
        routing_text: Optional[str] = None,
        workflow_mode: Optional[str] = None,
        task_mode: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {"prompt": prompt, "user_id": user_id}
        if task_mode is not None:
            payload["task_mode"] = task_mode
        if workflow_mode is not None:
            payload["workflow_mode"] = workflow_mode
        if routing_text is not None:
            payload["routing_text"] = routing_text
        result = await self._request(
            "POST",
            f"/v1/sessions/{session_id}/prompts",
            json=payload,
            expected_status=202,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
        )
        return self._validate_accepted(result)

    async def resume(self, run_id: str, user_id: str) -> Dict[str, Any]:
        result = await self._request(
            "POST",
            f"/v1/runs/{run_id}/resume",
            json={"user_id": user_id},
            expected_status=202,
        )
        return self._validate_accepted(result)

    async def get_run_patch(self, run_id: str, user_id: str) -> Dict[str, Any]:
        """The owner's retained, redacted patch of an undelivered run.

        Returns ``{"patch": bytes, "truncated": bool, "bytes": int}``; a patch over
        the runtime's download cap ends in a truncation marker line.
        """
        response = await self._send(
            "GET",
            f"/v1/runs/{quote(run_id, safe='')}/patch",
            params={"user_id": user_id},
            timeout=self._control_timeout,
        )
        if response.status_code != 200:
            self._raise_for_error(response)
        try:
            size = int(response.headers.get("x-patch-bytes", len(response.content)))
        except ValueError as exc:
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned an invalid patch"
            ) from exc
        return {
            "patch": response.content,
            "truncated": response.headers.get("x-patch-truncated") == "true",
            "bytes": size,
        }

    async def resume_from_patch(
        self, run_id: str, user_id: str, idempotency_key: str
    ) -> Dict[str, Any]:
        """Start a new run on the session from a terminal run's retained patch."""
        result = await self._request(
            "POST",
            f"/v1/runs/{quote(run_id, safe='')}/resume-from-patch",
            json={"user_id": user_id},
            expected_status=202,
            idempotency_key=idempotency_key,
        )
        return self._validate_accepted(result)

    async def tasks(self, run_id: str) -> List[Dict[str, Any]]:
        result = await self._request("GET", f"/v1/runs/{run_id}/tasks", control=True)
        if not isinstance(result, list):
            raise AgentRuntimeInvalidResponse("Invalid workflow tasks")
        return result

    async def get_run(self, run_id: str) -> Dict[str, Any]:
        result = await self._request("GET", f"/v1/runs/{run_id}", control=True)
        if not isinstance(result, dict):
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned invalid run state"
            )
        for field in ("run_id", "session_id", "status"):
            if not isinstance(result.get(field), str):
                raise AgentRuntimeInvalidResponse(
                    "Agent runtime returned invalid run state"
                )
        return result

    async def get_events(self, run_id: str, after: int = 0) -> List[Dict[str, Any]]:
        text = await self._request_text(
            "GET", f"/v1/runs/{run_id}/events?after={max(0, int(after))}"
        )
        events: List[Dict[str, Any]] = []
        event_id: Optional[int] = None
        for block in text.split("\n\n"):
            data_lines = []
            for line in block.split("\n"):
                if line.startswith("id: "):
                    try:
                        event_id = int(line[4:])
                    except ValueError:
                        event_id = None
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
            if not data_lines:
                continue
            try:
                payload = json.loads("\n".join(data_lines))
            except ValueError as exc:
                raise AgentRuntimeInvalidResponse(
                    "Agent runtime returned invalid SSE"
                ) from exc
            if not isinstance(payload, dict):
                raise AgentRuntimeInvalidResponse("Agent runtime returned invalid SSE")
            if event_id is not None:
                payload["event_id"] = event_id
            events.append(payload)
        return events

    async def decide(
        self,
        run_id: str,
        approval_id: str,
        user_id: str,
        decision: str,
    ) -> None:
        await self._request(
            "POST",
            f"/v1/runs/{run_id}/decisions",
            json={
                "approval_id": approval_id,
                "user_id": user_id,
                "decision": decision,
            },
            expected_status=202,
            control=True,
            # One approval card yields one decision; a retried click or
            # request must not be applied twice.
            idempotency_key=f"decision:{run_id}:{approval_id}:{user_id}:{decision}",
        )

    async def feedback(
        self,
        run_id: str,
        user_id: str,
        verdict: str,
        feedback_id: str,
        reason_category: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        body = {"user_id": user_id, "verdict": verdict, "feedback_id": feedback_id}
        if reason_category is not None:
            body["reason_category"] = reason_category
        if comment is not None:
            body["comment"] = comment
        await self._request(
            "POST",
            f"/v1/runs/{run_id}/feedback",
            json=body,
            expected_status=202,
            control=True,
        )

    async def cancel(self, run_id: str, user_id: str) -> None:
        await self._request(
            "POST",
            f"/v1/runs/{run_id}/cancel",
            json={"user_id": user_id},
            expected_status=202,
            control=True,
        )

    async def close_session(self, session_id: str, user_id: str) -> None:
        await self._request(
            "POST",
            f"/v1/sessions/{session_id}/close",
            json={"user_id": user_id},
            expected_status=202,
            control=True,
        )

    async def wait_for_update(
        self,
        run_id: str,
        *,
        poll_seconds: float = 1.0,
        max_error_seconds: float = 300.0,
        on_progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Poll until the run needs Slack attention, riding out transient errors.

        With ``on_progress``, polls between full run reads follow the event log
        from a cursor, and the run is read again only when an event shows a
        status or phase change. Each poll reports the latest known state.
        """
        backoff = 1.0
        errors_since: Optional[float] = None
        run: Optional[Dict[str, Any]] = None
        cursor = 0
        while True:
            try:
                if run is None or on_progress is None:
                    run = await self.get_run(run_id)
                    # The runtime keeps the newest events; cursors are absolute.
                    cursor = len(run.get("events") or []) + int(
                        run.get("events_truncated") or 0
                    )
                    events = None
                else:
                    events = await self.get_events(run_id, after=cursor)
            except AgentRuntimeError as exc:
                if not self._is_transient(exc):
                    raise
                now = self._monotonic()
                if errors_since is None:
                    errors_since = now
                if now - errors_since + backoff > max_error_seconds:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.MAX_RETRY_BACKOFF_SECONDS)
                continue
            backoff = 1.0
            errors_since = None
            if events is None:
                if (
                    run["status"] in self.TERMINAL_STATES
                    or run["status"] == "awaiting_approval"
                ):
                    return run
            elif any(
                event.get("type") in self.RUN_CHANGE_EVENT_TYPES for event in events
            ):
                run = None
                continue
            else:
                last_id = events[-1].get("event_id") if events else None
                cursor = last_id if isinstance(last_id, int) else cursor + len(events)
                run = self._with_tool_events(run, events)
            if on_progress is not None:
                await on_progress(run)
            await asyncio.sleep(poll_seconds)

    @staticmethod
    def _with_tool_events(
        run: Dict[str, Any], events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Count the tool calls the event log shows since the last full read."""
        started = [
            event
            for event in events
            if event.get("type") in ("tool", "mcp") and event.get("phase") == "start"
        ]
        if not started:
            return run
        return {
            **run,
            "tool_count": int(run.get("tool_count") or 0) + len(started),
            "last_tool": str(started[-1].get("tool") or ""),
        }

    @staticmethod
    def _is_transient(exc: AgentRuntimeError) -> bool:
        if isinstance(
            exc, (AgentRuntimeTimeout, AgentRuntimeUnavailable, AgentRuntimeBusy)
        ):
            return True
        return exc.status is not None and exc.status >= 500

    async def close(self) -> None:
        await self._client.aclose()

    async def shutdown(self) -> None:
        await self.close()

    def _headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        headers = {"authorization": f"Bearer {self._token}"}
        if idempotency_key:
            headers["idempotency-key"] = idempotency_key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        expected_status: int = 200,
        control: bool = False,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        response = await self._send(
            method,
            path,
            json=json,
            timeout=self._control_timeout if control else None,
            idempotency_key=idempotency_key,
        )
        if response.status_code != expected_status:
            self._raise_for_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned invalid JSON"
            ) from exc

    async def _request_text(self, method: str, path: str) -> str:
        response = await self._send(method, path, timeout=self._control_timeout)
        if response.status_code != 200:
            self._raise_for_error(response)
        return response.text

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        idempotency_key: Optional[str] = None,
    ) -> httpx.Response:
        extra: Dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        try:
            return await self._client.request(
                method,
                path,
                headers=self._headers(idempotency_key),
                json=json,
                params=params,
                **extra,
            )
        except httpx.TimeoutException as exc:
            raise AgentRuntimeTimeout("Agent runtime request timed out") from exc
        except httpx.HTTPError as exc:
            raise AgentRuntimeUnavailable("Agent runtime is unavailable") from exc

    def _raise_for_error(self, response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        status = response.status_code
        details = {"status": status, "code": code}
        if status == 401:
            raise AgentRuntimeUnauthorized(
                "Agent runtime authentication failed", **details
            )
        if status == 409:
            raise AgentRuntimeConflict(
                "Agent runtime rejected a conflicting action", **details
            )
        if status == 429:
            raise AgentRuntimeBusy("Agent runtime is busy", **details)
        if status == 504 or code == "run_timeout":
            raise AgentRuntimeTimeout("Agent runtime request timed out", **details)
        raise AgentRuntimeError(
            f"Agent runtime rejected the request with status {status}", **details
        )

    @staticmethod
    def _validate_accepted(
        result: Any, expected_status: str = "starting"
    ) -> Dict[str, Any]:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("session_id"), str)
            or not isinstance(result.get("run_id"), str)
            or result.get("status") != expected_status
        ):
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned an invalid acceptance response"
            )
        return result
