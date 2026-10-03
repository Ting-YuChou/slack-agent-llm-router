"""Authenticated async client for the local Pi coding-agent orchestrator."""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx


class AgentRuntimeError(RuntimeError):
    pass


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

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        base_url = str(config.get("base_url", "http://127.0.0.1:3001")).rstrip("/")
        connect_timeout = float(config.get("connect_timeout_seconds", 2))
        request_timeout = float(config.get("request_timeout_seconds", 910))
        token_env = str(config.get("token_env", "AGENT_RUNTIME_TOKEN"))
        self._token = os.getenv(token_env, "")
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_timeout_seconds = min(connect_timeout, 2.0)
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
        routing_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "team_id": team_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "user_id": user_id,
            "prompt": prompt,
        }
        if model:
            payload["model"] = model
        if routing_text is not None:
            payload["routing_text"] = routing_text
        result = await self._request(
            "POST",
            "/v1/sessions",
            json=payload,
            expected_status=202,
        )
        return self._validate_accepted(result)

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
        self, session_id: str, prompt: str, user_id: str, *, routing_text: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = {"prompt": prompt, "user_id": user_id}
        if routing_text is not None:
            payload["routing_text"] = routing_text
        result = await self._request(
            "POST",
            f"/v1/sessions/{session_id}/prompts",
            json=payload,
            expected_status=202,
        )
        return self._validate_accepted(result)

    async def get_run(self, run_id: str) -> Dict[str, Any]:
        result = await self._request("GET", f"/v1/runs/{run_id}")
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
        )

    async def cancel(self, run_id: str, user_id: str) -> None:
        await self._request(
            "POST",
            f"/v1/runs/{run_id}/cancel",
            json={"user_id": user_id},
            expected_status=202,
        )

    async def close_session(self, session_id: str, user_id: str) -> None:
        await self._request(
            "POST",
            f"/v1/sessions/{session_id}/close",
            json={"user_id": user_id},
            expected_status=202,
        )

    async def wait_for_update(
        self, run_id: str, *, poll_seconds: float = 1.0
    ) -> Dict[str, Any]:
        while True:
            run = await self.get_run(run_id)
            if (
                run["status"] in self.TERMINAL_STATES
                or run["status"] == "awaiting_approval"
            ):
                return run
            await asyncio.sleep(poll_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def shutdown(self) -> None:
        await self.close()

    def _headers(self) -> Dict[str, str]:
        return {"authorization": f"Bearer {self._token}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        expected_status: int = 200,
    ) -> Any:
        response = await self._send(method, path, json=json)
        if response.status_code != expected_status:
            self._raise_for_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned invalid JSON"
            ) from exc

    async def _request_text(self, method: str, path: str) -> str:
        response = await self._send(method, path)
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
    ) -> httpx.Response:
        try:
            return await self._client.request(
                method, path, headers=self._headers(), json=json, params=params
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
        if response.status_code == 401:
            raise AgentRuntimeUnauthorized("Agent runtime authentication failed")
        if response.status_code == 409:
            raise AgentRuntimeConflict("Agent runtime rejected a conflicting action")
        if response.status_code == 429:
            raise AgentRuntimeBusy("Agent runtime is busy")
        if response.status_code == 504 or code == "run_timeout":
            raise AgentRuntimeTimeout("Agent runtime request timed out")
        raise AgentRuntimeError(
            f"Agent runtime rejected the request with status {response.status_code}"
        )

    @staticmethod
    def _validate_accepted(result: Any) -> Dict[str, Any]:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("session_id"), str)
            or not isinstance(result.get("run_id"), str)
            or result.get("status") != "starting"
        ):
            raise AgentRuntimeInvalidResponse(
                "Agent runtime returned an invalid acceptance response"
            )
        return result
