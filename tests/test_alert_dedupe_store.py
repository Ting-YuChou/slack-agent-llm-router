"""O3: stream alert dedupe survives a restart through a shared Redis key."""

import logging

import pytest

from src.llm_router_part4_monitor import (
    ALERT_DEDUPE_STORE_ERRORS,
    AlertDedupeStore,
    AlertManager,
    MonitoringService,
)


class _RecordingHandler:
    def __init__(self):
        self.calls = []

    async def send_alert(self, payload):
        self.calls.append(payload)


class _Server(dict):
    """String keys of one fake Redis; sets and expiries ride along."""

    def __init__(self):
        super().__init__()
        self.sets = {}
        self.expires_at = {}
        self.now = 0.0

    def expire_keys(self):
        for name, at in list(self.expires_at.items()):
            if at <= self.now:
                self.pop(name, None)
                self.sets.pop(name, None)
                del self.expires_at[name]


class _FakeRedis:
    """Emulates the dedupe store's Redis commands and Lua scripts in Python.

    Instances built over the same ``data`` share one server, like processes
    sharing one Redis.
    """

    def __init__(self, data=None):
        self.data = _Server() if data is None else data
        self.ttls = {}
        self.calls = []
        self.closed = False
        self.before_eval = None

    async def eval(self, script, numkeys, *args):
        self.calls.append(("eval",))
        if self.before_eval is not None:
            self.before_eval(script)
        keys, argv = list(args[:numkeys]), [str(a) for a in args[numkeys:]]
        self.data.expire_keys()
        if script == AlertDedupeStore.CLAIM_SCRIPT:
            return self._claim(keys, argv)
        if script == AlertDedupeStore.RELEASE_SCRIPT:
            return self._release(keys, argv)
        if script == AlertDedupeStore.EXTEND_SCRIPT:
            return self._extend(keys, argv)
        raise AssertionError("unknown script")

    def _extend(self, keys, argv):
        ttl = int(argv[0]) / 1000
        for name in keys:
            if name in self.data or name in self.data.sets:
                remaining = self.data.expires_at.get(name, 0) - self.data.now
                if remaining < ttl:
                    self.data.expires_at[name] = self.data.now + ttl
        return 1

    @staticmethod
    def _parse(value):
        """(action, severity, notified_ms or None) of a stored value, or None."""
        parts = value.split(",")
        try:
            action, severity = int(parts[0]), int(parts[1])
            notified = int(parts[2]) if len(parts) > 2 else None
        except (ValueError, IndexError):
            return None
        return action, severity, notified

    def _claim(self, keys, argv):
        key, scope_set = keys
        action, severity, ttl_ms = int(argv[0]), int(argv[1]), int(argv[2])
        now_ms = int(argv[3])
        current = self.data.get(key)
        if current is not None:
            stored = self._parse(current)
            if stored is not None and (action, severity) <= stored[:2]:
                remaining = self.data.expires_at[key] - self.data.now
                notified = -1 if stored[2] is None else stored[2]
                return [0, stored[0], stored[1], int(remaining * 1000), notified]
        self.data[key] = f"{action},{severity},{now_ms}"
        self.data.expires_at[key] = self.data.now + ttl_ms / 1000
        self.ttls[key] = ttl_ms / 1000
        self.data.sets.setdefault(scope_set, set()).add(key)
        self.data.expires_at[scope_set] = self.data.now + ttl_ms / 1000
        return [1]

    def _paged_before(self, value, cleared_ms):
        stored = self._parse(value)
        return stored is None or stored[2] is None or stored[2] <= cleared_ms

    def _drop(self, member, members):
        removed = [member] if self.data.pop(member, None) is not None else []
        self.data.expires_at.pop(member, None)
        members.discard(member)
        return removed

    def _release(self, keys, argv):
        (scope_set,) = keys
        avoid_rank, cleared_ms, explicit = argv[0], int(argv[1]), argv[2:]
        members = self.data.sets.get(scope_set, set())
        removed = []
        for member in sorted(members):
            value = self.data.get(member)
            if value is None:
                members.discard(member)
            elif value.startswith(f"{avoid_rank},") and self._paged_before(
                value, cleared_ms
            ):
                removed += self._drop(member, members)
        for member in explicit:
            value = self.data.get(member)
            if value is not None and self._paged_before(value, cleared_ms):
                removed += self._drop(member, members)
            elif value is None:
                members.discard(member)
        if not members:
            self.data.sets.pop(scope_set, None)
            self.data.expires_at.pop(scope_set, None)
        return removed

    async def aclose(self):
        self.closed = True


class _DownRedis(_FakeRedis):
    async def eval(self, *args, **kwargs):
        self.calls.append(("eval",))
        raise ConnectionError("redis down")


STORE_CONFIG = {"enabled": True, "key_prefix": "test:alert:"}


def _manager(client, cooldown=900, **store_overrides):
    manager = AlertManager(
        {
            "alerts": {
                "dedupe_cooldown_seconds": cooldown,
                "dedupe_store": {**STORE_CONFIG, **store_overrides},
            }
        }
    )
    manager.dedupe_store._client = client
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    return manager, handler


def _guardrail(**overrides):
    event = {
        "event_type": "routing.guardrails",
        "alert_type": "routing_guardrail",
        "trigger_type": "latency_spike",
        "severity": "warning",
        "description": "Latency spike",
        "scope_type": "model",
        "scope_key": "gpt-5",
        "guardrail_action": "warn",
        "model_name": "gpt-5",
        "provider": "openai",
        "window_start_ms": 1_710_000_000_000,
        "window_end_ms": 1_710_000_060_000,
    }
    event.update(overrides)
    return event


def _next_window(n, **overrides):
    start = 1_710_000_000_000 + n * 60_000
    return _guardrail(window_start_ms=start, window_end_ms=start + 60_000, **overrides)


def _avoid(n):
    return _next_window(n, guardrail_action="avoid", severity="error")


def test_dedupe_store_is_off_by_default():
    assert AlertManager({}).dedupe_store is None
    assert (
        AlertManager({"alerts": {"dedupe_store": {"enabled": False}}}).dedupe_store
        is None
    )


@pytest.mark.asyncio
async def test_restarted_manager_does_not_renotify_an_active_incident():
    redis = _FakeRedis()
    before, before_handler = _manager(redis)
    await before.record_external_alert(_guardrail())
    assert len(before_handler.calls) == 1

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_next_window(1))
    await after.record_external_alert(_next_window(2))

    assert after_handler.calls == []
    assert after.suppressed_alert_count == 2
    # A different scope is a different incident.
    await after.record_external_alert(_next_window(3, scope_key="claude-sonnet"))
    assert [c["scope_key"] for c in after_handler.calls] == ["claude-sonnet"]


@pytest.mark.asyncio
async def test_incident_key_expires_with_the_cooldown():
    redis = _FakeRedis()
    manager, _handler = _manager(redis, cooldown=600)

    await manager.record_external_alert(_guardrail())

    (key,) = redis.data
    assert key.startswith("test:alert:")
    assert redis.ttls[key] == 600


@pytest.mark.asyncio
async def test_escalation_after_restart_still_notifies_once():
    redis = _FakeRedis()
    before, before_handler = _manager(redis)
    await before.record_external_alert(_guardrail())

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_avoid(1))
    await after.record_external_alert(_avoid(2))
    assert [c["guardrail_action"] for c in after_handler.calls] == ["avoid"]

    # A second restart remembers the escalated level.
    again, again_handler = _manager(_FakeRedis(redis.data))
    await again.record_external_alert(_avoid(3))
    await again.record_external_alert(_next_window(4))
    assert again_handler.calls == []
    assert len(before_handler.calls) == 1


@pytest.mark.asyncio
async def test_restart_remembers_a_higher_stored_level_than_the_next_event():
    redis = _FakeRedis()
    before, _ = _manager(redis)
    await before.record_external_alert(_avoid(0))

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_next_window(1))
    await after.record_external_alert(_avoid(2))

    assert after_handler.calls == []


@pytest.mark.asyncio
async def test_in_process_escalation_refreshes_the_stored_level():
    redis = _FakeRedis()
    manager, handler = _manager(redis)
    await manager.record_external_alert(_guardrail())
    await manager.record_external_alert(_avoid(1))
    assert [c["guardrail_action"] for c in handler.calls] == ["warn", "avoid"]

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_avoid(2))
    assert after_handler.calls == []


@pytest.mark.asyncio
async def test_resolve_deletes_the_incident_key():
    redis = _FakeRedis()
    manager, handler = _manager(redis)
    await manager.record_external_alert(_guardrail())
    await manager.record_external_alert(_next_window(1, guardrail_action="clear"))

    assert redis.data == {}
    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_next_window(2))
    assert len(after_handler.calls) == 1


@pytest.mark.asyncio
async def test_clear_after_restart_deletes_the_key_it_names():
    redis = _FakeRedis()
    before, _ = _manager(redis)
    await before.record_external_alert(_guardrail())

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_next_window(1, guardrail_action="clear"))
    assert redis.data == {}
    # Nothing was active in this process, so there is nothing to announce.
    assert after_handler.calls == []

    await after.record_external_alert(_next_window(2))
    assert len(after_handler.calls) == 1


@pytest.mark.asyncio
async def test_suppressed_after_restart_incident_still_resolves():
    redis = _FakeRedis()
    before, _ = _manager(redis)
    await before.record_external_alert(_avoid(0))

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(_avoid(1))
    await after.record_external_alert(
        _next_window(2, guardrail_action="clear", trigger_type="error_rate")
    )

    assert [c["status"] for c in after_handler.calls] == ["resolved"]
    assert redis.data == {}


@pytest.mark.asyncio
async def test_redis_down_falls_back_to_in_memory_dedupe(caplog):
    redis = _DownRedis()
    manager, handler = _manager(redis)
    errors_before = ALERT_DEDUPE_STORE_ERRORS.labels(operation="claim")._value.get()

    with caplog.at_level(logging.WARNING, logger="src.llm_router_part4_monitor"):
        await manager.record_external_alert(_guardrail())
        await manager.record_external_alert(_next_window(1))
        await manager.record_external_alert(_next_window(2, scope_key="claude-sonnet"))
        await manager.record_external_alert(_avoid(3))
        await manager.record_external_alert(_next_window(4, guardrail_action="clear"))

    assert [(c["scope_key"], c["status"]) for c in handler.calls] == [
        ("gpt-5", "firing"),
        ("claude-sonnet", "firing"),
        ("gpt-5", "firing"),
        ("gpt-5", "resolved"),
    ]
    warnings = [r for r in caplog.records if "dedupe store" in r.getMessage()]
    assert len(warnings) == 1
    assert (
        ALERT_DEDUPE_STORE_ERRORS.labels(operation="claim")._value.get() > errors_before
    )


@pytest.mark.asyncio
async def test_redis_errors_back_off_before_retrying():
    redis = _DownRedis()
    manager, _handler = _manager(redis, retry_after_seconds=30)
    now = [1000.0]
    manager.dedupe_store._clock = lambda: now[0]

    await manager.record_external_alert(_guardrail())
    calls_after_failure = len(redis.calls)
    await manager.record_external_alert(_next_window(1, scope_key="m2"))
    assert len(redis.calls) == calls_after_failure

    now[0] += 31
    await manager.record_external_alert(_next_window(2, scope_key="m3"))
    assert len(redis.calls) > calls_after_failure


@pytest.mark.asyncio
async def test_store_recovers_after_an_outage():
    redis = _FakeRedis()
    store = AlertDedupeStore(STORE_CONFIG)
    store._client = _DownRedis()
    assert await store.claim(("s", "t", ("model", "a", "b")), "1,1", 60) is None

    store._client = redis
    store._retry_at = None
    assert await store.claim(("s", "t", ("model", "a", "b")), "1,1", 60) is True
    assert await store.claim(("s", "t", ("model", "a", "b")), "1,1", 60) is False


@pytest.mark.asyncio
async def test_close_closes_the_store_client():
    redis = _FakeRedis()
    manager, _handler = _manager(redis)

    await manager.close()

    assert redis.closed is True


def test_store_client_reads_password_env_and_short_timeouts(monkeypatch):
    redis_asyncio = pytest.importorskip("redis.asyncio")
    captured = {}

    def fake_redis(**kwargs):
        captured.update(kwargs)
        return _FakeRedis()

    monkeypatch.setattr(redis_asyncio, "Redis", fake_redis)
    monkeypatch.setenv("TEST_ALERT_REDIS_PASSWORD", "s3cret")
    store = AlertDedupeStore(
        {
            "enabled": True,
            "host": "redis",
            "port": 6380,
            "db": 2,
            "password_env": "TEST_ALERT_REDIS_PASSWORD",
        }
    )

    store._get_client()

    assert captured["host"] == "redis"
    assert captured["port"] == 6380
    assert captured["db"] == 2
    assert captured["password"] == "s3cret"
    assert captured["socket_connect_timeout"] == pytest.approx(0.1)
    assert captured["socket_timeout"] == pytest.approx(0.1)
    assert captured["decode_responses"] is True


# --- review round 1 -----------------------------------------------------------


class _HeldButUnreadable(_FakeRedis):
    """The incident looks held, but the level can never be read back."""

    async def set(self, *args, **kwargs):
        return None

    async def get(self, *args, **kwargs):
        raise ConnectionError("redis read timed out")

    async def eval(self, *args, **kwargs):
        raise ConnectionError("redis read timed out")


@pytest.mark.asyncio
async def test_unreadable_store_never_suppresses_a_page():
    manager, handler = _manager(_HeldButUnreadable())

    await manager.record_external_alert(_guardrail())

    assert len(handler.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_escalation_pages_once_across_processes():
    redis = _FakeRedis()
    first, first_handler = _manager(redis)
    second, second_handler = _manager(_FakeRedis(redis.data))

    await first.record_external_alert(_guardrail())
    await second.record_external_alert(_avoid(1))
    # first still holds its warn in memory: avoid looks like an escalation
    # to it, but second already paged at avoid.
    await first.record_external_alert(_avoid(2))

    assert [c["guardrail_action"] for c in first_handler.calls] == ["warn"]
    assert [c["guardrail_action"] for c in second_handler.calls] == ["avoid"]


@pytest.mark.asyncio
async def test_release_skipped_during_backoff_is_retried_before_the_next_claim():
    redis = _FakeRedis()
    manager, handler = _manager(redis, retry_after_seconds=30)
    now = [1000.0]
    manager.dedupe_store._clock = lambda: now[0]

    await manager.record_external_alert(_avoid(0))
    manager.dedupe_store._client = _DownRedis()
    await manager.record_external_alert(_next_window(1, scope_key="other"))
    # In backoff now: the clear cannot reach Redis.
    await manager.record_external_alert(_next_window(2, guardrail_action="clear"))
    assert redis.data != {}

    manager.dedupe_store._client = redis
    now[0] += 31
    await manager.record_external_alert(_avoid(3))

    assert [(c["scope_key"], c["status"]) for c in handler.calls] == [
        ("gpt-5", "firing"),
        ("other", "firing"),
        ("gpt-5", "resolved"),
        ("gpt-5", "firing"),
    ]


@pytest.mark.asyncio
async def test_pending_release_reaches_other_processes_once_redis_recovers():
    redis = _FakeRedis()
    manager, _ = _manager(redis, retry_after_seconds=30)
    now = [1000.0]
    manager.dedupe_store._clock = lambda: now[0]
    await manager.record_external_alert(_avoid(0))

    manager.dedupe_store._client = _DownRedis()
    await manager.record_external_alert(_next_window(1, guardrail_action="clear"))
    manager.dedupe_store._client = redis
    now[0] += 31
    # Any later claim (another scope here) flushes the pending release.
    await manager.record_external_alert(_next_window(2, scope_key="other"))

    other, other_handler = _manager(_FakeRedis(redis.data))
    await other.record_external_alert(_avoid(3))
    assert len(other_handler.calls) == 1


@pytest.mark.asyncio
async def test_clear_after_restart_releases_every_avoid_incident_of_its_scope():
    redis = _FakeRedis()
    before, _ = _manager(redis)
    await before.record_external_alert(_avoid(0))
    await before.record_external_alert(_next_window(1, trigger_type="queue_depth"))

    after, after_handler = _manager(_FakeRedis(redis.data))
    await after.record_external_alert(
        _next_window(2, guardrail_action="clear", trigger_type="error_rate")
    )

    # The avoid incident (latency_spike) is gone; the warn one is not, as
    # Flink never clears warn-only incidents.
    assert len(redis.data) == 1
    await after.record_external_alert(_avoid(3))
    await after.record_external_alert(_next_window(4, trigger_type="queue_depth"))
    assert [c["rule_name"] for c in after_handler.calls] == ["latency_spike"]


@pytest.mark.asyncio
async def test_suppressed_incident_keeps_the_original_page_time():
    redis = _FakeRedis()
    clocks = _Clocks(redis, 1_710_000_000)

    before, before_handler = _manager(redis)
    clocks.attach(before)
    await before.record_external_alert(_avoid(0))

    clocks.tick(800)
    after, after_handler = _manager(_FakeRedis(redis.data))
    clocks.attach(after)
    await after.record_external_alert(_avoid(1))
    assert after_handler.calls == []

    # The page happened at t=0, so the cooldown ends at t=900, not t=1700.
    clocks.tick(101)
    await after.record_external_alert(_avoid(2))
    assert len(after_handler.calls) == 1
    assert len(before_handler.calls) == 1


@pytest.mark.asyncio
async def test_resolve_survives_targets_evicted_while_releasing():
    manager, handler = _manager(_FakeRedis())
    await manager.record_external_alert(_avoid(0))

    async def evicting_release(*args, **kwargs):
        manager._active_external_alerts.clear()

    manager.dedupe_store.release = evicting_release

    await manager.record_external_alert(_next_window(1, guardrail_action="clear"))

    assert [c["status"] for c in handler.calls] == ["firing", "resolved"]


# --- review round 1 add-on: a missed clear must not outlive the incident ------


class _Clocks:
    """One fake time for the manager, the wall clock and the fake Redis."""

    def __init__(self, redis, start_wall_seconds):
        self.redis = redis
        self.offset = 0.0
        self.start_wall = start_wall_seconds

    def attach(self, manager):
        manager._clock = lambda: self.offset
        manager._wall_clock = lambda: self.start_wall + self.offset
        return manager

    def tick(self, seconds):
        self.offset += seconds
        self.redis.data.now = self.offset


def _ttl_manager(redis, clocks, **alerts):
    manager = AlertManager(
        {
            "alerts": {
                "dedupe_cooldown_seconds": 900,
                "incident_ttl_seconds": 180,
                "incident_ttl_grace_seconds": 60,
                "dedupe_store": dict(STORE_CONFIG),
                **alerts,
            }
        }
    )
    manager.dedupe_store._client = redis
    handler = _RecordingHandler()
    manager.notification_handlers = {"test": handler}
    return clocks.attach(manager), handler


def _avoid_ending_at(window_end_ms, **overrides):
    fields = {
        "guardrail_action": "avoid",
        "severity": "error",
        "window_start_ms": window_end_ms - 60_000,
        "window_end_ms": window_end_ms,
    }
    fields.update(overrides)
    return _guardrail(**fields)


@pytest.mark.asyncio
async def test_avoid_key_expires_with_the_incident_not_the_cooldown():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    manager, _ = _ttl_manager(redis, clocks)

    await manager.record_external_alert(_avoid_ending_at(end_ms))

    (key,) = redis.data
    # window end + active TTL (180 s) + grace (60 s), from the window end.
    assert redis.ttls[key] == pytest.approx(240)


@pytest.mark.asyncio
async def test_missed_clear_does_not_suppress_a_new_avoid_after_the_incident_ttl():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    manager, handler = _ttl_manager(redis, clocks)
    await manager.record_external_alert(_avoid_ending_at(end_ms))

    # The clear never arrives. Past the incident's own expiry a new avoid
    # pages again.
    clocks.tick(241)
    await manager.record_external_alert(_avoid_ending_at(end_ms + 241_000))

    assert len(handler.calls) == 2


@pytest.mark.asyncio
async def test_missed_clear_does_not_suppress_a_restarted_process_either():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    before, _ = _ttl_manager(redis, clocks)
    await before.record_external_alert(_avoid_ending_at(end_ms))

    clocks.tick(241)
    restarted, restarted_handler = _ttl_manager(_FakeRedis(redis.data), clocks)
    await restarted.record_external_alert(_avoid_ending_at(end_ms + 241_000))

    assert len(restarted_handler.calls) == 1


@pytest.mark.asyncio
async def test_a_continuing_avoid_incident_keeps_its_dedupe():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    manager, handler = _ttl_manager(redis, clocks)
    await manager.record_external_alert(_avoid_ending_at(end_ms))

    # Flink keeps re-entering the scope every minute: one page per cooldown,
    # not one per incident TTL, here and in a restarted process.
    for minute in range(1, 8):
        clocks.tick(60)
        await manager.record_external_alert(_avoid_ending_at(end_ms + minute * 60_000))
    restarted, restarted_handler = _ttl_manager(_FakeRedis(redis.data), clocks)
    clocks.tick(60)
    await restarted.record_external_alert(_avoid_ending_at(end_ms + 480_000))

    assert len(handler.calls) == 1
    assert restarted_handler.calls == []


@pytest.mark.asyncio
async def test_event_active_until_bounds_the_key():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    manager, _ = _ttl_manager(redis, clocks, incident_ttl_seconds=None)

    await manager.record_external_alert(
        _avoid_ending_at(end_ms, active_until_ms=end_ms + 30_000)
    )

    (key,) = redis.data
    assert redis.ttls[key] == pytest.approx(90)


@pytest.mark.asyncio
async def test_incident_ttl_never_extends_the_cooldown_or_touches_warns():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    manager, _ = _ttl_manager(redis, clocks, dedupe_cooldown_seconds=120)

    await manager.record_external_alert(_avoid_ending_at(end_ms))
    await manager.record_external_alert(
        _guardrail(
            scope_key="warn-only",
            window_start_ms=end_ms - 60_000,
            window_end_ms=end_ms,
        )
    )

    assert sorted(redis.ttls.values()) == [120, 120]


# --- review round 2 -----------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_during_a_continuing_incident_waits_out_the_cooldown():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    before, before_handler = _ttl_manager(redis, clocks)
    await before.record_external_alert(_avoid_ending_at(end_ms))

    restarted, restarted_handler = _ttl_manager(_FakeRedis(redis.data), clocks)
    # An avoid every minute for ten minutes, all inside the 900 s cooldown.
    for minute in range(1, 11):
        clocks.tick(60)
        await restarted.record_external_alert(
            _avoid_ending_at(end_ms + minute * 60_000)
        )

    assert len(before_handler.calls) == 1
    assert restarted_handler.calls == []


@pytest.mark.asyncio
async def test_lagged_avoid_windows_still_deduplicate():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    # Every event arrives 300 s after its window ended.
    clocks = _Clocks(redis, end_ms / 1000 + 300)
    manager, handler = _ttl_manager(redis, clocks)

    for minute in range(6):
        await manager.record_external_alert(_avoid_ending_at(end_ms + minute * 60_000))
        clocks.tick(60)

    assert len(handler.calls) == 1


def _store_manager(redis, clocks, **store):
    return _ttl_manager(redis, clocks, dedupe_store={**STORE_CONFIG, **store})


@pytest.mark.asyncio
async def test_replayed_release_spares_incidents_paged_after_the_clear():
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    first, _ = _store_manager(redis, clocks, retry_after_seconds=0)
    await first.record_external_alert(_avoid_ending_at(end_ms))

    first.dedupe_store._client = _DownRedis()
    clocks.tick(10)
    await first.record_external_alert(
        _avoid_ending_at(end_ms + 10_000, guardrail_action="clear")
    )
    # first's key expires; another replica pages a new incident.
    clocks.tick(250)
    second, second_handler = _store_manager(_FakeRedis(redis.data), clocks)
    await second.record_external_alert(_avoid_ending_at(end_ms + 260_000))
    assert len(second_handler.calls) == 1

    # first recovers and replays its old release before its next claim.
    first.dedupe_store._client = redis
    await first.record_external_alert(
        _avoid_ending_at(end_ms + 270_000, scope_key="other")
    )
    assert first.dedupe_store._pending_releases == {}

    third, third_handler = _store_manager(_FakeRedis(redis.data), clocks)
    await third.record_external_alert(_avoid_ending_at(end_ms + 280_000))
    assert third_handler.calls == []


class ResponseError(Exception):
    """Named like redis.exceptions.ResponseError, which the store detects."""


class _RejectsRelease(_FakeRedis):
    def __init__(self, data=None, error=None):
        super().__init__(data)
        self.error = error or ResponseError("WRONGTYPE Operation against a key")

    async def eval(self, script, numkeys, *args):
        if script == AlertDedupeStore.RELEASE_SCRIPT:
            self.calls.append(("eval",))
            raise self.error
        return await super().eval(script, numkeys, *args)


INCIDENT = ("routing_guardrails", "latency_spike", ("scope", "model", "gpt-5"))
OTHER = ("routing_guardrails", "latency_spike", ("scope", "model", "other"))


@pytest.mark.asyncio
async def test_rejected_release_is_dropped_not_retried_forever(caplog):
    store = AlertDedupeStore({**STORE_CONFIG, "retry_after_seconds": 0})
    store._client = _RejectsRelease()

    with caplog.at_level(logging.WARNING, logger="src.llm_router_part4_monitor"):
        await store.release(INCIDENT, [], cleared_ms=1_000)

    assert store._pending_releases == {}
    assert any("release" in r.getMessage() for r in caplog.records)
    assert await store.try_claim(OTHER, (1, 1), 60, now_ms=2_000) is not None


@pytest.mark.asyncio
async def test_failing_release_is_dropped_after_max_attempts():
    store = AlertDedupeStore(
        {**STORE_CONFIG, "retry_after_seconds": 0, "max_release_attempts": 3}
    )
    store._client = _RejectsRelease(error=ConnectionError("redis down"))

    await store.release(INCIDENT, [], cleared_ms=1_000)
    assert await store.try_claim(OTHER, (1, 1), 60, now_ms=2_000) is None
    # Third attempt fails too: the release is given up and the claim proceeds.
    assert await store.try_claim(OTHER, (1, 1), 60, now_ms=3_000) is not None
    assert store._pending_releases == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("first_to_clear", ["pager", "suppressed"])
async def test_replicas_page_and_resolve_once_between_them(first_to_clear):
    redis = _FakeRedis()
    end_ms = 1_710_000_060_000
    clocks = _Clocks(redis, end_ms / 1000)
    pager, pager_handler = _ttl_manager(redis, clocks)
    other, other_handler = _ttl_manager(_FakeRedis(redis.data), clocks)

    for replica in (pager, other):
        await replica.record_external_alert(_avoid_ending_at(end_ms))
    clocks.tick(30)
    clear = _avoid_ending_at(end_ms + 30_000, guardrail_action="clear")
    order = (pager, other) if first_to_clear == "pager" else (other, pager)
    for replica in order:
        await replica.record_external_alert(clear)

    statuses = [c["status"] for c in pager_handler.calls + other_handler.calls]
    assert sorted(statuses) == ["firing", "resolved"]
    assert other._active_external_alerts == {}
    assert pager._active_external_alerts == {}


@pytest.mark.asyncio
async def test_monitoring_without_dedupe_store_warns_at_startup(caplog):
    with caplog.at_level(logging.WARNING, logger="src.llm_router_part4_monitor"):
        await MonitoringService({}).initialize()

    assert any(
        "multiple worker replicas will each page" in r.getMessage()
        for r in caplog.records
    )
