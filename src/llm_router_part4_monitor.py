"""
LLM Router Part 4: Monitoring Service - Prometheus, Grafana, and Slack Integration
Comprehensive observability, alerting, and performance monitoring
"""

import asyncio
from collections import Counter as CollectionCounter, OrderedDict
import hashlib
import json
import logging
import math
import os
import time
import psutil
import GPUtil
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Sequence, Tuple
from dataclasses import dataclass
from pathlib import Path

import httpx
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import yaml

from src.utils.logger import setup_logging
from src.utils.metrics import (
    SYSTEM_METRICS,
    ROUTER_METRICS,
    INFERENCE_METRICS,
    PIPELINE_METRICS,
)

logger = logging.getLogger(__name__)

# Stream alerts recorded but not notified because they were older than
# alerts.max_event_age_seconds (for example on a consumer replay).
STALE_ALERT_EVENTS = Counter(
    "llm_router_alerts_stale_events_total",
    "Stream alert events too old to notify",
)

# Failed Redis calls of the shared alert dedupe store; each one falls back to
# the in-process dedupe state.
ALERT_DEDUPE_STORE_ERRORS = Counter(
    "llm_router_alert_dedupe_store_errors_total",
    "Alert dedupe store (Redis) operations that failed",
    ["operation"],
)


@dataclass(frozen=True)
class DedupeClaim:
    """Answer of ``AlertDedupeStore.try_claim``."""

    notify: bool
    # When held: the level that already paged and the cooldown left on it.
    stored_level: Optional[Tuple[int, int]] = None
    remaining_seconds: Optional[float] = None


class AlertDedupeStore:
    """Shared record of notified stream-alert incidents, kept in Redis.

    One key per incident ``(source, type, scope)`` holding the paged level
    ``"<action rank>,<severity rank>"`` with a TTL of the dedupe cooldown,
    so a restarted (or second) process sees that the incident already
    paged. Claiming is one compare-and-set script: it pages only when the
    key is missing or holds a lower level. Each scope keeps a set of its
    incident keys so a clear can close every avoid incident of the scope.
    Every call returns ``None`` when Redis is unavailable, and the caller
    falls back to its in-memory dedupe; after a failure Redis is skipped for
    ``retry_after_seconds``. Releases Redis could not take are retried
    before the next claim.
    """

    # KEYS: incident, scope set. ARGV: action rank, severity rank, ttl ms.
    # Returns {1} to page, or {0, action, severity, pttl} when already held
    # at the same or a higher level.
    CLAIM_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local action = tonumber(ARGV[1])
local severity = tonumber(ARGV[2])
if current then
  local held_action, held_severity = string.match(current, '^(%d+),(%d+)$')
  if held_action then
    held_action = tonumber(held_action)
    held_severity = tonumber(held_severity)
    if action < held_action
      or (action == held_action and severity <= held_severity) then
      return {0, held_action, held_severity, redis.call('PTTL', KEYS[1])}
    end
  end
end
redis.call('SET', KEYS[1], ARGV[1] .. ',' .. ARGV[2], 'PX', ARGV[3])
redis.call('SADD', KEYS[2], KEYS[1])
if redis.call('PTTL', KEYS[2]) < tonumber(ARGV[3]) then
  redis.call('PEXPIRE', KEYS[2], ARGV[3])
end
return {1}
"""

    # KEYS: scope set. ARGV: avoid action rank, then incident keys to close.
    # Closes every avoid incident of the scope plus the named ones.
    RELEASE_SCRIPT = """
local removed = 0
local prefix = ARGV[1] .. ','
for _, member in ipairs(redis.call('SMEMBERS', KEYS[1])) do
  local value = redis.call('GET', member)
  if not value then
    redis.call('SREM', KEYS[1], member)
  elseif string.sub(value, 1, string.len(prefix)) == prefix then
    removed = removed + redis.call('DEL', member)
    redis.call('SREM', KEYS[1], member)
  end
end
for i = 2, #ARGV do
  removed = removed + redis.call('DEL', ARGV[i])
  redis.call('SREM', KEYS[1], ARGV[i])
end
return removed
"""

    def __init__(self, config: Dict[str, Any], avoid_rank: int = 2):
        self.host = str(config.get("host", "localhost"))
        self.port = int(config.get("port", 6379))
        self.db = int(config.get("db", 0))
        self.password_env = config.get("password_env")
        self.key_prefix = str(config.get("key_prefix", "llm_router:alert_incident:"))
        self.connect_timeout_ms = int(config.get("connect_timeout_ms", 100))
        self.socket_timeout_ms = int(config.get("socket_timeout_ms", 100))
        self.operation_timeout_ms = int(config.get("operation_timeout_ms", 500))
        self.max_connections = int(config.get("max_connections", 10))
        self.retry_after_seconds = max(
            0.0, float(config.get("retry_after_seconds", 30))
        )
        self._client: Any = None
        self._clock: Callable[[], float] = time.monotonic
        self._retry_at: Optional[float] = None
        self._unavailable_logged = False
        self.avoid_rank = int(avoid_rank)
        self.max_pending_releases = max(
            1, int(config.get("max_pending_releases", 1024))
        )
        # scope set key -> incident keys still to delete.
        self._pending_releases: "OrderedDict[str, set]" = OrderedDict()

    def _get_client(self) -> Any:
        if self._client is None:
            from redis.asyncio import Redis

            password = None
            if self.password_env:
                password = os.getenv(str(self.password_env))
            self._client = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=self.connect_timeout_ms / 1000.0,
                socket_timeout=self.socket_timeout_ms / 1000.0,
                max_connections=self.max_connections,
                health_check_interval=30,
            )
        return self._client

    def _scope_tag(self, incident: tuple) -> str:
        source, _alert_type, scope = incident
        digest = hashlib.sha256(
            json.dumps(list(scope), default=str).encode("utf-8")
        ).hexdigest()[:32]
        # Hash tag: an incident and its scope set live in one cluster slot.
        return f"{{{source}:{digest}}}"

    def key(self, incident: tuple) -> str:
        return f"{self.key_prefix}{self._scope_tag(incident)}:{incident[1]}"

    def scope_key(self, incident: tuple) -> str:
        return f"{self.key_prefix}{self._scope_tag(incident)}:incidents"

    async def _call(self, operation: str, factory: Callable[[Any], Any]) -> Any:
        """Run one Redis command; raise ``_StoreUnavailable`` on any failure."""
        now = self._clock()
        if self._retry_at is not None and now < self._retry_at:
            raise _StoreUnavailable()
        try:
            result = await asyncio.wait_for(
                factory(self._get_client()),
                timeout=self.operation_timeout_ms / 1000.0,
            )
        except Exception as exc:
            ALERT_DEDUPE_STORE_ERRORS.labels(operation=operation).inc()
            self._retry_at = now + self.retry_after_seconds
            if not self._unavailable_logged:
                self._unavailable_logged = True
                logger.warning(
                    "Alert dedupe store unavailable (%s); using in-memory "
                    "dedupe: %s",
                    operation,
                    exc,
                )
            else:
                logger.debug("Alert dedupe store %s failed: %s", operation, exc)
            raise _StoreUnavailable() from exc
        if self._unavailable_logged:
            self._unavailable_logged = False
            logger.info("Alert dedupe store recovered")
        self._retry_at = None
        return result

    async def _flush_pending_releases(self) -> None:
        """Retry releases that failed earlier; raise if Redis is still out."""
        while self._pending_releases:
            scope_key, keys = next(iter(self._pending_releases.items()))
            await self._call(
                "release",
                lambda client: client.eval(
                    self.RELEASE_SCRIPT,
                    1,
                    scope_key,
                    self.avoid_rank,
                    *sorted(keys),
                ),
            )
            self._pending_releases.pop(scope_key, None)

    async def try_claim(
        self, incident: tuple, level: Tuple[int, int], ttl_seconds: float
    ) -> Optional[DedupeClaim]:
        """Atomically record that ``incident`` pages at ``level``.

        Returns whether this caller should page (the incident is not held, or
        held at a lower level), or ``None`` when Redis cannot answer, in
        which case the caller keeps its in-memory decision.
        """
        ttl_ms = max(1, int(math.ceil(ttl_seconds * 1000)))
        key, scope_key = self.key(incident), self.scope_key(incident)
        try:
            await self._flush_pending_releases()
            result = await self._call(
                "claim",
                lambda client: client.eval(
                    self.CLAIM_SCRIPT,
                    2,
                    key,
                    scope_key,
                    int(level[0]),
                    int(level[1]),
                    ttl_ms,
                ),
            )
        except _StoreUnavailable:
            return None
        try:
            if int(result[0]) == 1:
                return DedupeClaim(notify=True)
            stored = (int(result[1]), int(result[2]))
            remaining = max(0, int(result[3])) / 1000.0
        except (TypeError, ValueError, IndexError):
            logger.debug("Unexpected dedupe store reply: %r", result)
            return None
        return DedupeClaim(
            notify=False, stored_level=stored, remaining_seconds=remaining
        )

    async def claim(
        self, incident: tuple, level: Any, ttl_seconds: float
    ) -> Optional[bool]:
        """``try_claim`` reduced to True (page), False (held) or None."""
        if isinstance(level, str):
            level = tuple(int(part) for part in level.split(","))
        result = await self.try_claim(incident, level, ttl_seconds)
        return None if result is None else result.notify

    async def release(self, incident: tuple, targets: Sequence[tuple] = ()) -> None:
        """Close a scope: its avoid incidents, ``incident`` and ``targets``.

        A release Redis cannot take (an error or the backoff) is kept and
        retried before the next claim, so a cleared incident pages again.
        """
        scope_key = self.scope_key(incident)
        keys = set(self._pending_releases.pop(scope_key, set()))
        keys.update(self.key(item) for item in [incident, *targets])
        self._pending_releases[scope_key] = keys
        while len(self._pending_releases) > self.max_pending_releases:
            self._pending_releases.pop(next(iter(self._pending_releases)))
        try:
            await self._flush_pending_releases()
        except _StoreUnavailable:
            return

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if callable(closer):
            try:
                await closer()
            except Exception:
                logger.debug("Failed to close alert dedupe store", exc_info=True)


class _StoreUnavailable(Exception):
    """The alert dedupe store could not answer; fall back to memory."""


def _parse_event_timestamp(value: Any) -> datetime:
    """Parse event timestamps emitted by the streaming pipeline as aware values.

    Naive values are read as UTC, like the rest of the analytics pipeline.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug(f"Failed to parse event timestamp: {value}")
            parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _stream_model_provider_key(model_name: Any, provider: Any) -> str:
    """Build a stable key for per-provider stream model metrics."""
    model = str(model_name or "unknown")
    resolved_provider = str(provider or "unknown")
    return f"{model}@{resolved_provider}"


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    condition: str
    threshold: float
    duration: int  # seconds; re-notify cooldown while the rule keeps firing
    severity: str  # critical, warning, info
    description: str
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    # How long the condition must hold before the first notification.
    pending_seconds: float = 0
    pending_since: Optional[datetime] = None
    # Whether a firing notification is outstanding (a resolve is owed).
    firing: bool = False
    # How long the condition must stay false before RESOLVED is sent; None
    # means the rule's duration, or 300s when that is unset.
    resolve_after_seconds: Optional[float] = None
    recovering_since: Optional[datetime] = None

    def effective_resolve_after_seconds(self) -> float:
        if self.resolve_after_seconds is not None:
            return float(self.resolve_after_seconds)
        return float(self.duration) if self.duration else 300.0


@dataclass
class SystemHealth:
    """System health status"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: List[float]
    gpu_memory: List[float]
    network_io: Dict[str, float]
    process_count: int
    uptime: float
    timestamp: datetime


class SystemMonitor:
    """System resource monitoring"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_interval = config.get("collection_interval", 15)  # seconds
        self.metrics_registry = CollectorRegistry()
        self.running = False

        # Initialize metrics
        self._init_system_metrics()

    def _init_system_metrics(self):
        """Initialize system monitoring metrics"""
        self.cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "CPU Usage Percentage",
            registry=self.metrics_registry,
        )
        self.memory_usage = Gauge(
            "system_memory_usage_percent",
            "Memory Usage Percentage",
            registry=self.metrics_registry,
        )
        self.disk_usage = Gauge(
            "system_disk_usage_percent",
            "Disk Usage Percentage",
            ["mount_point"],
            registry=self.metrics_registry,
        )
        self.gpu_usage = Gauge(
            "system_gpu_usage_percent",
            "GPU Usage Percentage",
            ["gpu_id"],
            registry=self.metrics_registry,
        )
        self.gpu_memory = Gauge(
            "system_gpu_memory_usage_percent",
            "GPU Memory Usage Percentage",
            ["gpu_id"],
            registry=self.metrics_registry,
        )
        self.network_io = Gauge(
            "system_network_io_bytes",
            "Network I/O Bytes",
            ["direction"],
            registry=self.metrics_registry,
        )
        self.process_count = Gauge(
            "system_process_count",
            "Number of Running Processes",
            registry=self.metrics_registry,
        )
        self.uptime = Gauge(
            "system_uptime_seconds",
            "System Uptime in Seconds",
            registry=self.metrics_registry,
        )

    async def start_monitoring(self):
        """Start system monitoring loop"""
        self.running = True
        logger.info("System monitoring started")

        while self.running:
            try:
                health = await self._collect_system_metrics()
                await self._update_metrics(health)
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self) -> SystemHealth:
        """Collect system metrics"""
        # CPU usage (non-blocking sample)
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
        }

        # Process count
        process_count = len(psutil.pids())

        # Uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time

        # GPU metrics (if available)
        gpu_usage = []
        gpu_memory = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_usage.append(gpu.load * 100)
                gpu_memory.append(gpu.memoryUtil * 100)
        except:
            # GPU monitoring not available
            pass

        return SystemHealth(
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            disk_usage=disk_percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            network_io=network_io,
            process_count=process_count,
            uptime=uptime,
            timestamp=datetime.now(),
        )

    async def _update_metrics(self, health: SystemHealth):
        """Update Prometheus metrics"""
        self.cpu_usage.set(health.cpu_usage)
        self.memory_usage.set(health.memory_usage)
        self.disk_usage.labels(mount_point="/").set(health.disk_usage)
        self.process_count.set(health.process_count)
        self.uptime.set(health.uptime)

        # Network I/O
        self.network_io.labels(direction="sent").set(health.network_io["bytes_sent"])
        self.network_io.labels(direction="received").set(
            health.network_io["bytes_recv"]
        )

        # GPU metrics
        for i, usage in enumerate(health.gpu_usage):
            self.gpu_usage.labels(gpu_id=str(i)).set(usage)

        for i, memory in enumerate(health.gpu_memory):
            self.gpu_memory.labels(gpu_id=str(i)).set(memory)

    async def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health asynchronously"""
        try:
            return await self._collect_system_metrics()
        except Exception as exc:
            logger.debug(f"Failed to collect system health: {exc}")
            return None

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        logger.info("System monitoring stopped")


class AlertManager:
    """Alert management and notification system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = []
        self.notification_handlers = {}
        self.alert_history = []
        self.max_history_size = 1000

        alerts_config = dict(config.get("alerts", {}) or {})
        # External (stream) alerts: dedupe by (type, scope, window) and hold a
        # cooldown per (type, scope) so a storm or a replay notifies once.
        self.dedupe_cooldown_seconds = max(
            0.0, float(alerts_config.get("dedupe_cooldown_seconds", 900))
        )
        self.dedupe_max_entries = max(
            1, int(alerts_config.get("dedupe_max_entries", 4096))
        )
        self.notification_timeout_seconds = max(
            0.001, float(alerts_config.get("notification_timeout_seconds", 10))
        )
        self.notification_queue_size = max(
            1, int(alerts_config.get("notification_queue_size", 1000))
        )
        # Replay guard: older stream events are recorded but never notify.
        # None (the default) disables the check.
        max_event_age = alerts_config.get("max_event_age_seconds")
        self.max_event_age_seconds: Optional[float] = (
            None if max_event_age is None else max(0.0, float(max_event_age))
        )
        self.stale_event_count = 0
        self.stale_warning_interval_seconds = 60.0
        self._last_stale_warning_at: Optional[float] = None
        self.sender_drain_timeout_seconds = max(
            0.0, float(alerts_config.get("sender_drain_timeout_seconds", 5))
        )
        self._clock: Callable[[], float] = time.monotonic
        self._seen_alert_identities: "OrderedDict[tuple, None]" = OrderedDict()
        self._active_external_alerts: "OrderedDict[tuple, Dict[str, Any]]" = (
            OrderedDict()
        )
        # Optional Redis record of notified incidents, so a restart or a
        # second process does not page an active incident again.
        store_config = dict(alerts_config.get("dedupe_store", {}) or {})
        self.dedupe_store: Optional[AlertDedupeStore] = (
            AlertDedupeStore(store_config, avoid_rank=self._ACTION_RANK["avoid"])
            if store_config.get("enabled", False)
            else None
        )
        self.suppressed_alert_count = 0
        self.dropped_notification_count = 0
        self._notification_queue: Optional[asyncio.Queue] = None
        self._sender_task: Optional[asyncio.Task] = None

        # Load alert rules
        self._load_alert_rules()

        # Initialize notification handlers
        self._init_notification_handlers()

    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        rules_config = self.config.get("alerts", {})

        # Default alert rules
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > threshold",
                threshold=rules_config.get("error_rate_threshold", 0.05),
                duration=300,  # 5 minutes
                severity="critical",
                description="Error rate is above acceptable threshold",
            ),
            AlertRule(
                name="high_latency",
                condition="avg_latency_ms > threshold",
                threshold=rules_config.get("latency_threshold_ms", 5000),
                duration=300,
                severity="warning",
                description="Average latency is high",
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage > threshold",
                threshold=rules_config.get("memory_usage_threshold", 0.9),
                duration=600,  # 10 minutes
                severity="warning",
                description="System memory usage is high",
            ),
            AlertRule(
                name="high_disk_usage",
                condition="disk_usage > threshold",
                threshold=rules_config.get("disk_usage_threshold", 0.8),
                duration=1800,  # 30 minutes
                severity="warning",
                description="Disk usage is high",
            ),
            AlertRule(
                name="model_unavailable",
                condition="model_health_status == 'unhealthy'",
                threshold=1,
                duration=60,
                severity="critical",
                description="Model is unavailable",
            ),
            AlertRule(
                # Flink metrics drive error-rate and latency alerts; when they
                # stop arriving those rules would silently read zero.
                name="stream_metrics_stale",
                condition="stream_metrics_age_seconds > threshold",
                threshold=float(
                    self.config.get("stream_metrics_staleness_seconds", 300)
                ),
                duration=1800,
                severity="warning",
                description="No stream analytics metrics received recently",
                # Opt-in: Flink only emits windows while there is traffic.
                enabled=bool(rules_config.get("stream_metrics_stale_enabled", False)),
            ),
        ]

        pending_seconds = max(0.0, float(rules_config.get("pending_seconds", 0) or 0))
        resolve_after = rules_config.get("resolve_after_seconds")
        for rule in default_rules:
            rule.pending_seconds = pending_seconds
            if resolve_after is not None:
                rule.resolve_after_seconds = max(0.0, float(resolve_after))

        self.alert_rules.extend(default_rules)
        logger.info(f"Loaded {len(self.alert_rules)} alert rules")

    def _init_notification_handlers(self):
        """Initialize notification handlers"""
        # Slack notification handler
        if self.config.get("slack", {}).get("enabled"):
            self.notification_handlers["slack"] = SlackNotificationHandler(
                self.config["slack"]
            )

        # Email notification handler (placeholder)
        if self.config.get("email", {}).get("enabled"):
            self.notification_handlers["email"] = EmailNotificationHandler(
                self.config["email"]
            )

        # Webhook notification handler
        if self.config.get("webhook", {}).get("enabled"):
            self.notification_handlers["webhook"] = WebhookNotificationHandler(
                self.config["webhook"]
            )

    async def check_alerts(
        self, metrics: Dict[str, float], now: Optional[datetime] = None
    ):
        """Check all alert rules against current metrics.

        A breach must hold for ``pending_seconds`` before it notifies; while
        it keeps firing, ``duration`` is the re-notify cooldown. After a
        firing notification, RESOLVED is sent once the condition has stayed
        false for the rule's resolve_after_seconds (hysteresis against flaps).
        """
        current_time = now or datetime.now(timezone.utc)

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # Evaluate alert condition; None means it could not be
                # evaluated, which is neither a breach nor a recovery.
                triggered = self._evaluate_condition_state(rule, metrics)
                if triggered is None:
                    continue

                if not triggered:
                    rule.pending_since = None
                    if not rule.firing:
                        continue
                    # Resolve hysteresis: a flapping metric must stay healthy
                    # for resolve_after_seconds before RESOLVED is sent.
                    if rule.recovering_since is None:
                        rule.recovering_since = current_time
                    recovered_for = (
                        current_time - rule.recovering_since
                    ).total_seconds()
                    if recovered_for >= rule.effective_resolve_after_seconds():
                        rule.firing = False
                        rule.recovering_since = None
                        # A breach after a real resolve is a new incident, so
                        # it notifies without waiting out the cooldown.
                        rule.last_triggered = None
                        await self._resolve_alert(rule, metrics)
                    continue

                rule.recovering_since = None

                if rule.pending_since is None:
                    rule.pending_since = current_time
                pending_for = (current_time - rule.pending_since).total_seconds()
                if pending_for < rule.pending_seconds:
                    continue

                # Check if we should send notification (debouncing)
                if (
                    not rule.last_triggered
                    or (current_time - rule.last_triggered).total_seconds()
                    > rule.duration
                ):
                    await self._trigger_alert(rule, metrics)
                    rule.last_triggered = current_time
                    rule.trigger_count += 1
                    rule.firing = True

            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")

    def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """Evaluate if alert condition is met"""
        return bool(self._evaluate_condition_state(rule, metrics))

    def _evaluate_condition_state(
        self, rule: AlertRule, metrics: Dict[str, float]
    ) -> Optional[bool]:
        """Evaluate a condition: True/False, or None when evaluation fails."""
        try:
            # Simple condition evaluation
            # In production, use a proper expression parser
            condition = rule.condition.replace("threshold", str(rule.threshold))

            # Map metric names to values
            context = {
                "error_rate": self._get_metric_value(
                    metrics, ["application.error_rate", "error_rate"], 0
                ),
                "avg_latency_ms": self._get_metric_value(
                    metrics, ["application.avg_latency_ms", "avg_latency_ms"], 0
                ),
                "memory_usage": self._get_metric_value(
                    metrics, ["system.memory_usage", "memory_usage"], 0
                ),
                "disk_usage": self._get_metric_value(
                    metrics, ["system.disk_usage", "disk_usage"], 0
                ),
                "cpu_usage": self._get_metric_value(
                    metrics, ["system.cpu_usage", "cpu_usage"], 0
                ),
                "model_health_status": self._get_metric_value(
                    metrics,
                    ["application.model_health_status", "model_health_status"],
                    "healthy",
                ),
                "stream_metrics_age_seconds": self._get_metric_value(
                    metrics,
                    ["streaming.metrics_age_seconds", "stream_metrics_age_seconds"],
                    0,
                ),
            }

            # Evaluate condition
            return bool(eval(condition, {"__builtins__": {}}, context))

        except Exception as e:
            logger.error(f"Failed to evaluate condition for rule {rule.name}: {e}")
            return None

    @staticmethod
    def _get_metric_value(
        metrics: Dict[str, Any], keys: List[str], default: Any
    ) -> Any:
        """Return first available metric value for given keys"""
        for key in keys:
            if key in metrics:
                return metrics[key]
        return default

    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, float]):
        """Trigger alert and send notifications"""
        alert_data = {
            "rule_name": rule.name,
            "severity": rule.severity,
            "description": rule.description,
            "threshold": rule.threshold,
            "current_value": self._get_current_value(rule, metrics),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_count": rule.trigger_count,
            "status": "firing",
        }

        self._append_alert_to_history(alert_data)
        await self._dispatch_alert(alert_data)

    async def _resolve_alert(self, rule: AlertRule, metrics: Dict[str, float]):
        """Notify that a rule which fired has recovered."""
        alert_data = {
            "rule_name": rule.name,
            "severity": rule.severity,
            "description": f"Resolved: {rule.description}",
            "threshold": rule.threshold,
            "current_value": self._get_current_value(rule, metrics),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_count": rule.trigger_count,
            "status": "resolved",
        }
        self._append_alert_to_history(alert_data)
        await self._dispatch_alert(alert_data)

    def _append_alert_to_history(self, alert_data: Dict[str, Any]):
        """Persist an alert in the in-memory alert history."""
        self.alert_history.append(alert_data)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)

    @property
    def sender_running(self) -> bool:
        """Whether notifications go through the background sender."""
        return self._sender_task is not None and not self._sender_task.done()

    def start_sender(self) -> None:
        """Send notifications from one background task fed by a bounded queue.

        Until this runs (for example in tests), alerts are sent inline.
        """
        if self.sender_running:
            return
        self._notification_queue = asyncio.Queue(maxsize=self.notification_queue_size)
        self._sender_task = asyncio.create_task(
            self._sender_loop(self._notification_queue), name="alert_sender"
        )

    async def _sender_loop(self, queue: asyncio.Queue) -> None:
        while True:
            alert_data = await queue.get()
            try:
                await self._send_to_handlers(alert_data)
            finally:
                queue.task_done()

    async def drain_notifications(self) -> None:
        """Wait until every queued notification has been attempted."""
        queue = self._notification_queue
        if queue is not None and self.sender_running:
            await queue.join()

    async def stop_sender(self) -> None:
        """Drain queued notifications briefly, then stop the sender."""
        task = self._sender_task
        if task is None:
            return
        if not task.done():
            try:
                await asyncio.wait_for(
                    self.drain_notifications(),
                    timeout=self.sender_drain_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("Alert sender stopped with notifications queued")
        # Python 3.10's wait_for can swallow a cancel that races completion,
        # so keep cancelling until the task is really done.
        while not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=0.1)
        await asyncio.gather(task, return_exceptions=True)
        self._sender_task = None
        self._notification_queue = None

    async def close(self) -> None:
        """Stop the sender and close handler HTTP clients."""
        await self.stop_sender()
        if self.dedupe_store is not None:
            await self.dedupe_store.close()
        for handler in self.notification_handlers.values():
            aclose = getattr(handler, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    logger.debug("Failed to close alert handler", exc_info=True)

    async def _dispatch_alert(self, alert_data: Dict[str, Any]):
        """Send an alert to all configured notification handlers.

        With the background sender running this only enqueues, so a slow
        handler never blocks the caller (the Kafka consumer); a full queue
        drops the notification and counts it.
        """
        queue = self._notification_queue
        if queue is not None and self.sender_running:
            try:
                queue.put_nowait(alert_data)
            except asyncio.QueueFull:
                self.dropped_notification_count += 1
                logger.warning(
                    "Alert notification queue full; dropped %s",
                    alert_data.get("rule_name"),
                )
            return
        await self._send_to_handlers(alert_data)

    async def _send_to_handlers(self, alert_data: Dict[str, Any]) -> None:
        for handler_name, handler in self.notification_handlers.items():
            try:
                await asyncio.wait_for(
                    handler.send_alert(alert_data),
                    timeout=self.notification_timeout_seconds,
                )
                logger.info(
                    f"Alert sent via {handler_name}: {alert_data.get('rule_name')}"
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Alert via {handler_name} timed out after "
                    f"{self.notification_timeout_seconds}s"
                )
            except Exception as e:
                logger.error(f"Failed to send alert via {handler_name}: {e}")

    @staticmethod
    def _external_alert_keys(
        alert: Dict[str, Any],
    ) -> tuple[tuple, Optional[tuple]]:
        """Return the (type, scope) incident key and the windowed identity."""
        alert_type = str(alert.get("rule_name") or "external_alert")
        if alert.get("scope_type") or alert.get("scope_key"):
            scope: tuple = ("scope", alert.get("scope_type"), alert.get("scope_key"))
        else:
            scope = ("model", alert.get("model_name"), alert.get("provider"))
        incident = (alert.get("source"), alert_type, scope)
        window = (alert.get("window_start_ms"), alert.get("window_end_ms"))
        identity = None if window == (None, None) else (*incident, window)
        return incident, identity

    def _remember_identity(self, identity: tuple) -> None:
        self._seen_alert_identities[identity] = None
        self._seen_alert_identities.move_to_end(identity)
        while len(self._seen_alert_identities) > self.dedupe_max_entries:
            self._seen_alert_identities.popitem(last=False)

    _ACTION_RANK = {"warn": 1, "avoid": 2}
    _SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "critical": 3}

    @classmethod
    def _alert_level(cls, alert: Dict[str, Any]) -> tuple[int, int]:
        """Rank an alert by guardrail action, then severity."""
        return (
            cls._ACTION_RANK.get(str(alert.get("guardrail_action") or ""), 0),
            cls._SEVERITY_RANK.get(str(alert.get("severity") or "").lower(), 1),
        )

    @staticmethod
    def _event_time(alert_data: Dict[str, Any]) -> Optional[datetime]:
        """Return when a stream event happened: emitted_at, else window end."""
        emitted_at = alert_data.get("emitted_at")
        if emitted_at:
            # Unparseable values read as now, i.e. fresh.
            return _parse_event_timestamp(emitted_at)
        window_end_ms = alert_data.get("window_end_ms")
        if isinstance(window_end_ms, (int, float)) and window_end_ms > 0:
            return datetime.fromtimestamp(window_end_ms / 1000, tz=timezone.utc)
        return None

    def _is_stale_event(self, alert_data: Dict[str, Any]) -> bool:
        if self.max_event_age_seconds is None:
            return False
        event_time = self._event_time(alert_data)
        if event_time is None:
            return False
        age = (datetime.now(timezone.utc) - event_time).total_seconds()
        return age > self.max_event_age_seconds

    def _record_stale_event(self, normalized_alert: Dict[str, Any]) -> None:
        """Count a stale firing event; warn at most once per minute."""
        self.stale_event_count += 1
        STALE_ALERT_EVENTS.inc()
        now = self._clock()
        last = self._last_stale_warning_at
        if last is None or now - last >= self.stale_warning_interval_seconds:
            self._last_stale_warning_at = now
            logger.warning(
                "Stale stream alert not notified (older than %ss): %s; "
                "%d stale so far",
                self.max_event_age_seconds,
                normalized_alert.get("rule_name"),
                self.stale_event_count,
            )

    async def _resolve_external_alert(self, normalized_alert: Dict[str, Any]):
        """Resolve the active incidents of the scope a ``clear`` closes.

        Flink's hysteresis tracks only avoids, so a clear closes every avoid
        incident of its scope (it carries only the trigger that entered) plus
        any incident of its own trigger type. Warn-only incidents are never
        cleared by Flink; they stay open until their cooldown runs out.
        """
        incident, _identity = self._external_alert_keys(normalized_alert)
        source, alert_type, scope = incident
        targets = [
            key
            for key, active in self._active_external_alerts.items()
            if key[0] == source
            and key[2] == scope
            and (key[1] == alert_type or active.get("guardrail_action") == "avoid")
        ]
        if targets:
            # Forget the windows seen for the resolved incidents, so a
            # retraction (avoid W, clear W, avoid W) pages again.
            resolved_keys = set(targets)
            for identity in [
                identity
                for identity in self._seen_alert_identities
                if identity[:3] in resolved_keys
            ]:
                del self._seen_alert_identities[identity]
        # Take the targets out before any await, so a concurrent handler
        # evicting one cannot break the resolve.
        closed = [
            active
            for active in (
                self._active_external_alerts.pop(key, None) for key in targets
            )
            if active is not None
        ]
        if self.dedupe_store is not None:
            # The store closes every avoid incident of the scope and the
            # clear's own, which a restarted process may not hold in memory.
            await self.dedupe_store.release(incident, targets)
        for active in closed:
            resolved = {
                **{k: v for k, v in active.items() if not k.startswith("_")},
                "description": f"Resolved: {active.get('description')}",
                "timestamp": normalized_alert["timestamp"],
                "guardrail_action": normalized_alert.get("guardrail_action"),
                "status": "resolved",
            }
            self._append_alert_to_history(resolved)
            await self._dispatch_alert(resolved)

    async def record_external_alert(self, alert_data: Dict[str, Any]):
        """Record and optionally notify on alert events emitted by the stream layer.

        Identical events (same type, scope and window) are recorded once, so
        a guardrail storm or a consumer replay notifies once. A new window of
        an incident that already notified stays quiet for the dedupe cooldown.
        ``guardrail_action == "clear"`` resolves the incident instead.
        """
        metrics = alert_data.get("metrics", {}) if isinstance(alert_data, dict) else {}
        anomaly_type = alert_data.get("anomaly_type")
        event_type = alert_data.get("event_type")
        trigger_type = alert_data.get("trigger_type")
        metric_name = anomaly_type or trigger_type or alert_data.get("alert_type")
        current_value = None

        if anomaly_type == "high_latency":
            current_value = metrics.get("avg_latency_ms")
        elif anomaly_type == "high_volume":
            current_value = metrics.get("queries_per_second")
        elif anomaly_type == "high_error_rate":
            current_value = metrics.get("error_rate")

        normalized_alert = {
            "rule_name": anomaly_type
            or trigger_type
            or alert_data.get("alert_type", "external_alert"),
            "severity": alert_data.get("severity", "warning"),
            "description": alert_data.get("description", "External alert"),
            "threshold": alert_data.get("threshold"),
            "current_value": current_value,
            "timestamp": alert_data.get(
                "emitted_at", datetime.now(timezone.utc).isoformat()
            ),
            "trigger_count": 1,
            "metric_name": metric_name,
            "alert_type": alert_data.get("alert_type"),
            "event_type": event_type,
            "source": (
                "routing_guardrails"
                if event_type == "routing.guardrails"
                else "stream_pipeline"
            ),
            "model_name": alert_data.get("model_name"),
            "provider": alert_data.get("provider"),
            "scope_type": alert_data.get("scope_type"),
            "scope_key": alert_data.get("scope_key"),
            "guardrail_action": alert_data.get("guardrail_action"),
            "window_start_ms": alert_data.get("window_start_ms"),
            "window_end_ms": alert_data.get("window_end_ms"),
            "status": "firing",
        }

        if normalized_alert["guardrail_action"] == "clear":
            # Clears are processed however late they arrive; dropping one
            # would leave its incident open for good.
            await self._resolve_external_alert(normalized_alert)
            return

        incident, identity = self._external_alert_keys(normalized_alert)
        if identity is not None:
            if identity in self._seen_alert_identities:
                self.suppressed_alert_count += 1
                return
            self._remember_identity(identity)

        self._append_alert_to_history(normalized_alert)

        if self._is_stale_event(alert_data):
            # Replayed from an old offset: keep the record, skip the page.
            self._record_stale_event(normalized_alert)
            return

        now = self._clock()
        active = self._active_external_alerts.get(incident)
        if (
            active is not None
            and now - float(active["_notified_at"]) < self.dedupe_cooldown_seconds
            and self._alert_level(normalized_alert) <= self._notified_level(active)
        ):
            # Same incident, no escalation: stay quiet but keep it fresh so
            # a storm does not evict the active incident first.
            self._active_external_alerts.move_to_end(incident)
            self.suppressed_alert_count += 1
            return

        claim = await self._claim_in_dedupe_store(incident, normalized_alert)

        self._active_external_alerts[incident] = {
            **normalized_alert,
            "_notified_at": now,
        }
        if claim is not None and not claim.notify:
            entry = self._active_external_alerts[incident]
            if claim.stored_level is not None:
                entry["_notified_level"] = claim.stored_level
            if claim.remaining_seconds is not None:
                # The page happened when the key was set, not now.
                elapsed = self.dedupe_cooldown_seconds - claim.remaining_seconds
                entry["_notified_at"] = now - max(0.0, elapsed)
        self._active_external_alerts.move_to_end(incident)
        while len(self._active_external_alerts) > self.dedupe_max_entries:
            self._active_external_alerts.popitem(last=False)
        if claim is not None and not claim.notify:
            # Already paged before a restart or by another process: track it
            # (so a clear resolves it) without paging again.
            self.suppressed_alert_count += 1
            return
        await self._dispatch_alert(normalized_alert)

    @classmethod
    def _notified_level(cls, active: Dict[str, Any]) -> tuple[int, int]:
        """The level an active incident paged at, here or (per the store)
        before a restart."""
        level = cls._alert_level(active)
        stored = active.get("_notified_level")
        return max(level, stored) if stored is not None else level

    async def _claim_in_dedupe_store(
        self, incident: tuple, alert: Dict[str, Any]
    ) -> Optional[DedupeClaim]:
        """Ask the shared store whether this page is still due.

        In-memory dedupe already decided this alert should page (new, past
        its cooldown, or an escalation); the store only adds suppression for
        an incident another process (or this one before a restart) already
        paged at the same or a higher level. ``None`` (no store, or Redis
        unavailable) keeps the in-memory decision.
        """
        store = self.dedupe_store
        if store is None or self.dedupe_cooldown_seconds <= 0:
            return None
        return await store.try_claim(
            incident, self._alert_level(alert), self.dedupe_cooldown_seconds
        )

    def _get_current_value(self, rule: AlertRule, metrics: Dict[str, float]) -> float:
        """Get current value for the metric being alerted on"""
        if "error_rate" in rule.condition:
            return metrics.get("error_rate", 0)
        elif "latency" in rule.condition:
            return metrics.get("avg_latency_ms", 0)
        elif "memory" in rule.condition:
            return metrics.get("memory_usage", 0)
        elif "disk" in rule.condition:
            return metrics.get("disk_usage", 0)
        elif "stream_metrics_age_seconds" in rule.condition:
            return self._get_metric_value(
                metrics,
                ["streaming.metrics_age_seconds", "stream_metrics_age_seconds"],
                0,
            )
        else:
            return 0

    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        return {
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules if r.enabled]),
            "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
            "notification_handlers": list(self.notification_handlers.keys()),
        }


class _SharedHTTPClientMixin:
    """One lazily created httpx client per handler, with an explicit timeout."""

    _client: Optional[httpx.AsyncClient] = None
    timeout_seconds: float = 10.0

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds)
            )
        return self._client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()


class SlackNotificationHandler(_SharedHTTPClientMixin):
    """Slack notification handler"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "LLM Router Monitor")
        self.timeout_seconds = float(config.get("timeout_seconds", 10) or 10)
        self._client = None

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack"""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        # Format alert message
        severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}

        emoji = severity_emoji.get(alert_data["severity"], "📊")

        message = {
            "channel": self.channel,
            "username": self.username,
            "text": (
                f"✅ *RESOLVED: {alert_data['rule_name'].upper()}*"
                if alert_data.get("status") == "resolved"
                else f"{emoji} *{alert_data['rule_name'].upper()}*"
            ),
            "attachments": [
                {
                    "color": (
                        "good"
                        if alert_data.get("status") == "resolved"
                        else "danger"
                        if alert_data["severity"] == "critical"
                        else "warning"
                    ),
                    "fields": [
                        {
                            "title": "Description",
                            "value": alert_data["description"],
                            "short": False,
                        },
                        {
                            "title": "Current Value",
                            "value": str(alert_data["current_value"]),
                            "short": True,
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert_data["threshold"]),
                            "short": True,
                        },
                        {
                            "title": "Timestamp",
                            "value": alert_data["timestamp"],
                            "short": True,
                        },
                        {
                            "title": "Trigger Count",
                            "value": str(alert_data["trigger_count"]),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        response = await self._http_client().post(self.webhook_url, json=message)
        response.raise_for_status()


class EmailNotificationHandler:
    """Email notification handler (placeholder)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email"""
        # Placeholder for email implementation
        logger.info(f"Email alert would be sent: {alert_data['rule_name']}")


class WebhookNotificationHandler(_SharedHTTPClientMixin):
    """Generic webhook notification handler"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get("url")
        self.headers = config.get("headers", {})
        self.timeout_seconds = float(config.get("timeout_seconds", 10) or 10)
        self._client = None

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert via webhook"""
        if not self.webhook_url:
            return

        response = await self._http_client().post(
            self.webhook_url, json=alert_data, headers=self.headers
        )
        response.raise_for_status()


class MetricsCollector:
    """Collects and aggregates metrics from various sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_interval = config.get("collection_interval", 30)
        self.metrics_history = []
        self.max_history_size = 1000
        self.stream_metrics_history = []
        self.latest_stream_metrics_by_key: Dict[str, Dict[str, Any]] = {}
        self.routing_feature_history = []
        self.routing_guardrail_history = []
        self.routing_policy_state_history = []
        self.stream_metrics_staleness_seconds = config.get(
            "stream_metrics_staleness_seconds", 300
        )
        # Freshness of the stream itself uses this monitor's receive time, not
        # the producer's emitted_at, so clock skew or replays cannot hide it.
        self._clock: Callable[[], float] = time.monotonic
        self._started_at = self._clock()
        self._last_stream_metrics_received_at: Optional[float] = None

    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all sources"""
        system_metrics = await self._collect_system_metrics()
        application_metrics = await self._collect_application_metrics()
        business_metrics = await self._collect_business_metrics()
        streaming_metrics = {
            **self._collect_streaming_metrics(),
            **self.stream_freshness_metrics()["streaming"],
        }
        routing_metrics = self.get_routing_feature_summary(hours=1)
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "application": application_metrics,
            "business": business_metrics,
            "streaming": streaming_metrics,
            "routing": routing_metrics,
        }

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)

        return metrics

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # CPU, Memory, Disk usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_usage": cpu_percent / 100,
                "memory_usage": memory.percent / 100,
                "disk_usage": (disk.used / disk.total),
                "load_average": psutil.getloadavg()[0]
                if hasattr(psutil, "getloadavg")
                else 0,
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-level metrics"""
        try:
            # Get metrics from Prometheus collectors
            router_metrics = self._extract_prometheus_metrics(ROUTER_METRICS)
            inference_metrics = self._extract_prometheus_metrics(INFERENCE_METRICS)
            pipeline_metrics = self._extract_prometheus_metrics(PIPELINE_METRICS)
            streaming_metrics = self._collect_streaming_metrics()
            application_metrics = {
                **router_metrics,
                **inference_metrics,
                **pipeline_metrics,
            }

            if streaming_metrics.get("request_count", 0) > 0:
                application_metrics.update(
                    {
                        "avg_latency_ms": streaming_metrics.get("avg_latency_ms", 0.0),
                        "error_rate": streaming_metrics.get("error_rate", 0.0),
                        "queries_per_second": streaming_metrics.get(
                            "queries_per_second", 0.0
                        ),
                        "cache_hit_rate": streaming_metrics.get("cache_hit_rate", 0.0),
                    }
                )

            return application_metrics
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {}

    async def _collect_business_metrics(self) -> Dict[str, float]:
        """Collect business-level metrics.

        There is no data source for these yet. Report nothing rather than
        fixed placeholder values that look like real measurements.
        """
        return {}

    def _extract_prometheus_metrics(self, metrics_object) -> Dict[str, float]:
        """Extract values from Prometheus metrics"""
        extracted = {}

        # This is a simplified extraction
        # In practice, you'd iterate through the metrics registry
        try:
            for attr_name in dir(metrics_object):
                attr = getattr(metrics_object, attr_name)
                if hasattr(attr, "_value"):
                    extracted[attr_name] = attr._value._value
        except Exception as e:
            logger.debug(f"Metrics extraction error: {e}")

        return extracted

    def record_stream_model_metrics(self, metric_event: Dict[str, Any]):
        """Store a consumed analytics.model_metrics_1m event for dashboard use."""
        event = dict(metric_event)
        event.setdefault(
            "timestamp",
            event.get("emitted_at", datetime.now(timezone.utc).isoformat()),
        )
        model_name = str(event.get("model_name", "unknown"))
        provider = str(event.get("provider", "unknown"))
        model_provider_key = _stream_model_provider_key(model_name, provider)

        self.stream_metrics_history.append(event)
        if len(self.stream_metrics_history) > self.max_history_size:
            self.stream_metrics_history.pop(0)

        self.latest_stream_metrics_by_key[model_provider_key] = event
        self._last_stream_metrics_received_at = self._clock()

    def stream_metrics_age_seconds(self) -> float:
        """Seconds since the last stream metrics window arrived.

        Before the first window it reads 0, so a deployment without Flink, or
        one still waiting for traffic, never looks stale.
        """
        last = self._last_stream_metrics_received_at
        if last is None:
            return 0.0
        return max(0.0, self._clock() - last)

    def stream_freshness_metrics(self) -> Dict[str, Any]:
        """Return stream freshness in the nested shape alert checks flatten."""
        return {
            "streaming": {
                "metrics_age_seconds": self.stream_metrics_age_seconds(),
                "metrics_received": int(
                    self._last_stream_metrics_received_at is not None
                ),
            }
        }

    def _recent_stream_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Return stream metrics still within the configured freshness window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            metric
            for metric in self.stream_metrics_history
            if _parse_event_timestamp(metric.get("timestamp")) >= cutoff
        ]

    def _fresh_stream_metrics(self) -> List[Dict[str, Any]]:
        """Return the latest non-stale stream window for each model."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.stream_metrics_staleness_seconds
        )
        return [
            metric
            for metric in self.latest_stream_metrics_by_key.values()
            if _parse_event_timestamp(metric.get("timestamp")) >= cutoff
        ]

    def _collect_streaming_metrics(self) -> Dict[str, Any]:
        """Build a lightweight aggregate view from recent Flink model windows."""
        fresh_metrics = self._fresh_stream_metrics()
        if not fresh_metrics:
            return {
                "models": {},
                "request_count": 0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
                "queries_per_second": 0.0,
                "cache_hit_rate": 0.0,
            }

        total_requests = sum(int(m.get("request_count", 0) or 0) for m in fresh_metrics)
        total_errors = sum(int(m.get("error_count", 0) or 0) for m in fresh_metrics)
        total_cached = sum(int(m.get("cached_count", 0) or 0) for m in fresh_metrics)
        weighted_latency = sum(
            float(m.get("avg_latency_ms", 0.0) or 0.0)
            * int(m.get("request_count", 0) or 0)
            for m in fresh_metrics
        )
        total_qps = sum(
            float(m.get("queries_per_second", 0.0) or 0.0) for m in fresh_metrics
        )

        latest_models = {}
        for metric in fresh_metrics:
            model_name = str(metric.get("model_name", "unknown"))
            provider = str(metric.get("provider", "unknown"))
            model_provider_key = _stream_model_provider_key(model_name, provider)
            latest_models[model_provider_key] = {
                "model_name": model_name,
                "provider": metric.get("provider"),
                "request_count": int(metric.get("request_count", 0) or 0),
                "avg_latency_ms": float(metric.get("avg_latency_ms", 0.0) or 0.0),
                "error_rate": float(metric.get("error_rate", 0.0) or 0.0),
                "queries_per_second": float(
                    metric.get("queries_per_second", 0.0) or 0.0
                ),
                "cache_hit_rate": float(metric.get("cache_hit_rate", 0.0) or 0.0),
                "window_end_ms": int(metric.get("window_end_ms", 0) or 0),
            }

        return {
            "models": latest_models,
            "request_count": total_requests,
            "avg_latency_ms": (
                weighted_latency / total_requests if total_requests > 0 else 0.0
            ),
            "error_rate": (
                total_errors / total_requests if total_requests > 0 else 0.0
            ),
            "queries_per_second": total_qps,
            "cache_hit_rate": (
                total_cached / total_requests if total_requests > 0 else 0.0
            ),
        }

    def get_stream_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Expose a dashboard-friendly summary of recent Flink model windows."""
        recent_metrics = self._recent_stream_metrics(hours=hours)
        summary = self._collect_streaming_metrics()
        summary["period_hours"] = hours
        summary["sample_count"] = len(recent_metrics)
        summary["total_requests_in_period"] = sum(
            int(metric.get("request_count", 0) or 0) for metric in recent_metrics
        )
        return summary

    def record_request_routing_features(self, routing_event: Dict[str, Any]):
        """Store a consumed requests.enriched event for dashboard use."""
        event = dict(routing_event)
        event.setdefault(
            "timestamp",
            event.get("emitted_at", datetime.now(timezone.utc).isoformat()),
        )
        self.routing_feature_history.append(event)
        if len(self.routing_feature_history) > self.max_history_size:
            self.routing_feature_history.pop(0)

    def _recent_request_routing_features(self, hours: int = 1) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            event
            for event in self.routing_feature_history
            if _parse_event_timestamp(event.get("timestamp")) >= cutoff
        ]

    def get_routing_feature_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Summarize recent request-side Flink routing features."""
        recent_events = self._recent_request_routing_features(hours=hours)
        if not recent_events:
            return {
                "period_hours": hours,
                "request_count": 0,
                "fast_lane_count": 0,
                "requires_high_reasoning_count": 0,
                "query_type_breakdown": {},
                "query_complexity_breakdown": {},
                "session_hotness_breakdown": {},
                "top_preferred_models": {},
                "top_avoid_models": {},
                "top_avoid_providers": {},
                "recent_requests": [],
            }

        query_type_counts = CollectionCounter(
            str(event.get("query_type", "unknown")) for event in recent_events
        )
        query_complexity_counts = CollectionCounter(
            str(event.get("query_complexity", "unknown")) for event in recent_events
        )
        session_hotness_counts = CollectionCounter(
            str(event.get("session_hotness", "unknown")) for event in recent_events
        )
        preferred_model_counts = CollectionCounter()
        avoid_model_counts = CollectionCounter()
        avoid_provider_counts = CollectionCounter()

        for event in recent_events:
            preferred_model_counts.update(list(event.get("preferred_models", []) or []))
            avoid_model_counts.update(list(event.get("avoid_models", []) or []))
            avoid_provider_counts.update(list(event.get("avoid_providers", []) or []))

        recent_requests = []
        for event in recent_events[-10:]:
            recent_requests.append(
                {
                    "request_id": event.get("request_id"),
                    "query_type": event.get("query_type"),
                    "query_complexity": event.get("query_complexity"),
                    "priority": event.get("priority"),
                    "route_to_fast_lane": bool(event.get("route_to_fast_lane", False)),
                    "requires_low_latency": bool(
                        event.get("requires_low_latency", False)
                    ),
                    "requires_high_reasoning": bool(
                        event.get("requires_high_reasoning", False)
                    ),
                    "session_hotness": event.get("session_hotness"),
                    "cost_sensitivity": event.get("cost_sensitivity"),
                    "error_sensitivity": event.get("error_sensitivity"),
                    "timestamp": event.get("timestamp"),
                }
            )

        return {
            "period_hours": hours,
            "request_count": len(recent_events),
            "fast_lane_count": sum(
                1 for event in recent_events if event.get("route_to_fast_lane")
            ),
            "requires_high_reasoning_count": sum(
                1 for event in recent_events if event.get("requires_high_reasoning")
            ),
            "query_type_breakdown": dict(query_type_counts),
            "query_complexity_breakdown": dict(query_complexity_counts),
            "session_hotness_breakdown": dict(session_hotness_counts),
            "top_preferred_models": dict(preferred_model_counts.most_common(5)),
            "top_avoid_models": dict(avoid_model_counts.most_common(5)),
            "top_avoid_providers": dict(avoid_provider_counts.most_common(5)),
            "recent_requests": recent_requests,
        }

    def record_routing_guardrail(self, guardrail_event: Dict[str, Any]):
        """Store a consumed routing.guardrails event for dashboard use."""
        event = dict(guardrail_event)
        event.setdefault(
            "timestamp",
            event.get("emitted_at", datetime.now(timezone.utc).isoformat()),
        )
        self.routing_guardrail_history.append(event)
        if len(self.routing_guardrail_history) > self.max_history_size:
            self.routing_guardrail_history.pop(0)

    def _recent_routing_guardrails(self, hours: int = 1) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            event
            for event in self.routing_guardrail_history
            if _parse_event_timestamp(event.get("timestamp")) >= cutoff
        ]

    def get_routing_guardrail_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Summarize recent routing guardrails for dashboard use."""
        recent_events = self._recent_routing_guardrails(hours=hours)
        if not recent_events:
            return {
                "period_hours": hours,
                "guardrail_count": 0,
                "scope_breakdown": {},
                "trigger_breakdown": {},
                "recent_guardrails": [],
            }

        scope_counts = CollectionCounter(
            str(event.get("scope_type", "unknown")) for event in recent_events
        )
        trigger_counts = CollectionCounter(
            str(event.get("trigger_type", "unknown")) for event in recent_events
        )
        recent_guardrails = []
        for event in recent_events[-10:]:
            recent_guardrails.append(
                {
                    "scope_type": event.get("scope_type"),
                    "scope_key": event.get("scope_key"),
                    "trigger_type": event.get("trigger_type"),
                    "guardrail_action": event.get("guardrail_action"),
                    "severity": event.get("severity"),
                    "description": event.get("description"),
                    "model_name": event.get("model_name"),
                    "provider": event.get("provider"),
                    "timestamp": event.get("timestamp"),
                }
            )

        return {
            "period_hours": hours,
            "guardrail_count": len(recent_events),
            "scope_breakdown": dict(scope_counts),
            "trigger_breakdown": dict(trigger_counts),
            "recent_guardrails": recent_guardrails,
        }

    def record_routing_policy_state(self, state_event: Dict[str, Any]):
        """Store a consumed routing.policy_state event for dashboard use."""
        event = dict(state_event)
        event.setdefault(
            "timestamp",
            event.get("emitted_at", datetime.now(timezone.utc).isoformat()),
        )
        self.routing_policy_state_history.append(event)
        if len(self.routing_policy_state_history) > self.max_history_size:
            self.routing_policy_state_history.pop(0)

    def _recent_routing_policy_states(self, hours: int = 1) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            event
            for event in self.routing_policy_state_history
            if _parse_event_timestamp(event.get("timestamp")) >= cutoff
        ]

    def get_routing_policy_state_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Summarize recent rolling routing policy state for dashboard use."""
        recent_events = self._recent_routing_policy_states(hours=hours)
        if not recent_events:
            return {
                "period_hours": hours,
                "state_count": 0,
                "scope_breakdown": {},
                "query_complexity_breakdown": {},
                "dominant_query_type_breakdown": {},
                "burst_protection_count": 0,
                "enterprise_priority_count": 0,
                "route_to_fast_lane_count": 0,
                "top_preferred_models": {},
                "top_avoid_models": {},
                "top_avoid_providers": {},
                "recent_states": [],
            }

        scope_counts = CollectionCounter(
            str(event.get("scope_type", "unknown")) for event in recent_events
        )
        complexity_counts = CollectionCounter(
            str(event.get("query_complexity", "unknown")) for event in recent_events
        )
        query_type_counts = CollectionCounter(
            str(event.get("dominant_query_type", "unknown")) for event in recent_events
        )
        preferred_model_counts = CollectionCounter()
        avoid_model_counts = CollectionCounter()
        avoid_provider_counts = CollectionCounter()

        for event in recent_events:
            preferred_model_counts.update(list(event.get("preferred_models", []) or []))
            avoid_model_counts.update(list(event.get("avoid_models", []) or []))
            avoid_provider_counts.update(list(event.get("avoid_providers", []) or []))

        recent_states = []
        for event in recent_events[-10:]:
            recent_states.append(
                {
                    "scope_type": event.get("scope_type"),
                    "scope_key": event.get("scope_key"),
                    "user_tier": event.get("user_tier"),
                    "recent_request_count": int(
                        event.get("recent_request_count", 0) or 0
                    ),
                    "recent_error_rate": float(
                        event.get("recent_error_rate", 0.0) or 0.0
                    ),
                    "avg_latency_ms": float(event.get("avg_latency_ms", 0.0) or 0.0),
                    "fast_lane_hit_rate": float(
                        event.get("fast_lane_hit_rate", 0.0) or 0.0
                    ),
                    "dominant_query_type": event.get("dominant_query_type"),
                    "query_complexity": event.get("query_complexity"),
                    "requires_low_latency": bool(
                        event.get("requires_low_latency", False)
                    ),
                    "requires_high_reasoning": bool(
                        event.get("requires_high_reasoning", False)
                    ),
                    "route_to_fast_lane": bool(event.get("route_to_fast_lane", False)),
                    "burst_protection_active": bool(
                        event.get("burst_protection_active", False)
                    ),
                    "enterprise_priority_active": bool(
                        event.get("enterprise_priority_active", False)
                    ),
                    "preferred_models": list(event.get("preferred_models", []) or []),
                    "avoid_models": list(event.get("avoid_models", []) or []),
                    "avoid_providers": list(event.get("avoid_providers", []) or []),
                    "hint_reason": event.get("hint_reason"),
                    "timestamp": event.get("timestamp"),
                }
            )

        return {
            "period_hours": hours,
            "state_count": len(recent_events),
            "scope_breakdown": dict(scope_counts),
            "query_complexity_breakdown": dict(complexity_counts),
            "dominant_query_type_breakdown": dict(query_type_counts),
            "burst_protection_count": sum(
                1 for event in recent_events if event.get("burst_protection_active")
            ),
            "enterprise_priority_count": sum(
                1 for event in recent_events if event.get("enterprise_priority_active")
            ),
            "route_to_fast_lane_count": sum(
                1 for event in recent_events if event.get("route_to_fast_lane")
            ),
            "top_preferred_models": dict(preferred_model_counts.most_common(5)),
            "top_avoid_models": dict(avoid_model_counts.most_common(5)),
            "top_avoid_providers": dict(avoid_provider_counts.most_common(5)),
            "recent_states": recent_states,
        }

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

        if not recent_metrics:
            return {}

        # Calculate averages and trends
        summary = {
            "period_hours": hours,
            "sample_count": len(recent_metrics),
            "avg_cpu_usage": self._average_metric(recent_metrics, "system.cpu_usage"),
            "avg_memory_usage": self._average_metric(
                recent_metrics, "system.memory_usage"
            ),
            "avg_latency": self._average_metric(
                recent_metrics, "application.avg_latency_ms"
            ),
            "error_rate": self._average_metric(
                recent_metrics, "application.error_rate"
            ),
        }

        return summary

    def _average_metric(self, metrics_list: List[Dict], metric_path: str) -> float:
        """Calculate average for a nested metric"""
        values = []
        for metrics in metrics_list:
            try:
                keys = metric_path.split(".")
                value = metrics
                for key in keys:
                    value = value[key]
                if isinstance(value, (int, float)):
                    values.append(value)
            except (KeyError, TypeError):
                continue

        return sum(values) / len(values) if values else 0


class PerformanceProfiler:
    """Performance profiling and optimization recommendations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profiling_enabled = config.get("profiling_enabled", False)
        self.performance_data = []

    async def profile_inference_request(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile a single inference request"""
        if not self.profiling_enabled:
            return {}

        profile_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_data.get("request_id"),
            "model_name": request_data.get("model_name"),
            "token_count": request_data.get("token_count", 0),
            "latency_ms": request_data.get("latency_ms", 0),
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_usage_percent": psutil.cpu_percent(),
        }

        self.performance_data.append(profile_data)

        # Keep only recent data
        if len(self.performance_data) > 10000:
            self.performance_data = self.performance_data[-5000:]

        return profile_data

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0

    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        if not self.performance_data:
            return {"recommendations": []}

        recommendations = []

        # Analyze latency patterns
        recent_data = self.performance_data[-1000:]  # Last 1000 requests
        avg_latency = sum(d["latency_ms"] for d in recent_data) / len(recent_data)

        if avg_latency > 2000:  # 2 seconds
            recommendations.append(
                {
                    "type": "latency",
                    "priority": "high",
                    "description": f"Average latency is {avg_latency:.0f}ms, consider model optimization",
                    "suggestions": [
                        "Enable response caching",
                        "Use smaller models for simple queries",
                        "Implement request batching",
                    ],
                }
            )

        # Analyze memory usage
        avg_memory = sum(d["memory_usage_mb"] for d in recent_data) / len(recent_data)
        if avg_memory > 8000:  # 8GB
            recommendations.append(
                {
                    "type": "memory",
                    "priority": "medium",
                    "description": f"High memory usage detected: {avg_memory:.0f}MB",
                    "suggestions": [
                        "Implement context compression",
                        "Reduce batch sizes",
                        "Consider model quantization",
                    ],
                }
            )

        # Analyze model usage patterns
        model_usage = {}
        for data in recent_data:
            model = data["model_name"]
            model_usage[model] = model_usage.get(model, 0) + 1

        if len(model_usage) > 1:
            most_used = max(model_usage.keys(), key=model_usage.get)
            usage_percent = model_usage[most_used] / len(recent_data) * 100

            if usage_percent > 80:
                recommendations.append(
                    {
                        "type": "routing",
                        "priority": "low",
                        "description": f"Model {most_used} handles {usage_percent:.1f}% of requests",
                        "suggestions": [
                            "Review routing rules",
                            "Consider load balancing",
                            "Evaluate model performance differences",
                        ],
                    }
                )

        return {
            "analysis_period": "last_1000_requests",
            "avg_latency_ms": avg_latency,
            "avg_memory_mb": avg_memory,
            "model_distribution": model_usage,
            "recommendations": recommendations,
        }


class MonitoringService:
    """Main monitoring service coordinator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_monitor = SystemMonitor(config)
        self.alert_manager = AlertManager(config)
        self.metrics_collector = MetricsCollector(config)
        self.performance_profiler = PerformanceProfiler(config)
        self.running = False

        # Monitoring intervals
        self.health_check_interval = config.get("health_checks", {}).get(
            "interval_seconds", 30
        )
        self.metrics_collection_interval = config.get("metrics_collection_interval", 60)

    async def initialize(self):
        """Initialize monitoring service"""
        logger.info("Initializing monitoring service...")
        # Monitoring service is ready immediately
        logger.info("Monitoring service initialized successfully")

    async def start(self):
        """Start all monitoring components"""
        logger.info("Starting monitoring service...")
        self.running = True
        self.alert_manager.start_sender()

        # Start monitoring tasks
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(self._monitoring_loop(), name="monitor_health")
            task_group.create_task(
                self.system_monitor.start_monitoring(), name="monitor_system"
            )
            task_group.create_task(self._alert_checking_loop(), name="monitor_alerts")
            task_group.create_task(
                self._metrics_collection_loop(), name="monitor_metrics"
            )

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Perform health checks
                health_status = await self._perform_health_checks()

                # Log health status
                if not health_status["overall_healthy"]:
                    logger.warning(f"System health issues detected: {health_status}")

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _alert_checking_loop(self):
        """Alert checking loop"""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = await self.metrics_collector.collect_all_metrics()

                # Flatten metrics for alert checking
                flat_metrics = self._flatten_metrics(current_metrics)

                # Check alerts
                await self.alert_manager.check_alerts(flat_metrics)

                await asyncio.sleep(60)  # Check alerts every minute

            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.running:
            try:
                # Collect and store metrics
                await self.metrics_collector.collect_all_metrics()

                await asyncio.sleep(self.metrics_collection_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.metrics_collection_interval)

    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "components": {},
        }

        # System health
        system_health = await self.system_monitor.get_current_health()
        if system_health:
            health_status["components"]["system"] = {
                "healthy": system_health.cpu_usage < 90
                and system_health.memory_usage < 90,
                "cpu_usage": system_health.cpu_usage,
                "memory_usage": system_health.memory_usage,
                "disk_usage": system_health.disk_usage,
            }

            if not health_status["components"]["system"]["healthy"]:
                health_status["overall_healthy"] = False

        # Add other component health checks here
        # (router, inference engine, pipeline, etc.)

        return health_status

    def _flatten_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested metrics dictionary"""
        flat = {}

        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, (int, float)):
                    flat[new_key] = value

        flatten_dict(metrics)
        return flat

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return await self.metrics_collector.collect_all_metrics()

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report"""
        return await self.performance_profiler.generate_optimization_report()

    async def ingest_stream_model_metrics(self, metric_event: Dict[str, Any]):
        """Consume windowed analytics events forwarded by the pipeline."""
        self.metrics_collector.record_stream_model_metrics(metric_event)

    async def ingest_stream_request_enriched(self, routing_event: Dict[str, Any]):
        """Consume request enrichment events forwarded by the pipeline."""
        self.metrics_collector.record_request_routing_features(routing_event)

    async def ingest_stream_routing_guardrail(self, guardrail_event: Dict[str, Any]):
        """Consume routing guardrail events forwarded by the pipeline."""
        self.metrics_collector.record_routing_guardrail(guardrail_event)
        await self.alert_manager.record_external_alert(guardrail_event)

    async def ingest_stream_routing_policy_state(self, state_event: Dict[str, Any]):
        """Consume routing.policy_state events forwarded by the pipeline."""
        self.metrics_collector.record_routing_policy_state(state_event)

    async def ingest_stream_alert(self, alert_event: Dict[str, Any]):
        """Consume alert events forwarded by the pipeline."""
        await self.alert_manager.record_external_alert(alert_event)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        current_metrics = await self.metrics_collector.collect_all_metrics()
        metrics_summary = self.metrics_collector.get_metrics_summary(hours=1)
        stream_analytics = self.metrics_collector.get_stream_metrics_summary(hours=1)
        routing_features = self.metrics_collector.get_routing_feature_summary(hours=1)
        routing_guardrails = self.metrics_collector.get_routing_guardrail_summary(
            hours=1
        )
        routing_policy_state = self.metrics_collector.get_routing_policy_state_summary(
            hours=1
        )
        alert_status = self.alert_manager.get_alert_status()
        performance_report = (
            await self.performance_profiler.generate_optimization_report()
        )

        return {
            "current_metrics": current_metrics,
            "metrics_summary": metrics_summary,
            "stream_analytics": stream_analytics,
            "routing_features": routing_features,
            "routing_guardrails": routing_guardrails,
            "routing_policy_state": routing_policy_state,
            "alert_status": alert_status,
            "performance_report": performance_report,
            "timestamp": datetime.now().isoformat(),
        }

    def is_healthy(self) -> bool:
        """Check if monitoring service is healthy"""
        return self.running

    async def shutdown(self):
        """Shutdown monitoring service"""
        logger.info("Shutting down monitoring service...")
        self.running = False
        self.system_monitor.stop_monitoring()
        await self.alert_manager.close()
        logger.info("Monitoring service shutdown complete")
