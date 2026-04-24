"""
Policy cache materialization for Flink routing hints.

This layer keeps Kafka and Flink off the synchronous API path:
- background workers consume requests.enriched / fast_lane_hints
- background workers consume routing.guardrails
- hints are materialized into Redis with a tiny in-process L1 cache
- API workers read the shared policy cache during routing
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition
import redis.asyncio as redis

from src.utils.metrics import PIPELINE_METRICS

logger = logging.getLogger(__name__)


class RoutingPolicyCache:
    """Shared routing policy cache with Redis durability and a small L1 cache."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        redis_config = self.config.get("redis", {})
        self.redis_host = redis_config.get("host", "localhost")
        self.redis_port = int(redis_config.get("port", 6379))
        self.redis_db = int(redis_config.get("db", 1))
        self.key_prefix = redis_config.get("key_prefix", "routing_policy")
        self.request_ttl_seconds = int(self.config.get("request_ttl_seconds", 300))
        self.session_ttl_seconds = int(self.config.get("session_ttl_seconds", 600))
        self.user_ttl_seconds = int(self.config.get("user_ttl_seconds", 900))
        self.guardrail_ttl_seconds = int(self.config.get("guardrail_ttl_seconds", 180))
        self.local_cache_ttl_seconds = int(
            self.config.get("local_cache_ttl_seconds", 5)
        )
        self.redis_client = None
        self._initialized = False
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._local_guardrail_index: Dict[str, set[str]] = {
            "model": set(),
            "provider": set(),
        }

    async def initialize(self):
        """Initialize Redis if the policy cache is enabled."""
        if self._initialized or not self.enabled:
            self._initialized = True
            return

        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
            )
            await self.redis_client.ping()
            self._initialized = True
            logger.info("Routing policy cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize routing policy cache: {e}")
            self.enabled = False
            self._initialized = True

    def is_healthy(self) -> bool:
        """Return whether the shared policy cache is available."""
        return not self.enabled or self.redis_client is not None

    def _cache_key(self, namespace: str, identifier: str) -> str:
        return f"{self.key_prefix}:{namespace}:{identifier}"

    def _guardrail_index_key(self, scope_type: str) -> str:
        return self._cache_key("guardrails", scope_type)

    def _guardrail_cache_key(self, scope_type: str, identifier: str) -> str:
        return self._cache_key(f"guardrail:{scope_type}", identifier)

    def _user_hint_cache_key(self, user_id: str) -> str:
        return self._cache_key("user_hint", user_id)

    def _user_state_cache_key(self, user_id: str) -> str:
        return self._cache_key("user_state", user_id)

    def _local_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        record = self._local_cache.get(cache_key)
        if not record:
            return None

        if record["expires_at"] <= time.time():
            self._local_cache.pop(cache_key, None)
            return None

        return dict(record["value"])

    def _local_set(self, cache_key: str, value: Dict[str, Any], ttl_seconds: int):
        self._local_cache[cache_key] = {
            "value": dict(value),
            "expires_at": time.time() + max(1, ttl_seconds),
        }

    async def _get_json(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cached = self._local_get(cache_key)
        if cached is not None:
            return cached

        if not self.enabled or not self.redis_client:
            return None

        try:
            payload = await self.redis_client.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to read routing policy cache key {cache_key}: {e}")
            return None

        if not payload:
            return None

        try:
            value = json.loads(payload)
            self._local_set(cache_key, value, self.local_cache_ttl_seconds)
            return value
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON payload in routing policy cache: {cache_key}")
            return None

    async def _set_json(self, cache_key: str, value: Dict[str, Any], ttl_seconds: int):
        self._local_set(cache_key, value, ttl_seconds)
        if not self.enabled or not self.redis_client:
            return

        try:
            await self.redis_client.setex(cache_key, ttl_seconds, json.dumps(value))
        except Exception as e:
            logger.warning(f"Failed to write routing policy cache key {cache_key}: {e}")

    async def get_request_policy(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request-scoped policy hints if present."""
        if not request_id:
            return None
        return await self._get_json(self._cache_key("request", request_id))

    async def get_user_policy(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the merged user-scoped policy view."""
        if not user_id:
            return None
        user_state_policy = await self.get_user_state_policy(user_id)
        user_hint_policy = await self.get_user_hint_policy(user_id)

        if user_state_policy and user_hint_policy:
            return self._merge_policy_records(user_state_policy, user_hint_policy)
        if user_hint_policy:
            return user_hint_policy
        return user_state_policy

    async def get_user_hint_policy(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user-scoped fast-lane hint policy if present."""
        if not user_id:
            return None
        return await self._get_json(self._user_hint_cache_key(user_id))

    async def get_user_state_policy(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get rolling user-scoped policy state if present."""
        if not user_id:
            return None
        return await self._get_json(self._user_state_cache_key(user_id))

    async def get_session_policy(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session-scoped rolling routing policy if present."""
        if not session_id:
            return None
        return await self._get_json(self._cache_key("session", session_id))

    async def get_effective_policy(
        self,
        request_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge request, session, user, and guardrail policy views for the router."""
        request_policy = await self.get_request_policy(request_id or "")
        session_policy = await self.get_session_policy(session_id or "")
        user_policy = await self.get_user_policy(user_id or "")
        active_guardrails = await self.get_active_guardrails()
        merged: Dict[str, Any] = {}
        policy_sources: List[str] = []

        if user_policy:
            merged = dict(user_policy)
            policy_sources.append("user")
        if session_policy:
            merged = self._merge_policy_records(merged, session_policy)
            policy_sources.append("session")
        if request_policy:
            merged = self._merge_policy_records(merged, request_policy)
            policy_sources.append("request")

        if policy_sources:
            merged["policy_source"] = "+".join(reversed(policy_sources))
            merged.update(active_guardrails)
            return merged

        return dict(active_guardrails)

    def _merge_policy_records(
        self, base_policy: Dict[str, Any], override_policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge request and user policies while preserving list semantics."""
        merged = dict(base_policy)
        list_fields = {
            "preferred_models",
            "avoid_models",
            "avoid_providers",
        }

        for key, value in override_policy.items():
            if value in (None, "", [], {}):
                continue

            if key in list_fields:
                combined = []
                for item in list(value) + list(merged.get(key, []) or []):
                    if item not in combined:
                        combined.append(item)
                merged[key] = combined
                continue

            merged[key] = value

        return merged

    async def materialize_request_enriched(self, event: Dict[str, Any]):
        """Materialize requests.enriched events into request/user caches."""
        if not event:
            return

        request_id = str(event.get("request_id", "") or "")
        user_id = str(event.get("user_id", "") or "")
        route_to_fast_lane = bool(event.get("route_to_fast_lane", False))
        priority = str(event.get("priority", "") or "")
        updated_at = event.get("emitted_at")

        request_policy = {
            "request_id": request_id,
            "user_id": user_id,
            "priority": priority,
            "route_to_fast_lane": route_to_fast_lane,
            "query_type": event.get("query_type"),
            "query_complexity": event.get("query_complexity"),
            "requires_low_latency": bool(event.get("requires_low_latency", False)),
            "requires_high_reasoning": bool(
                event.get("requires_high_reasoning", False)
            ),
            "long_context": bool(event.get("long_context", False)),
            "attachment_heavy": bool(event.get("attachment_heavy", False)),
            "code_heavy": bool(event.get("code_heavy", False)),
            "session_hotness": event.get("session_hotness"),
            "cost_sensitivity": event.get("cost_sensitivity"),
            "error_sensitivity": event.get("error_sensitivity"),
            "preferred_models": list(event.get("preferred_models", []) or []),
            "avoid_models": list(event.get("avoid_models", []) or []),
            "avoid_providers": list(event.get("avoid_providers", []) or []),
            "hint_reason": event.get("enrichment_stage", "priority_classification"),
            "source_event_type": event.get("event_type", "requests.enriched"),
            "updated_at": updated_at,
        }

        if request_id:
            await self._set_json(
                self._cache_key("request", request_id),
                request_policy,
                self.request_ttl_seconds,
            )

        # User-level promotions should be short-lived and only written on fast-lane wins.
        if route_to_fast_lane and user_id:
            existing_user_policy = await self.get_user_hint_policy(user_id)
            user_policy = {
                "request_id": request_id,
                "user_id": user_id,
                "priority": priority,
                "route_to_fast_lane": True,
                "query_type": event.get("query_type"),
                "query_complexity": event.get("query_complexity"),
                "requires_low_latency": bool(event.get("requires_low_latency", False)),
                "requires_high_reasoning": bool(
                    event.get("requires_high_reasoning", False)
                ),
                "long_context": bool(event.get("long_context", False)),
                "attachment_heavy": bool(event.get("attachment_heavy", False)),
                "code_heavy": bool(event.get("code_heavy", False)),
                "session_hotness": event.get("session_hotness"),
                "cost_sensitivity": event.get("cost_sensitivity"),
                "error_sensitivity": event.get("error_sensitivity"),
                "preferred_models": list(event.get("preferred_models", []) or []),
                "avoid_models": list(event.get("avoid_models", []) or []),
                "avoid_providers": list(event.get("avoid_providers", []) or []),
                "hint_reason": f"requests.enriched priority={priority}",
                "source_event_type": event.get("event_type", "requests.enriched"),
                "updated_at": updated_at,
            }
            if existing_user_policy:
                user_policy = self._merge_policy_records(
                    existing_user_policy, user_policy
                )
            await self._set_json(
                self._user_hint_cache_key(user_id),
                user_policy,
                self.user_ttl_seconds,
            )

    async def materialize_fast_lane_hint(self, event: Dict[str, Any]):
        """Materialize fast_lane_hints into request/user caches."""
        if not event:
            return

        request_id = str(event.get("request_id", "") or "")
        user_id = str(event.get("user_id", "") or "")
        existing_request_policy = await self.get_request_policy(request_id)
        existing_user_policy = await self.get_user_hint_policy(user_id)
        selected_model = event.get("selected_model")
        preferred_models = [selected_model] if selected_model else []
        updated_at = event.get("emitted_at")
        route_to_fast_lane = bool(event.get("route_to_fast_lane", False))
        requires_low_latency = event.get("requires_low_latency")
        requires_high_reasoning = event.get("requires_high_reasoning")
        long_context = event.get("long_context")
        attachment_heavy = event.get("attachment_heavy")
        code_heavy = event.get("code_heavy")

        request_policy = {
            "request_id": request_id,
            "user_id": user_id,
            "priority": event.get("priority"),
            "route_to_fast_lane": route_to_fast_lane,
            "query_type": event.get("query_type"),
            "query_complexity": event.get("query_complexity"),
            "requires_low_latency": (
                bool(requires_low_latency) if requires_low_latency is not None else None
            ),
            "requires_high_reasoning": (
                bool(requires_high_reasoning)
                if requires_high_reasoning is not None
                else None
            ),
            "long_context": bool(long_context) if long_context is not None else None,
            "attachment_heavy": (
                bool(attachment_heavy) if attachment_heavy is not None else None
            ),
            "code_heavy": bool(code_heavy) if code_heavy is not None else None,
            "session_hotness": event.get("session_hotness"),
            "cost_sensitivity": event.get("cost_sensitivity"),
            "error_sensitivity": event.get("error_sensitivity"),
            "preferred_models": preferred_models,
            "avoid_models": list(event.get("avoid_models", []) or []),
            "avoid_providers": list(event.get("avoid_providers", []) or []),
            "hint_reason": event.get("hint_reason", "fast_lane_candidate"),
            "source_event_type": event.get("event_type", "fast_lane_hints"),
            "updated_at": updated_at,
        }
        if existing_request_policy:
            request_policy = self._merge_policy_records(
                existing_request_policy, request_policy
            )

        if request_id:
            await self._set_json(
                self._cache_key("request", request_id),
                request_policy,
                self.request_ttl_seconds,
            )

        if route_to_fast_lane and user_id:
            user_policy = {
                "request_id": request_id,
                "user_id": user_id,
                "priority": event.get("priority"),
                "route_to_fast_lane": True,
                "query_type": event.get("query_type"),
                "query_complexity": event.get("query_complexity"),
                "requires_low_latency": (
                    bool(requires_low_latency)
                    if requires_low_latency is not None
                    else None
                ),
                "requires_high_reasoning": (
                    bool(requires_high_reasoning)
                    if requires_high_reasoning is not None
                    else None
                ),
                "long_context": (
                    bool(long_context) if long_context is not None else None
                ),
                "attachment_heavy": (
                    bool(attachment_heavy) if attachment_heavy is not None else None
                ),
                "code_heavy": bool(code_heavy) if code_heavy is not None else None,
                "session_hotness": event.get("session_hotness"),
                "cost_sensitivity": event.get("cost_sensitivity"),
                "error_sensitivity": event.get("error_sensitivity"),
                "preferred_models": preferred_models,
                "avoid_models": list(event.get("avoid_models", []) or []),
                "avoid_providers": list(event.get("avoid_providers", []) or []),
                "hint_reason": event.get("hint_reason", "fast_lane_candidate"),
                "source_event_type": event.get("event_type", "fast_lane_hints"),
                "updated_at": updated_at,
            }
            if existing_user_policy:
                user_policy = self._merge_policy_records(
                    existing_user_policy, user_policy
                )
            await self._set_json(
                self._user_hint_cache_key(user_id),
                user_policy,
                self.user_ttl_seconds,
            )

    async def materialize_routing_policy_state(self, event: Dict[str, Any]):
        """Materialize rolling user/session policy state emitted by Flink analytics."""
        if not event:
            return

        scope_type = str(event.get("scope_type", "") or "").lower()
        scope_key = str(event.get("scope_key", "") or "")
        if scope_type not in {"user", "session"} or not scope_key:
            return

        payload = {
            "scope_type": scope_type,
            "scope_key": scope_key,
            "user_id": event.get("user_id"),
            "session_id": event.get("session_id"),
            "user_tier": event.get("user_tier"),
            "query_type": event.get("query_type"),
            "query_complexity": event.get("query_complexity"),
            "requires_low_latency": bool(event.get("requires_low_latency", False)),
            "requires_high_reasoning": bool(
                event.get("requires_high_reasoning", False)
            ),
            "route_to_fast_lane": bool(event.get("route_to_fast_lane", False)),
            "session_hotness": event.get("session_hotness"),
            "cost_sensitivity": event.get("cost_sensitivity"),
            "error_sensitivity": event.get("error_sensitivity"),
            "burst_protection_active": bool(
                event.get("burst_protection_active", False)
            ),
            "enterprise_priority_active": bool(
                event.get("enterprise_priority_active", False)
            ),
            "recent_request_count": int(event.get("recent_request_count", 0) or 0),
            "recent_error_rate": float(event.get("recent_error_rate", 0.0) or 0.0),
            "avg_total_tokens": float(event.get("avg_total_tokens", 0.0) or 0.0),
            "avg_cost_usd": float(event.get("avg_cost_usd", 0.0) or 0.0),
            "avg_latency_ms": float(event.get("avg_latency_ms", 0.0) or 0.0),
            "fast_lane_hit_rate": float(event.get("fast_lane_hit_rate", 0.0) or 0.0),
            "query_type_breakdown": dict(event.get("query_type_breakdown", {}) or {}),
            "preferred_models": list(event.get("preferred_models", []) or []),
            "avoid_models": list(event.get("avoid_models", []) or []),
            "avoid_providers": list(event.get("avoid_providers", []) or []),
            "hint_reason": event.get("hint_reason", "rolling_policy_state"),
            "source_event_type": event.get("event_type", "routing.policy_state"),
            "updated_at": event.get("emitted_at"),
        }

        cache_key = self._cache_key("session", scope_key)
        ttl_seconds = self.session_ttl_seconds
        if scope_type == "user":
            cache_key = self._user_state_cache_key(scope_key)
            ttl_seconds = self.user_ttl_seconds
        await self._set_json(
            cache_key,
            payload,
            ttl_seconds,
        )

    async def materialize_routing_guardrail(self, event: Dict[str, Any]):
        """Materialize routing.guardrails events into model/provider guardrail views."""
        if not event:
            return

        scope_type = str(event.get("scope_type", "") or "").lower()
        scope_key = str(event.get("scope_key", "") or "")
        if scope_type not in {"model", "provider"} or not scope_key:
            return

        payload = {
            "scope_type": scope_type,
            "scope_key": scope_key,
            "guardrail_action": str(event.get("guardrail_action", "avoid") or "avoid"),
            "severity": str(event.get("severity", "warning") or "warning"),
            "trigger_type": str(event.get("trigger_type", "unknown") or "unknown"),
            "description": str(event.get("description", "") or ""),
            "provider": event.get("provider"),
            "model_name": event.get("model_name"),
            "updated_at": event.get("emitted_at"),
            "source_event_type": event.get("event_type", "routing.guardrails"),
        }

        if not self.enabled or not self.redis_client:
            self._local_guardrail_index.setdefault(scope_type, set()).add(scope_key)
            self._local_set(
                self._guardrail_cache_key(scope_type, scope_key),
                payload,
                self.guardrail_ttl_seconds,
            )
            return

        cache_key = self._guardrail_cache_key(scope_type, scope_key)
        index_key = self._guardrail_index_key(scope_type)
        await self._set_json(cache_key, payload, self.guardrail_ttl_seconds)
        try:
            await self.redis_client.sadd(index_key, scope_key)
        except Exception as e:
            logger.warning(
                f"Failed to update routing guardrail index {scope_type}:{scope_key}: {e}"
            )

    async def get_active_guardrails(self) -> Dict[str, Any]:
        """Return active model/provider guardrails for router candidate filtering."""
        active_model_guardrails = await self._read_guardrail_scope("model")
        active_provider_guardrails = await self._read_guardrail_scope("provider")

        policy = {
            "blocked_models": [],
            "warn_models": [],
            "blocked_providers": [],
            "warn_providers": [],
            "guardrail_reasons": {},
        }

        for scope_type, guardrails in [
            ("model", active_model_guardrails),
            ("provider", active_provider_guardrails),
        ]:
            for scope_key, payload in guardrails.items():
                guardrail_action = str(
                    payload.get("guardrail_action", "avoid") or "avoid"
                )
                reason = payload.get("description") or payload.get("trigger_type")
                if scope_type == "model":
                    target_key = (
                        "blocked_models"
                        if guardrail_action == "avoid"
                        else "warn_models"
                    )
                else:
                    target_key = (
                        "blocked_providers"
                        if guardrail_action == "avoid"
                        else "warn_providers"
                    )
                if scope_key not in policy[target_key]:
                    policy[target_key].append(scope_key)
                policy["guardrail_reasons"][f"{scope_type}:{scope_key}"] = reason

        return policy

    async def _read_guardrail_scope(self, scope_type: str) -> Dict[str, Dict[str, Any]]:
        """Read active guardrails for one scope and prune expired entries."""
        active: Dict[str, Dict[str, Any]] = {}

        if not self.enabled or not self.redis_client:
            for scope_key in list(self._local_guardrail_index.get(scope_type, set())):
                payload = self._local_get(
                    self._guardrail_cache_key(scope_type, scope_key)
                )
                if payload is None:
                    self._local_guardrail_index.setdefault(scope_type, set()).discard(
                        scope_key
                    )
                    continue
                active[scope_key] = payload
            return active

        index_key = self._guardrail_index_key(scope_type)
        try:
            members = await self.redis_client.smembers(index_key)
        except Exception as e:
            logger.warning(f"Failed to read routing guardrail index {scope_type}: {e}")
            return active

        for member in members:
            scope_key = str(member)
            payload = await self._get_json(
                self._guardrail_cache_key(scope_type, scope_key)
            )
            if payload is None:
                try:
                    await self.redis_client.srem(index_key, scope_key)
                except Exception as e:
                    logger.warning(
                        f"Failed to prune expired routing guardrail {scope_type}:{scope_key}: {e}"
                    )
                continue
            active[scope_key] = payload

        return active

    async def shutdown(self):
        """Shutdown the shared policy cache."""
        if self.redis_client and hasattr(self.redis_client, "close"):
            await self.redis_client.close()
        self.redis_client = None


class PolicyMaterializer:
    """Background worker that materializes Flink hints into the shared policy cache."""

    def __init__(self, kafka_config: Dict[str, Any], policy_cache: RoutingPolicyCache):
        self.kafka_config = kafka_config or {}
        self.policy_cache = policy_cache
        self.topics = self.kafka_config.get("topics", {})
        consumer_config = self.kafka_config.get("consumer", {})
        materializer_config = policy_cache.config.get("consumer", {})
        self.consumer_config = {
            "bootstrap_servers": self.kafka_config.get(
                "bootstrap_servers", ["localhost:9092"]
            ),
            "group_id": materializer_config.get(
                "group_id",
                consumer_config.get("group_id", "llm-router-policy-materializer"),
            ),
            "auto_offset_reset": materializer_config.get(
                "auto_offset_reset",
                consumer_config.get("auto_offset_reset", "latest"),
            ),
            "enable_auto_commit": materializer_config.get(
                "enable_auto_commit",
                consumer_config.get("enable_auto_commit", True),
            ),
            "max_poll_records": materializer_config.get(
                "max_poll_records", consumer_config.get("max_poll_records", 500)
            ),
        }
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.topic_names_by_key: Dict[str, str] = {}
        self.consumer_task_status: Dict[str, Dict[str, Any]] = {}
        self.consumer_restart_counts: Dict[str, int] = {}
        self.supervisor_initial_backoff_seconds = float(
            materializer_config.get(
                "supervisor_initial_backoff_seconds",
                consumer_config.get("supervisor_initial_backoff_seconds", 1.0),
            )
        )
        self.supervisor_max_backoff_seconds = float(
            materializer_config.get(
                "supervisor_max_backoff_seconds",
                consumer_config.get("supervisor_max_backoff_seconds", 30.0),
            )
        )
        self.running = False

    async def initialize(self):
        """Initialize Kafka consumers for the policy materializer."""
        if not self.policy_cache.enabled:
            logger.info("Policy materializer disabled")
            return

        await self.policy_cache.initialize()
        if not self.policy_cache.enabled:
            return

        topic_keys = (
            "requests_enriched",
            "fast_lane_hints",
            "routing_guardrails",
            "routing_policy_state",
        )
        for topic_key in topic_keys:
            topic_name = self.topics.get(topic_key)
            if not topic_name:
                continue
            consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
            await consumer.start()
            self.consumers[topic_key] = consumer
            self.topic_names_by_key[topic_key] = topic_name
            self.consumer_task_status[topic_key] = {
                "running": False,
                "last_error": None,
                "last_started_at": None,
                "last_stopped_at": None,
            }
            logger.info(f"Policy materializer consumer initialized: {topic_name}")

    async def start(self):
        """Start consuming policy events."""
        if not self.consumers:
            return

        self.running = True
        tasks = []
        for topic_key, consumer in self.consumers.items():
            tasks.append(
                asyncio.create_task(
                    self._supervise_consumer(topic_key, consumer),
                    name=f"policy_materializer_{topic_key}",
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _supervise_consumer(self, topic_key: str, consumer: AIOKafkaConsumer):
        """Restart policy materializer consumers after task-level failures."""
        backoff_seconds = self.supervisor_initial_backoff_seconds
        while self.running:
            self.consumer_task_status.setdefault(topic_key, {})
            self.consumer_task_status[topic_key].update(
                {
                    "running": True,
                    "last_error": None,
                    "last_started_at": time.time(),
                }
            )
            try:
                await self._run_consumer(topic_key, consumer)
                if not self.running:
                    break
                raise RuntimeError("policy materializer consumer exited unexpectedly")
            except asyncio.CancelledError:
                self.consumer_task_status[topic_key].update(
                    {"running": False, "last_stopped_at": time.time()}
                )
                raise
            except Exception as e:
                self.consumer_task_status[topic_key].update(
                    {
                        "running": False,
                        "last_error": str(e),
                        "last_stopped_at": time.time(),
                    }
                )
                self.consumer_restart_counts[topic_key] = (
                    self.consumer_restart_counts.get(topic_key, 0) + 1
                )
                PIPELINE_METRICS.consumer_restarts.labels(topic=topic_key).inc()
                logger.error(
                    "Policy materializer consumer for %s stopped; restarting in %.1fs: %s",
                    topic_key,
                    backoff_seconds,
                    e,
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    max(backoff_seconds * 2, self.supervisor_initial_backoff_seconds),
                    self.supervisor_max_backoff_seconds,
                )
                consumer = await self._restart_consumer(topic_key)

    async def _restart_consumer(self, topic_key: str) -> AIOKafkaConsumer:
        old_consumer = self.consumers.get(topic_key)
        if old_consumer is not None:
            try:
                await old_consumer.stop()
            except Exception as e:
                logger.warning(
                    f"Failed to stop unhealthy policy consumer {topic_key}: {e}"
                )
        topic_name = self.topic_names_by_key.get(topic_key) or self.topics[topic_key]
        consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
        await consumer.start()
        self.consumers[topic_key] = consumer
        return consumer

    async def _run_consumer(self, topic_key: str, consumer: AIOKafkaConsumer):
        handlers = {
            "requests_enriched": self._consume_requests_enriched,
            "fast_lane_hints": self._consume_fast_lane_hints,
            "routing_guardrails": self._consume_routing_guardrails,
            "routing_policy_state": self._consume_routing_policy_state,
        }
        handler = handlers.get(topic_key)
        if handler is None:
            raise RuntimeError(f"No policy materializer handler for {topic_key}")
        await handler(consumer)

    async def _commit_processed_message(self, consumer: AIOKafkaConsumer, message: Any):
        """Commit a processed Kafka message when manual commits are enabled."""
        if self.consumer_config.get("enable_auto_commit", True):
            return

        topic = getattr(message, "topic", None)
        partition = getattr(message, "partition", None)
        offset = getattr(message, "offset", None)
        if topic is None or partition is None or offset is None:
            return

        await consumer.commit({TopicPartition(topic, partition): int(offset) + 1})

    @staticmethod
    def _decode_message_value(message: Any) -> Dict[str, Any]:
        value = getattr(message, "value", None)
        if isinstance(value, dict):
            return value
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        raise ValueError(f"Unsupported Kafka message payload type: {type(value)}")

    async def _consume_requests_enriched(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    await self.policy_cache.materialize_request_enriched(payload)
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Skipping invalid requests.enriched message: {e}")
                    await self._commit_processed_message(consumer, message)
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize requests.enriched message: {e}"
                    )
        except Exception as e:
            logger.error(f"requests.enriched consumer error: {e}")
            raise

    async def _consume_fast_lane_hints(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    await self.policy_cache.materialize_fast_lane_hint(payload)
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Skipping invalid fast_lane_hints message: {e}")
                    await self._commit_processed_message(consumer, message)
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize fast_lane_hints message: {e}"
                    )
        except Exception as e:
            logger.error(f"fast_lane_hints consumer error: {e}")
            raise

    async def _consume_routing_guardrails(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    await self.policy_cache.materialize_routing_guardrail(payload)
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Skipping invalid routing_guardrails message: {e}")
                    await self._commit_processed_message(consumer, message)
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize routing_guardrails message: {e}"
                    )
        except Exception as e:
            logger.error(f"routing_guardrails consumer error: {e}")
            raise

    async def _consume_routing_policy_state(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    payload = self._decode_message_value(message)
                    await self.policy_cache.materialize_routing_policy_state(payload)
                    await self._commit_processed_message(consumer, message)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"Skipping invalid routing_policy_state message: {e}"
                    )
                    await self._commit_processed_message(consumer, message)
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize routing_policy_state message: {e}"
                    )
        except Exception as e:
            logger.error(f"routing_policy_state consumer error: {e}")
            raise

    @staticmethod
    def _deserialize_message(message: bytes) -> Dict[str, Any]:
        return json.loads(message.decode("utf-8"))

    def is_healthy(self) -> bool:
        """Return whether the materializer is configured and ready."""
        if not self.policy_cache.enabled:
            return True
        if not self.consumers:
            return False
        return not any(
            status.get("last_error") for status in self.consumer_task_status.values()
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Return materializer consumer supervisor status."""
        return {
            "healthy": self.is_healthy(),
            "topics": dict(self.consumer_task_status),
            "restart_counts": dict(self.consumer_restart_counts),
        }

    async def shutdown(self):
        """Shutdown Kafka consumers. The shared cache is owned by the platform."""
        self.running = False
        for consumer in self.consumers.values():
            await consumer.stop()
        self.consumers.clear()
