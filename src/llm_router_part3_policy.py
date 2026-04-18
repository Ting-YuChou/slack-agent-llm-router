"""
Policy cache materialization for Flink routing hints.

This layer keeps Kafka and Flink off the synchronous API path:
- background workers consume requests.enriched / fast_lane_hints
- hints are materialized into Redis with a tiny in-process L1 cache
- API workers read the shared policy cache during routing
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from aiokafka import AIOKafkaConsumer
import redis.asyncio as redis


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
        self.user_ttl_seconds = int(self.config.get("user_ttl_seconds", 900))
        self.local_cache_ttl_seconds = int(
            self.config.get("local_cache_ttl_seconds", 5)
        )
        self.redis_client = None
        self._initialized = False
        self._local_cache: Dict[str, Dict[str, Any]] = {}

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
        """Get user-scoped fast-lane promotions if present."""
        if not user_id:
            return None
        return await self._get_json(self._cache_key("user", user_id))

    async def get_effective_policy(
        self, request_id: Optional[str], user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Merge request and user policy views for the router hot path."""
        request_policy = await self.get_request_policy(request_id or "")
        user_policy = await self.get_user_policy(user_id or "")

        if request_policy and user_policy:
            merged = dict(user_policy)
            merged.update(
                {
                    key: value
                    for key, value in request_policy.items()
                    if value not in (None, "", [], {})
                }
            )
            merged["policy_source"] = "request+user"
            return merged

        if request_policy:
            request_policy = dict(request_policy)
            request_policy["policy_source"] = "request"
            return request_policy

        if user_policy:
            user_policy = dict(user_policy)
            user_policy["policy_source"] = "user"
            return user_policy

        return {}

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
            "preferred_models": [],
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
            user_policy = {
                "request_id": request_id,
                "user_id": user_id,
                "priority": priority,
                "route_to_fast_lane": True,
                "query_type": event.get("query_type"),
                "preferred_models": [],
                "hint_reason": f"requests.enriched priority={priority}",
                "source_event_type": event.get("event_type", "requests.enriched"),
                "updated_at": updated_at,
            }
            await self._set_json(
                self._cache_key("user", user_id),
                user_policy,
                self.user_ttl_seconds,
            )

    async def materialize_fast_lane_hint(self, event: Dict[str, Any]):
        """Materialize fast_lane_hints into request/user caches."""
        if not event:
            return

        request_id = str(event.get("request_id", "") or "")
        user_id = str(event.get("user_id", "") or "")
        selected_model = event.get("selected_model")
        preferred_models = [selected_model] if selected_model else []
        updated_at = event.get("emitted_at")
        route_to_fast_lane = bool(event.get("route_to_fast_lane", False))

        request_policy = {
            "request_id": request_id,
            "user_id": user_id,
            "priority": event.get("priority"),
            "route_to_fast_lane": route_to_fast_lane,
            "query_type": event.get("query_type"),
            "preferred_models": preferred_models,
            "hint_reason": event.get("hint_reason", "fast_lane_candidate"),
            "source_event_type": event.get("event_type", "fast_lane_hints"),
            "updated_at": updated_at,
        }

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
                "preferred_models": preferred_models,
                "hint_reason": event.get("hint_reason", "fast_lane_candidate"),
                "source_event_type": event.get("event_type", "fast_lane_hints"),
                "updated_at": updated_at,
            }
            await self._set_json(
                self._cache_key("user", user_id),
                user_policy,
                self.user_ttl_seconds,
            )

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
            "value_deserializer": self._deserialize_message,
        }
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.running = False

    async def initialize(self):
        """Initialize Kafka consumers for the policy materializer."""
        if not self.policy_cache.enabled:
            logger.info("Policy materializer disabled")
            return

        await self.policy_cache.initialize()
        if not self.policy_cache.enabled:
            return

        topic_keys = ("requests_enriched", "fast_lane_hints")
        for topic_key in topic_keys:
            topic_name = self.topics.get(topic_key)
            if not topic_name:
                continue
            consumer = AIOKafkaConsumer(topic_name, **self.consumer_config)
            await consumer.start()
            self.consumers[topic_key] = consumer
            logger.info(f"Policy materializer consumer initialized: {topic_name}")

    async def start(self):
        """Start consuming policy events."""
        if not self.consumers:
            return

        self.running = True
        tasks = []
        if "requests_enriched" in self.consumers:
            tasks.append(
                asyncio.create_task(
                    self._consume_requests_enriched(self.consumers["requests_enriched"])
                )
            )
        if "fast_lane_hints" in self.consumers:
            tasks.append(
                asyncio.create_task(
                    self._consume_fast_lane_hints(self.consumers["fast_lane_hints"])
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume_requests_enriched(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    await self.policy_cache.materialize_request_enriched(message.value)
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize requests.enriched message: {e}"
                    )
        except Exception as e:
            logger.error(f"requests.enriched consumer error: {e}")

    async def _consume_fast_lane_hints(self, consumer: AIOKafkaConsumer):
        try:
            async for message in consumer:
                if not self.running:
                    break
                try:
                    await self.policy_cache.materialize_fast_lane_hint(message.value)
                except Exception as e:
                    logger.warning(
                        f"Failed to materialize fast_lane_hints message: {e}"
                    )
        except Exception as e:
            logger.error(f"fast_lane_hints consumer error: {e}")

    @staticmethod
    def _deserialize_message(message: bytes) -> Dict[str, Any]:
        return json.loads(message.decode("utf-8"))

    def is_healthy(self) -> bool:
        """Return whether the materializer is configured and ready."""
        return not self.policy_cache.enabled or bool(self.consumers)

    async def shutdown(self):
        """Shutdown Kafka consumers. The shared cache is owned by the platform."""
        self.running = False
        for consumer in self.consumers.values():
            await consumer.stop()
        self.consumers.clear()
