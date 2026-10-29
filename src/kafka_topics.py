"""Provision the chat pipeline's Kafka topics and their dead-letter topics.

Run as ``python -m src.kafka_topics --config config/config.compose.yaml``.
Every ``kafka.topics`` entry and its DLQ is created with explicit partitions,
replication and retention, so the broker can run with topic auto-creation
disabled. Topics that already exist get their managed configs re-applied, which
makes the command safe to run on every start.

Settings come from ``kafka.provisioning`` (all optional)::

    provisioning:
      partitions: 1
      replication_factor: 1
      retention_ms: 604800000
      dlq_retention_ms: 2592000000
      topics:
        routing_policy_state: {retention_ms: 86400000, cleanup_policy: delete}
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PARTITIONS = 1
DEFAULT_REPLICATION_FACTOR = 1
DEFAULT_RETENTION_MS = 7 * 24 * 3600 * 1000
DEFAULT_DLQ_RETENTION_MS = 30 * 24 * 3600 * 1000
DEFAULT_DLQ_SUFFIX = ".dlq"
# Built-in per-topic defaults; config overrides them.
#
# routing.policy_state is a latest-state stream, so it only needs a short
# history. It stays on ``delete`` rather than ``compact``: a compacted topic
# rejects records without a key, and the Flink job's FlinkKafkaProducer cannot
# set one. Compact it once the policy-state sink writes ``scope_key`` as key.
BUILTIN_TOPIC_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "routing_policy_state": {
        "retention_ms": 24 * 3600 * 1000,
        "cleanup_policy": "delete",
    },
}
TOPIC_ALREADY_EXISTS = 36


@dataclass(frozen=True)
class TopicSpec:
    """One topic the pipeline needs, with the configs this module manages."""

    name: str
    partitions: int
    replication_factor: int
    configs: Dict[str, str] = field(default_factory=dict)
    topic_key: str = ""


class TopicAdmin(Protocol):
    async def existing_topics(self) -> List[str]:
        ...

    async def create_topics(self, specs: Sequence[TopicSpec]) -> None:
        ...

    async def apply_configs(self, specs: Sequence[TopicSpec]) -> None:
        ...


def _dlq_topic_name(kafka_config: Mapping[str, Any], topic_key: str, name: str) -> str:
    """Mirror KafkaConsumerManager._dead_letter_topic for one source topic."""
    dead_letter_topics = kafka_config.get("dead_letter_topics", {}) or {}
    if topic_key in dead_letter_topics:
        return str(dead_letter_topics[topic_key])
    topics = kafka_config.get("topics", {}) or {}
    if f"{topic_key}_dlq" in topics:
        return str(topics[f"{topic_key}_dlq"])
    suffix = (kafka_config.get("dlq", {}) or {}).get("topic_suffix", DEFAULT_DLQ_SUFFIX)
    return f"{name}{suffix}"


def _spec(
    name: str,
    topic_key: str,
    settings: Mapping[str, Any],
    *,
    retention_ms: int,
) -> TopicSpec:
    configs = {
        "cleanup.policy": str(settings.get("cleanup_policy", "delete")),
        "retention.ms": str(int(settings.get("retention_ms", retention_ms))),
    }
    for key, value in (settings.get("configs", {}) or {}).items():
        configs[str(key)] = str(value)
    return TopicSpec(
        name=name,
        partitions=int(settings.get("partitions", DEFAULT_PARTITIONS)),
        replication_factor=int(
            settings.get("replication_factor", DEFAULT_REPLICATION_FACTOR)
        ),
        configs=configs,
        topic_key=topic_key,
    )


def topic_specs(kafka_config: Mapping[str, Any]) -> List[TopicSpec]:
    """Return every topic in ``kafka.topics`` plus its DLQ, deduplicated by name."""
    provisioning = dict(kafka_config.get("provisioning", {}) or {})
    overrides = dict(provisioning.get("topics", {}) or {})
    defaults = {
        "partitions": provisioning.get("partitions", DEFAULT_PARTITIONS),
        "replication_factor": provisioning.get(
            "replication_factor", DEFAULT_REPLICATION_FACTOR
        ),
    }
    retention_ms = int(provisioning.get("retention_ms", DEFAULT_RETENTION_MS))
    dlq_retention_ms = int(
        provisioning.get("dlq_retention_ms", DEFAULT_DLQ_RETENTION_MS)
    )
    dlq_enabled = bool((kafka_config.get("dlq", {}) or {}).get("enabled", True))

    specs: Dict[str, TopicSpec] = {}
    for topic_key, name in (kafka_config.get("topics", {}) or {}).items():
        topic_key, name = str(topic_key), str(name)
        is_dlq = topic_key.endswith("_dlq")
        settings = {
            **defaults,
            **BUILTIN_TOPIC_OVERRIDES.get(topic_key, {}),
            **(overrides.get(topic_key, {}) or {}),
        }
        specs.setdefault(
            name,
            _spec(
                name,
                topic_key,
                settings,
                retention_ms=dlq_retention_ms if is_dlq else retention_ms,
            ),
        )
        if is_dlq or not dlq_enabled:
            continue
        dlq_key = f"{topic_key}_dlq"
        dlq_name = _dlq_topic_name(kafka_config, topic_key, name)
        dlq_settings = {**defaults, **(overrides.get(dlq_key, {}) or {})}
        specs.setdefault(
            dlq_name,
            _spec(dlq_name, dlq_key, dlq_settings, retention_ms=dlq_retention_ms),
        )
    return list(specs.values())


async def provision_topics(
    admin: TopicAdmin, specs: Iterable[TopicSpec]
) -> Dict[str, List[str]]:
    """Create missing topics and re-apply managed configs to existing ones."""
    wanted = list(specs)
    existing = set(await admin.existing_topics())
    missing = [spec for spec in wanted if spec.name not in existing]
    present = [spec for spec in wanted if spec.name in existing]
    if missing:
        await admin.create_topics(missing)
    if present:
        await admin.apply_configs(present)
    result = {
        "created": [spec.name for spec in missing],
        "updated": [spec.name for spec in present],
    }
    logger.info(
        "Kafka topics provisioned: %d created, %d already present",
        len(result["created"]),
        len(result["updated"]),
    )
    return result


class AIOKafkaTopicAdmin:
    """TopicAdmin backed by aiokafka's admin client."""

    def __init__(self, client: Any):
        self.client = client

    async def existing_topics(self) -> List[str]:
        return list(await self.client.list_topics())

    async def create_topics(self, specs: Sequence[TopicSpec]) -> None:
        from aiokafka.admin import NewTopic

        response = await self.client.create_topics(
            [
                NewTopic(
                    name=spec.name,
                    num_partitions=spec.partitions,
                    replication_factor=spec.replication_factor,
                    topic_configs=dict(spec.configs),
                )
                for spec in specs
            ]
        )
        errors = []
        for topic_error in getattr(response, "topic_errors", []) or []:
            topic, error_code = topic_error[0], int(topic_error[1])
            if error_code in (0, TOPIC_ALREADY_EXISTS):
                continue
            message = topic_error[2] if len(topic_error) > 2 else ""
            errors.append(f"{topic}: error {error_code} {message}".strip())
        if errors:
            raise RuntimeError("Failed to create Kafka topics: " + "; ".join(errors))

    async def apply_configs(self, specs: Sequence[TopicSpec]) -> None:
        from aiokafka.admin.config_resource import ConfigResource, ConfigResourceType

        responses = await self.client.alter_configs(
            [
                ConfigResource(ConfigResourceType.TOPIC, spec.name, dict(spec.configs))
                for spec in specs
            ]
        )
        if not isinstance(responses, (list, tuple)):
            responses = [responses]
        errors = []
        for response in responses:
            for resource in getattr(response, "resources", []) or []:
                error_code, message, _type, name = resource[:4]
                if int(error_code) != 0:
                    errors.append(f"{name}: error {error_code} {message}".strip())
        if errors:
            raise RuntimeError(
                "Failed to update Kafka topic configs: " + "; ".join(errors)
            )


def load_kafka_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return dict(config.get("kafka", {}) or {})


async def start_admin(bootstrap_servers: str, timeout_seconds: float):
    from aiokafka.admin import AIOKafkaAdminClient

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    delay = 1.0
    while True:
        client = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)
        try:
            await client.start()
            return client
        except Exception as exc:
            await client.close()
            if asyncio.get_running_loop().time() + delay > deadline:
                raise
            logger.warning(
                "Kafka not reachable yet (%s); retrying in %.0fs", exc, delay
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)


async def run(
    config_path: str,
    bootstrap_servers: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> Dict[str, List[str]]:
    kafka_config = load_kafka_config(config_path)
    servers = bootstrap_servers or ",".join(
        kafka_config.get("bootstrap_servers", ["localhost:9092"])
    )
    client = await start_admin(servers, timeout_seconds)
    try:
        return await provision_topics(
            AIOKafkaTopicAdmin(client), topic_specs(kafka_config)
        )
    finally:
        await client.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default=os.environ.get("LLM_ROUTER_CONFIG", "config/config.yaml"),
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS"),
        help="comma-separated; defaults to kafka.bootstrap_servers in the config",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = asyncio.run(run(args.config, args.bootstrap_servers, args.timeout_seconds))
    for name in result["created"]:
        print(f"created {name}")
    for name in result["updated"]:
        print(f"verified {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
