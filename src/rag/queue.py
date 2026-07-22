"""Queue adapters for RAG ingestion work."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.metrics import RAG_METRICS


@dataclass(frozen=True)
class QueueDelivery:
    message_id: str
    receipt_handle: str
    payload: Dict[str, Any]
    receive_count: int = 1


class SqsIngestionQueue:
    """Amazon SQS Standard queue adapter."""

    def __init__(self, config: Dict[str, Any], *, client: Optional[Any] = None):
        self.config = dict(config or {})
        self.queue_url = str(self.config.get("queue_url") or "").strip()
        if not self.queue_url:
            raise ValueError("rag.ingestion_queue.sqs.queue_url is required")
        self.wait_time_seconds = min(
            20, max(0, int(self.config.get("wait_time_seconds", 20)))
        )
        self.visibility_timeout_seconds = min(
            43_200,
            max(1, int(self.config.get("visibility_timeout_seconds", 900))),
        )
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("boto3 is required for the SQS RAG backend") from exc
            kwargs = {}
            region = self.config.get("region")
            endpoint_url = self.config.get("endpoint_url")
            if region:
                kwargs["region_name"] = region
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            self._client = boto3.client("sqs", **kwargs)
        return self._client

    async def publish(self, payload: Dict[str, Any]) -> str:
        try:
            response = await asyncio.to_thread(
                self._get_client().send_message,
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
        except BaseException:
            RAG_METRICS.queue_operations.labels("sqs", "publish", "error").inc()
            raise
        RAG_METRICS.queue_operations.labels("sqs", "publish", "success").inc()
        return str(response["MessageId"])

    async def receive(self, *, max_messages: int) -> List[QueueDelivery]:
        try:
            response = await asyncio.to_thread(
                self._get_client().receive_message,
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=min(10, max(1, int(max_messages))),
                WaitTimeSeconds=self.wait_time_seconds,
                VisibilityTimeout=self.visibility_timeout_seconds,
                AttributeNames=["ApproximateReceiveCount"],
            )
        except BaseException:
            RAG_METRICS.queue_operations.labels("sqs", "receive", "error").inc()
            raise
        RAG_METRICS.queue_operations.labels("sqs", "receive", "success").inc()
        deliveries = []
        for message in response.get("Messages", []):
            deliveries.append(
                QueueDelivery(
                    message_id=str(message["MessageId"]),
                    receipt_handle=str(message["ReceiptHandle"]),
                    payload=dict(json.loads(message["Body"])),
                    receive_count=int(
                        message.get("Attributes", {}).get(
                            "ApproximateReceiveCount", "1"
                        )
                    ),
                )
            )
        return deliveries

    async def ack(self, delivery: QueueDelivery) -> None:
        try:
            await asyncio.to_thread(
                self._get_client().delete_message,
                QueueUrl=self.queue_url,
                ReceiptHandle=delivery.receipt_handle,
            )
        except BaseException:
            RAG_METRICS.queue_operations.labels("sqs", "ack", "error").inc()
            raise
        RAG_METRICS.queue_operations.labels("sqs", "ack", "success").inc()

    async def extend_visibility(
        self, delivery: QueueDelivery, timeout_seconds: int
    ) -> None:
        try:
            await asyncio.to_thread(
                self._get_client().change_message_visibility,
                QueueUrl=self.queue_url,
                ReceiptHandle=delivery.receipt_handle,
                VisibilityTimeout=min(43_200, max(0, int(timeout_seconds))),
            )
        except BaseException:
            RAG_METRICS.queue_operations.labels("sqs", "visibility", "error").inc()
            raise
        RAG_METRICS.queue_operations.labels("sqs", "visibility", "success").inc()
