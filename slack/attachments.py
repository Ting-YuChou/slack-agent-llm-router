"""SlackBot mixin that downloads a message's Slack files as attachments."""

import asyncio
import httpx
import logging
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from slack.files import (
    declares_html,
    download_slack_file,
    error_name,
    is_allowed_slack_file_url,
    is_slack_hosted_file,
)
from src.utils.schema import (
    Attachment,
    AttachmentType,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


# Slack files of one message downloaded at the same time (S6c).
ATTACHMENT_DOWNLOAD_CONCURRENCY = 4


class SlackBotAttachmentsMixin:
    """SlackBot's Slack file downloads."""

    async def _extract_query_attachments(
        self, event: Dict[str, Any]
    ) -> List[Attachment]:
        """Convert Slack file payloads into QueryRequest attachments."""
        if not self.attachment_settings.get("enabled", True):
            return []

        files = list(event.get("files", []) or [])
        if not files:
            return []

        max_files = min(
            int(self.attachment_settings.get("max_files", 10) or 10),
            10,
        )
        # Download side by side, a few at a time; results keep Slack's order.
        semaphore = asyncio.Semaphore(ATTACHMENT_DOWNLOAD_CONCURRENCY)

        async def build(file_payload: Dict[str, Any]) -> Optional[Attachment]:
            async with semaphore:
                return await self._build_attachment_from_slack_file(file_payload)

        results = await asyncio.gather(
            *(build(file_payload) for file_payload in files[:max_files]),
            return_exceptions=True,
        )
        attachments = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            if result is not None:
                attachments.append(result)
        return attachments

    async def _build_attachment_from_slack_file(
        self, file_payload: Dict[str, Any]
    ) -> Optional[Attachment]:
        """Build a normalized attachment object from Slack file metadata."""
        if not is_slack_hosted_file(file_payload):
            return None

        download_url = file_payload.get("url_private_download") or file_payload.get(
            "url_private"
        )
        mime_type = file_payload.get("mimetype") or "application/octet-stream"
        attachment_type = self._infer_attachment_type(mime_type)
        name = (
            file_payload.get("name")
            or file_payload.get("title")
            or file_payload.get("id")
        )
        if not name:
            return None

        size_bytes = int(file_payload.get("size") or 0)
        content = None
        max_file_size_bytes = int(
            self.attachment_settings.get("max_file_size_bytes", 10_000_000)
            or 10_000_000
        )

        if download_url and size_bytes <= max_file_size_bytes:
            content = await self._download_attachment_content(
                file_payload, str(download_url), max_file_size_bytes
            )
            if content is not None and size_bytes <= 0:
                size_bytes = len(content)

        if size_bytes <= 0:
            if content:
                size_bytes = len(content)
            else:
                logger.warning(
                    "Skipping Slack file without usable size metadata: %s", name
                )
                return None

        return Attachment(
            id=file_payload.get("id") or str(uuid.uuid4()),
            name=name,
            type=attachment_type,
            size_bytes=size_bytes,
            mime_type=mime_type,
            url=download_url,
            content=content,
        )

    async def _download_attachment_content(
        self, file_payload: Dict[str, Any], download_url: str, max_bytes: int
    ) -> Optional[bytes]:
        """Download a private Slack file using the bot token.

        The token only goes to Slack file hosts. Logs name the file id and the
        failure, never the URL.
        """
        if not download_url or not self.bot_token:
            return None

        file_id = str(file_payload.get("id") or "unknown")
        if not is_allowed_slack_file_url(download_url):
            logger.warning("Skipping Slack attachment %s: unsafe_download_url", file_id)
            return None

        try:
            return await asyncio.wait_for(
                download_slack_file(
                    self._get_attachment_http_client(),
                    download_url,
                    bot_token=self.bot_token,
                    max_bytes=max_bytes,
                    allow_html=declares_html(file_payload),
                ),
                timeout=self._attachment_download_timeout(),
            )
        except Exception as exc:
            logger.warning(
                "Failed to download Slack attachment %s: %s",
                file_id,
                error_name(exc),
            )
            return None

    def _attachment_download_timeout(self) -> float:
        return float(self.attachment_settings.get("download_timeout_seconds", 30) or 30)

    def _get_attachment_http_client(self) -> httpx.AsyncClient:
        if self._attachment_http_client is None:
            self._attachment_http_client = httpx.AsyncClient(
                timeout=self._attachment_download_timeout(),
                follow_redirects=False,
            )
        return self._attachment_http_client

    async def _close_attachment_http_client(self) -> None:
        client = self._attachment_http_client
        self._attachment_http_client = None
        if client is None:
            return
        try:
            await client.aclose()
        except Exception as exc:
            logger.warning(
                "Slack attachment HTTP client close failed: %s", error_name(exc)
            )

    def _infer_attachment_type(self, mime_type: str) -> AttachmentType:
        """Map a MIME type to the internal attachment type enum."""
        normalized = (mime_type or "").lower()
        if normalized.startswith("image/"):
            return AttachmentType.IMAGE
        if normalized.startswith("audio/"):
            return AttachmentType.AUDIO
        if (
            normalized.startswith("text/")
            or "pdf" in normalized
            or "document" in normalized
            or "spreadsheet" in normalized
            or "presentation" in normalized
            or "json" in normalized
            or "csv" in normalized
        ):
            return AttachmentType.DOCUMENT
        return AttachmentType.FILE
