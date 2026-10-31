"""Slack users: tiers, preferences, rate limits, and SlackBot's tier mixin."""

import logging
import time
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from src.utils.schema import UserTier
from slack.formatting import (
    _format_wait,
    _plural,
)

# The split modules log as the bot always has: one "slack.bot_real" logger.
logger = logging.getLogger("slack.bot_real")


# Requests older than this no longer count against a user's rate limit.
RATE_LIMIT_WINDOW_SECONDS = 3600
# The windows UserManager.check_rate_limit counts requests in.
RATE_LIMIT_WINDOW_SECONDS = 3600
RATE_LIMIT_BURST_WINDOW_SECONDS = 300


RATE_LIMIT_CHAT = "chat"
RATE_LIMIT_AGENT = "agent"


def _rate_limit_entry(at: float, bucket: str) -> Any:
    """One counted request; a chat request stays a bare timestamp.

    That is the shape saved before Agent requests had a bucket of their own.
    """
    return at if bucket == RATE_LIMIT_CHAT else {"at": at, "bucket": bucket}


def _read_rate_limit_entry(entry: Any) -> Optional[Tuple[float, str]]:
    """The (timestamp, bucket) of a counted request; None when malformed."""
    try:
        if isinstance(entry, dict):
            return float(entry["at"]), str(entry.get("bucket") or RATE_LIMIT_CHAT)
        return float(entry), RATE_LIMIT_CHAT
    except (KeyError, TypeError, ValueError):
        return None


def _rate_limit_entries(values: Any, since: float) -> List[Tuple[float, str]]:
    """The well-formed counted requests after `since`."""
    entries = (_read_rate_limit_entry(value) for value in values or [])
    return [entry for entry in entries if entry is not None and entry[0] > since]


class UserManager:
    """Manages user data and preferences"""

    def __init__(
        self,
        default_tier: str = UserTier.FREE.value,
        tier_overrides: Optional[Dict[str, str]] = None,
    ):
        self.default_tier = self._normalize_tier_name(default_tier)
        self.tier_overrides = {
            user_id: self._normalize_tier_name(tier_name)
            for user_id, tier_name in (tier_overrides or {}).items()
        }
        self.users = {}  # In production, use a database
        self.rate_limits = {}

    def get_user_tier(self, user_id: str) -> UserTier:
        """The user's tier from config: their override, else the default.

        The tier stored on a user record is only a record of it, so removing
        an override or lowering the default takes effect on the next request.
        """
        return UserTier(self._configured_tier_name(user_id))

    def _configured_tier_name(self, user_id: str) -> str:
        return self.tier_overrides.get(user_id) or self.default_tier

    def check_rate_limit(
        self, user_id: str, config: Dict[str, Any], bucket: str = RATE_LIMIT_CHAT
    ) -> bool:
        """Check if user is within rate limits.

        Chat queries and Agent requests fill separate buckets with the same
        limits, so a burst of one never blocks the other. Each request still
        counts as one Slack query in the user's list.
        """
        current_time = time.time()
        hour_window = current_time - 3600  # 1 hour window

        # Clean old requests
        entries = _rate_limit_entries(self.rate_limits.get(user_id), hour_window)
        self.rate_limits[user_id] = [_rate_limit_entry(*entry) for entry in entries]
        bucket_requests = [at for at, kind in entries if kind == bucket]

        # Check limits
        requests_per_hour = config.get("requests_per_hour", 100)
        burst_requests = config.get("burst_requests", 5)

        # Check hourly limit
        if len(bucket_requests) >= requests_per_hour:
            return False

        # Check burst limit (last 5 minutes)
        burst_window = current_time - 300  # 5 minutes
        recent_requests = [
            req_time for req_time in bucket_requests if req_time > burst_window
        ]

        if len(recent_requests) >= burst_requests:
            return False

        # Add current request
        self.rate_limits[user_id].append(_rate_limit_entry(current_time, bucket))
        return True

    def requests_since(
        self, user_id: str, since: float, bucket: str = RATE_LIMIT_CHAT
    ) -> List[float]:
        """Times of the user's counted requests in one bucket after `since`."""
        return [
            at
            for at, kind in _rate_limit_entries(self.rate_limits.get(user_id), since)
            if kind == bucket
        ]

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        return self.users.get(user_id, {}).get(
            "preferences",
            {
                "response_length": "medium",
                "technical_level": "intermediate",
                "preferred_models": [],
                "threading": True,
                "web_search_auto": True,
            },
        )

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        if user_id not in self.users:
            self.users[user_id] = {}

        if "preferences" not in self.users[user_id]:
            self.users[user_id]["preferences"] = {}

        self.users[user_id]["preferences"].update(preferences)
        self.users[user_id]["updated_at"] = datetime.now().isoformat()

    def ensure_user_record(self, user_id: str) -> bool:
        """Record the configured tier on the user's record; True if it changed.

        A user on the implicit free default gets no record for it alone.
        """
        if not user_id:
            return False

        previous_tier = self.users.get(user_id, {}).get("tier")
        resolved_tier = self._configured_tier_name(user_id)
        if previous_tier == resolved_tier:
            return False
        if (
            previous_tier is None
            and user_id not in self.tier_overrides
            and resolved_tier == UserTier.FREE.value
        ):
            return False

        if user_id not in self.users:
            self.users[user_id] = {}

        self.users[user_id]["tier"] = resolved_tier
        self.users[user_id]["updated_at"] = datetime.now().isoformat()
        return True

    def _normalize_tier_name(self, tier_name: Optional[str]) -> str:
        """Normalize tier names used in config and persisted user state."""
        normalized = (tier_name or UserTier.FREE.value).strip().lower()
        try:
            return UserTier(normalized).value
        except ValueError:
            return UserTier.FREE.value


class SlackBotUsersMixin:
    """SlackBot's user tiers and rate limits."""

    async def _send_rate_limit_message(
        self,
        channel_id: str,
        user_id: str,
        *,
        thread_ts: Optional[str] = None,
        agent: bool = False,
    ):
        """Tell only the user that they hit a limit, in their thread if any.

        Pass thread_ts only for a reply in an existing thread: Slack never
        shows an ephemeral sent into a thread that has no replies yet, so a
        notice for a new top-level request stays in the channel.
        """
        await self.web_client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text=self._build_rate_limit_message(user_id, agent=agent),
            thread_ts=thread_ts,
        )

    def _build_rate_limit_message(self, user_id: str, *, agent: bool = False) -> str:
        """Say which of the user's limits was hit and when to try again.

        The limits are a rolling 60-minute window and a 5-minute burst, each
        counted per bucket: chat queries and Agent requests apart.
        """
        user_tier = self.user_manager.get_user_tier(user_id)
        limits = self._get_rate_limit_config_for_tier(user_tier)
        hourly_limit = limits["requests_per_hour"]
        burst_limit = limits["burst_requests"]
        kind = "Agent request" if agent else "chat request"
        now = time.time()
        counted = sorted(
            self.user_manager.requests_since(
                user_id,
                now - RATE_LIMIT_WINDOW_SECONDS,
                bucket=RATE_LIMIT_AGENT if agent else RATE_LIMIT_CHAT,
            )
        )
        recent = [at for at in counted if at > now - RATE_LIMIT_BURST_WINDOW_SECONDS]

        # The first request that must leave a full window frees a slot in it;
        # with both full, the later of the two is when a retry goes through.
        reason = None
        retry_at = 0.0
        if 0 < hourly_limit <= len(counted):
            reason = (
                f"You've sent {_plural(len(counted), kind)} in the last "
                "60 minutes, the hourly limit for your tier."
            )
            retry_at = counted[len(counted) - hourly_limit] + RATE_LIMIT_WINDOW_SECONDS
        if 0 < burst_limit <= len(recent):
            burst_retry_at = (
                recent[len(recent) - burst_limit] + RATE_LIMIT_BURST_WINDOW_SECONDS
            )
            if reason is None or burst_retry_at > retry_at:
                reason = (
                    f"You've sent {_plural(len(recent), kind)} in the last "
                    "5 minutes, the burst limit for your tier."
                )
                retry_at = burst_retry_at
        if hourly_limit <= 0 or burst_limit <= 0:
            # check_rate_limit refuses every request against a zero limit.
            summary = (
                f"The {user_tier.value.title()} tier doesn't allow {kind}s. "
                "Ask an admin if you need access."
            )
        elif reason is None:
            article = "an" if agent else "a"
            summary = (
                f"You've reached {article} {kind} limit. Try again in a few minutes."
            )
        else:
            summary = f"{reason} Try again in {_format_wait(retry_at - now)}."

        return "\n".join(
            [
                "🚫 *Rate Limit Exceeded*",
                "",
                summary,
                "",
                f"*Your tier:* {user_tier.value.title()}",
                f"*Limits:* {hourly_limit} requests per rolling 60 minutes and "
                f"{burst_limit} per 5 minutes, for chat and Agent requests each",
            ]
        )

    def _synchronize_user_tier(self, user_id: Optional[str]) -> bool:
        """Apply configured user-tier defaults/overrides to the in-memory state."""
        return self.user_manager.ensure_user_record(user_id or "")

    def _get_rate_limit_config_for_tier(self, user_tier: UserTier) -> Dict[str, int]:
        """Resolve rate-limit settings for a concrete tier."""
        base_limits = {
            "requests_per_hour": int(
                self.rate_limiting.get("requests_per_hour", 100) or 100
            ),
            "burst_requests": int(self.rate_limiting.get("burst_requests", 5) or 5),
        }
        tier_overrides = self.rate_limiting.get("by_tier", {}).get(user_tier.value, {})
        if isinstance(tier_overrides, dict):
            for key in ("requests_per_hour", "burst_requests"):
                if key in tier_overrides:
                    base_limits[key] = int(tier_overrides[key])
        return base_limits

    def _get_rate_limit_config_for_user(self, user_id: str) -> Dict[str, int]:
        """Resolve rate-limit settings for the user's current tier."""
        return self._get_rate_limit_config_for_tier(
            self.user_manager.get_user_tier(user_id)
        )

    def _within_rate_limit(self, user_id: str, *, agent: bool) -> bool:
        """Count one request in the user's chat or Agent bucket, unless it is full."""
        return self.user_manager.check_rate_limit(
            user_id,
            self._get_rate_limit_config_for_user(user_id),
            **({"bucket": RATE_LIMIT_AGENT} if agent else {}),
        )

    def _format_tier_limit_summary(self) -> str:
        """Render the configured tier limits for Slack help text."""
        summaries = []
        for tier in (UserTier.FREE, UserTier.PREMIUM, UserTier.ENTERPRISE):
            limits = self._get_rate_limit_config_for_tier(tier)
            summaries.append(
                f"{self._tier_emoji(tier)} {tier.value.title()}: "
                f"{limits['requests_per_hour']} requests/hour, "
                f"{limits['burst_requests']} burst requests / 5 min"
            )
        return "\n".join(summaries)

    def _tier_emoji(self, tier: UserTier) -> str:
        if tier == UserTier.PREMIUM:
            return "💎"
        if tier == UserTier.ENTERPRISE:
            return "🏢"
        return "🆓"
