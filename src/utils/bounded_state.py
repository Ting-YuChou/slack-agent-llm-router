"""Small bounded process-local state containers."""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import Callable, Generic, Optional, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class BoundedTTLMap(MutableMapping[K, V], Generic[K, V]):
    """A monotonic TTL map with least-recently-used capacity eviction."""

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        on_evict: Optional[Callable[[K, V, str], None]] = None,
        metric_name: Optional[str] = None,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = int(max_entries)
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._on_evict = on_evict
        self.metric_name = metric_name
        self._entries: OrderedDict[K, tuple[float, float, V]] = OrderedDict()
        self.evictions: Counter[str] = Counter()
        self._update_metrics()

    def _update_metrics(self) -> None:
        if not self.metric_name:
            return
        from src.utils.metrics import BOUNDED_STATE_METRICS

        BOUNDED_STATE_METRICS.entries.labels(state=self.metric_name).set(
            len(self._entries)
        )
        BOUNDED_STATE_METRICS.capacity.labels(state=self.metric_name).set(
            self.max_entries
        )
        oldest_age = 0.0
        if self._entries:
            oldest_created = min(created for _, created, _ in self._entries.values())
            oldest_age = max(0.0, self._clock() - oldest_created)
        BOUNDED_STATE_METRICS.oldest_item_age.labels(state=self.metric_name).set(
            oldest_age
        )

    def _evict(self, key: K, reason: str) -> None:
        _, _, value = self._entries.pop(key)
        self.evictions[reason] += 1
        if self.metric_name:
            from src.utils.metrics import BOUNDED_STATE_METRICS

            BOUNDED_STATE_METRICS.evictions.labels(
                state=self.metric_name, reason=reason
            ).inc()
        if self._on_evict:
            self._on_evict(key, value, reason)

    def prune_expired(self) -> int:
        now = self._clock()
        expired = [
            key
            for key, (expires_at, _, _) in self._entries.items()
            if expires_at <= now
        ]
        for key in expired:
            self._evict(key, "ttl")
        self._update_metrics()
        return len(expired)

    def set(self, key: K, value: V, *, ttl_seconds: Optional[float] = None) -> None:
        self.prune_expired()
        ttl = self.ttl_seconds if ttl_seconds is None else max(0.0, ttl_seconds)
        now = self._clock()
        if key in self._entries:
            self._entries.pop(key)
        self._entries[key] = (now + ttl, now, value)
        while len(self._entries) > self.max_entries:
            oldest = next(iter(self._entries))
            self._evict(oldest, "lru")
        self._update_metrics()

    def __getitem__(self, key: K) -> V:
        self.prune_expired()
        expires_at, created_at, value = self._entries[key]
        self._entries.move_to_end(key)
        self._entries[key] = (expires_at, created_at, value)
        return value

    def get(self, key: K, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def __delitem__(self, key: K) -> None:
        del self._entries[key]
        self._update_metrics()

    def __iter__(self) -> Iterator[K]:
        self.prune_expired()
        return iter(tuple(self._entries))

    def __len__(self) -> int:
        self.prune_expired()
        return len(self._entries)

    def oldest_item_age(self) -> float:
        self.prune_expired()
        if not self._entries:
            return 0.0
        oldest_created = min(created for _, created, _ in self._entries.values())
        return max(0.0, self._clock() - oldest_created)
