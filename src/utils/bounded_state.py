"""Small bounded process-local state containers."""

from __future__ import annotations

import time
import heapq
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
        self._entries: OrderedDict[K, tuple[float, float, int, V]] = OrderedDict()
        self._expiry_heap: list[tuple[float, int, K]] = []
        self._created_heap: list[tuple[float, int, K]] = []
        self._version = 0
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
            oldest_created = self._oldest_created()
            oldest_age = max(0.0, self._clock() - oldest_created)
        BOUNDED_STATE_METRICS.oldest_item_age.labels(state=self.metric_name).set(
            oldest_age
        )

    def _evict(self, key: K, reason: str) -> None:
        _, _, _, value = self._entries.pop(key)
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
        expired = 0
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            expires_at, version, key = heapq.heappop(self._expiry_heap)
            current = self._entries.get(key)
            if current is None or current[0] != expires_at or current[2] != version:
                continue
            self._evict(key, "ttl")
            expired += 1
        self._update_metrics()
        return expired

    def set(self, key: K, value: V, *, ttl_seconds: Optional[float] = None) -> None:
        self.prune_expired()
        ttl = self.ttl_seconds if ttl_seconds is None else max(0.0, ttl_seconds)
        now = self._clock()
        self._version += 1
        if key in self._entries:
            self._entries.pop(key)
        expires_at = now + ttl
        self._entries[key] = (expires_at, now, self._version, value)
        heapq.heappush(self._expiry_heap, (expires_at, self._version, key))
        heapq.heappush(self._created_heap, (now, self._version, key))
        while len(self._entries) > self.max_entries:
            oldest = next(iter(self._entries))
            self._evict(oldest, "lru")
        self._compact_heaps_if_needed()
        self._update_metrics()

    def __getitem__(self, key: K) -> V:
        self.prune_expired()
        expires_at, created_at, version, value = self._entries[key]
        self._entries.move_to_end(key)
        self._entries[key] = (expires_at, created_at, version, value)
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
        oldest_created = self._oldest_created()
        return max(0.0, self._clock() - oldest_created)

    def _oldest_created(self) -> float:
        while self._created_heap:
            created_at, version, key = self._created_heap[0]
            current = self._entries.get(key)
            if (
                current is not None
                and current[1] == created_at
                and current[2] == version
            ):
                return created_at
            heapq.heappop(self._created_heap)
        return self._clock()

    def _compact_heaps_if_needed(self) -> None:
        threshold = max(4, self.max_entries * 4)
        if len(self._expiry_heap) <= threshold and len(self._created_heap) <= threshold:
            return
        self._expiry_heap = [
            (expires_at, version, key)
            for key, (expires_at, _, version, _) in self._entries.items()
        ]
        self._created_heap = [
            (created_at, version, key)
            for key, (_, created_at, version, _) in self._entries.items()
        ]
        heapq.heapify(self._expiry_heap)
        heapq.heapify(self._created_heap)
