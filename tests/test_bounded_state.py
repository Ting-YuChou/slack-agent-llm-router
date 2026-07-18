from src.utils.bounded_state import BoundedTTLMap


def test_bounded_ttl_map_prunes_expired_before_lru_eviction():
    now = [0.0]
    state = BoundedTTLMap(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    state["expired"] = 1
    now[0] = 11.0
    state["fresh"] = 2
    state["new"] = 3

    assert list(state.keys()) == ["fresh", "new"]
    assert state.evictions["ttl"] == 1
    assert state.evictions["lru"] == 0


def test_bounded_ttl_map_get_refreshes_lru_order_without_refreshing_ttl():
    now = [0.0]
    state = BoundedTTLMap(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    state["first"] = 1
    state["second"] = 2
    assert state.get("first") == 1
    state["third"] = 3

    assert list(state.keys()) == ["first", "third"]
    now[0] = 10.0
    assert state.get("first") is None
