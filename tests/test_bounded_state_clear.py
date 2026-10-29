"""BoundedTTLMap.clear empties the map in one step."""

from src.utils.bounded_state import BoundedTTLMap


class NoIterMap(BoundedTTLMap):
    def __iter__(self):
        raise AssertionError("clear must not pop entries one by one")


def test_clear_empties_the_map_without_popping_each_entry():
    state = NoIterMap(max_entries=10_000, ttl_seconds=60)
    for index in range(5_000):
        state.set(f"key-{index}", index)

    state.clear()

    assert len(state) == 0
    assert state._expiry_heap == []
    assert state._created_heap == []
    state["again"] = 1
    assert state.get("again") == 1
    assert state.evictions["ttl"] == 0
