"""The immutable mapping that model objects, schedules, and solver metadata hold."""

from __future__ import annotations

import copy
import pickle
from types import MappingProxyType

import pytest

from dispatchatlas.core._frozen import FrozenMap


def test_a_frozen_map_is_a_read_only_copy() -> None:
    source = {"b": 2, "a": 1}
    frozen = FrozenMap(source)
    source["c"] = 3
    assert dict(frozen) == {"b": 2, "a": 1}
    assert list(frozen) == ["b", "a"]
    assert (len(frozen), frozen["a"], frozen.get("z"), frozen.get("z", 0)) == (2, 1, None, 0)
    assert "a" in frozen
    assert "z" not in frozen
    assert list(frozen.keys()) == ["b", "a"]
    assert list(frozen.values()) == [2, 1]
    assert list(frozen.items()) == [("b", 2), ("a", 1)]
    assert FrozenMap([("x", 1)]) == {"x": 1}
    assert FrozenMap() == {}
    assert repr(FrozenMap({"a": 1})) == "FrozenMap({'a': 1})"


def test_a_frozen_map_cannot_change() -> None:
    frozen = FrozenMap({"a": 1})
    with pytest.raises(TypeError):
        frozen["a"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        del frozen["a"]  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="immutable"):
        frozen._data = {}
    with pytest.raises(AttributeError, match="immutable"):
        del frozen._data
    with pytest.raises(AttributeError):
        frozen.update({"b": 2})  # type: ignore[attr-defined]
    assert frozen == {"a": 1}


def test_frozen_maps_equal_any_mapping_with_the_same_items() -> None:
    frozen = FrozenMap({"a": 1, "b": 2.0})
    assert frozen == {"b": 2.0, "a": 1}
    assert frozen == MappingProxyType({"a": 1, "b": 2})
    assert frozen == FrozenMap({"b": 2, "a": 1})
    assert frozen != {"a": 1}
    assert frozen != FrozenMap({"a": 1, "b": 3})
    assert frozen != [("a", 1), ("b", 2.0)]
    plain = {"a": 1, "b": 2.0}
    assert plain == frozen


def test_frozen_maps_hash_by_content_when_their_values_do() -> None:
    first, second = FrozenMap({"a": 1, "b": (2, 3)}), FrozenMap({"b": (2, 3), "a": 1})
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    with pytest.raises(TypeError):
        hash(FrozenMap({"a": [1]}))


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_frozen_maps_pickle_and_copy(protocol: int) -> None:
    frozen = FrozenMap({("m", None, "a"): 1.5, ("m", "a", "b"): 2.0})
    again = pickle.loads(pickle.dumps(frozen, protocol=protocol))
    assert again == frozen
    assert type(again) is FrozenMap
    assert hash(again) == hash(frozen)
    nested = FrozenMap({"tags": ["x"]})
    deep = copy.deepcopy(nested)
    assert deep == nested
    assert deep["tags"] is not nested["tags"]
    assert copy.copy(nested)["tags"] is nested["tags"]
