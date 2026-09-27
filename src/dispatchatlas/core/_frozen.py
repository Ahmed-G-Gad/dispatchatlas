"""An immutable mapping for the fields of immutable objects.

:class:`types.MappingProxyType` is read-only, but it cannot be pickled or
deep-copied, so neither could the problems, schedules, and results that hold
one. :class:`FrozenMap` keeps the same read-only interface and adds both.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Any, NoReturn

__all__ = ["FrozenMap"]


class FrozenMap[K, V](Mapping[K, V]):
    """A read-only mapping that pickles, copies, and hashes.

    It copies its input and offers no way to change it. It equals any mapping
    with the same items, hashes when its values are hashable, and survives
    ``pickle`` and ``copy.deepcopy``.

    Args:
        items: A mapping, or an iterable of key and value pairs.
    """

    __slots__ = ("_data", "_hash")

    _data: dict[K, V]
    _hash: int | None

    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] = (), /) -> None:
        object.__setattr__(self, "_data", dict(items))
        object.__setattr__(self, "_hash", None)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self) -> KeysView[K]:
        """Return a view of the keys."""
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        """Return a view of the values."""
        return self._data.values()

    def items(self) -> ItemsView[K, V]:
        """Return a view of the key and value pairs."""
        return self._data.items()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMap):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other.items())
        return NotImplemented

    def __hash__(self) -> int:
        cached = self._hash
        if cached is None:
            cached = hash(frozenset(self._data.items()))
            object.__setattr__(self, "_hash", cached)
        return cached

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __reduce__(self) -> tuple[type[FrozenMap[K, V]], tuple[dict[K, V]]]:
        return (type(self), (self._data,))

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"{type(self).__name__} is immutable")
