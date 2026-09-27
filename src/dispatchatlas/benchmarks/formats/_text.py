"""Token handling shared by the instance-file readers.

Every reader fails closed: a malformed file raises
:class:`~dispatchatlas.core.errors.SerializationError` naming the source and the
offending value, rather than producing a partial problem.
"""

from __future__ import annotations

from collections.abc import Sequence

from dispatchatlas.core.errors import ConfigError, SerializationError

__all__ = ["Tokens", "check_first_machine", "is_integer", "parse_error", "to_integer"]


def parse_error(source: str, message: str) -> SerializationError:
    """Return the error a reader raises for a malformed ``source``."""
    return SerializationError(f"{source}: {message}")


def is_integer(token: str) -> bool:
    """Return whether ``token`` is a base-10 integer, optionally signed."""
    body = token[1:] if token[:1] in "+-" else token
    return body.isdigit()


def to_integer(token: str, what: str, source: str, *, minimum: int | None = None) -> int:
    """Return ``token`` as an integer of at least ``minimum``.

    Raises:
        SerializationError: If the token is not an integer or is too small.
    """
    if not is_integer(token):
        raise parse_error(source, f"{what} must be an integer, got {token!r}")
    value = int(token)
    if minimum is not None and value < minimum:
        raise parse_error(source, f"{what} must be at least {minimum}, got {value}")
    return value


def check_first_machine(first_machine: object) -> int:
    """Return the machine numbering base, which must be 0 or 1.

    Raises:
        ConfigError: If it is neither.
    """
    if first_machine not in (0, 1) or isinstance(first_machine, bool):
        raise ConfigError(f"first_machine must be 0 or 1, got {first_machine!r}")
    return int(first_machine)


class Tokens:
    """A cursor over whitespace-separated tokens that reads integers."""

    def __init__(self, tokens: Sequence[str], source: str) -> None:
        self._tokens = tokens
        self._position = 0
        self.source = source

    @property
    def remaining(self) -> int:
        """Number of tokens not read yet."""
        return len(self._tokens) - self._position

    def integer(self, what: str, *, minimum: int | None = None) -> int:
        """Read the next token as an integer.

        Raises:
            SerializationError: If no token is left or it is not a valid integer.
        """
        if self._position >= len(self._tokens):
            raise parse_error(self.source, f"the data ends before the {what}")
        token = self._tokens[self._position]
        self._position += 1
        return to_integer(token, what, self.source, minimum=minimum)
