"""Alphabets: the option set Jev picks from at every generation step.

An alphabet is a list of symbols plus a STOP symbol. Each symbol has a ``key``
(what Jev sees and returns) and an ``emit`` string (what gets appended to the
answer). Alphabets are JSON files so they are swappable without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from pathlib import Path

# The Jev API rejects a choice question with more than this many options. This is a
# limit on one *question*, not on an alphabet: the `buckets` strategy splits a larger
# alphabet across several questions, so only `choice` has to fit inside it.
MAX_CHOICES = 255

_BUILTIN_PACKAGE = "jevchat.alphabets"


class AlphabetError(Exception):
    """Raised when an alphabet cannot be loaded or is not usable."""


@dataclass(frozen=True)
class Symbol:
    key: str
    emit: str
    description: str | None = None


@dataclass(frozen=True)
class Alphabet:
    name: str
    description: str
    symbols: tuple[Symbol, ...]
    stop_key: str = "STOP"
    stop_description: str = "The reply is complete: nothing more should be added."
    source: str | None = None
    # Settings this alphabet prefers, applied only where the user has not said
    # otherwise. A character alphabet wants no repetition penalty; a subword one
    # gains nothing from hypothesis options and pays 43% more for them.
    defaults: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.symbols:
            raise AlphabetError(f"alphabet {self.name!r} has no symbols")
        seen: set[str] = set()
        for sym in self.symbols:
            if not sym.key:
                raise AlphabetError(f"alphabet {self.name!r} has a symbol with an empty key")
            if sym.key in seen:
                raise AlphabetError(f"alphabet {self.name!r} repeats the key {sym.key!r}")
            seen.add(sym.key)
        if self.stop_key in seen:
            raise AlphabetError(
                f"alphabet {self.name!r} uses the stop key {self.stop_key!r} as a symbol"
            )

    @cached_property
    def _by_key(self) -> dict[str, str]:
        return {s.key: s.emit for s in self.symbols}

    @property
    def size(self) -> int:
        """Number of options sent to Jev, STOP included."""
        return len(self.symbols) + 1

    def criteria(self) -> dict[str, str | None]:
        """The ``criteria`` map for a Jev choice question."""
        out: dict[str, str | None] = {s.key: s.description for s in self.symbols}
        out[self.stop_key] = self.stop_description
        return out

    def fits_one_question(self) -> bool:
        """Whether the whole alphabet can go in a single Jev choice question."""
        return self.size <= MAX_CHOICES

    def emit_for(self, key: str) -> str | None:
        """Text to append for ``key``; ``None`` means the key is STOP."""
        if key == self.stop_key:
            return None
        try:
            return self._by_key[key]
        except KeyError:
            raise AlphabetError(f"{key!r} is not in alphabet {self.name!r}") from None


def _symbol_from_json(raw: object, index: int, alphabet_name: str) -> Symbol:
    if isinstance(raw, str):
        return Symbol(key=raw, emit=raw, description=None)
    if not isinstance(raw, dict):
        raise AlphabetError(
            f"alphabet {alphabet_name!r} symbol #{index} must be a string or an object"
        )
    try:
        key = raw["key"]
    except KeyError:
        raise AlphabetError(f"alphabet {alphabet_name!r} symbol #{index} is missing 'key'") from None
    emit = raw.get("emit", key)
    description = raw.get("description")
    return Symbol(key=str(key), emit=str(emit), description=description)


def from_json(data: dict, *, source: str | None = None) -> Alphabet:
    name = str(data.get("name") or Path(source or "alphabet").stem)
    raw_symbols = data.get("symbols")
    if not isinstance(raw_symbols, list):
        raise AlphabetError(f"alphabet {name!r} needs a 'symbols' list")
    stop = data.get("stop") or {}
    defaults = Alphabet.__dataclass_fields__
    return Alphabet(
        name=name,
        description=str(data.get("description", "")),
        symbols=tuple(_symbol_from_json(s, i, name) for i, s in enumerate(raw_symbols)),
        stop_key=str(stop.get("key", defaults["stop_key"].default)),
        stop_description=str(stop.get("description", defaults["stop_description"].default)),
        defaults=tuple(sorted((data.get("defaults") or {}).items())),
        source=source,
    )


def builtin_names() -> list[str]:
    root = resources.files(_BUILTIN_PACKAGE)
    return sorted(p.name[: -len(".json")] for p in root.iterdir() if p.name.endswith(".json"))


def _search_dirs(extra: Path | None = None) -> list[Path]:
    dirs = []
    if extra is not None:
        dirs.append(extra)
    dirs.append(Path.cwd() / "alphabets")
    return dirs


def load(ref: str, *, search_dir: Path | None = None) -> Alphabet:
    """Load an alphabet by builtin name, bare name in ``alphabets/``, or file path."""
    path = Path(ref).expanduser()
    if path.suffix == ".json" or path.is_file():
        if not path.is_file():
            raise AlphabetError(f"no alphabet file at {path}")
        return from_json(json.loads(path.read_text()), source=str(path))

    for directory in _search_dirs(search_dir):
        candidate = directory / f"{ref}.json"
        if candidate.is_file():
            return from_json(json.loads(candidate.read_text()), source=str(candidate))

    resource = resources.files(_BUILTIN_PACKAGE) / f"{ref}.json"
    if resource.is_file():
        return from_json(json.loads(resource.read_text()), source=f"builtin:{ref}")

    raise AlphabetError(
        f"unknown alphabet {ref!r}; builtins are {', '.join(builtin_names())}, "
        "or pass a path to a .json file"
    )
