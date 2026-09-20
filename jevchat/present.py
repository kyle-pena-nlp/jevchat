"""How the options are shown to Jev.

Two ways to ask the same question:

`symbol`      the options are the symbols themselves — 'a', 'i', ' the'. Jev has to
              append the option to `answer_so_far` in its head and judge the result.

`hypothesis`  the options are the *resulting texts* — '…France is Para',
              '…France is Pari'. The append is already done, so Jev only has to rank
              finished strings, which is what a decision model is built for. Options
              show the tail of the reply rather than all of it, to keep their cost
              constant as the reply grows; the whole reply still travels in
              `state.answer_so_far`.

Hypothesis presentation roughly doubled both top-1 accuracy and the probability mass
landing on the right symbol, for slightly fewer input tokens. See `jevchat bench`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .alphabet import Alphabet, Symbol

PRESENTATIONS = ("symbol", "hypothesis")
ELLIPSIS = "…"


@dataclass(frozen=True)
class Options:
    """The criteria map sent to Jev, plus how to read its answer back."""

    criteria: dict[str, str | None]
    key_for: dict[str, str] = field(default_factory=dict)

    def fold(self, probabilities: dict[str, float]) -> dict[str, float]:
        """Translate Jev's answer back into probabilities over alphabet keys."""
        if not self.key_for:
            return {str(k): float(v) for k, v in probabilities.items()}
        out: dict[str, float] = {}
        for label, value in probabilities.items():
            key = self.key_for.get(str(label))
            if key is not None:
                out[key] = out.get(key, 0.0) + float(value)
        return out


def window(text: str, size: int) -> str:
    """The last `size` characters of `text`, marked when something was cut."""
    if size <= 0 or len(text) <= size:
        return text
    return ELLIPSIS + text[-size:]


def build(
    alphabet: Alphabet,
    symbols: tuple[Symbol, ...] | list[Symbol],
    *,
    presentation: str = "symbol",
    text: str = "",
    size: int = 40,
    describe: bool = True,
    include_stop: bool = True,
    extra: dict[str, str | None] | None = None,
    rng: random.Random | None = None,
) -> Options:
    """Build the options for one question over `symbols`.

    With `include_stop`, STOP is offered too: as its own key under `symbol`, or as
    the reply left unchanged under `hypothesis`.
    """
    if presentation not in PRESENTATIONS:
        raise ValueError(f"unknown presentation {presentation!r}")

    pairs: list[tuple[str, str | None, str]] = []  # label, description, key
    if presentation == "symbol":
        for s in symbols:
            pairs.append((s.key, s.description if describe else None, s.key))
        if include_stop:
            pairs.append((alphabet.stop_key, alphabet.stop_description, alphabet.stop_key))
        key_for: dict[str, str] = {}
    else:
        seen: set[str] = set()
        if include_stop:
            # The reply with nothing added: choosing it means "this is finished".
            label = window(text, size)
            seen.add(label)
            pairs.append((label, None, alphabet.stop_key))
        for s in symbols:
            label = window(text + s.emit, size)
            if label in seen:
                # Two symbols that render identically; the first one keeps the slot.
                continue
            seen.add(label)
            pairs.append((label, None, s.key))
        key_for = {label: key for label, _, key in pairs}

    if rng is not None:
        rng.shuffle(pairs)

    criteria: dict[str, str | None] = {label: description for label, description, _ in pairs}
    if extra:
        criteria.update(extra)
    return Options(criteria=criteria, key_for=key_for)
