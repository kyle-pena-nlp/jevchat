"""Bisection sampling: narrow the alphabet with earlier/later questions.

Instead of one choice question over the whole alphabet, the alphabet is sorted
and split in half repeatedly. Each split is a yes/no (`noul`) question — does the
next symbol fall in the earlier group or the later one? — and the answer is a
probability, not a verdict, so the two branches keep their weight. Splitting stops
once a group is small enough to put in one choice question.

Every question in the tree travels in a single request. Jev answers them in
parallel and in isolation, so the whole tree costs about one round trip, and the
walk down it happens locally from the returned probabilities.

Two things make this worth doing over a flat choice question:

* Jev rounds probabilities to two decimals. Over 255 options that zeroes about
  230 of them; over a 16-option leaf almost nothing is lost.
* A leaf question is small, so Jev's bias towards the first options in a criteria
  map has far less room to distort the result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .alphabet import Alphabet, Symbol
from .client import ChoiceAnswer, JevClient, JevError

STOP_QUESTION = "stop"


@dataclass(frozen=True)
class Split:
    """One earlier/later decision over a group of symbols."""

    node: str
    lo: tuple[Symbol, ...]
    hi: tuple[Symbol, ...]


@dataclass(frozen=True)
class Leaf:
    """A group small enough to resolve with a single choice question."""

    node: str
    symbols: tuple[Symbol, ...]
    path: tuple[tuple[str, bool], ...]  # (split node, took the later half)


@dataclass(frozen=True)
class Tree:
    splits: tuple[Split, ...]
    leaves: tuple[Leaf, ...]

    @property
    def questions_per_step(self) -> int:
        """Splits (before any swapping) + non-trivial leaves + the STOP question."""
        return len(self.splits) + sum(1 for lf in self.leaves if len(lf.symbols) > 1) + 1

    @property
    def depth(self) -> int:
        return max((len(lf.path) for lf in self.leaves), default=0)


def build(alphabet: Alphabet, cutoff: int) -> Tree:
    """Split the sorted alphabet in half until each group fits in `cutoff`."""
    if cutoff < 2:
        raise ValueError("bisect cutoff must be >= 2")

    symbols = sorted(alphabet.symbols, key=lambda s: (s.emit, s.key))
    splits: list[Split] = []
    leaves: list[Leaf] = []

    def walk(group: list[Symbol], node: str, path: list[tuple[str, bool]]) -> None:
        if len(group) <= cutoff:
            leaves.append(Leaf(node, tuple(group), tuple(path)))
            return
        mid = len(group) // 2
        lo, hi = group[:mid], group[mid:]
        splits.append(Split(node, tuple(lo), tuple(hi)))
        walk(lo, node + "L", [*path, (node, False)])
        walk(hi, node + "H", [*path, (node, True)])

    walk(symbols, "n", [])
    return Tree(tuple(splits), tuple(leaves))


def _describe(group: tuple[Symbol, ...]) -> str:
    listing = ", ".join(repr(s.emit) for s in group)
    return f"the next symbol is one of these {len(group)}: {listing}"


def build_questions(
    tree: Tree,
    alphabet: Alphabet,
    instructions: str,
    *,
    swap: bool = True,
    rng: random.Random | None = None,
) -> dict[str, dict]:
    """Every split and every leaf as one question map."""
    questions: dict[str, dict] = {
        STOP_QUESTION: {
            "type": "noul",
            "instructions": "Is the reply already finished?",
            "criteria": {
                "true": alphabet.stop_description,
                "false": "The reply is not finished: at least one more symbol follows.",
            },
        }
    }

    split_instructions = (
        f"{instructions} Which of the two groups does that option belong to?"
    )
    for split in tree.splits:
        questions[f"{split.node}:a"] = {
            "type": "noul",
            "instructions": split_instructions,
            "criteria": {"true": _describe(split.hi), "false": _describe(split.lo)},
        }
        if swap:
            # The same split with the groups exchanged, to cancel any lean
            # towards `true`. The two answers are averaged.
            questions[f"{split.node}:b"] = {
                "type": "noul",
                "instructions": split_instructions,
                "criteria": {"true": _describe(split.lo), "false": _describe(split.hi)},
            }

    for leaf in tree.leaves:
        if len(leaf.symbols) < 2:
            continue
        criteria: list[tuple[str, str | None]] = [
            (s.key, s.description) for s in leaf.symbols
        ]
        if rng is not None:
            rng.shuffle(criteria)
        questions[f"{leaf.node}:pick"] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(criteria),
        }
    return questions


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def distribution(
    tree: Tree, alphabet: Alphabet, answers: dict, *, swap: bool = True
) -> dict[str, float]:
    """Fold the tree's answers into one distribution over the whole alphabet.

    p(symbol) = p(not stop) x prod(branch probabilities) x p(symbol | leaf),
    which is the distribution a top-down walk would sample from, computed in full
    so temperature and the top-k/top-p filters act on the real thing.
    """
    try:
        p_stop = _clamp(float(answers[STOP_QUESTION]["noul"]))

        p_hi: dict[str, float] = {}
        for split in tree.splits:
            forward = _clamp(float(answers[f"{split.node}:a"]["noul"]))
            if swap:
                reverse = _clamp(float(answers[f"{split.node}:b"]["noul"]))
                forward = (forward + (1.0 - reverse)) / 2.0
            p_hi[split.node] = forward

        out: dict[str, float] = {alphabet.stop_key: p_stop}
        for leaf in tree.leaves:
            reach = 1.0 - p_stop
            for node, took_hi in leaf.path:
                reach *= p_hi[node] if took_hi else 1.0 - p_hi[node]

            if len(leaf.symbols) < 2:
                out[leaf.symbols[0].key] = reach
                continue

            within = answers[f"{leaf.node}:pick"]["probabilities"]
            total = sum(float(v) for v in within.values()) or 1.0
            for symbol in leaf.symbols:
                share = float(within.get(symbol.key, 0.0)) / total
                out[symbol.key] = reach * share
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"unexpected Jev response shape: {answers!r}") from exc
    return out


def ask(
    client: JevClient,
    tree: Tree,
    alphabet: Alphabet,
    state: dict | str,
    instructions: str,
    *,
    swap: bool = True,
    rng: random.Random | None = None,
    cancel=None,
) -> ChoiceAnswer:
    """One request for the whole tree, returned as a flat distribution."""
    questions = build_questions(tree, alphabet, instructions, swap=swap, rng=rng)
    answers, latency = client.ask(
        state, questions, cancel=cancel,
        overflow_hint=(
            f"{len(questions)} bisection questions over {alphabet.size} options. "
            "Raise `bisect_cutoff`, turn off `bisect_swap`, or use a smaller alphabet."
        ),
    )
    probabilities = distribution(tree, alphabet, answers, swap=swap)
    return ChoiceAnswer(
        choice=max(probabilities, key=probabilities.__getitem__),
        probabilities=probabilities,
        confidence=max(probabilities.values(), default=0.0),
        model=client.model,
        latency=latency,
        variants=len(questions),
    )
