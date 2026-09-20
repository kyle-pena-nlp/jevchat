"""A fixed set of continuations, used to compare modes on the same ground.

Each case is a question plus a prefix of a correct answer, and the symbol that
should come next. Scoring one case is one step of real generation — the same
`Scorer` the loop uses — so the numbers describe the modes as they actually run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .alphabet import Alphabet
from .client import JevClient
from .config import Config
from .generate import Scorer, build_state


@dataclass(frozen=True)
class Case:
    question: str
    answer_so_far: str
    expected: str


CASES: tuple[Case, ...] = (
    Case("what is the capital of france?", "The capital of France is Par", "i"),
    Case("what is the capital of france?", "The capital of France is ", "P"),
    Case("what is the capital of france?", "The cap", "i"),
    Case("what is the capital of france?", "The capital of Fra", "n"),
    Case("is the sky blue?", "Yes, the sky is bl", "u"),
    Case("is the sky blue?", "Yes, the s", "k"),
    Case("what colour is grass?", "Grass is gre", "e"),
    Case("what colour is grass?", "Grass is g", "r"),
    Case("how many legs does a dog have?", "A dog has fo", "u"),
    Case("what is 2+2?", "2 + 2 = ", "4"),
    Case("who wrote hamlet?", "Hamlet was written by Shake", "s"),
    Case("what is water made of?", "Water is made of hydro", "g"),
)

# Word-level continuations, for the word and subword alphabets. Every target is a
# whole option in words1k, so a miss is a ranking failure rather than a vocabulary gap.
WORD_CASES: tuple[Case, ...] = (
    Case("is the sky blue?", " Yes, the sky is", " blue"),
    Case("what do you need to live?", " You need", " water"),
    Case("how many eyes do people have?", " People have", " two"),
    Case("what colour is grass?", " Grass is", " green"),
    Case("how many legs does a dog have?", " A dog has", " four"),
    Case("what time is it?", " I do not", " know"),
    Case("where do fish live?", " Fish live in", " water"),
    Case("what do bees make?", " Bees make", " honey"),
    Case("what is the opposite of hot?", " The opposite of hot is", " cold"),
    Case("what colour is snow?", " Snow is", " white"),
)

CASE_SETS = {"char": CASES, "word": WORD_CASES}


# The modes the README quotes. Each is a name plus Config overrides.
MODES: tuple[tuple[str, dict], ...] = (
    ("choice, symbol, no shuffle", {"strategy": "choice", "presentation": "symbol",
                                    "ensemble": 1, "shuffle_criteria": False}),
    ("choice, symbol", {"strategy": "choice", "presentation": "symbol", "ensemble": 1}),
    ("choice, symbol, ensemble 4", {"strategy": "choice", "presentation": "symbol",
                                    "ensemble": 4}),
    ("choice, hypothesis w24", {"strategy": "choice", "presentation": "hypothesis",
                                "window": 24}),
    ("choice, hypothesis w40", {"strategy": "choice", "presentation": "hypothesis",
                                "window": 40}),
    ("bisect, cutoff 20", {"strategy": "bisect", "bisect_cutoff": 20,
                           "presentation": "symbol"}),
    ("buckets, symbol", {"strategy": "buckets", "bucket_size": 127,
                         "presentation": "symbol"}),
    ("buckets, hypothesis w40", {"strategy": "buckets", "bucket_size": 127,
                                 "presentation": "hypothesis", "window": 40}),
    # bucket_size is pinned so `refine` has several buckets to weigh against each
    # other even on the small alphabets.
    ("buckets, size 16", {"strategy": "buckets", "bucket_size": 16}),
    ("refine, 0 rounds", {"strategy": "refine", "bucket_size": 16,
                          "refine_rounds": 0}),
    ("refine, 1 round m=3", {"strategy": "refine", "bucket_size": 16,
                             "refine_rounds": 1, "refine_nucleus": 3}),
    ("refine, 2 rounds m=3", {"strategy": "refine", "bucket_size": 16,
                              "refine_rounds": 2, "refine_nucleus": 3}),
    ("refine, 1 round m=6", {"strategy": "refine", "bucket_size": 16,
                             "refine_rounds": 1, "refine_nucleus": 6}),
    ("refine, buckets sorted", {"strategy": "refine", "bucket_size": 16,
                                "bucket_order": "sorted"}),
    ("refine, buckets shuffled", {"strategy": "refine", "bucket_size": 16,
                                  "bucket_order": "shuffled"}),
)


@dataclass
class Result:
    label: str
    cases: int
    top1: int
    top3: int
    mass: float        # mean probability put on the right symbol
    nonzero: float     # mean number of symbols with any probability at all
    ms: float          # mean wall time per step
    input_tokens: int  # mean input tokens per step
    questions: int     # questions sent per step
    requests: int      # requests sent per step
    offered: int       # cases whose answer the alphabet can even express

    @property
    def top1_pct(self) -> float:
        return 100.0 * self.top1 / self.cases


def run_mode(
    client: JevClient,
    alphabet: Alphabet,
    config: Config,
    label: str,
    cases: tuple[Case, ...] = CASES,
) -> Result:
    scorer = Scorer.build(client, alphabet, config)
    before_tokens = client.usage.input_tokens
    top1 = top3 = 0
    mass = nonzero = 0.0

    # A case whose answer is not in the alphabet cannot be won, and scoring it would
    # measure vocabulary coverage while pretending to measure ranking. Skip it, and
    # report how many were skipped.
    scored = [(c, _key_for(alphabet, c.expected)) for c in cases]
    scored = [(c, key) for c, key in scored if key is not None]

    started = time.monotonic()
    for case, expected in scored:
        state = build_state(config, case.question, case.answer_so_far)
        probabilities = scorer.score(state, case.answer_so_far).probabilities
        ranked = [k for k, _ in sorted(probabilities.items(), key=lambda kv: -kv[1])]
        top1 += ranked[:1] == [expected]
        top3 += expected in ranked[:3]
        mass += probabilities.get(expected, 0.0)
        nonzero += sum(1 for v in probabilities.values() if v > 0)

    n = max(1, len(scored))
    return Result(
        label=label,
        requests=scorer.requests_per_step,
        offered=len(scored),
        cases=len(scored),
        top1=top1,
        top3=top3,
        mass=mass / n,
        nonzero=nonzero / n,
        ms=1000.0 * (time.monotonic() - started) / n,
        input_tokens=(client.usage.input_tokens - before_tokens) // n,
        questions=scorer.questions_per_step,
    )


def _key_for(alphabet: Alphabet, emit: str) -> str | None:
    """The alphabet key that emits `emit` ('SPACE' rather than ' '), or None."""
    for symbol in alphabet.symbols:
        if symbol.emit == emit:
            return symbol.key
    return None
