"""Bucketed sampling: one alphabet spread across many questions.

A Jev choice question holds at most 255 options, but that is a limit on a
*question*, not on a vocabulary. This strategy splits the alphabet into buckets of
at most 254 symbols and asks one question per bucket, each with an extra OTHER
option meaning "the next symbol is not in this bucket".

OTHER is what makes the buckets comparable. Jev normalises within a question, so
without it you get N unrelated conditional distributions with no way to weigh them
against each other. With it, each question reports how much of its mass belongs to
its own members versus everything else, and the in-bucket numbers can simply be
concatenated.

Questions in one request are answered in parallel, and latency is close to flat in
their number, so a vocabulary of a few thousand symbols still costs about one round
trip. Beyond what the context window holds, the buckets are split across as few
requests as will fit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import present
from .alphabet import MAX_CHOICES, Alphabet, Symbol
from .client import ChoiceAnswer, JevClient, JevError

OTHER_KEY = "OTHER"
OTHER_DESCRIPTION = (
    "None of the options listed in this question. The symbol that comes next is "
    "something else entirely, not present among these options."
)
STOP_QUESTION = "stop"

# One slot in every bucket is spent on OTHER.
MAX_BUCKET = MAX_CHOICES - 1

OTHER_INSTRUCTION = (
    " Only a fraction of the possible symbols are listed in this question. If the "
    "symbol that should come next is not among them, choose OTHER."
)


@dataclass(frozen=True)
class Plan:
    """How one alphabet is cut into questions."""

    buckets: tuple[tuple[Symbol, ...], ...]
    batch: int  # questions per request

    @property
    def questions_per_step(self) -> int:
        return len(self.buckets) + 1  # + the STOP question

    @property
    def requests_per_step(self) -> int:
        return max(1, -(-self.questions_per_step // self.batch))


def build(alphabet: Alphabet, bucket_size: int, batch: int) -> Plan:
    """Cut `alphabet` into buckets of at most `bucket_size` symbols."""
    if bucket_size < 1:
        raise ValueError("bucket_size must be >= 1")
    if bucket_size > MAX_BUCKET:
        raise ValueError(
            f"bucket_size {bucket_size} leaves no room for OTHER; "
            f"the most Jev allows is {MAX_BUCKET}"
        )
    if batch < 1:
        raise ValueError("bucket_batch must be >= 1")

    symbols = list(alphabet.symbols)
    buckets = tuple(
        tuple(symbols[i : i + bucket_size]) for i in range(0, len(symbols), bucket_size)
    )
    return Plan(buckets=buckets, batch=batch)


def build_questions(
    plan: Plan,
    alphabet: Alphabet,
    config,
    *,
    text: str = "",
    rng: random.Random | None = None,
) -> tuple[dict[str, dict], dict[str, present.Options]]:
    """One question per bucket, plus the STOP question.

    STOP is asked once, on its own, rather than offered inside every bucket where it
    would be counted many times over. So the buckets themselves never include it.
    """
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
    instructions = config.active_instructions + OTHER_INSTRUCTION
    options: dict[str, present.Options] = {}

    for index, bucket in enumerate(plan.buckets):
        name = f"b{index}"
        options[name] = present.build(
            alphabet, bucket,
            presentation=config.presentation,
            text=text,
            size=config.window,
            describe=config.bucket_describe,
            include_stop=False,
            extra={OTHER_KEY: OTHER_DESCRIPTION},
            rng=rng,
        )
        questions[name] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": options[name].criteria,
        }
    return questions, options


def distribution(
    plan: Plan,
    alphabet: Alphabet,
    answers: dict,
    options: dict[str, present.Options],
) -> dict[str, float]:
    """Fold the buckets into one distribution over the whole alphabet.

    OTHER is dropped and what remains is concatenated, then rescaled to leave room
    for STOP. Weighting each bucket by 1 - P(OTHER) instead scored identically in
    testing, so the simpler form is used.
    """
    try:
        p_stop = min(1.0, max(0.0, float(answers[STOP_QUESTION]["noul"])))

        merged: dict[str, float] = {}
        for index in range(len(plan.buckets)):
            name = f"b{index}"
            raw = dict(answers[name]["probabilities"])
            raw.pop(OTHER_KEY, None)
            for key, value in options[name].fold(raw).items():
                merged[key] = merged.get(key, 0.0) + value
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"unexpected Jev response shape: {answers!r}") from exc

    total = sum(merged.values())
    scale = (1.0 - p_stop) / total if total > 0 else 0.0
    out = {key: value * scale for key, value in merged.items()}
    out[alphabet.stop_key] = p_stop
    return out


def ask(
    client: JevClient,
    plan: Plan,
    alphabet: Alphabet,
    state: dict | str,
    config,
    *,
    text: str = "",
    rng: random.Random | None = None,
    cancel=None,
) -> ChoiceAnswer:
    """Ask every bucket, in as few requests as the context window allows."""
    questions, options = build_questions(plan, alphabet, config, text=text, rng=rng)
    names = list(questions)
    answers: dict = {}
    latency = 0.0
    batch = plan.batch
    index = 0

    while index < len(names):
        slice_ = {name: questions[name] for name in names[index : index + batch]}
        try:
            part, took = client.ask(
                state, slice_, cancel=cancel,
                overflow_hint=(
                    f"{len(slice_)} bucket questions. Lower `bucket_batch` or "
                    "`bucket_size`."
                ),
            )
        except JevError as exc:
            # Back off to smaller requests rather than failing the whole step.
            if "overflowed" in str(exc) and batch > 1:
                batch //= 2
                continue
            raise
        answers.update(part)
        latency += took
        index += batch

    probabilities = distribution(plan, alphabet, answers, options)
    return ChoiceAnswer(
        choice=max(probabilities, key=probabilities.__getitem__),
        probabilities=probabilities,
        confidence=max(probabilities.values(), default=0.0),
        model=client.model,
        latency=latency,
        variants=len(questions),
    )
