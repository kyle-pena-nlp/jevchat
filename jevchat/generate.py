"""The sampling loop: ask Jev to score the alphabet, draw a symbol, repeat."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from . import bisect as bisect_mod
from . import buckets as buckets_mod
from . import present
from .alphabet import MAX_CHOICES, Alphabet, AlphabetError
from .client import Cancelled, ChoiceAnswer, JevClient, JevError
from .config import Config
from .sampler import Draw, sample


@dataclass(frozen=True)
class Turn:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class Stats:
    steps: int = 0
    chars: int = 0
    elapsed: float = 0.0
    api_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def symbols_per_sec(self) -> float:
        return self.steps / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.chars / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def mean_latency(self) -> float:
        return self.api_time / self.steps if self.steps else 0.0


@dataclass
class Step:
    """One sampled symbol."""

    index: int
    key: str
    emit: str | None       # None when the key is STOP
    text: str              # the whole reply after this step
    draw: Draw
    latency: float
    stats: Stats


@dataclass
class Done:
    """Emitted once, after the last Step."""

    text: str
    reason: str            # stop | max_steps | max_chars | cancelled
    stats: Stats


@dataclass
class Scorer:
    """Turns one state into a distribution over the alphabet, the configured way.

    Generation and the benchmark both go through this, so what the benchmark
    measures is exactly what generation does.
    """

    client: JevClient
    alphabet: Alphabet
    config: Config
    order_rng: random.Random
    tree: object | None
    criteria_items: list

    @classmethod
    def build(cls, client: JevClient, alphabet: Alphabet, config: Config) -> "Scorer":
        if config.strategy == "choice" and not alphabet.fits_one_question():
            raise AlphabetError(
                f"alphabet {alphabet.name!r} has {alphabet.size} options, more than the "
                f"{MAX_CHOICES} a single Jev question allows. Use --strategy buckets "
                "(or bisect) to spread it across several questions."
            )
        return cls(
            client=client,
            alphabet=alphabet,
            config=config,
            # Kept separate from the sampling rng so toggling option order does not
            # change the sequence of draws for a given seed.
            order_rng=random.Random(config.seed),
            tree=(
                bisect_mod.build(alphabet, config.bisect_cutoff)
                if config.strategy == "bisect"
                else buckets_mod.build(alphabet, config.bucket_size, config.bucket_batch)
                if config.strategy == "buckets"
                else None
            ),
            criteria_items=list(alphabet.criteria().items()),
        )

    @property
    def requests_per_step(self) -> int:
        if self.config.strategy == "buckets":
            return self.tree.requests_per_step
        return 1

    @property
    def questions_per_step(self) -> int:
        if self.config.strategy == "buckets":
            return self.tree.questions_per_step
        if self.tree is not None:
            swap = 2 if self.config.bisect_swap else 1
            return len(self.tree.splits) * swap + sum(
                1 for leaf in self.tree.leaves if len(leaf.symbols) > 1
            ) + 1
        return max(1, self.config.ensemble) if self.config.shuffle_criteria else 1

    def score(self, state: dict | str, text: str = "", *, cancel=None) -> ChoiceAnswer:
        rng = self.order_rng if self.config.shuffle_criteria else None
        if self.config.strategy == "buckets":
            return buckets_mod.ask(
                self.client, self.tree, self.alphabet, state, self.config,
                text=text, rng=rng, cancel=cancel,
            )
        if self.tree is not None:
            return bisect_mod.ask(
                self.client, self.tree, self.alphabet, state,
                self.config.instructions,
                swap=self.config.bisect_swap, rng=rng, cancel=cancel,
            )

        variants = []
        options = None
        for _ in range(self.questions_per_step):
            options = present.build(
                self.alphabet, self.alphabet.symbols,
                presentation=self.config.presentation,
                text=text, size=self.config.window, rng=rng,
            )
            variants.append(options.criteria)
        answer = self.client.choice(
            state, self.config.active_instructions, variants, cancel=cancel
        )
        # Every variant shares the same labels, so any of them can fold the answer.
        folded = options.fold(answer.probabilities)
        if not folded:
            raise JevError(
                "none of Jev's answers matched the options that were sent; got "
                f"{sorted(answer.probabilities)[:5]}"
            )
        answer.probabilities = folded
        answer.choice = max(folded, key=folded.__getitem__)
        return answer


def build_state(
    config: Config,
    question: str,
    answer_so_far: str,
    history: list[Turn] | None = None,
) -> dict:
    """The `state` object Jev scores the alphabet against."""
    state: dict = {"task": config.task, "question": question}
    if history:
        recent = history[-config.history_turns :] if config.history_turns else history
        if recent:
            state["conversation"] = [{"role": t.role, "content": t.content} for t in recent]
    state["answer_so_far"] = answer_so_far
    return state


def generate(
    client: JevClient,
    alphabet: Alphabet,
    config: Config,
    question: str,
    *,
    history: list[Turn] | None = None,
    cancel: threading.Event | None = None,
    rng: random.Random | None = None,
) -> Iterator[Step | Done]:
    """Yield a Step per sampled symbol, then a single Done.

    Set ``cancel`` at any time to stop: the in-flight request is allowed to
    finish, then generation ends with reason ``cancelled`` and the partial text.
    """
    rng = rng or random.Random(config.seed)
    scorer = Scorer.build(client, alphabet, config)
    # Symbols that would open with whitespace, e.g. " the" in the token alphabet.
    leading_space = {s.key for s in alphabet.symbols if s.emit[:1].isspace()}
    recent: deque[str] = deque(maxlen=config.repetition_window or 1)

    text = ""
    stats = Stats()
    started = time.monotonic()
    reason = "max_steps"

    for index in range(config.max_steps):
        if cancel is not None and cancel.is_set():
            reason = "cancelled"
            break

        state = build_state(config, question, text, history)
        try:
            answer = scorer.score(state, text, cancel=cancel)
        except Cancelled:
            reason = "cancelled"
            break

        banned: set[str] = set()
        if index < config.min_steps:
            banned.add(alphabet.stop_key)
        if config.no_repeat_space and text[-1:].isspace():
            # Without this, whitespace is self-reinforcing and generation stalls.
            banned |= leading_space

        bias = {alphabet.stop_key: config.stop_bias} if config.stop_bias != 1.0 else {}
        if config.repetition_penalty != 1.0 and config.repetition_window:
            for key in recent:
                bias[key] = bias.get(key, 1.0) / config.repetition_penalty

        draw = sample(
            answer.probabilities,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            banned=banned,
            bias=bias or None,
            rng=rng,
        )

        emit = alphabet.emit_for(draw.key)
        if draw.key != alphabet.stop_key:
            recent.append(draw.key)
        if emit is not None:
            text += emit

        stats.steps = index + 1
        stats.chars = len(text)
        stats.api_time += answer.latency
        stats.elapsed = time.monotonic() - started
        stats.input_tokens = client.usage.input_tokens
        stats.output_tokens = client.usage.output_tokens

        yield Step(
            index=index,
            key=draw.key,
            emit=emit,
            text=text,
            draw=draw,
            latency=answer.latency,
            stats=stats,
        )

        if emit is None:
            reason = "stop"
            break
        if config.max_chars and len(text) >= config.max_chars:
            reason = "max_chars"
            break

    stats.elapsed = time.monotonic() - started
    yield Done(text=text, reason=reason, stats=stats)
