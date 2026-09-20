"""Beam search: keep several candidate replies alive instead of committing.

Every other strategy here commits to one symbol per step. Each Jev call is
independent and stateless, so a bad symbol is never repaired — it just conditions
everything after it. That compounding is the reason long replies fall apart while
short ones come out right.

Beam search keeps the `beam_width` best partial replies and lets later evidence
settle earlier ambiguity: a prefix that looked fine at step 2 can be abandoned at
step 5 when nothing good follows it. Beams are ranked by mean log probability, so a
short confident reply is not automatically beaten by a long hesitant one.

The cost is one score per live beam per step, and a beam carries its own
repetition history and whitespace state, so the guards act per beam.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from .alphabet import Alphabet
from .client import Cancelled, JevClient
from .config import Config
from .generate import Done, Scorer, Stats, Step, Turn, build_state
from .sampler import Draw, shape

LOG_FLOOR = 1e-12


@dataclass(frozen=True)
class Beam:
    text: str = ""
    logprob: float = 0.0
    steps: int = 0
    keys: tuple[str, ...] = ()
    finished: bool = False
    last: Draw | None = None

    def score(self, length_penalty: float) -> float:
        """Mean log probability, so length does not decide the winner by itself."""
        if self.steps == 0:
            return 0.0
        return self.logprob / (self.steps ** length_penalty)


def _guards(beam: Beam, alphabet: Alphabet, config: Config, index: int,
            leading_space: set[str]) -> tuple[set[str], dict[str, float]]:
    banned: set[str] = set()
    if index < config.min_steps:
        banned.add(alphabet.stop_key)
    if config.no_repeat_space and beam.text[-1:].isspace():
        banned |= leading_space

    bias: dict[str, float] = {}
    if config.stop_bias != 1.0:
        bias[alphabet.stop_key] = config.stop_bias
    if config.repetition_penalty != 1.0 and config.repetition_window:
        for key in beam.keys[-config.repetition_window :]:
            bias[key] = bias.get(key, 1.0) / config.repetition_penalty
    return banned, bias


def search(
    client: JevClient,
    alphabet: Alphabet,
    config: Config,
    question: str,
    *,
    history: list[Turn] | None = None,
    cancel: threading.Event | None = None,
    rng: random.Random | None = None,
) -> Iterator[Step | Done]:
    """Yield the leading beam after each step, then a single Done."""
    scorer = Scorer.build(client, alphabet, config)
    leading_space = {s.key for s in alphabet.symbols if s.emit[:1].isspace()}
    width = max(1, config.beam_width)

    live = [Beam()]
    done: list[Beam] = []
    stats = Stats()
    started = time.monotonic()
    reason = "max_steps"

    for index in range(config.max_steps):
        if cancel is not None and cancel.is_set():
            reason = "cancelled"
            break

        expanded: list[Beam] = []
        try:
            for beam in live:
                state = build_state(config, question, beam.text, history)
                answer = scorer.score(state, beam.text, cancel=cancel)
                stats.api_time += answer.latency
                banned, bias = _guards(beam, alphabet, config, index, leading_space)
                # Bias and bans are semantic guards and still apply. Temperature,
                # top-p and top-k are sampling controls: at temperature 0 they
                # collapse the distribution to one symbol, which would leave beam
                # search with nothing to branch on. Beam ranks by probability.
                shaped, raw = shape(
                    answer.probabilities,
                    temperature=1.0, top_p=1.0, top_k=0,
                    banned=banned, bias=bias or None,
                )
                best = sorted(shaped.items(), key=lambda kv: -kv[1])[:width]
                for key, probability in best:
                    emit = alphabet.emit_for(key)
                    draw = Draw(key=key, probability=probability,
                                distribution=shaped, raw=raw)
                    expanded.append(Beam(
                        text=beam.text if emit is None else beam.text + emit,
                        logprob=beam.logprob + math.log(max(LOG_FLOOR, probability)),
                        steps=beam.steps + 1,
                        keys=beam.keys if emit is None else (*beam.keys, key),
                        finished=emit is None,
                        last=draw,
                    ))
        except Cancelled:
            reason = "cancelled"
            break

        if not expanded:
            break
        expanded.sort(key=lambda b: -b.score(config.beam_length_penalty))
        keep = expanded[:width]
        done.extend(b for b in keep if b.finished)
        live = [b for b in keep if not b.finished]

        leader = max(keep, key=lambda b: b.score(config.beam_length_penalty))
        stats.steps = index + 1
        stats.chars = len(leader.text)
        stats.elapsed = time.monotonic() - started
        stats.input_tokens = client.usage.input_tokens
        stats.output_tokens = client.usage.output_tokens
        yield Step(index=index, key=leader.last.key,
                   emit=alphabet.emit_for(leader.last.key),
                   text=leader.text, draw=leader.last,
                   latency=stats.api_time, stats=stats)

        if not live:
            reason = "stop"
            break
        if config.max_chars and min(len(b.text) for b in live) >= config.max_chars:
            reason = "max_chars"
            break

    pool = done or live
    best = (max(pool, key=lambda b: b.score(config.beam_length_penalty))
            if pool else Beam())
    stats.elapsed = time.monotonic() - started
    stats.chars = len(best.text)
    yield Done(text=best.text, reason=reason, stats=stats)
