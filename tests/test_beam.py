import math
import random
import threading

import pytest

from jevchat import beam as B
from jevchat.alphabet import Alphabet, Symbol
from jevchat.client import Cancelled, ChoiceAnswer, Usage
from jevchat.config import Config
from jevchat.generate import Done, Step

ALPHA = Alphabet(name="four", description="",
                 symbols=tuple(Symbol(c, c, f"letter {c}") for c in "abxy"))


def cfg(**kwargs):
    kwargs.setdefault("presentation", "symbol")
    kwargs.setdefault("strategy", "choice")
    kwargs.setdefault("temperature", 1.0)      # beam ranks, it does not draw
    kwargs.setdefault("min_steps", 0)
    kwargs.setdefault("stop_bias", 1.0)
    kwargs.setdefault("repetition_penalty", 1.0)
    return Config(**kwargs)


class FakeClient:
    """Returns a distribution chosen by what has been written so far."""

    model = "fake"

    def __init__(self, table, default=None):
        self.table = table
        self.default = default or {"STOP": 1.0}
        self.usage = Usage()
        self.seen = []

    def choice(self, state, instructions, criteria_variants, *, cancel=None):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        text = state["answer_so_far"]
        self.seen.append(text)
        probs = self.table.get(text, self.default)
        self.usage.add({"input_tokens": 10, "output_tokens": 2},
                       questions=len(criteria_variants))
        return ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=dict(probs),
                            confidence=1.0, model="fake", latency=0.01,
                            variants=len(criteria_variants))


# Greedy takes 'a' (0.6) then stops. Going through 'b' scores better overall.
TABLE = {
    "":   {"a": 0.6, "b": 0.4},
    "a":  {"STOP": 0.8, "x": 0.1, "y": 0.1},
    "b":  {"x": 0.95, "STOP": 0.05},
    "bx": {"STOP": 0.99, "y": 0.01},
}


def run(client, config, question="q"):
    steps, done = [], None
    for event in B.search(client, ALPHA, config, question, rng=random.Random(0)):
        (steps if isinstance(event, Step) else []).append(event) if isinstance(
            event, Step) else None
        if isinstance(event, Done):
            done = event
    return steps, done


def test_beam_finds_the_path_greedy_misses():
    _, done = run(FakeClient(TABLE), cfg(beam_width=2, max_steps=3))
    assert done.text == "bx"


def test_width_one_is_greedy():
    _, done = run(FakeClient(TABLE), cfg(beam_width=1, max_steps=3))
    assert done.text == "a"


def test_beams_rank_by_mean_log_probability():
    a = B.Beam(text="a", logprob=math.log(0.6) + math.log(0.8), steps=2)
    bx = B.Beam(text="bx", logprob=math.log(0.4) + math.log(0.95) + math.log(0.99),
                steps=3)
    assert bx.score(1.0) > a.score(1.0)     # mean: -0.326 beats -0.367
    assert a.score(0.0) > bx.score(0.0)     # unnormalised, the shorter one wins


def test_a_finished_beam_stops_being_expanded():
    client = FakeClient(TABLE)
    run(client, cfg(beam_width=2, max_steps=3))
    # 'a' finishes at step 2 and is never scored again.
    assert client.seen.count("a") == 1


def test_every_step_reports_the_leader():
    steps, done = run(FakeClient(TABLE), cfg(beam_width=2, max_steps=3))
    assert steps
    assert all(isinstance(s.text, str) for s in steps)
    assert steps[-1].stats.steps == len(steps)


def test_min_steps_is_enforced_per_beam():
    client = FakeClient({"": {"STOP": 0.99, "a": 0.01}}, default={"STOP": 1.0})
    _, done = run(client, cfg(beam_width=2, min_steps=2, max_steps=4))
    assert len(done.text) >= 1        # STOP was blocked at step 0


def test_no_repeat_space_applies_per_beam():
    alpha = Alphabet(name="s", description="",
                     symbols=(Symbol("SPACE", " "), Symbol("a", "a")))
    client = FakeClient({}, default={"SPACE": 0.9, "a": 0.1})
    steps, done = [], None
    for ev in B.search(client, alpha, cfg(beam_width=1, max_steps=4,
                                          no_repeat_space=True), "q",
                       rng=random.Random(0)):
        if isinstance(ev, Done):
            done = ev
    assert "  " not in done.text


def test_cancellation_keeps_the_leading_beam():
    cancel = threading.Event()
    client = FakeClient({"": {"a": 1.0}}, default={"a": 1.0})
    seen = []
    done = None
    for ev in B.search(client, ALPHA, cfg(beam_width=2, max_steps=10), "q",
                       cancel=cancel, rng=random.Random(0)):
        if isinstance(ev, Step):
            seen.append(ev)
            if len(seen) == 2:
                cancel.set()
        else:
            done = ev
    assert done.reason == "cancelled"
    assert done.text == "aa"


def test_beam_width_is_validated():
    from jevchat.config import ConfigError
    with pytest.raises(ConfigError):
        Config(beam_width=0).validate()
    with pytest.raises(ConfigError):
        Config(beam_length_penalty=-1).validate()


def test_beam_expands_even_at_temperature_zero():
    """Temperature is a sampling control; it must not collapse the candidates."""
    client = FakeClient(TABLE)
    _, done = run(client, cfg(beam_width=2, max_steps=3, temperature=0.0, top_p=0.9))
    assert done.text == "bx"                 # would be "a" if expansion collapsed
    assert client.seen.count("b") == 1       # the second beam really was scored


def test_top_p_does_not_prune_the_beam():
    client = FakeClient(TABLE)
    run(client, cfg(beam_width=2, max_steps=2, top_p=0.5))
    assert "b" in client.seen                # 'b' holds 0.4, outside a 0.5 nucleus
