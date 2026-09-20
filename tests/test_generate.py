import random
import threading

import pytest

from jevchat.alphabet import Alphabet, AlphabetError, Symbol
from jevchat.client import Cancelled, ChoiceAnswer, Usage
from jevchat.config import Config
from jevchat.generate import Done, Scorer, Step, Turn, build_state, generate

def cfg(**kwargs):
    """Config for the loop tests.

    These exercise the sampling loop, not the defaults, so they pin the simplest
    strategy and presentation unless a test overrides them.
    """
    kwargs.setdefault("presentation", "symbol")
    kwargs.setdefault("strategy", "choice")
    return Config(**kwargs)


ALPHA = Alphabet(
    name="test",
    description="",
    symbols=(Symbol("a", "a"), Symbol("b", "b"), Symbol("SPACE", " ")),
)


class FakeClient:
    """Replays a scripted list of probability maps."""

    def __init__(self, script, *, cancel_after=None):
        self.script = list(script)
        self.calls = []
        self.cancel_after = cancel_after
        self.usage = Usage()
        self.criteria_orders = []
        self.variant_counts = []

    def choice(self, state, instructions, criteria_variants, *, cancel=None):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        self.calls.append(dict(state))
        self.criteria_orders.append(list(criteria_variants[0]))
        self.variant_counts.append(len(criteria_variants))
        probs = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if self.cancel_after is not None and len(self.calls) >= self.cancel_after:
            assert cancel is not None
            cancel.set()
        self.usage.add({"input_tokens": 100, "output_tokens": 20},
                       questions=len(criteria_variants))
        return ChoiceAnswer(
            choice=max(probs, key=probs.get),
            probabilities=probs,
            confidence=1.0,
            model="fake",
            latency=0.01,
            variants=len(criteria_variants),
        )


def run(client, config, question="hi", history=None):
    steps, done = [], None
    cancel = threading.Event()
    for event in generate(client, ALPHA, config, question, history=history,
                          cancel=cancel, rng=random.Random(0)):
        if isinstance(event, Step):
            steps.append(event)
        else:
            done = event
    return steps, done


def test_greedy_generation_stops_on_stop():
    script = [{"a": 1.0}, {"b": 1.0}, {"STOP": 1.0}]
    config = cfg(temperature=0.0, min_steps=0)
    steps, done = run(FakeClient(script), config)
    assert done.text == "ab"
    assert done.reason == "stop"
    assert len(steps) == 3
    assert steps[-1].emit is None


def test_min_steps_blocks_an_immediate_stop():
    client = FakeClient([{"STOP": 0.99, "a": 0.01}])
    config = cfg(temperature=0.0, min_steps=3, max_steps=10)
    steps, done = run(client, config)
    # STOP is banned for the first three draws, then taken.
    assert done.text == "aaa"
    assert done.reason == "stop"
    assert [s.key for s in steps] == ["a", "a", "a", "STOP"]


def test_max_steps_caps_generation():
    config = cfg(temperature=0.0, min_steps=0, max_steps=5)
    steps, done = run(FakeClient([{"a": 1.0}]), config)
    assert done.text == "aaaaa"
    assert done.reason == "max_steps"
    assert done.stats.steps == 5


def test_max_chars_caps_generation():
    config = cfg(temperature=0.0, min_steps=0, max_steps=100, max_chars=4)
    _, done = run(FakeClient([{"a": 1.0}]), config)
    assert done.text == "aaaa"
    assert done.reason == "max_chars"


def test_cancellation_keeps_the_partial_reply():
    client = FakeClient([{"a": 1.0}], cancel_after=3)
    config = cfg(temperature=0.0, min_steps=0, max_steps=100)
    _, done = run(client, config)
    assert done.reason == "cancelled"
    assert done.text == "aaa"
    assert len(client.calls) == 3


def test_stats_are_reported():
    config = cfg(temperature=0.0, min_steps=0, max_steps=3)
    steps, done = run(FakeClient([{"a": 1.0}]), config)
    assert done.stats.steps == 3
    assert done.stats.chars == 3
    assert done.stats.elapsed > 0
    assert done.stats.symbols_per_sec > 0
    assert done.stats.chars_per_sec > 0
    assert steps[-1].stats.mean_latency == pytest.approx(0.01, abs=1e-6)
    assert done.stats.input_tokens == 300
    assert done.stats.output_tokens == 60


def test_state_carries_question_history_and_text_so_far():
    client = FakeClient([{"a": 1.0}, {"STOP": 1.0}])
    config = cfg(temperature=0.0, min_steps=0)
    history = [Turn("user", "earlier q"), Turn("assistant", "earlier a")]
    run(client, config, question="what now?", history=history)

    first, second = client.calls
    assert first["question"] == "what now?"
    assert first["answer_so_far"] == ""
    assert second["answer_so_far"] == "a"
    assert first["conversation"] == [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
    ]


def test_history_is_trimmed_to_history_turns():
    config = cfg(history_turns=2)
    history = [Turn("user", f"q{i}") for i in range(10)]
    state = build_state(config, "now", "so far", history)
    assert [t["content"] for t in state["conversation"]] == ["q8", "q9"]


def test_state_omits_conversation_when_there_is_none():
    assert "conversation" not in build_state(cfg(), "q", "")


def test_no_repeat_space_blocks_a_second_space():
    # SPACE is the runaway favourite, but must not be drawn twice in a row.
    client = FakeClient([{"SPACE": 0.9, "a": 0.1}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=4, no_repeat_space=True)
    steps, done = run(client, config)
    assert done.text == " a a"
    assert [s.key for s in steps] == ["SPACE", "a", "SPACE", "a"]


def test_no_repeat_space_can_be_turned_off():
    client = FakeClient([{"SPACE": 0.9, "a": 0.1}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=3, no_repeat_space=False)
    _, done = run(client, config)
    assert done.text == "   "


def test_repetition_penalty_breaks_a_loop():
    # 'a' is preferred, but each repeat divides its weight by the penalty.
    client = FakeClient([{"a": 0.6, "b": 0.4}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=4,
                    repetition_penalty=3.0, repetition_window=4)
    _, done = run(client, config)
    assert done.text == "abab"


def test_repetition_penalty_off_by_configuration():
    client = FakeClient([{"a": 0.6, "b": 0.4}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=3, repetition_penalty=1.0)
    _, done = run(client, config)
    assert done.text == "aaa"


def test_criteria_order_is_shuffled_between_calls():
    client = FakeClient([{"a": 1.0}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=6, seed=3,
                    shuffle_criteria=True)
    run(client, config)
    orders = client.criteria_orders
    assert all(set(o) == set(ALPHA.criteria()) for o in orders)
    assert len({tuple(o) for o in orders}) > 1


def test_criteria_order_is_stable_when_shuffling_is_off():
    client = FakeClient([{"a": 1.0}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=4, shuffle_criteria=False)
    run(client, config)
    assert len({tuple(o) for o in client.criteria_orders}) == 1


def test_ensemble_sends_one_question_per_reordering():
    client = FakeClient([{"a": 1.0}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=3, ensemble=5)
    run(client, config)
    assert client.variant_counts == [5, 5, 5]
    # Five questions per step, still one request per step.
    assert client.usage.calls == 3
    assert client.usage.questions == 15


def test_ensemble_collapses_to_one_without_shuffling():
    # Identical orderings would just be five copies of the same question.
    client = FakeClient([{"a": 1.0}])
    config = cfg(temperature=0.0, min_steps=0, max_steps=2, ensemble=5,
                    shuffle_criteria=False)
    run(client, config)
    assert client.variant_counts == [1, 1]


class FakeBisectClient:
    """Answers a whole bisection tree, always leaning to the later half."""

    model = "fake"

    def __init__(self, stop_after):
        self.usage = Usage()
        self.stop_after = stop_after
        self.requests = []

    def ask(self, state, questions, *, cancel=None, overflow_hint=""):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        self.requests.append(questions)
        done = len(self.requests) > self.stop_after
        answers = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                if name == "stop":
                    answers[name] = {"noul": 1.0 if done else 0.0}
                else:
                    # ':b' is the same split with the groups exchanged, so it must
                    # answer the complement for the swap-averaging to preserve 0.9.
                    answers[name] = {"noul": 0.1 if name.endswith(":b") else 0.9}
            else:
                keys = list(q["criteria"])
                answers[name] = {"probabilities": {k: 1.0 / len(keys) for k in keys}}
        self.usage.add({"input_tokens": 50, "output_tokens": 10}, questions=len(questions))
        return answers, 0.02


def test_bisect_strategy_asks_one_request_per_symbol():
    client = FakeBisectClient(stop_after=3)
    config = cfg(temperature=0.0, min_steps=0, max_steps=10, strategy="bisect",
                    bisect_cutoff=2, seed=1)
    steps, done = run(client, config)

    assert done.reason == "stop"
    assert len(done.text) == 3                     # three symbols, then STOP
    assert len(client.requests) == 4               # one request per step
    assert client.usage.calls == 4
    # Every step sent the whole tree: splits (swapped) + leaf picks + stop.
    assert all(len(q) > 1 for q in client.requests)


def test_bisect_leans_to_the_later_half():
    # ALPHA sorted by emit is [SPACE, a, b]; p(later) = 0.9 at every split.
    client = FakeBisectClient(stop_after=2)
    config = cfg(temperature=0.0, min_steps=0, max_steps=5, strategy="bisect",
                    bisect_cutoff=2, seed=1)
    _, done = run(client, config)
    assert set(done.text) <= {"a", "b"}


def test_bisect_cancellation_keeps_partial_text():
    client = FakeBisectClient(stop_after=99)
    config = cfg(temperature=0.0, min_steps=0, max_steps=6, strategy="bisect",
                    bisect_cutoff=2)
    cancel = threading.Event()
    steps = []
    for event in generate(client, ALPHA, config, "hi", cancel=cancel,
                          rng=random.Random(0)):
        if isinstance(event, Step):
            steps.append(event)
            if len(steps) == 2:
                cancel.set()
        else:
            done = event
    assert done.reason == "cancelled"
    assert len(done.text) == 2


def test_choice_refuses_an_alphabet_too_big_for_one_question():
    big = Alphabet(name="big", description="",
                   symbols=tuple(Symbol(f"k{i}", "x") for i in range(300)))
    with pytest.raises(AlphabetError, match="--strategy buckets"):
        Scorer.build(FakeClient([{"a": 1.0}]), big, cfg(strategy="choice"))


def test_buckets_accepts_an_alphabet_too_big_for_one_question():
    big = Alphabet(name="big", description="",
                   symbols=tuple(Symbol(f"k{i}", "x") for i in range(300)))
    scorer = Scorer.build(FakeClient([{"a": 1.0}]), big,
                          cfg(strategy="buckets", bucket_size=64))
    assert scorer.questions_per_step == 5 + 1     # ceil(300/64) buckets + stop
    assert scorer.requests_per_step == 1


def test_hypothesis_presentation_offers_whole_texts():
    """Options become the resulting texts, and STOP becomes the text unchanged."""
    client = FakeClient([{"a": 1.0}])
    config = cfg(presentation="hypothesis", window=40, temperature=0.0,
                 min_steps=0, max_steps=1)
    list(generate(client, ALPHA, config, "hi", rng=random.Random(0)))
    offered = set(client.criteria_orders[0])
    # text is empty at the first step, so each option is just the symbol's emit,
    # and the unchanged option is the empty reply.
    assert offered == {"", "a", "b", " "}


def test_hypothesis_probabilities_fold_back_to_alphabet_keys():
    # Jev answers in labels; the loop must translate them back to keys.
    # Step 1 offers 'a'(=stop), 'aa', 'ab', 'a '. Picking 'a ' means the SPACE symbol.
    client = FakeClient([{"a": 1.0}, {"a ": 1.0}])
    config = cfg(presentation="hypothesis", temperature=0.0, min_steps=0, max_steps=2)
    steps, done = run(client, config)
    assert [s.key for s in steps] == ["a", "SPACE"]   # 'a ' folded back to SPACE
    assert done.text == "a "


def test_hypothesis_options_grow_with_the_reply():
    client = FakeClient([{"a": 1.0}, {"aa": 1.0}])
    config = cfg(presentation="hypothesis", temperature=0.0, min_steps=0, max_steps=2)
    run(client, config)
    assert set(client.criteria_orders[0]) == {"", "a", "b", " "}
    assert set(client.criteria_orders[1]) == {"a", "aa", "ab", "a "}


def test_an_unrecognised_answer_is_reported():
    from jevchat.client import JevError
    client = FakeClient([{"not-an-option": 1.0}])
    config = cfg(presentation="hypothesis", temperature=0.0, min_steps=0, max_steps=1)
    with pytest.raises(JevError, match="none of Jev's answers matched"):
        run(client, config)


def test_hypothesis_window_shows_only_the_tail():
    from jevchat.present import ELLIPSIS, window
    assert window("abcdef", 3) == ELLIPSIS + "def"
    assert window("ab", 10) == "ab"


def test_seed_makes_a_run_reproducible():
    config = cfg(temperature=1.0, min_steps=0, max_steps=12, seed=99)
    script = [{"a": 0.5, "b": 0.3, "SPACE": 0.2}]
    first = run(FakeClient(script), config)[1].text
    second = run(FakeClient(script), config)[1].text
    assert first == second
