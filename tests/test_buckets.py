import pytest

from jevchat import buckets as B
from jevchat.alphabet import Alphabet, Symbol
from jevchat.client import JevError
from jevchat.config import Config


def cfg(**kwargs):
    kwargs.setdefault("presentation", "symbol")
    return Config(**kwargs)

ALPHA = Alphabet(
    name="ten", description="",
    symbols=tuple(Symbol(c, c, f"letter {c}") for c in "abcdefghij"),
)


def test_buckets_cover_every_symbol_once():
    plan = B.build(ALPHA, bucket_size=3, batch=10)
    seen = [s.key for bucket in plan.buckets for s in bucket]
    assert seen == list("abcdefghij")
    assert [len(b) for b in plan.buckets] == [3, 3, 3, 1]


def test_questions_per_step_counts_the_stop_question():
    plan = B.build(ALPHA, bucket_size=5, batch=10)
    assert len(plan.buckets) == 2
    assert plan.questions_per_step == 3
    assert plan.requests_per_step == 1


def test_requests_are_split_when_the_batch_is_small():
    plan = B.build(ALPHA, bucket_size=1, batch=4)   # 10 buckets + stop = 11 questions
    assert plan.requests_per_step == 3


def test_bucket_size_must_leave_room_for_other():
    with pytest.raises(ValueError, match="no room for OTHER"):
        B.build(ALPHA, bucket_size=B.MAX_BUCKET + 1, batch=10)


@pytest.mark.parametrize("kwargs", [{"bucket_size": 0}, {"batch": 0}])
def test_invalid_plans_are_rejected(kwargs):
    args = {"bucket_size": 3, "batch": 3, **kwargs}
    with pytest.raises(ValueError):
        B.build(ALPHA, **args)


def test_every_bucket_carries_its_own_other():
    plan = B.build(ALPHA, bucket_size=4, batch=10)
    questions, _ = B.build_questions(plan, ALPHA, cfg())
    buckets = [q for name, q in questions.items() if name != B.STOP_QUESTION]
    assert len(buckets) == 3
    for question in buckets:
        assert B.OTHER_KEY in question["criteria"]
        assert len(question["criteria"]) <= 4 + 1
    assert questions[B.STOP_QUESTION]["type"] == "noul"


def test_descriptions_are_omitted_by_default():
    plan = B.build(ALPHA, bucket_size=10, batch=10)
    plain = B.build_questions(plan, ALPHA, cfg())[0]["b0"]["criteria"]
    described = B.build_questions(plan, ALPHA, cfg(bucket_describe=True))[0]["b0"]["criteria"]
    assert all(v is None for k, v in plain.items() if k != B.OTHER_KEY)
    assert all(v for k, v in described.items() if k != B.OTHER_KEY)


def answers_for(plan, *, stop, weights):
    """weights[i] maps a bucket's keys to raw probabilities (OTHER included)."""
    out = {B.STOP_QUESTION: {"noul": stop}}
    for i, bucket in enumerate(plan.buckets):
        out[f"b{i}"] = {"probabilities": weights(i, bucket)}
    return out


def options_for(plan, **kwargs):
    return B.build_questions(plan, ALPHA, cfg(**kwargs))[1]


def test_distribution_drops_other_and_normalises_around_stop():
    plan = B.build(ALPHA, bucket_size=5, batch=10)
    answers = answers_for(
        plan, stop=0.25,
        weights=lambda i, bucket: {**{s.key: 0.1 for s in bucket}, B.OTHER_KEY: 0.5},
    )
    dist = B.distribution(plan, ALPHA, answers, options_for(plan))

    assert B.OTHER_KEY not in dist
    assert dist[ALPHA.stop_key] == pytest.approx(0.25)
    assert sum(dist.values()) == pytest.approx(1.0)
    # Ten equally weighted symbols share what STOP leaves behind.
    assert dist["a"] == pytest.approx(0.75 / 10)


def test_a_confident_bucket_outweighs_a_vague_one():
    plan = B.build(ALPHA, bucket_size=5, batch=10)

    def weights(i, bucket):
        if i == 0:                       # this bucket says the answer is here
            return {**{s.key: 0.02 for s in bucket}, "a": 0.9, B.OTHER_KEY: 0.02}
        return {**{s.key: 0.02 for s in bucket}, B.OTHER_KEY: 0.9}

    dist = B.distribution(plan, ALPHA, answers_for(plan, stop=0.0, weights=weights),
                          options_for(plan))
    assert max(dist, key=dist.__getitem__) == "a"
    assert dist["a"] > dist["f"] * 10


def test_stop_of_one_leaves_no_mass_for_symbols():
    plan = B.build(ALPHA, bucket_size=5, batch=10)
    dist = B.distribution(plan, ALPHA, answers_for(
        plan, stop=1.0,
        weights=lambda i, b: {**{s.key: 0.2 for s in b}, B.OTHER_KEY: 0.0}),
        options_for(plan))
    assert dist[ALPHA.stop_key] == pytest.approx(1.0)
    assert sum(v for k, v in dist.items() if k != ALPHA.stop_key) == pytest.approx(0.0)


def test_a_missing_bucket_answer_is_reported():
    plan = B.build(ALPHA, bucket_size=5, batch=10)
    with pytest.raises(JevError, match="unexpected Jev response shape"):
        B.distribution(plan, ALPHA, {B.STOP_QUESTION: {"noul": 0.1}}, options_for(plan))


class RecordingClient:
    model = "fake"

    def __init__(self, overflow_above=None):
        self.batches = []
        self.overflow_above = overflow_above

    def ask(self, state, questions, *, cancel=None, overflow_hint=""):
        if self.overflow_above is not None and len(questions) > self.overflow_above:
            raise JevError("Jev's context window overflowed with " + overflow_hint)
        self.batches.append(len(questions))
        answers = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                answers[name] = {"noul": 0.0}
            else:
                answers[name] = {"probabilities": {k: 1.0 / len(q["criteria"])
                                                   for k in q["criteria"]}}
        return answers, 0.05


def test_questions_are_sent_in_batches():
    plan = B.build(ALPHA, bucket_size=1, batch=4)
    client = RecordingClient()
    answer = B.ask(client, plan, ALPHA, {}, cfg())
    assert client.batches == [4, 4, 3]        # 11 questions
    assert answer.variants == 11
    assert answer.latency == pytest.approx(0.15)


def test_an_overflow_backs_off_to_smaller_requests():
    plan = B.build(ALPHA, bucket_size=1, batch=8)
    client = RecordingClient(overflow_above=3)
    B.ask(client, plan, ALPHA, {}, cfg())
    assert max(client.batches) <= 3
    assert sum(client.batches) == 11


def test_buckets_never_offer_stop_inside_a_bucket():
    # STOP is asked once, separately; offering it per bucket would count it many times.
    plan = B.build(ALPHA, bucket_size=4, batch=10)
    questions, _ = B.build_questions(plan, ALPHA, cfg(presentation="hypothesis"),
                                     text="Par")
    for name, q in questions.items():
        if name != B.STOP_QUESTION:
            assert "Par" not in q["criteria"]          # the unchanged reply
            assert all(k.startswith("Par") or k == B.OTHER_KEY for k in q["criteria"])


def test_hypothesis_buckets_fold_back_to_keys():
    plan = B.build(ALPHA, bucket_size=10, batch=10)
    questions, options = B.build_questions(
        plan, ALPHA, cfg(presentation="hypothesis"), text="Par")
    labels = [k for k in questions["b0"]["criteria"] if k != B.OTHER_KEY]
    answers = {B.STOP_QUESTION: {"noul": 0.0},
               "b0": {"probabilities": {**{l: 0.0 for l in labels},
                                        labels[0]: 1.0, B.OTHER_KEY: 0.0}}}
    dist = B.distribution(plan, ALPHA, answers, options)
    winner = max(dist, key=dist.__getitem__)
    assert winner in {s.key for s in ALPHA.symbols}     # a key, not a label
    assert options["b0"].key_for[labels[0]] == winner


def test_an_overflow_that_cannot_shrink_is_raised():
    plan = B.build(ALPHA, bucket_size=1, batch=1)
    client = RecordingClient(overflow_above=0)
    with pytest.raises(JevError, match="overflowed"):
        B.ask(client, plan, ALPHA, {}, cfg())
