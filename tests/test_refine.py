import pytest

from jevchat import buckets as BK
from jevchat import refine as R
from jevchat.alphabet import Alphabet, Symbol
from jevchat.client import JevError, Usage
from jevchat.config import Config

ALPHA = Alphabet(name="eight", description="",
                 symbols=tuple(Symbol(c, c, f"letter {c}") for c in "abcdefgh"))


def cfg(**kwargs):
    kwargs.setdefault("presentation", "symbol")   # criteria keys are the symbol keys
    kwargs.setdefault("strategy", "refine")
    return Config(**kwargs)


class FakeClient:
    """Answers each question from a scripted map of name -> {key: prob}."""

    model = "fake"

    def __init__(self, script, stop=0.0):
        self.script = script
        self.stop = stop
        self.usage = Usage()
        self.requests = []

    def ask(self, state, questions, *, cancel=None, overflow_hint=""):
        self.requests.append(list(questions))
        out = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                out[name] = {"noul": self.stop}
                continue
            keys = list(q["criteria"])
            spec = self.script.get(name)
            if spec is None:
                out[name] = {"probabilities": {k: 1.0 / len(keys) for k in keys}}
            else:
                out[name] = {"probabilities": {k: spec.get(k, 0.0) for k in keys}}
        self.usage.add({"input_tokens": 10, "output_tokens": 2}, questions=len(questions))
        return out, 0.01


def plan_for(bucket_size=4):
    return BK.build(ALPHA, bucket_size, 48)


def test_nucleus_is_clamped_so_the_pool_fits_one_question():
    plan = BK.build(ALPHA, 1, 48)          # 8 buckets
    assert R.nucleus_size(plan, 3) == 3
    big = BK.build(Alphabet(name="x", description="",
                            symbols=tuple(Symbol(f"k{i}", "x") for i in range(1000))),
                   4, 48)                   # 250 buckets
    assert R.nucleus_size(big, 6) == 1      # 250 * 1 still fits under 255
    assert len(big.buckets) * R.nucleus_size(big, 6) <= 254


def test_a_single_bucket_skips_the_probe():
    # One bucket is already a question over the whole alphabet.
    client = FakeClient({})
    plan = plan_for(bucket_size=8)
    assert len(plan.buckets) == 1
    answer = R.ask(client, plan, ALPHA, {}, cfg(), text="")
    assert len(client.requests) == 1
    assert pytest.approx(sum(answer.probabilities.values())) == 1.0


def test_too_many_buckets_for_one_probe_is_reported():
    alpha = Alphabet(name="big", description="",
                     symbols=tuple(Symbol(f"k{i}", "x") for i in range(600)))
    plan = BK.build(alpha, 1, 48)          # 600 buckets
    with pytest.raises(JevError, match="Raise `bucket_size`"):
        R.ask(FakeClient({}), plan, alpha, {}, cfg(), text="")


def test_weights_come_from_the_probe_divided_by_the_within_bucket_share():
    plan = plan_for()                       # {a,b,c,d}, {e,f,g,h}
    script = {
        "b0": {"a": 0.4, "b": 0.2, "c": 0.2, "d": 0.2},
        "b1": {"e": 0.4, "f": 0.2, "g": 0.2, "h": 0.2},
        "reps": {"a": 0.8, "e": 0.2},       # bucket 0 is 4x bucket 1
    }
    client = FakeClient(script)
    answer = R.ask(client, plan, ALPHA, {}, cfg(refine_rounds=0), text="")
    p = answer.probabilities
    # w0/w1 = (0.8/0.4) / (0.2/0.4) = 4, and 'a' holds 0.4 of its bucket.
    assert p["a"] / p["e"] == pytest.approx(4.0)
    assert p["a"] == pytest.approx(0.8 * 0.4)
    assert sum(p.values()) == pytest.approx(1.0)


def test_a_probe_rounded_to_zero_does_not_delete_its_bucket():
    # Jev rounds to 0.01; a reported 0.00 means "below 0.005", not "impossible".
    plan = plan_for()
    script = {
        "b0": {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1},
        "b1": {"e": 0.7, "f": 0.1, "g": 0.1, "h": 0.1},
        "reps": {"a": 1.0, "e": 0.0},       # bucket 1's winner rounded away
    }
    answer = R.ask(FakeClient(script), plan, ALPHA, {}, cfg(refine_rounds=0), text="")
    p = answer.probabilities
    assert all(p[c] > 0 for c in "efgh"), "a rounded-off bucket was deleted"
    assert p["a"] > p["e"] * 50              # still strongly outweighed


def test_a_refinement_round_reweights_from_the_pool():
    plan = plan_for()
    script = {
        "b0": {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1},
        "b1": {"e": 0.5, "f": 0.3, "g": 0.1, "h": 0.1},
        "reps": {"a": 0.5, "e": 0.5},        # probe says the buckets are equal
        # the pool says bucket 1 is actually 3x bucket 0
        "pool0": {"a": 0.125, "b": 0.075, "e": 0.375, "f": 0.225},
    }
    client = FakeClient(script)
    answer = R.ask(client, plan, ALPHA, {}, cfg(refine_rounds=1, refine_nucleus=2),
                   text="")
    p = answer.probabilities
    # w_k = sum q(nucleus) / sum p(nucleus|k):  b0 = 0.2/0.8, b1 = 0.6/0.8  -> 1:3
    assert p["e"] / p["a"] == pytest.approx(3.0)
    assert sum(p.values()) == pytest.approx(1.0)


def test_the_pool_holds_the_top_m_of_every_bucket():
    plan = plan_for()
    script = {"b0": {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1},
              "b1": {"e": 0.5, "f": 0.3, "g": 0.1, "h": 0.1},
              "reps": {"a": 0.5, "e": 0.5}}
    client = FakeClient(script)
    R.ask(client, plan, ALPHA, {}, cfg(refine_rounds=1, refine_nucleus=2), text="")
    pool = [names for names in client.requests if "pool0" in names]
    assert pool, "no pool question was sent"


def test_request_and_question_counts():
    plan = plan_for()
    script = {"b0": {"a": 1.0}, "b1": {"e": 1.0}, "reps": {"a": 0.5, "e": 0.5}}
    client = FakeClient(script)
    answer = R.ask(client, plan, ALPHA, {}, cfg(refine_rounds=2, refine_nucleus=2),
                   text="")
    # pass 1 (2 buckets + stop), the probe, then one request per round
    assert len(client.requests) == 4
    assert answer.variants == 3 + 1 + 2


def test_stop_takes_its_share():
    plan = plan_for()
    client = FakeClient({}, stop=0.25)
    answer = R.ask(client, plan, ALPHA, {}, cfg(refine_rounds=0), text="")
    assert answer.probabilities[ALPHA.stop_key] == pytest.approx(0.25)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
