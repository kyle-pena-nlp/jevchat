import json
import threading

import httpx
import pytest

from jevchat.client import Cancelled, JevClient, JevError


def make_client(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    return JevClient("key", client=httpx.Client(transport=transport), **kwargs)


def answers(*distributions):
    """A Jev-shaped response with one answer per question."""
    return {
        "model": "jev-1.13.0",
        "answers": {
            f"next{i}": {"type": "choice", "choice": max(d, key=d.get),
                         "confidence": 0.5, "probabilities": d}
            for i, d in enumerate(distributions)
        },
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


def test_a_single_question_round_trips():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=answers({"a": 0.7, "b": 0.3}))

    client = make_client(handler)
    result = client.choice({"question": "q"}, "pick one", [{"a": None, "b": None}])

    assert result.probabilities == {"a": 0.7, "b": 0.3}
    assert result.choice == "a"
    assert result.variants == 1
    assert seen["model"] == "jev-latest"
    assert list(seen["questions"]) == ["next0"]
    assert seen["questions"]["next0"]["type"] == "choice"
    assert client.usage.input_tokens == 10


def test_ensemble_sends_parallel_questions_and_averages_them():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        # First ordering favours 'a', second favours 'b'.
        return httpx.Response(200, json=answers({"a": 0.8, "b": 0.2},
                                                {"a": 0.2, "b": 0.8},
                                                {"a": 0.5, "b": 0.5}))

    client = make_client(handler)
    orderings = [{"a": None, "b": None}, {"b": None, "a": None}, {"a": None, "b": None}]
    result = client.choice({"question": "q"}, "pick one", orderings)

    assert len(seen["questions"]) == 3          # one request, three questions
    assert result.probabilities == pytest.approx({"a": 0.5, "b": 0.5})
    assert result.variants == 3
    assert client.usage.calls == 1
    assert client.usage.questions == 3


def test_empty_variant_list_is_rejected():
    client = make_client(lambda r: httpx.Response(200, json=answers({"a": 1.0})))
    with pytest.raises(ValueError):
        client.choice({}, "pick", [])


def test_bad_key_is_reported_clearly():
    client = make_client(lambda r: httpx.Response(401, json={"detail": "nope"}))
    with pytest.raises(JevError, match="API key"):
        client.choice({}, "pick", [{"a": None}])


def test_validation_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"detail": "Too many choices."})

    client = make_client(handler, max_retries=3)
    with pytest.raises(JevError, match="Too many choices"):
        client.choice({}, "pick", [{"a": None}])
    assert len(calls) == 1


def test_rate_limits_are_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json=answers({"a": 1.0}))

    client = make_client(handler, max_retries=5)
    result = client.choice({}, "pick", [{"a": None}])
    assert result.choice == "a"
    assert len(calls) == 3


def test_retries_give_up_and_raise():
    client = make_client(lambda r: httpx.Response(529, json={"detail": "overloaded"}),
                         max_retries=2)
    with pytest.raises(JevError, match="529"):
        client.choice({}, "pick", [{"a": None}])


def test_a_set_cancel_event_stops_before_the_request():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=answers({"a": 1.0}))

    client = make_client(handler)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        client.choice({}, "pick", [{"a": None}], cancel=cancel)
    assert not calls


def test_context_overflow_names_the_dial_to_turn():
    client = make_client(
        lambda r: httpx.Response(400, json={"error_type": "max_tokens_exceeded"}))
    with pytest.raises(JevError, match="Lower `ensemble`"):
        client.choice({}, "pick", [{"a": None}, {"a": None}, {"a": None}])


def test_a_malformed_body_is_reported():
    client = make_client(lambda r: httpx.Response(200, json={"answers": {}}))
    with pytest.raises(JevError, match="unexpected Jev response shape"):
        client.choice({}, "pick", [{"a": None}])
