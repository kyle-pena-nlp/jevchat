import random

import pytest

from jevchat.sampler import (
    SamplerError,
    apply_temperature,
    normalize,
    sample,
    top_k_filter,
    top_p_filter,
)


def test_normalize_rescales_and_drops_nonpositive():
    out = normalize({"a": 2.0, "b": 2.0, "c": 0.0, "d": -1.0})
    assert out == {"a": 0.5, "b": 0.5}
    assert sum(out.values()) == pytest.approx(1.0)


def test_normalize_falls_back_to_uniform_when_all_zero():
    out = normalize({"a": 0.0, "b": 0.0})
    assert out == {"a": 0.5, "b": 0.5}


def test_normalize_rejects_empty():
    with pytest.raises(SamplerError):
        normalize({})


def test_temperature_zero_is_greedy():
    assert apply_temperature({"a": 0.4, "b": 0.6}, 0.0) == {"b": 1.0}


def test_low_temperature_sharpens_high_temperature_flattens():
    probs = {"a": 0.7, "b": 0.3}
    assert apply_temperature(probs, 0.5)["a"] > 0.7
    assert apply_temperature(probs, 2.0)["a"] < 0.7
    assert apply_temperature(probs, 1.0) == probs


def test_top_k_keeps_the_best_and_renormalises():
    out = top_k_filter({"a": 0.5, "b": 0.3, "c": 0.2}, 2)
    assert set(out) == {"a", "b"}
    assert sum(out.values()) == pytest.approx(1.0)


def test_top_p_keeps_the_smallest_set_reaching_the_mass():
    out = top_p_filter({"a": 0.6, "b": 0.3, "c": 0.1}, 0.85)
    assert set(out) == {"a", "b"}
    assert sum(out.values()) == pytest.approx(1.0)


def test_sample_respects_banned_keys():
    draw = sample({"STOP": 0.99, "a": 0.01}, banned={"STOP"}, rng=random.Random(0))
    assert draw.key == "a"
    assert "STOP" not in draw.distribution
    # raw keeps what Jev actually said, before any shaping.
    assert draw.raw["STOP"] == pytest.approx(0.99)


def test_sample_survives_banning_everything():
    draw = sample({"STOP": 1.0}, banned={"STOP"}, rng=random.Random(0))
    assert draw.key == "STOP"


def test_bias_shifts_mass():
    probs = {"STOP": 0.5, "a": 0.5}
    low = sample(probs, bias={"STOP": 0.0}, rng=random.Random(0))
    assert low.key == "a"


def test_sampling_is_seeded_and_matches_the_distribution():
    probs = {"a": 0.25, "b": 0.75}
    rng = random.Random(1234)
    counts = {"a": 0, "b": 0}
    for _ in range(4000):
        counts[sample(probs, rng=rng).key] += 1
    assert counts["b"] / 4000 == pytest.approx(0.75, abs=0.03)

    first = [sample(probs, rng=random.Random(7)).key for _ in range(5)]
    second = [sample(probs, rng=random.Random(7)).key for _ in range(5)]
    assert first == second
