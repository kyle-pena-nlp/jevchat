"""Turn Jev's probability map over the alphabet into one sampled symbol."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Probabilities below this are treated as zero after filtering.
EPS = 1e-12


class SamplerError(Exception):
    """The distribution could not be sampled."""


@dataclass(frozen=True)
class Draw:
    key: str
    probability: float          # probability of the drawn key, after shaping
    distribution: dict[str, float]  # the shaped distribution it was drawn from
    raw: dict[str, float]           # normalised, unshaped, as Jev returned it


def normalize(probs: dict[str, float]) -> dict[str, float]:
    """Clamp negatives, drop non-finite values, and rescale to sum to 1."""
    clean = {k: float(v) for k, v in probs.items() if math.isfinite(float(v)) and float(v) > 0}
    total = sum(clean.values())
    if total <= 0:
        # Jev gave us nothing to go on: fall back to uniform over the options.
        if not probs:
            raise SamplerError("empty probability distribution")
        share = 1.0 / len(probs)
        return {k: share for k in probs}
    return {k: v / total for k, v in clean.items()}


def apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """p' proportional to p**(1/T). T=0 is greedy, T=1 leaves p unchanged."""
    if temperature == 1.0:
        return dict(probs)
    if temperature <= 0:
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        return {best: 1.0}
    scaled = {k: v ** (1.0 / temperature) for k, v in probs.items()}
    return normalize(scaled)


def top_k_filter(probs: dict[str, float], k: int) -> dict[str, float]:
    if k <= 0 or k >= len(probs):
        return dict(probs)
    kept = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return normalize(dict(kept))


def top_p_filter(probs: dict[str, float], p: float) -> dict[str, float]:
    """Nucleus filter: keep the smallest set whose mass reaches ``p``."""
    if p >= 1.0:
        return dict(probs)
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    kept: dict[str, float] = {}
    total = 0.0
    for key, value in ordered:
        kept[key] = value
        total += value
        if total >= p:
            break
    return normalize(kept)


def shape(
    probs: dict[str, float],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    banned: set[str] | None = None,
    bias: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply bias, bans and the filters. Returns (shaped, normalised raw).

    Order: normalise -> multiplicative bias -> drop banned keys -> temperature ->
    top-k -> top-p.
    """
    if not probs:
        raise SamplerError("empty probability distribution")
    raw = normalize(probs)

    shaped = dict(raw)
    for key, factor in (bias or {}).items():
        if key in shaped:
            shaped[key] *= max(0.0, factor)
    for key in banned or ():
        shaped.pop(key, None)
    if not shaped or sum(shaped.values()) <= EPS:
        # Everything was banned or zeroed: fall back to the unshaped distribution
        # minus the banned keys, then to the unshaped distribution itself.
        shaped = {k: v for k, v in raw.items() if k not in (banned or ())} or dict(raw)

    shaped = normalize(shaped)
    shaped = apply_temperature(shaped, temperature)
    shaped = top_k_filter(shaped, top_k)
    shaped = top_p_filter(shaped, top_p)
    return shaped, raw


def sample(
    probs: dict[str, float],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    banned: set[str] | None = None,
    bias: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> Draw:
    """Sample one key from Jev's scores over the alphabet."""
    shaped, raw = shape(probs, temperature=temperature, top_p=top_p, top_k=top_k,
                        banned=banned, bias=bias)
    rng = rng or random.Random()
    target = rng.random()
    cumulative = 0.0
    key = next(iter(shaped))
    for key, value in shaped.items():
        cumulative += value
        if target <= cumulative:
            break
    return Draw(key=key, probability=shaped[key], distribution=shaped, raw=raw)
