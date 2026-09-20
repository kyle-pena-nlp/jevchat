"""Refinement sampling: bucket, probe, project back down, rescore, repeat.

`buckets` asks each slice of the alphabet how much of the mass is *not* in it
(OTHER) and stitches the answers together. That number is asserted rather than
measured, and it came back miscalibrated. This strategy measures it instead, in
three passes:

1. **down** — one plain question per bucket (no OTHER) gives `p(x | bucket)`.
2. **up** — one question over the buckets' winners puts those winners on a single
   normalised scale. Each winner is a *probe*: `w_k = r_k / p(rep_k | k)` recovers
   its bucket's total, exactly under Luce, without assuming the winner dominates.
3. **down, then up again** — with a first estimate in hand, take the top few of
   every bucket, pool them into one question, and re-estimate every bucket's
   weight from that pool:

       w_k  =  sum q(x) / sum p(x | k)      over x in bucket k's nucleus

   which is exact for the same reason and averages over several members rather
   than dividing by one quantised number. The corrected weight is applied to the
   whole bucket, so the tail moves with the head.

Step 3 can repeat; each extra round costs one request.
"""

from __future__ import annotations

import random

from . import present
from .alphabet import MAX_CHOICES, Alphabet
from .buckets import STOP_QUESTION, Plan, send
from .client import ChoiceAnswer, JevClient, JevError


# Jev rounds probabilities to 0.01, so a reported 0.00 means "somewhere below
# 0.005", not "impossible". Taking it literally sets a bucket's weight to zero and
# silently deletes every symbol in it. Using the midpoint of the censored interval
# keeps those buckets alive at a plausible magnitude.
CENSORED = 0.0025


def nucleus_size(plan: Plan, requested: int) -> int:
    """Largest nucleus per bucket whose pool still fits in one question."""
    return max(1, min(requested, (MAX_CHOICES - 1) // max(1, len(plan.buckets))))


def _stop_question(alphabet: Alphabet) -> dict:
    return {
        "type": "noul",
        "instructions": "Is the reply already finished?",
        "criteria": {
            "true": alphabet.stop_description,
            "false": "The reply is not finished: at least one more symbol follows.",
        },
    }


def _score(
    client: JevClient,
    state: dict | str,
    alphabet: Alphabet,
    config,
    symbols,
    text: str,
    rng: random.Random | None,
    name: str,
    cancel,
) -> tuple[dict[str, float], float]:
    """One choice question over `symbols`, folded back to alphabet keys."""
    options = present.build(
        alphabet, symbols,
        presentation=config.presentation,
        text=text, size=config.window,
        describe=config.bucket_describe, include_stop=False, rng=rng,
    )
    answers, latency = client.ask(
        state,
        {name: {"type": "choice", "instructions": config.active_instructions,
                "criteria": options.criteria}},
        cancel=cancel,
        overflow_hint=f"{len(options.criteria)} options in one question.",
    )
    return options.fold(answers[name]["probabilities"]), latency


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
    by_key = {s.key: s for s in alphabet.symbols}
    n_buckets = len(plan.buckets)
    if n_buckets > MAX_CHOICES - 1:
        raise JevError(
            f"refine needs one question over all {n_buckets} bucket winners, more "
            f"than the {MAX_CHOICES} a question allows. Raise `bucket_size`."
        )

    # --- pass 1: every bucket on its own terms, plus the STOP question ----------
    questions: dict[str, dict] = {STOP_QUESTION: _stop_question(alphabet)}
    options: dict[str, present.Options] = {}
    for index, bucket in enumerate(plan.buckets):
        options[f"b{index}"] = present.build(
            alphabet, bucket,
            presentation=config.presentation,
            text=text, size=config.window,
            describe=config.bucket_describe, include_stop=False, rng=rng,
        )
        questions[f"b{index}"] = {
            "type": "choice",
            "instructions": config.active_instructions,
            "criteria": options[f"b{index}"].criteria,
        }
    answers, latency = send(client, state, questions, plan.batch, cancel=cancel)

    try:
        p_stop = min(1.0, max(0.0, float(answers[STOP_QUESTION]["noul"])))
        conditional = [
            options[f"b{i}"].fold(answers[f"b{i}"]["probabilities"])
            for i in range(n_buckets)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"unexpected Jev response shape: {answers!r}") from exc

    # A single bucket is already one question over the whole alphabet: there is
    # nothing to weigh it against, and a one-option question is not well formed.
    if n_buckets < 2:
        probabilities = _combine(plan, conditional, [1.0])
        total = sum(probabilities.values())
        scale = (1.0 - p_stop) / total if total > 0 else 0.0
        probabilities = {k: v * scale for k, v in probabilities.items()}
        probabilities[alphabet.stop_key] = p_stop
        return ChoiceAnswer(
            choice=max(probabilities, key=probabilities.__getitem__),
            probabilities=probabilities, confidence=max(probabilities.values()),
            model=client.model, latency=latency, variants=len(questions),
        )

    # --- pass 2: the winners, measured against each other on one scale ---------
    winners = [max(c, key=c.get) if c else None for c in conditional]
    live = [i for i, w in enumerate(winners) if w is not None]
    probe, took = _score(
        client, state, alphabet, config, [by_key[winners[i]] for i in live],
        text, rng, "reps", cancel,
    )
    latency += took

    weights = [0.0] * n_buckets
    for i in live:
        share = conditional[i].get(winners[i], 0.0)
        seen = probe.get(winners[i], 0.0) or CENSORED
        weights[i] = seen / share if share > 0 else 0.0

    questions_sent = len(questions) + 1

    # --- pass 3: pool each bucket's nucleus, re-estimate every weight ----------
    size = nucleus_size(plan, config.refine_nucleus)
    for round_ in range(max(0, config.refine_rounds)):
        estimate = _combine(plan, conditional, weights)
        pool: list[str] = []
        for i, bucket in enumerate(plan.buckets):
            ranked = sorted((s.key for s in bucket), key=lambda k: -estimate.get(k, 0.0))
            pool.extend(ranked[:size])
        if len(pool) < 2:
            break

        pooled, took = _score(
            client, state, alphabet, config, [by_key[k] for k in pool],
            text, rng, f"pool{round_}", cancel,
        )
        latency += took
        questions_sent += 1

        chosen = set(pool)
        for i, bucket in enumerate(plan.buckets):
            members = [s.key for s in bucket if s.key in chosen]
            numerator = sum(pooled.get(k, 0.0) for k in members) or CENSORED
            denominator = sum(conditional[i].get(k, 0.0) for k in members)
            if denominator > 0:
                weights[i] = numerator / denominator

    probabilities = _combine(plan, conditional, weights)
    total = sum(probabilities.values())
    scale = (1.0 - p_stop) / total if total > 0 else 0.0
    probabilities = {k: v * scale for k, v in probabilities.items()}
    probabilities[alphabet.stop_key] = p_stop

    return ChoiceAnswer(
        choice=max(probabilities, key=probabilities.__getitem__),
        probabilities=probabilities,
        confidence=max(probabilities.values(), default=0.0),
        model=client.model,
        latency=latency,
        variants=questions_sent,
    )


def _combine(plan: Plan, conditional, weights) -> dict[str, float]:
    """p(x) = w_k * p(x | k), normalised over the whole alphabet."""
    total = sum(weights) or 1.0
    out: dict[str, float] = {}
    for i in range(len(plan.buckets)):
        share = weights[i] / total
        for key, value in conditional[i].items():
            out[key] = out.get(key, 0.0) + share * value
    return out
