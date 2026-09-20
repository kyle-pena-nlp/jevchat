"""Thin HTTP client for the Jev System One endpoint."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504, 529}


class JevError(Exception):
    """A Jev API call failed."""


class Cancelled(Exception):
    """Generation was cancelled by the caller."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    questions: int = 0

    def add(self, payload: dict, questions: int = 1) -> None:
        self.calls += 1
        self.questions += questions
        self.input_tokens += int(payload.get("input_tokens", 0))
        self.output_tokens += int(payload.get("output_tokens", 0))


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    model: str
    latency: float
    variants: int = 1


class JevClient:
    """Calls ``POST {api_base}/systemone`` with one choice question at a time."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-latest",
        api_base: str = "https://api.typesafe.ai/v1",
        timeout: float = 60.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.max_retries = max_retries
        self.usage = Usage()
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def choice(
        self,
        state: dict | str,
        instructions: str,
        criteria_variants: Sequence[dict[str, str | None]],
        *,
        cancel: threading.Event | None = None,
    ) -> ChoiceAnswer:
        """Ask the same choice question once per criteria ordering and average.

        Jev answers every question in a request in parallel and in isolation, and
        request latency is close to flat in the number of questions, so asking k
        re-orderings of the same alphabet costs about one call. Averaging their
        distributions cancels out Jev's preference for whichever options come
        first in the criteria map.
        """
        if not criteria_variants:
            raise ValueError("choice() needs at least one criteria ordering")

        n = len(criteria_variants)
        questions = {
            f"next{i}": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
            for i, criteria in enumerate(criteria_variants)
        }
        answers, latency = self.ask(
            state, questions, cancel=cancel,
            overflow_hint=(
                f"{n} question(s) of {len(criteria_variants[0])} options. Lower "
                "`ensemble` or use a smaller alphabet."
            ),
        )

        totals: dict[str, float] = {}
        confidence = 0.0
        try:
            for i in range(n):
                answer = answers[f"next{i}"]
                confidence += float(answer.get("confidence", 0.0)) / n
                for key, value in answer["probabilities"].items():
                    totals[str(key)] = totals.get(str(key), 0.0) + float(value) / n
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(f"unexpected Jev response shape: {answers!r}") from exc

        return ChoiceAnswer(
            choice=max(totals, key=totals.__getitem__) if totals else "",
            probabilities=totals,
            confidence=confidence,
            model=self.model,
            latency=latency,
            variants=n,
        )

    def ask(
        self,
        state: dict | str,
        questions: dict[str, dict],
        *,
        cancel: threading.Event | None = None,
        overflow_hint: str = "",
    ) -> tuple[dict, float]:
        """Post an arbitrary question map and return its `answers` and latency.

        Jev answers every question in a request in parallel and in isolation, so
        a caller can put a whole decision tree in one round trip.
        """
        if not questions:
            raise ValueError("ask() needs at least one question")
        body = {"model": self.model, "state": state, "questions": questions}
        try:
            payload, latency = self._post(body, cancel=cancel)
        except JevError as exc:
            if "max_tokens_exceeded" in str(exc):
                raise JevError(
                    "Jev's context window overflowed with "
                    + (overflow_hint or f"{len(questions)} questions.")
                ) from exc
            raise

        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise JevError(f"unexpected Jev response shape: {payload!r}")
        self.usage.add(payload.get("usage") or {}, questions=len(questions))
        return answers, latency

    # ------------------------------------------------------------------ http
    def _post(self, body: dict, *, cancel: threading.Event | None) -> tuple[dict, float]:
        url = f"{self.api_base}/systemone"
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            started = time.monotonic()
            try:
                response = self._client.post(url, json=body)
            except httpx.HTTPError as exc:
                last = JevError(f"request to {url} failed: {exc}")
            else:
                latency = time.monotonic() - started
                if response.status_code < 400:
                    try:
                        return response.json(), latency
                    except ValueError as exc:
                        raise JevError("Jev returned a non-JSON body") from exc
                detail = _detail(response)
                if response.status_code == 401:
                    raise JevError("Jev rejected the API key (401). Check .env.")
                if response.status_code not in RETRY_STATUS:
                    raise JevError(f"Jev returned {response.status_code}: {detail}")
                last = JevError(f"Jev returned {response.status_code}: {detail}")

            if attempt == self.max_retries:
                break
            delay = min(8.0, 0.5 * 2**attempt) * (0.5 + random.random())
            if cancel is not None and cancel.wait(delay):
                raise Cancelled()
            elif cancel is None:
                time.sleep(delay)

        raise last or JevError("request failed")


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error_type") or payload)[:300]
    return str(payload)[:300]
