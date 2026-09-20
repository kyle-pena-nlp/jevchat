"""Ctrl-C handling: the first press cancels, the second aborts."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def sigint_cancel(on_first: Callable[[], None] | None = None) -> Iterator[threading.Event]:
    """Route SIGINT to a cancel Event instead of raising KeyboardInterrupt.

    The first Ctrl-C sets the event so the caller can stop cleanly and keep the
    partial result. A second Ctrl-C restores the default handler's behaviour and
    raises KeyboardInterrupt, so a wedged request can still be escaped.
    """
    cancel = threading.Event()

    def handler(signum: int, frame: object) -> None:
        if cancel.is_set():
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        cancel.set()
        if on_first is not None:
            on_first()

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Not on the main thread: cancellation still works, just not via signals.
        yield cancel
        return

    try:
        yield cancel
    finally:
        signal.signal(signal.SIGINT, previous)
