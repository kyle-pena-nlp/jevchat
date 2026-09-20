"""Terminal rendering: the reply as it is written, plus the generation rate."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .alphabet import Alphabet
from .generate import Done, Step

# Visible stand-ins for symbols that would otherwise be invisible in the readout.
_VISIBLE = {" ": "␣", "\n": "⏎", "\n\n": "¶", "\t": "⇥"}

REASON_LABEL = {
    "stop": "STOP sampled",
    "max_steps": "hit max_steps",
    "max_chars": "hit max_chars",
    "cancelled": "cancelled",
}


def show_key(key: str) -> str:
    return _VISIBLE.get(key, key)


def format_rate(step_or_done: Step | Done) -> str:
    s = step_or_done.stats
    return (
        f"{s.steps} sym · {s.symbols_per_sec:.1f} sym/s · "
        f"{s.chars_per_sec:.1f} char/s · {s.mean_latency * 1000:.0f} ms/call · "
        f"{s.elapsed:.1f}s"
    )


def format_dist(step: Step, alphabet: Alphabet, top: int) -> str:
    if top <= 0:
        return ""
    ranked = sorted(step.draw.raw.items(), key=lambda kv: kv[1], reverse=True)[:top]
    parts = [f"{show_key(k)} {v:.2f}" for k, v in ranked]
    return "  ".join(parts)


class Renderer:
    """Base interface so the CLI does not care which mode is active."""

    def start(self, question: str) -> None: ...
    def update(self, step: Step) -> None: ...
    def finish(self, done: Done) -> None: ...
    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None: ...


class LiveRenderer(Renderer):
    """A panel holding the reply so far, with a rate line underneath."""

    def __init__(
        self,
        console: Console,
        alphabet: Alphabet,
        *,
        show_dist: int = 3,
        title: str = "reply",
    ) -> None:
        self.console = console
        self.alphabet = alphabet
        self.show_dist = show_dist
        self.title = title
        self._live: Live | None = None
        self._text = ""
        self._status = Text("starting…", style="dim")
        self._final: Done | None = None
        self._printed = False

    # -- geometry ---------------------------------------------------------
    def _tail(self) -> str:
        """Keep the panel inside the window by showing only the recent tail."""
        height = max(4, self.console.size.height - 8)
        width = max(20, self.console.size.width - 6)
        budget = height * width
        if len(self._text) <= budget:
            return self._text
        return "…" + self._text[-budget:]

    def _renderable(self) -> Group:
        body = Text(self._tail() or " ", style="bold")
        return Group(Panel(body, title=self.title, border_style="cyan"), self._status)

    # -- lifecycle --------------------------------------------------------
    def start(self, question: str) -> None:
        self._live = Live(
            self._renderable(),
            console=self.console,
            refresh_per_second=10,
            vertical_overflow="crop",
            # The live panel is scratch space: it holds only the tail and is erased
            # on stop, so that `finish` can leave the reply on screen exactly once.
            transient=True,
        )
        self._live.start()

    def update(self, step: Step) -> None:
        if step.emit is not None:
            self._text = step.text
        status = Text()
        status.append(format_rate(step), style="dim")
        if self.show_dist:
            status.append("   ")
            status.append(f"↦ {show_key(step.key)} ", style="magenta")
            status.append(format_dist(step, self.alphabet, self.show_dist), style="dim cyan")
        self._status = status
        if self._live is not None:
            self._live.update(self._renderable())

    def finish(self, done: Done) -> None:
        self._final = done
        self.close()
        self._printed = True
        # Reprint the reply in full: the live panel only ever showed the tail.
        self.console.print(Panel(Text(done.text or "(empty)", style="bold"),
                                 title=self.title, border_style="cyan"))
        self.console.print(
            Text(f"{format_rate(done)} · {REASON_LABEL.get(done.reason, done.reason)}",
                 style="dim"))

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._final is None and self._text and not self._printed:
            # Stopped without a Done (aborted mid-reply): keep what was written,
            # since the transient panel has just been erased.
            self._printed = True
            self.console.print(
                Panel(Text(self._tail(), style="bold"), title=self.title, border_style="cyan")
            )


class PlainRenderer(Renderer):
    """Raw streaming to stdout, with the rate on stderr. Good for pipes and logs."""

    def __init__(self, console: Console, alphabet: Alphabet, *, show_dist: int = 0) -> None:
        self.console = console
        self.alphabet = alphabet
        self.show_dist = show_dist
        self._last_report = 0

    def update(self, step: Step) -> None:
        if step.emit:
            sys.stdout.write(step.emit)
            sys.stdout.flush()
        if step.stats.steps - self._last_report >= 25:
            self._last_report = step.stats.steps
            print(f"  [{format_rate(step)}]", file=sys.stderr)

    def finish(self, done: Done) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(
            f"[{format_rate(done)} · {REASON_LABEL.get(done.reason, done.reason)}]",
            file=sys.stderr,
        )


def make_renderer(
    console: Console, alphabet: Alphabet, *, live: bool, show_dist: int, title: str = "reply"
) -> Renderer:
    if live and console.is_terminal:
        return LiveRenderer(console, alphabet, show_dist=show_dist, title=title)
    return PlainRenderer(console, alphabet, show_dist=show_dist)
