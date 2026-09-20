"""Configuration: defaults, ``jevchat.toml``, environment, then CLI overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from dotenv import load_dotenv

from .buckets import BUCKET_ORDERS
from .present import PRESENTATIONS

CONFIG_FILENAME = "jevchat.toml"

DEFAULT_TASK = (
    "A correct English answer to `question` is being written out left to right. "
    "`answer_so_far` is exactly the text written so far, character for character; "
    "`conversation` is what was said earlier in the chat. Exactly one option is "
    "appended to `answer_so_far` next, verbatim, with no edits and no extra spacing."
)

HYPOTHESIS_INSTRUCTIONS = (
    "`answer_so_far` is the reply as written so far. Every option below is that reply "
    "with one more symbol added, shown as its tail rather than in full. Pick the option "
    "that reads as the best beginning of a correct, grammatical, factually right English "
    "answer to `question`. Exactly one option is the reply with nothing added to it: "
    "pick that one only if the reply already answers `question` completely and ends on "
    "a finished sentence."
)

DEFAULT_INSTRUCTIONS = (
    "Append one option to `answer_so_far`. Choose the option that makes "
    "`answer_so_far` + option a valid prefix of a grammatical, factually correct "
    "English answer to `question`. Judge grammar and spelling on the concatenation, "
    "not on the option on its own. Choose STOP only if `answer_so_far` already answers "
    "`question` completely and ends on a finished sentence."
)

STRATEGIES = ("choice", "bisect", "buckets", "refine")

# One option in every bucket is spent on OTHER.
MAX_BUCKET = 254

# The most bucket winners `refine` can weigh against each other in one question.
MAX_PROBE = 254

# Environment variable names checked for the API key, in order.
API_KEY_VARS = ("api_key", "JEV_API_KEY", "TYPESAFE_API_KEY", "API_KEY")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    # --- what Jev is asked -------------------------------------------------
    alphabet: str = "words1k"
    model: str = "jev-latest"
    task: str = DEFAULT_TASK
    instructions: str = DEFAULT_INSTRUCTIONS
    hypothesis_instructions: str = HYPOTHESIS_INSTRUCTIONS
    history_turns: int = 6
    # How the options are worded: "symbol" offers the symbols themselves, "hypothesis"
    # offers the resulting texts. See jevchat/present.py.
    presentation: str = "hypothesis"
    # Characters of the reply's tail shown in each hypothesis option.
    window: int = 40
    # Jev favours whichever options come first in the criteria map. Re-ordering the
    # alphabet on every call cancels that bias out; it roughly doubles how often the
    # right symbol is scored highest.
    shuffle_criteria: bool = True
    # How many re-orderings to score per step. They travel as parallel questions in
    # one request, so latency stays close to flat, but input tokens scale with it.
    # Raising this widens the usable support (Jev rounds probabilities to 0.01, so a
    # single call zeroes most of a large alphabet); it does not improve accuracy.
    ensemble: int = 1

    # How the next symbol's distribution is obtained.
    #   choice  one choice question over the whole alphabet (x `ensemble`)
    #   bisect  a tree of earlier/later noul questions down to `bisect_cutoff`,
    #           then one choice question inside the group that is reached
    #   buckets one question per slice of the alphabet, each with an OTHER option
    #   refine  buckets without OTHER, weighted by a question over the winners, then
    #           refined by rescoring each bucket's nucleus (see jevchat/refine.py)
    strategy: str = "buckets"
    bisect_cutoff: int = 20
    bisect_swap: bool = True
    bucket_size: int = 127
    bucket_batch: int = 48
    bucket_describe: bool = False
    # How the alphabet is cut into buckets: "given" (its own order), "sorted"
    # (look-alikes grouped together), or "shuffled".
    bucket_order: str = "given"
    refine_nucleus: int = 3
    refine_rounds: int = 1

    # --- how the next symbol is sampled ------------------------------------
    temperature: float = 0.4
    top_p: float = 0.9
    top_k: int = 0
    stop_bias: float = 0.5
    repetition_penalty: float = 1.1
    repetition_window: int = 8
    no_repeat_space: bool = True
    min_steps: int = 3
    max_steps: int = 400
    max_chars: int = 1500
    # Candidate replies kept alive at once. 1 samples one symbol at a time; more
    # lets later evidence abandon a prefix that turned out badly, at one score per
    # live beam per step. Above 1, `temperature`, `top_p` and `top_k` no longer
    # apply: beams are ranked by probability, not drawn.
    beam_width: int = 1
    beam_length_penalty: float = 1.0
    seed: int | None = None

    # --- transport ---------------------------------------------------------
    api_base: str = "https://api.typesafe.ai/v1"
    timeout: float = 60.0
    max_retries: int = 4

    # --- display -----------------------------------------------------------
    live: bool = True
    show_dist: int = 3

    # Which options the user actually named, in the file or on the command line.
    # Everything else is open to being resolved once the alphabet is known.
    explicit: frozenset[str] = frozenset()

    @property
    def active_instructions(self) -> str:
        """The wording that matches the chosen presentation."""
        if self.presentation == "hypothesis":
            return self.hypothesis_instructions
        return self.instructions

    @classmethod
    def load(
        cls,
        *,
        config_path: Path | None = None,
        start_dir: Path | None = None,
        overrides: dict | None = None,
    ) -> "Config":
        data: dict = {}
        path = config_path or find_config_file(start_dir or Path.cwd())
        if path is not None:
            if not path.is_file():
                raise ConfigError(f"no config file at {path}")
            raw = tomllib.loads(path.read_text())
            data = raw.get("jevchat", raw)

        known = {f.name for f in fields(cls)} - {"explicit"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"{path}: unknown option(s) {', '.join(sorted(unknown))}; "
                f"valid options are {', '.join(sorted(known))}"
            )

        clean = {k: v for k, v in (overrides or {}).items() if v is not None}
        cfg = replace(cls(**data), **clean) if clean else cls(**data)
        cfg = replace(cfg, explicit=frozenset(data) | frozenset(clean))
        cfg.validate()
        return cfg

    def with_overrides(self, **kwargs) -> "Config":
        """Apply overrides and mark them as deliberately chosen."""
        return replace(self, **kwargs,
                       explicit=self.explicit | frozenset(kwargs))

    def resolve(self, alphabet) -> "Config":
        """Fill in settings that depend on the alphabet and the strategy.

        Precedence: command line > jevchat.toml > the alphabet's declared
        preferences > what the strategy requires > the plain defaults.
        """
        known = {f.name for f in fields(self)} - {"explicit"}
        auto: dict = {}
        for name, value in alphabet.defaults:
            if name in known and name not in self.explicit:
                auto[name] = value

        if self.strategy == "refine" and "bucket_size" not in self.explicit:
            # refine asks one question over every bucket's winner, so the number of
            # buckets has to fit inside a single question.
            needed = -(-(alphabet.size - 1) // MAX_PROBE)
            auto["bucket_size"] = min(
                MAX_BUCKET, max(auto.get("bucket_size", self.bucket_size), needed)
            )

        if not auto:
            return self
        resolved = replace(self, **auto)
        resolved.validate()
        return resolved

    def validate(self) -> None:
        if self.temperature < 0:
            raise ConfigError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ConfigError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ConfigError("top_k must be >= 0 (0 disables it)")
        if self.stop_bias < 0:
            raise ConfigError("stop_bias must be >= 0")
        if self.repetition_penalty <= 0:
            raise ConfigError("repetition_penalty must be > 0")
        if self.repetition_window < 0:
            raise ConfigError("repetition_window must be >= 0")
        if self.ensemble < 1:
            raise ConfigError("ensemble must be >= 1")
        if self.strategy not in STRATEGIES:
            raise ConfigError(
                f"strategy must be one of {', '.join(sorted(STRATEGIES))}, "
                f"not {self.strategy!r}"
            )
        if self.bisect_cutoff < 2:
            raise ConfigError("bisect_cutoff must be >= 2")
        if not 1 <= self.bucket_size <= MAX_BUCKET:
            raise ConfigError(f"bucket_size must be between 1 and {MAX_BUCKET}")
        if self.bucket_batch < 1:
            raise ConfigError("bucket_batch must be >= 1")
        if self.bucket_order not in BUCKET_ORDERS:
            raise ConfigError(
                f"bucket_order must be one of {', '.join(BUCKET_ORDERS)}, "
                f"not {self.bucket_order!r}"
            )
        if self.beam_width < 1:
            raise ConfigError("beam_width must be >= 1")
        if self.beam_length_penalty < 0:
            raise ConfigError("beam_length_penalty must be >= 0")
        if self.refine_nucleus < 1:
            raise ConfigError("refine_nucleus must be >= 1")
        if self.refine_rounds < 0:
            raise ConfigError("refine_rounds must be >= 0")
        if self.presentation not in PRESENTATIONS:
            raise ConfigError(
                f"presentation must be one of {', '.join(PRESENTATIONS)}, "
                f"not {self.presentation!r}"
            )
        if self.window < 1:
            raise ConfigError("window must be >= 1")
        if self.max_steps < 1:
            raise ConfigError("max_steps must be >= 1")
        if self.min_steps < 0:
            raise ConfigError("min_steps must be >= 0")


def find_config_file(start: Path) -> Path | None:
    """Look for jevchat.toml in ``start`` and its parents."""
    for directory in [start, *start.parents]:
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_api_key(*, env_file: Path | None = None) -> str:
    """Read the Jev API key from .env / the environment.

    The .env file is the source of truth: values there win over anything already
    exported, so editing .env is enough to switch keys.
    """
    load_dotenv(dotenv_path=env_file, override=True)
    for var in API_KEY_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    raise ConfigError(
        "no API key found. Put `api_key=\"...\"` in a .env file next to pyproject.toml "
        f"(or set one of {', '.join(API_KEY_VARS[1:])})."
    )
