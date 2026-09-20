#!/usr/bin/env python3
"""Generate jevchat/alphabets/bigrams.json — every ASCII character pair.

One option per two-character string over the `ascii` alphabet's single characters,
plus the single characters themselves so the reply can end on an odd length. A
middle ground between spelling one character at a time and a word vocabulary:
every decision carries twice the context and a reply needs half the steps, while
still being able to spell anything.

    python tools/make_ngram_alphabet.py [--n 2]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ALPHABETS = Path(__file__).parent.parent / "jevchat" / "alphabets"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="characters per option")
    parser.add_argument("--min-n", type=int, default=1,
                        help="shortest option. Set equal to --n for fixed-length "
                             "options: mixing lengths biases hypothesis scoring "
                             "towards the short ones, since a shorter prefix is "
                             "compatible with more continuations.")
    parser.add_argument("--out", default="bigrams.json")
    args = parser.parse_args()

    source = json.loads((ALPHABETS / "ascii.json").read_text())
    # Only single characters: the paragraph break is two newlines and would make
    # options longer than n.
    chars = sorted({s["emit"] for s in source["symbols"] if len(s["emit"]) == 1})

    symbols: list[str] = []
    for length in range(args.min_n, args.n + 1):
        symbols += ["".join(c) for c in itertools.product(chars, repeat=length)]

    assert len(symbols) == len(set(symbols))
    out = ALPHABETS / args.out
    out.write_text(json.dumps({
        "name": out.stem,
        "description": (
            (f"Every string of exactly {args.n} characters"
             if args.min_n == args.n else
             f"Every string of {args.min_n} to {args.n} characters")
            + f" over the {len(chars)} single "
            "ASCII characters: letters, digits, punctuation, space and newline. "
            f"{len(symbols)} options, so it needs --strategy buckets or refine. Half "
            "the steps of a character alphabet, and it can still spell anything."
        ),
        "stop": {"key": "<|STOP|>",
                 "description": "The reply is already complete and well formed: stop writing."},
        "defaults": {"repetition_penalty": 1.0, "bucket_order": "sorted"},
        "symbols": symbols,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(symbols) + 1} options "
          f"({len(chars)} characters, up to {args.n} per option)")


if __name__ == "__main__":
    main()
