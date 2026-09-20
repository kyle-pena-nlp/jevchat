#!/usr/bin/env python3
"""Generate jevchat/alphabets/bpe2k.json from GPT-2's real BPE vocabulary.

GPT-2 assigns ids 0-255 to single bytes and then numbers the merged pieces in the
order the BPE training added them, which is by descending frequency. So the lowest
ids are the byte-level fallback plus the most common word pieces in English — a
principled "most useful N tokens" without needing a frequency corpus.

The output is committed, so jevchat itself never needs this script or a network.

    python tools/make_bpe_alphabet.py [--limit 2600] [--out bpe2k.json]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

VOCAB_URL = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
ALPHABETS = Path(__file__).parent.parent / "jevchat" / "alphabets"


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte <-> printable-character mapping."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def decode(token: str, byte_for: dict[str, int]) -> str | None:
    """Turn a GPT-2 vocabulary entry back into the text it stands for."""
    try:
        raw = bytes(byte_for[ch] for ch in token)
    except KeyError:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None  # half of a multi-byte character; useless on its own
    if not text:
        return None
    # Keep space and newline, drop every other control character.
    if any(ch not in " \n" and not ch.isprintable() for ch in text):
        return None
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2600,
                        help="how many of the lowest token ids to consider; pass the "
                             "full vocabulary size (50257 for GPT-2) for all of it")
    parser.add_argument("--out", default="bpe2k.json")
    args = parser.parse_args()

    with urllib.request.urlopen(VOCAB_URL, timeout=60) as response:
        vocab: dict[str, int] = json.load(response)

    byte_for = {ch: b for b, ch in bytes_to_unicode().items()}
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])[: args.limit]

    symbols: list[dict] = []
    seen: set[str] = set()
    for token, _id in by_id:
        text = decode(token, byte_for)
        if text is None or text in seen:
            continue
        seen.add(text)
        # A bare string is shorthand for key = emit with no description, which
        # keeps a 50k-entry file to a sane size.
        symbols.append(text)

    out = ALPHABETS / args.out
    out.write_text(json.dumps({
        "name": out.stem,
        "description": (
            (f"All {len(symbols)} usable tokens of GPT-2's real BPE vocabulary."
             if args.limit >= len(vocab)
             else f"The {len(symbols)} lowest-numbered tokens of GPT-2's BPE "
                  "vocabulary: single bytes plus the word pieces BPE merged first, "
                  "which are the most frequent ones.")
            + " A real subword vocabulary, so it can spell anything by falling back "
              "to shorter pieces. Too large for one Jev question: use "
              "--strategy buckets."
        ),
        # Frequency order is arbitrary with respect to spelling; sorting groups
        # look-alikes into the same bucket, which measured 2/6 -> 5/6 top-1.
        "defaults": {"presentation": "symbol", "bucket_order": "sorted"},
        "stop": {"key": "<|STOP|>",
                 "description": "The reply is already complete and well formed: stop writing."},
        "symbols": symbols,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(symbols) + 1} options "
          f"(from the {args.limit} lowest of {len(vocab)} GPT-2 tokens)")


if __name__ == "__main__":
    main()
