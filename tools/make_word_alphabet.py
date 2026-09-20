#!/usr/bin/env python3
"""Generate jevchat/alphabets/words1k.json — common words, plus a spelling fallback.

Word-level options only work when the word you need is on the list. The bare
letters, suffixes and punctuation at the end are the escape hatch: anything out of
vocabulary can still be spelled out one character at a time.

    python tools/make_word_alphabet.py
"""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wordlist import words  # noqa: E402

OUT = Path(__file__).parent.parent / "jevchat" / "alphabets" / "words1k.json"

CHARNAMES = {
    " ": "a single space", "\n": "a newline (line break)",
    "\n\n": "a blank line (paragraph break)", ".": "a period", ",": "a comma",
    "!": "an exclamation mark", "?": "a question mark", "'": "an apostrophe",
    '"': "a double quote", "-": "a hyphen", ":": "a colon", ";": "a semicolon",
    "(": "an opening parenthesis", ")": "a closing parenthesis",
}
NAMED = {" ": "SPACE", "\n": "NEWLINE", "\n\n": "PARAGRAPH"}
SUFFIXES = ["ed", "ing", "ly", "er", "es", "tion", "ment", "ness", "'s", "n't"]


def main() -> None:
    symbols: list[dict] = []
    vocabulary = words()

    for word in vocabulary:
        symbols.append({"key": f" {word}", "emit": f" {word}",
                        "description": f"the word {word!r}, with a leading space"})
        capitalised = word.capitalize()
        if word in {"i", "the", "it", "this", "that", "you", "we", "they", "there",
                    "he", "she", "a", "an", "yes", "no", "but", "so", "if", "when",
                    "what", "how", "why", "most", "some", "one", "two", "three"}:
            symbols.append({"key": f" {capitalised}", "emit": f" {capitalised}",
                            "description": f"the capitalised word {capitalised!r}, "
                                           "with a leading space"})

    for c in string.ascii_lowercase:
        symbols.append({"key": c, "emit": c, "description":
                        f"the bare letter {c!r}, for spelling a word that is not listed"})
    for c in string.ascii_uppercase:
        symbols.append({"key": c, "emit": c,
                        "description": f"the capital letter {c!r}, for spelling a name"})
    for suffix in SUFFIXES:
        symbols.append({"key": suffix, "emit": suffix,
                        "description": f"the suffix {suffix!r}, joined to the word before it"})
    for digit in string.digits:
        symbols.append({"key": digit, "emit": digit, "description": f"the digit {digit}"})
    for ch, name in CHARNAMES.items():
        symbols.append({"key": NAMED.get(ch, ch), "emit": ch, "description": name})

    keys = [s["key"] for s in symbols]
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert not duplicates, f"duplicate keys: {sorted(duplicates)[:10]}"

    OUT.write_text(json.dumps({
        "name": "words1k",
        "description": (
            f"{len(vocabulary)} common English words (with a leading space) plus "
            "capitalised forms of the commonest, and bare letters, digits, suffixes "
            "and punctuation so anything out of vocabulary can still be spelled out. "
            "Too large for one Jev question: use --strategy buckets."
        ),
        "stop": {"key": "STOP",
                 "description": "The reply is already complete and well formed: stop writing."},
        "symbols": symbols,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"{OUT.name}: {len(symbols) + 1} options ({len(vocabulary)} words)")


if __name__ == "__main__":
    main()
