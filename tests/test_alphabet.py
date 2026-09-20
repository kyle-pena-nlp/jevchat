import json

import pytest

from jevchat import alphabet as alphabet_mod
from jevchat.alphabet import MAX_CHOICES, Alphabet, AlphabetError, Symbol


@pytest.mark.parametrize("name", alphabet_mod.builtin_names())
def test_builtin_alphabets_load(name):
    alpha = alphabet_mod.load(name)
    criteria = alpha.criteria()
    assert len(criteria) == alpha.size
    assert alpha.stop_key in criteria
    assert alpha.emit_for(alpha.stop_key) is None


def test_builtins_include_the_character_and_word_alphabets():
    assert {"lower26", "ascii", "tokens", "words1k", "bpe2k", "bpe5k"} <= set(
        alphabet_mod.builtin_names())


@pytest.mark.parametrize("name", ["words1k", "bpe2k", "bpe5k"])
def test_large_alphabets_exceed_one_question(name):
    alpha = alphabet_mod.load(name)
    assert not alpha.fits_one_question()
    assert alpha.size > MAX_CHOICES


def test_bpe_alphabet_holds_real_subword_pieces():
    alpha = alphabet_mod.load("bpe2k")
    keys = {s.key for s in alpha.symbols}
    assert {" the", " of", "ing", " a"} <= keys      # frequent whole pieces
    assert {"a", "z", " "} <= keys                   # single-byte fallback
    assert alpha.emit_for(" the") == " the"


def test_words1k_can_spell_out_of_vocabulary_words():
    alpha = alphabet_mod.load("words1k")
    assert alpha.emit_for(" water") == " water"
    assert all(alpha.emit_for(c) == c for c in "qxz")
    assert alpha.emit_for("SPACE") == " "


def test_lower26_is_letters_plus_space():
    alpha = alphabet_mod.load("lower26")
    assert alpha.size == 28
    assert alpha.emit_for("q") == "q"
    assert alpha.emit_for("SPACE") == " "


def test_tokens_alphabet_emits_whole_words_and_can_still_spell():
    alpha = alphabet_mod.load("tokens")
    assert alpha.emit_for(" the") == " the"
    assert alpha.emit_for("z") == "z"          # out-of-vocabulary fallback
    assert alpha.emit_for("ing") == "ing"      # suffix piece


def test_unknown_key_is_rejected():
    alpha = alphabet_mod.load("lower26")
    with pytest.raises(AlphabetError):
        alpha.emit_for("!!")


def test_duplicate_keys_are_rejected():
    with pytest.raises(AlphabetError):
        Alphabet(name="x", description="", symbols=(Symbol("a", "a"), Symbol("a", "b")))


def test_stop_key_cannot_double_as_a_symbol():
    with pytest.raises(AlphabetError):
        Alphabet(name="x", description="", symbols=(Symbol("STOP", "s"),))


def test_an_alphabet_may_exceed_one_question_worth_of_options():
    # The 255 cap is per question, not per alphabet: `buckets` splits across questions.
    symbols = tuple(Symbol(f"k{i}", "x") for i in range(MAX_CHOICES * 4))
    big = Alphabet(name="big", description="", symbols=symbols)
    assert big.size == MAX_CHOICES * 4 + 1
    assert not big.fits_one_question()


def test_a_small_alphabet_fits_one_question():
    assert alphabet_mod.load("tokens").fits_one_question()
    assert alphabet_mod.load("lower26").fits_one_question()


def test_custom_alphabet_from_a_file(tmp_path):
    path = tmp_path / "vowels.json"
    path.write_text(json.dumps({
        "name": "vowels",
        "description": "vowels only",
        "stop": {"key": "END", "description": "done"},
        "symbols": ["a", "e", {"key": "SP", "emit": " ", "description": "space"}],
    }))
    alpha = alphabet_mod.load(str(path))
    assert alpha.size == 4
    assert alpha.stop_key == "END"
    assert alpha.emit_for("SP") == " "
    assert alpha.emit_for("a") == "a"


def test_missing_alphabet_mentions_the_builtins():
    with pytest.raises(AlphabetError, match="lower26"):
        alphabet_mod.load("nope")
