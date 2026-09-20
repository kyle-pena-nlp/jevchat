import pytest

from jevchat import bisect as B
from jevchat.alphabet import Alphabet, Symbol
from jevchat.client import JevError

ALPHA = Alphabet(
    name="eight",
    description="",
    symbols=tuple(Symbol(c, c, f"letter {c}") for c in "abcdefgh"),
)


def test_leaves_cover_every_symbol_exactly_once():
    tree = B.build(ALPHA, cutoff=2)
    seen = [s.key for leaf in tree.leaves for s in leaf.symbols]
    assert sorted(seen) == list("abcdefgh")
    assert all(len(leaf.symbols) <= 2 for leaf in tree.leaves)


def test_tree_shape():
    tree = B.build(ALPHA, cutoff=2)
    assert len(tree.leaves) == 4
    assert len(tree.splits) == 3
    assert tree.depth == 2


def test_a_cutoff_larger_than_the_alphabet_needs_no_splits():
    tree = B.build(ALPHA, cutoff=20)
    assert tree.splits == ()
    assert len(tree.leaves) == 1


def test_cutoff_below_two_is_rejected():
    with pytest.raises(ValueError):
        B.build(ALPHA, cutoff=1)


def test_symbols_are_sorted_so_earlier_and_later_mean_something():
    scrambled = Alphabet(
        name="s", description="",
        symbols=(Symbol("d", "d"), Symbol("a", "a"), Symbol("c", "c"), Symbol("b", "b")),
    )
    tree = B.build(scrambled, cutoff=2)
    assert [s.key for s in tree.leaves[0].symbols] == ["a", "b"]
    assert [s.key for s in tree.leaves[1].symbols] == ["c", "d"]


def test_question_count_with_and_without_swapping():
    tree = B.build(ALPHA, cutoff=2)
    swapped = B.build_questions(tree, ALPHA, "pick", swap=True)
    plain = B.build_questions(tree, ALPHA, "pick", swap=False)
    # 3 splits (x2 when swapped) + 4 leaf questions + 1 stop question
    assert len(swapped) == 3 * 2 + 4 + 1
    assert len(plain) == 3 + 4 + 1
    assert swapped[B.STOP_QUESTION]["type"] == "noul"
    assert all(q["type"] in {"noul", "choice"} for q in swapped.values())


def test_a_single_symbol_leaf_needs_no_question():
    small = Alphabet(name="t", description="",
                     symbols=(Symbol("a", "a"), Symbol("b", "b"), Symbol("c", "c")))
    tree = B.build(small, cutoff=2)
    lone = [leaf for leaf in tree.leaves if len(leaf.symbols) == 1]
    assert lone
    questions = B.build_questions(tree, small, "pick", swap=False)
    assert f"{lone[0].node}:pick" not in questions


def answers_for(tree, *, stop, p_hi, leaf_probs):
    out = {B.STOP_QUESTION: {"noul": stop}}
    for split in tree.splits:
        out[f"{split.node}:a"] = {"noul": p_hi}
        out[f"{split.node}:b"] = {"noul": 1.0 - p_hi}
    for leaf in tree.leaves:
        if len(leaf.symbols) > 1:
            out[f"{leaf.node}:pick"] = {"probabilities": dict(leaf_probs(leaf))}
    return out


def test_distribution_is_normalised():
    tree = B.build(ALPHA, cutoff=2)
    answers = answers_for(
        tree, stop=0.2, p_hi=0.75,
        leaf_probs=lambda leaf: {s.key: 0.5 for s in leaf.symbols},
    )
    dist = B.distribution(tree, ALPHA, answers, swap=True)
    assert dist[ALPHA.stop_key] == pytest.approx(0.2)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert set(dist) == set("abcdefgh") | {ALPHA.stop_key}


def test_branch_probabilities_multiply_down_the_path():
    tree = B.build(ALPHA, cutoff=2)
    answers = answers_for(
        tree, stop=0.0, p_hi=0.75,
        leaf_probs=lambda leaf: {s.key: 0.5 for s in leaf.symbols},
    )
    dist = B.distribution(tree, ALPHA, answers, swap=True)
    # 'h' is the last symbol: later half twice, then half of its leaf.
    assert dist["h"] == pytest.approx(0.75 * 0.75 * 0.5)
    # 'a' is the first: earlier half twice.
    assert dist["a"] == pytest.approx(0.25 * 0.25 * 0.5)


def test_swapping_averages_the_two_phrasings():
    tree = B.build(ALPHA, cutoff=4)
    # Forward says 0.9 later; the swapped question says 0.5, i.e. 0.5 later.
    answers = {
        B.STOP_QUESTION: {"noul": 0.0},
        f"{tree.splits[0].node}:a": {"noul": 0.9},
        f"{tree.splits[0].node}:b": {"noul": 0.5},
    }
    for leaf in tree.leaves:
        answers[f"{leaf.node}:pick"] = {"probabilities": {s.key: 0.25 for s in leaf.symbols}}

    swapped = B.distribution(tree, ALPHA, answers, swap=True)
    plain = B.distribution(tree, ALPHA, answers, swap=False)
    assert sum(swapped[c] for c in "efgh") == pytest.approx(0.7)   # (0.9 + 0.5) / 2
    assert sum(plain[c] for c in "efgh") == pytest.approx(0.9)


def test_leaf_probabilities_are_renormalised():
    tree = B.build(ALPHA, cutoff=4)
    answers = answers_for(tree, stop=0.0, p_hi=0.5,
                          leaf_probs=lambda leaf: {s.key: 0.1 for s in leaf.symbols})
    dist = B.distribution(tree, ALPHA, answers, swap=True)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_out_of_range_probabilities_are_clamped():
    tree = B.build(ALPHA, cutoff=4)
    answers = answers_for(tree, stop=1.4, p_hi=0.5,
                          leaf_probs=lambda leaf: {s.key: 0.25 for s in leaf.symbols})
    dist = B.distribution(tree, ALPHA, answers, swap=True)
    assert dist[ALPHA.stop_key] == pytest.approx(1.0)
    assert all(v >= 0 for v in dist.values())


def test_a_missing_answer_is_reported():
    tree = B.build(ALPHA, cutoff=2)
    with pytest.raises(JevError, match="unexpected Jev response shape"):
        B.distribution(tree, ALPHA, {B.STOP_QUESTION: {"noul": 0.1}}, swap=True)
