"""Answer matching, consistency, answer classes, the train/test split and the perturbation sample.

Labelling is the weakest link of the whole sprint: a correct answer that the matcher marks wrong is
usually also consistent, so it lands in CW and fakes exactly the hard regime the paper is about.
"""

import numpy as np
import pandas as pd
import pytest

from sgta.data import ANSWER_PREFIXES, assign_etype, clean_answer, is_correct, token_f1
from sgta.store import Run, assign_split, stratified_pick


# ------------------------------------------------------------------ correctness


@pytest.mark.parametrize(
    "pred, aliases",
    [
        ("The Eiffel Tower.", ["Eiffel Tower"]),  # article, punctuation, case
        ("Paris, France", ["Paris"]),  # alias inside a longer answer
        ("1,000", ["1000"]),  # thousands separator, both ways
        ("1000", ["1,000"]),
        ("St. Louis", ["St Louis"]),
        ("U.S.", ["US"]),
        ("Jean Paul Sartre", ["Jean-Paul Sartre"]),  # hyphen in the alias only
        ("Jean-Paul Sartre", ["Jean Paul Sartre"]),  # hyphen in the answer only
        ("AC DC", ["AC/DC"]),
        ("ACDC", ["AC/DC"]),
        ("Pele", ["Pelé"]),  # accents, both ways
        ("Pelé", ["Pele"]),
        ("Ireland’s", ["Ireland's"]),  # typographic apostrophe
        ("Gödel", ["Godel"]),
        ("A. A. Milne", ["A.A. Milne"]),
    ],
)
def test_spelling_variants_of_a_correct_answer_are_correct(pred, aliases):
    assert is_correct(pred, aliases)


@pytest.mark.parametrize(
    "pred, aliases",
    [
        ("Parisian", ["Paris"]),  # whole words only
        ("London", ["New London"]),  # part of the alias is not the alias
        ("", ["Paris"]),
        ("The", ["The Who"]),  # an answer that is only an article
        ("Who", ["The Who of Leeds"]),
        ("Iron", ["Fe"]),
    ],
)
def test_wrong_answers_stay_wrong(pred, aliases):
    assert not is_correct(pred, aliases)


@pytest.mark.parametrize("prefix", list(ANSWER_PREFIXES) + [""])
def test_clean_answer_keeps_the_first_line_without_prefix_and_trailing_period(prefix):
    assert clean_answer(f"{prefix} Paris.\nExplanation follows") == "Paris"
    assert clean_answer("  Iron.  ") == "Iron"


# ------------------------------------------------------------------ consistency


def test_token_f1_basics():
    assert token_f1("Leo Tolstoy", "Leo Tolstoy") == 1.0
    assert token_f1("Paris", "Rome") == 0.0
    assert token_f1("Leo Tolstoy", "Tolstoy") == pytest.approx(2 / 3)
    for a, b in [("Leo Tolstoy", "Tolstoy"), ("the red planet", "Mars"), ("Jean-Paul", "Jean Paul")]:
        assert token_f1(a, b) == token_f1(b, a)


def test_spelling_variants_count_as_the_same_answer_for_consistency():
    assert token_f1("Jean-Paul Sartre", "Jean Paul Sartre") >= 0.5
    assert token_f1("Pelé", "Pele") == 1.0


# ------------------------------------------------------------------ classes


def test_assign_etype_rule():
    correct = np.array([1, 1, 0, 0, 0, 0])
    cons = np.array([0.2, 1.0, 0.8, 0.6, np.nan, 1.0])
    assert list(assign_etype(correct, cons, 0.8)) == ["FA", "FA", "CW", "UW", "UW", "CW"]


def test_every_cw_is_consistent_and_fac_matches_the_definition(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    correct = rng.random(n) < 0.6
    cons = np.round(rng.random(n) * 5) / 5
    cons[rng.random(n) < 0.05] = np.nan
    meta = pd.DataFrame(
        {"correct": correct.astype(int), "consistency": cons, "etype": assign_etype(correct, cons, 0.8)}
    )
    meta.to_csv(tmp_path / "meta.csv", index=False)
    run = Run(tmp_path)
    et, c = run.etype, run.consistent
    assert c[et == "CW"].all()
    assert not c[et == "UW"].any()
    fac = (et == "FA") & c
    assert np.array_equal(fac, correct & (np.nan_to_num(cons) >= 0.8))


# ------------------------------------------------------------------ split


def test_split_is_deterministic_order_free_and_about_forty_percent():
    ids = [f"tq_{i}" for i in range(10000)]
    a = assign_split(ids, 0.4, 0)
    b = assign_split(ids[::-1], 0.4, 0)[::-1]
    assert np.array_equal(a, b)
    assert abs((a == "test").mean() - 0.4) < 0.02
    assert (assign_split(ids, 0.4, 1) != a).any()  # the seed matters


# ------------------------------------------------------------------ perturbation sample


def _classes(n_fa_c, n_fa_i, n_cw, n_uw, seed=0):
    et = np.array(["FA"] * (n_fa_c + n_fa_i) + ["CW"] * n_cw + ["UW"] * n_uw)
    cons = np.array([True] * n_fa_c + [False] * n_fa_i + [True] * n_cw + [False] * n_uw)
    p = np.random.default_rng(seed).permutation(len(et))
    return et[p], cons[p]


def test_stratified_pick_quotas_and_no_duplicates():
    et, cons = _classes(4000, 2000, 1500, 1500)
    idx = stratified_pick(et, cons, 3000, seed=0)
    assert len(idx) == 3000 and len(np.unique(idx)) == 3000
    assert np.array_equal(idx, np.sort(idx))
    assert (et[idx] == "CW").sum() == 1000
    assert ((et[idx] == "FA") & cons[idx]).sum() == 1000
    assert (et[idx] == "UW").sum() == 500
    assert ((et[idx] == "FA") & ~cons[idx]).sum() == 500
    assert np.array_equal(idx, stratified_pick(et, cons, 3000, seed=0))


def test_short_stratum_is_taken_whole_and_the_rest_is_filled():
    et, cons = _classes(5000, 2000, 300, 1500)
    idx = stratified_pick(et, cons, 3000, seed=1)
    assert len(idx) == 3000 and len(np.unique(idx)) == 3000
    assert (et[idx] == "CW").sum() == 300  # every CW the model produced


def test_more_requested_than_available_returns_everything():
    et, cons = _classes(50, 20, 10, 20)
    idx = stratified_pick(et, cons, 3000, seed=0)
    assert np.array_equal(idx, np.arange(100))
