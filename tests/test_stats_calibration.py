"""Size and power of the statistics behind every verdict.

The central bet is judged by one procedure: a feature block is collapsed into one stacked score
(out-of-fold on train), then base vs base + score is compared by DeLong on test and by a likelihood
ratio on train, and the H1 criterion needs delta AUROC >= 0.02, DeLong p < 0.05 and LR p < 0.01.
These tests run that exact procedure on simulated data where the truth is known.

  null       the block is pure noise                              -> must (almost) never pass
  redundant  the block is a non-linear function of the base         -> must (almost) never pass
  nonlinear  the error depends on |confidence|, the block measures  -> must (almost) never pass;
             |confidence|: what a flip rate does when it only        this failed before the base
             re-measures the margin                                  went through a spline basis (v1.5)
  signal     the block carries information the base does not have -> must usually pass
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from sgta.hypotheses import THRESHOLDS as T
from sgta.hypotheses import _stack_score
from sgta.stats import auroc, delong, flex_basis, lr_increment, nested_test


def _h1_criterion(kind: str, seed: int, n_tr: int = 900, n_te: int = 600):
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    m, c, extra = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    if kind == "nonlinear":
        logit = -1.0 + 1.5 * np.abs(m) - 0.3 * m + 0.5 * c
    else:
        logit = 1.0 * m + 0.6 * c + (0.8 * extra if kind == "signal" else 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    if kind == "null":
        block = rng.standard_normal((n, 12))
    elif kind == "redundant":
        block = np.c_[np.exp(m / 2), m**3, np.tanh(2 * c), c * m] + 0.05 * rng.standard_normal((n, 4))
    elif kind == "nonlinear":
        block = np.c_[np.abs(m), m**2, np.exp(-(m**2))] + 0.05 * rng.standard_normal((n, 3))
    else:
        block = np.c_[extra + 0.7 * rng.standard_normal(n), rng.standard_normal((n, 6))]
    tr = np.arange(n) < n_tr
    te = ~tr
    Xb = flex_basis(np.c_[m, c], tr)
    sc = _stack_score(block, y, tr, te, seed)
    Xf = np.c_[Xb, sc]
    nt = nested_test(Xb[tr], Xf[tr], y[tr], Xb[te], Xf[te], y[te])
    passed = nt["delta_auc"] >= T["H1_delta_auc"] and nt["p_delong"] < T["H1_p_delong"] and nt["p_lr"] < T["H1_p_lr"]
    return passed, nt


@pytest.mark.slow
@pytest.mark.parametrize("kind, reps", [("null", 80), ("redundant", 50), ("nonlinear", 50)])
def test_h1_criterion_does_not_pass_without_new_information(kind, reps):
    rate = np.mean([_h1_criterion(kind, s)[0] for s in range(reps)])
    assert rate <= 0.03, f"{kind}: H1 criterion passed in {rate:.0%} of runs"


@pytest.mark.slow
def test_h1_criterion_has_power_when_the_block_carries_new_information():
    rate = np.mean([_h1_criterion("signal", 100 + s)[0] for s in range(30)])
    assert rate >= 0.6, f"power {rate:.0%}"


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.standard_normal(500) + 0.5 * y, 1)  # many ties
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_delong_size_under_null():
    rng = np.random.default_rng(1)
    hits = 0
    reps = 300
    for _ in range(reps):
        y = rng.integers(0, 2, 400)
        s1 = 0.5 * y + rng.standard_normal(400)
        s2 = 0.5 * y + rng.standard_normal(400)
        hits += delong(y, s1, s2)[2] < 0.05
    assert 0.01 <= hits / reps <= 0.10


def test_lr_increment_size_and_power_for_a_raw_feature():
    """H3b uses the LR test on a raw feature (commitment depth); there it must be calibrated."""
    rng = np.random.default_rng(2)
    null_hits, power_hits = 0, 0
    for _ in range(150):
        n = 600
        x, z = rng.standard_normal(n), rng.standard_normal(n)
        y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
        Xb = x[:, None]
        null_hits += lr_increment(Xb, np.c_[Xb, z], y)[1] < 0.05
        y2 = (rng.random(n) < 1 / (1 + np.exp(-(x + 0.4 * z)))).astype(int)
        power_hits += lr_increment(Xb, np.c_[Xb, z], y2)[1] < 0.01
    assert null_hits / 150 <= 0.10
    assert power_hits / 150 >= 0.8


def test_flex_basis_properties():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.standard_normal(n)
    cons = rng.choice([0.8, 1.0], n)  # consistency inside CW vs FAc: two values
    x_nan = x.copy()
    x_nan[:7] = np.nan
    fit = np.arange(n) < 250
    B = flex_basis(np.c_[x_nan, cons], fit)
    assert np.array_equal(B[:, 0], x_nan, equal_nan=True)  # raw column kept
    assert np.isnan(B[:7]).any(1).all() and np.isfinite(B[7:]).all()
    assert B.shape[1] > 2  # spline columns for x
    assert np.array_equal(B[:, -1], cons)  # too few values: stays linear
    # knots come from the fitting rows only: changing the other rows changes nothing for the fitted map
    x2 = x.copy()
    x2[300:] = x2[300:] * 5
    B2 = flex_basis(x2[:, None], fit)
    B1 = flex_basis(x[:, None], fit)
    assert np.allclose(B1[:300], B2[:300])
