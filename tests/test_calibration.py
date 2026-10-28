"""End-to-end false-positive / false-negative calibration of the tournament on synthetic runs.

planted:  effects for H0, H1, H1b, H2, H3b, H4 are built into the generator -> they must SURVIVE.
null:     labels independent of everything, second-order depth dynamics, unfaithful interventions
          -> none of H0, H1, H1b, H2, H3b, H4 may SURVIVE.
confound: stability follows answer consistency only. CW are consistent by definition and FA are a mix,
          so a naive "CW vs all FA" comparison sees a basin effect that is pure selection. The real tests
          (CW vs consistent FA, consistency in the base) must not survive, nor may anything else.
"""

import pytest

from sgta.config import FitCfg
from sgta.fit import fit_run
from sgta.hypotheses import run_all
from sgta.synthetic import make_interventions, make_run


def _run(tmp_path, mode, markov, faithful, full=False, seed=3):
    d = tmp_path / f"{mode}_{seed}"
    make_run(str(d), mode=mode, markov=markov, N=2400, seed=seed)
    fit_run(str(d), FitCfg(ks=[4, 6, 8, 16], pca_dim=32), log=lambda *a: None)
    make_interventions(str(d), faithful=faithful, k=6)
    vs = run_all(str(d), only=["H0", "H1", "H1b", "H2", "H3b", "H4"], n_perm=60, log=lambda *a: None)
    return vs if full else {v.hid: v.verdict for v in vs}


@pytest.mark.slow
def test_planted_effects_are_found(tmp_path):
    v = _run(tmp_path, "planted", True, True)
    assert v == {
        "H0": "SURVIVES",
        "H1": "SURVIVES",
        "H1b": "SURVIVES",
        "H2": "SURVIVES",
        "H3b": "SURVIVES",
        "H4": "SURVIVES",
    }, v


@pytest.mark.slow
def test_null_produces_no_false_positives(tmp_path):
    v = _run(tmp_path, "null", False, False)
    assert all(x != "SURVIVES" for x in v.values()), v


@pytest.mark.slow
def test_consistency_confound_is_not_a_finding(tmp_path):
    vs = {v.hid: v for v in _run(tmp_path, "confound", False, False, full=True)}
    assert all(v.verdict != "SURVIVES" for v in vs.values()), {h: v.verdict for h, v in vs.items()}
    m = vs["H1"].metrics
    # the trap is real in this data: against ALL correct answers the basin score alone separates CW ...
    assert m["cw_all"]["auc_basin_only"] >= 0.6, m["cw_all"]
    # ... and against consistent correct answers it does not
    assert m["cw"]["auc_basin_only"] < 0.56, m["cw"]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["null", "confound"])
def test_no_false_positives_on_a_second_seed(tmp_path, mode):
    v = _run(tmp_path, mode, False, False, seed=17)
    assert all(x != "SURVIVES" for x in v.values()), v
