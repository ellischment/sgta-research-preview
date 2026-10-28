import numpy as np
import pytest

from sgta.geometry import rotate, toward, unit, random_tangent
from sgta.info import bh_fdr, cmi_test, entropy, interaction_info, mi, synergy_test
from sgta.markov import committor, transition
from sgta.stats import auroc, delong, stratified_auroc
from sgta.sst import fisher_id


def test_entropy_uniform():
    x = np.repeat(np.arange(4), 1000)
    assert abs(entropy(x, mm=False) - np.log(4)) < 1e-9


def test_mi_independent_near_zero_and_dependent_positive():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 5000)
    b = rng.integers(0, 4, 5000)
    assert abs(mi(a, b)) < 0.01
    assert mi(a, a) > 1.2


def test_cmi_markov_chain_is_zero_but_second_order_is_not():
    rng = np.random.default_rng(1)
    n = 6000
    x0 = rng.integers(0, 3, n)
    x1 = (x0 + (rng.random(n) < 0.3)) % 3
    x2 = (x1 + (rng.random(n) < 0.3)) % 3  # Markov: x2 depends on x1 only
    obs, null, p = cmi_test(x0, x2, x1, n_perm=100)
    assert p > 0.01 and obs - null < 0.01
    y2 = np.where(rng.random(n) < 0.5, x0, x1)  # second order
    obs, null, p = cmi_test(x0, y2, x1, n_perm=100)
    assert p < 0.02 and obs - null > 0.05


def test_xor_is_pure_synergy():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 2, 4000)
    b = rng.integers(0, 2, 4000)
    e = a ^ b
    assert interaction_info(e, a, b) > 0.6
    _, _, p = synergy_test(e, a, b, n_perm=100)
    assert p < 0.02
    _, _, p0 = synergy_test(rng.integers(0, 2, 4000), a, b, n_perm=100)
    assert p0 > 0.01


def test_bh():
    p = np.array([0.001, 0.002, 0.5, 0.9])
    assert bh_fdr(p).tolist() == [True, True, False, False]


def test_transition_rows_sum_to_one():
    P = transition(np.array([0, 0, 1]), np.array([1, 1, 0]), 2, 2)
    assert np.allclose(P.sum(1), 1)
    assert P[0, 1] > P[0, 0]


def test_committor_matches_direct_on_markov_chain():
    rng = np.random.default_rng(3)
    n, L, k = 8000, 5, 3
    T = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    c = np.zeros((n, L), int)
    c[:, 0] = rng.integers(0, k, n)
    for l in range(1, L):
        c[:, l] = [rng.choice(k, p=T[x]) for x in c[:, l - 1]]
    err = (rng.random(n) < np.where(c[:, -1] == 0, 0.7, 0.1)).astype(int)
    tr = rng.random(n) < 0.6
    Q, E = committor(c, err, tr, list(range(L)), k)
    te = ~tr
    for i in range(L):
        assert abs(auroc(err[te], Q[te, i]) - auroc(err[te], E[te, i])) < 0.03


def test_delong_detects_better_score():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 1500)
    good = y + rng.normal(0, 0.5, 1500)
    bad = y + rng.normal(0, 2.0, 1500)
    d, z, p = delong(y, good, bad)
    assert d > 0 and p < 1e-6


def test_stratified_auroc_removes_confounder():
    rng = np.random.default_rng(6)
    conf = rng.random(4000)
    y = (rng.random(4000) < conf).astype(int)
    s = conf + 0.01 * rng.normal(size=4000)  # pure proxy of the confounder
    assert auroc(y, s) > 0.75
    assert abs(stratified_auroc(y, s, conf, n_bins=10) - 0.5) < 0.08


def test_rotation_preserves_norm_and_angle():
    rng = np.random.default_rng(7)
    u = unit(rng.normal(size=(5, 32)))
    v = random_tangent(u, rng)
    w = rotate(u, v, 0.3)
    assert np.allclose(np.linalg.norm(w, axis=1), 1)
    assert np.allclose(np.arccos(np.clip((u * w).sum(1), -1, 1)), 0.3, atol=1e-6)
    t, ang = toward(u, unit(rng.normal(size=(5, 32))))
    assert np.allclose((t * u).sum(1), 0, atol=1e-9)


@pytest.mark.parametrize("d", [5, 20, 60])
def test_fisher_id_recovers_dimension(d):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(1500, d)) @ rng.normal(size=(d, 100))
    est = fisher_id(X)["id"]
    assert abs(est - d) / d < 0.15, est


def _verdict(hid, verdict, p_delong, p_lr, n_pos=200, n_neg=200):
    return {
        "hid": hid,
        "verdict": verdict,
        "criteria": [{"passed": True}, {"passed": True}],
        "metrics": {
            "cw": {"delta_auc": 0.03, "p_delong": p_delong, "p_lr": p_lr, "n_test_pos": n_pos, "n_test_neg": n_neg}
        },
    }


def test_central_bet_needs_the_strict_p_values_not_just_a_raw_survives():
    from sgta.report import decision_rules

    weak = _verdict("H1", "SURVIVES", p_delong=0.03, p_lr=0.001)  # passes H1's own 0.05, not the strict 0.025
    assert decision_rules([weak, _verdict("H1b", "INCONCLUSIVE", 0.4, 0.3)])["central_bet"] == "NOT CONFIRMED"


def test_central_bet_is_confirmed_by_one_strict_survivor():
    from sgta.report import decision_rules

    strong = _verdict("H1", "SURVIVES", p_delong=0.01, p_lr=0.001)
    assert decision_rules([strong, _verdict("H1b", "KILLED", 0.9, 0.9)])["central_bet"] == "CONFIRMED"


def test_killed_on_a_small_test_set_is_read_as_inconclusive():
    from sgta.report import decision_rules

    small = _verdict("H1", "KILLED", 0.9, 0.9, n_pos=60, n_neg=300)
    dr = decision_rules([small, _verdict("H1b", "KILLED", 0.9, 0.9)])
    assert dr["killed_read_as_inconclusive"] == ["H1"] and dr["central_bet"] == "NOT CONFIRMED"


def test_no_verdicts_means_not_measured():
    from sgta.report import decision_rules

    assert decision_rules([])["central_bet"] == "NOT MEASURED"


def test_epgs_comparison_random_scores_lose_to_planted_basins(tmp_path):
    import json
    from sgta.config import FitCfg
    from sgta.epgs import epgs_compare
    from sgta.fit import fit_run
    from sgta.synthetic import make_run

    d = tmp_path / "syn"
    make_run(str(d), mode="planted", N=1500, seed=4)
    fit_run(str(d), FitCfg(ks=[4, 8], pca_dim=16), log=lambda *a: None)
    pt = np.load(d / "perturb.npz")
    n = len(pt["idx"])
    rng = np.random.default_rng(0)
    np.savez(
        d / "epgs.npz",
        idx=pt["idx"],
        score=rng.lognormal(0, 1, n).astype(np.float32),  # EPGS that knows nothing
        gnorm=np.ones(n, np.float32),
        cos=np.ones(n, np.float32),
        sigma=np.float32(0.01),
        n_noise=np.int32(2),
    )
    assert epgs_compare(str(d)).exists()
    cw = json.loads((d / "results" / "epgs_comparison.json").read_text())["tasks"]["cw"]
    assert abs(cw["auc_epgs_stacked"] - 0.5) < 0.12
    gaps = [v for k, v in cw.items() if "minus EPGS" in k]  # H1 and H1b head-to-head, whatever the column is called
    assert gaps and min(gaps) > 0.2
