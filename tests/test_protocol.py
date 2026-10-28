"""The code must do what HYPOTHESES.md promises, and the protocol must not drift after it was sent.

* the SHA-256 of HYPOTHESES.md equals the one in HYPOTHESES.sha256 (the value sent to the supervisor);
* every threshold in the code appears, as a number, in the protocol section it governs;
* configs match the protocol (models, datasets, sample sizes, consistency threshold);
* fitting uses training rows only, and stacked scores never see test labels;
* the label audit sheet is written for every analysed run.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from sgta.config import FitCfg, SprintCfg
from sgta.hypotheses import THRESHOLDS as T
from sgta.hypotheses import _stack_score
from sgta.report import MIN_N_FOR_KILL, STRICT_P_DELONG, STRICT_P_LR

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "HYPOTHESES.md").read_text()


def _section(hid: str) -> str:
    m = re.search(rf"^## {re.escape(hid)}\..*?(?=^## |\Z)", PROTOCOL, flags=re.S | re.M)
    assert m, f"section {hid} not found in HYPOTHESES.md"
    return m.group(0)


# ------------------------------------------------------------------ protocol is pinned


def test_protocol_hash_is_the_one_that_was_sent():
    want = (ROOT / "HYPOTHESES.sha256").read_text().split()[0]
    got = hashlib.sha256((ROOT / "HYPOTHESES.md").read_bytes()).hexdigest()
    assert got == want, (
        "HYPOTHESES.md changed after its hash was fixed. If the change is intended, record it in "
        "the changelog, update HYPOTHESES.sha256 and send the new hash to the supervisor."
    )


# ------------------------------------------------------------------ thresholds: code == text

# Every pre-registered number must appear, as that number, in the section of the hypothesis it governs.
# Only the numbers are checked: the wording around them may change freely.
SECTION_OF = {
    "H0_internal_gain": "H0",
    "H0_cheang_gap": "H0",
    "H1_delta_auc": "H1",
    "H1_p_delong": "H1",
    "H1_p_lr": "H1",
    "H1_basin_only": "H1",
    "H1b_delta_auc": "H1b",
    "H1b_margin_only": "H1b",
    "H2_gap_markov": "H2",
    "H2_gap_kill": "H2",
    "H2_faithful": "H2",
    "H2_faithful_kill": "H2",
    "H2_inertia_p": "H2",
    "H3a_lead": "H3a",
    "H3b_p": "H3b",
    "H4_min_pairs": "H4",
    "H4_low_label_gain": "H4",
    "H5_rho": "H5",
    "H7_norm_auc": "H7",
    "H7_norm_gain": "H7",
}


def _numbers(text: str) -> set[float]:
    # a number is digits with an optional fraction; a trailing "." that ends a sentence is not part of it
    return {float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])(?!\.\d)", text)}


def _spellings(x) -> set[float]:
    """A threshold may be written as a fraction (0.10) or, for shares of depth, as a percentage (10)."""
    return {float(x), round(float(x) * 100, 6)}


def test_every_threshold_has_the_same_number_in_its_protocol_section():
    assert set(SECTION_OF) == set(T), "a threshold was added or renamed: map it to its section"
    missing = []
    for key, hid in SECTION_OF.items():
        if not _spellings(T[key]) & _numbers(_section(hid)):
            missing.append((key, T[key], hid))
    assert not missing, missing


def test_decision_rule_numbers_in_text():
    rules = _numbers(PROTOCOL)
    assert {STRICT_P_DELONG, STRICT_P_LR, float(MIN_N_FOR_KILL)} <= rules


def test_perturbation_design_numbers_in_text():
    pc = SprintCfg().perturb
    h1, h1b = _numbers(_section("H1")), _numbers(_section("H1b"))
    assert {round(f * 100, 6) for f in pc.layer_fracs} <= h1, "layer depths (percent) missing from H1"
    assert set(map(float, pc.kappas)) <= h1, "perturbation strengths missing from H1"
    assert float(pc.n_random) in h1, "number of random directions missing from H1"
    assert {min(pc.margin_grid), max(pc.margin_grid)} <= h1b, "margin grid edges missing from H1b"


# ------------------------------------------------------------------ configs

CONFIGS = sorted(p for p in (ROOT / "configs").glob("*.json") if p.name != "monitor.json")


def test_configs_share_the_protocol_design():
    """Every sprint config runs the same data, sample sizes and class definition: only the model differs."""
    cfgs = {p.name: SprintCfg.load(str(p)) for p in CONFIGS}
    ref = SprintCfg()
    assert len({c.run_dir for c in cfgs.values()}) == len(cfgs), "two configs write to the same run_dir"
    for name, c in cfgs.items():
        assert c.perturb.n_examples == ref.perturb.n_examples, name
        assert c.gen.n_samples == ref.gen.n_samples, name
        assert c.gen.consistency_threshold == ref.gen.consistency_threshold, name
        assert c.data.datasets == ref.data.datasets, name
        assert c.data.test_fraction == ref.data.test_fraction, name
    protocol = _numbers(PROTOCOL)
    assert {float(ref.gen.n_samples), ref.gen.consistency_threshold} <= protocol
    base = [c for c in cfgs.values() if not c.model.lower().endswith("instruct")]
    assert base and all(c.data.prompt_style == "qa" for c in base), "a base model needs the few-shot prompt"


# ------------------------------------------------------------------ no leakage


def test_stacked_score_never_sees_test_labels():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal((n, 6))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(int)
    tr = np.arange(n) < 300
    te = ~tr
    a = _stack_score(X, y, tr, te, 0)
    y2 = y.copy()
    y2[te] = 1 - y2[te]
    b = _stack_score(X, y2, tr, te, 0)
    assert np.allclose(a, b, equal_nan=True)


@pytest.mark.slow
def test_fit_uses_training_rows_only(tmp_path):
    from sgta.fit import fit_run
    from sgta.store import Run
    from sgta.synthetic import make_run

    d = tmp_path / "syn"
    make_run(str(d), mode="planted", N=900, seed=5)
    cfg = FitCfg(ks=[4, 6], pca_dim=16)
    fit_run(str(d), cfg, log=lambda *a: None)
    run = Run(d)
    st1 = {k: v.copy() for k, v in run.fit_stats().items()}
    c1 = Run(d).partition("caps", 6)["centers"].copy()
    te = ~run.is_train
    H = np.load(d / "states.npy")
    H[te] = (H[te].astype(np.float32) * -3 + 7).astype(H.dtype)  # wreck every test row
    np.save(d / "states.npy", H)
    fit_run(str(d), cfg, log=lambda *a: None)
    run2 = Run(d)
    for k in st1:
        assert np.allclose(st1[k], run2.fit_stats()[k]), k
    assert np.allclose(c1, run2.partition("caps", 6)["centers"])


# ------------------------------------------------------------------ label audit sheet


def test_audit_sheet_has_twenty_rows_per_class(tmp_path):
    from sgta.audit import audit_sheet
    from sgta.synthetic import make_run

    d = tmp_path / "syn"
    make_run(str(d), mode="planted", N=600, seed=1)
    out = audit_sheet(str(d))
    text = out.read_text()
    for cls in ("CW", "FA", "UW"):
        block = text.split(f"## {cls}:")[1].split("## ")[0]
        assert sum(1 for line in block.splitlines() if re.match(r"^\| \d+ \|", line)) == 20, cls
