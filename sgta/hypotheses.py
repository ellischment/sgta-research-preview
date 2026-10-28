"""The hypothesis tournament.

Every hypothesis has a pre-registered claim, numeric pass criteria and an explicit kill rule.
Thresholds live in THRESHOLDS so they are fixed before anyone looks at real data. Changing them
after seeing results must be logged in results/threshold_changes.md (the report reminds you).

Verdicts:
    SURVIVES      all primary criteria pass
    KILLED        the kill rule fires
    INCONCLUSIVE  neither (usually: low power, missing stage, or mixed evidence)
    SKIPPED       required data not present

Map of this module:
    THRESHOLDS, Verdict         pre-registered numbers and the result record
    Ctx                         everything shared between hypotheses: labels, tasks (cw / cw_all / uw / err),
                                pointwise probes, TriLens-style score, the spline base of every increment test
    _stack_score, _increment    one feature block -> one score -> nested test of base vs base + score
    h0_gate .. h7_angle_norm    one function per hypothesis, each returns a Verdict
    run_all                     runs them in order and writes results/verdicts.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Callable

import numpy as np
from scipy import stats as sst

from .info import bh_fdr, synergy_test
from .markov import committor, enriched_paths, markov_gap, markov_power, transition
from .sst import fisher_fewshot_auc, fisher_id, whiten_sphere
from .stats import (
    auroc,
    auroc_oriented,
    bootstrap_ci,
    delong,
    fit_score,
    flex_basis,
    gbdt,
    logreg,
    lr_increment,
    make_probe,
    nested_test,
    oof_scores,
    spearman,
    stratified_auroc,
)
from .store import Run

THRESHOLDS = {
    "H0_internal_gain": 0.02,  # pointwise-internal AUROC minus output-confidence AUROC
    "H0_cheang_gap": 0.05,  # AUROC(UW vs FA) - AUROC(CW vs consistent FA) to call the hard regime replicated
    "H1_delta_auc": 0.02,  # basin profile increment over the full base, CW vs consistent FA
    "H1_p_delong": 0.05,
    "H1_p_lr": 0.01,
    "H1_basin_only": 0.60,
    "H1b_delta_auc": 0.02,  # gradient-margin profile increment over the same base, CW vs consistent FA
    "H1b_margin_only": 0.60,
    "H2_gap_markov": 0.10,  # median normalised Markov gap
    "H2_gap_kill": 0.30,
    "H2_faithful": 0.70,  # share of interventions where P(tgt,.) explains outcome better than P(src,.)
    "H2_faithful_kill": 0.55,
    "H2_inertia_p": 0.05,  # chain beats the best inertia baseline at the deepest horizon (McNemar)
    "H3a_lead": 0.10,  # decidability precedes content by >= 10% of depth
    "H3b_p": 0.01,
    "H4_min_pairs": 3,
    "H4_low_label_gain": 0.02,
    "H5_rho": 0.5,
    "H7_norm_auc": 0.60,
    "H7_norm_gain": 0.01,
}


@dataclass
class Verdict:
    hid: str
    title: str
    claim: str
    verdict: str
    criteria: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    meaning: str = ""
    seconds: float = 0.0

    def crit(self, name: str, passed: bool | None, value, target: str):
        self.criteria.append(
            {"name": name, "passed": None if passed is None else bool(passed), "value": _num(value), "target": target}
        )


def _num(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    return _num(o)


# ============================================================================ context


class Ctx:
    """Shared, lazily computed quantities. One per run."""

    def __init__(
        self, run: Run, k_disc: int = 16, k_syn: int = 8, n_disc_layers: int = 8, seed: int = 0, n_perm: int = 200
    ):
        self.run = run
        self.seed = seed
        self.k_disc, self.k_syn, self.n_perm = k_disc, k_syn, n_perm
        L = run.n_layers
        self.L = L
        self.layers = list(range(1, L + 1))  # skip embeddings
        self.disc_layers = sorted(set(np.linspace(1, L, n_disc_layers).round().astype(int).tolist()))
        m = run.meta
        self.tr = run.is_train
        self.te = ~self.tr
        self.err = run.y_err
        self.et = run.etype
        self.cons = run.consistent  # CW are consistent by definition, FA are a mix
        self.fac = (self.et == "FA") & self.cons  # correct AND consistent: the fair control for CW
        self.cons_value = m["consistency"].fillna(0).to_numpy(float)
        pf = np.clip(m["p_first"].to_numpy(float), 1e-6, 1 - 1e-6)
        self.F0 = np.c_[np.log(pf / (1 - pf)), m["mean_logprob"].to_numpy(float), m["ent_first"].to_numpy(float)]
        self.cache: dict[str, object] = {}

    # ---------------- tasks
    def task(self, name: str):
        """Returns (mask over examples, y) for a task.

        'cw'     : CW vs CONSISTENT FA. The main (hard-regime) comparison. CW are selected for consistency, so
                   comparing them with all FA would let any stability measure win through the class definition.
        'cw_all' : CW vs all FA. The old comparison, kept for reference only (selection-confounded).
        'uw'     : UW vs all FA.
        'err'    : any error vs FA.
        """
        fa = self.et == "FA"
        if name == "err":
            return np.ones(len(self.et), bool), self.err
        if name == "cw":
            sel = self.et == "CW"
            return self.fac | sel, sel.astype(int)
        if name == "cw_all":
            sel = self.et == "CW"
            return fa | sel, sel.astype(int)
        sel = self.et == name.upper()
        return fa | sel, sel.astype(int)

    # ---------------- pointwise probes (the strongest "where" baseline)
    def pointwise(self, task: str):
        """Per-layer linear probes + linear stack over layers.

        Returns dict with per-layer test AUROC, train-OOF and test stack scores (full-length arrays,
        NaN outside the task mask).
        """
        key = f"pw_{task}"
        if key in self.cache:
            return self.cache[key]
        mask, y = self.task(task)
        tr = self.tr & mask
        te = self.te & mask
        N = len(y)
        oof = np.full((N, len(self.layers)), np.nan)
        tst = np.full((N, len(self.layers)), np.nan)
        per_auc_te, per_auc_tr = [], []
        for j, l in enumerate(self.layers):
            X = self.run.layer_s(l)
            d = X.shape[1]
            mk = lambda d=d: make_probe(d, pca=64, seed=self.seed)
            oof[tr, j] = oof_scores(X[tr], y[tr], mk, seed=self.seed)
            tst[te, j] = fit_score(X[tr], y[tr], X[te], mk)
            per_auc_tr.append(auroc(y[tr], oof[tr, j]))
            per_auc_te.append(auroc(y[te], tst[te, j]))
        # stack: linear model over per-layer scores (logit space)
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        Ztr, Zte = lg(oof[tr]), lg(tst[te])
        stack_tr = oof_scores(Ztr, y[tr], logreg, seed=self.seed)
        stack_te = fit_score(Ztr, y[tr], Zte, logreg)
        s_tr = np.full(N, np.nan)
        s_te = np.full(N, np.nan)
        s_tr[tr] = stack_tr
        s_te[te] = stack_te
        best = int(np.nanargmax(per_auc_tr))
        out = {
            "layer_auc_test": per_auc_te,
            "layer_auc_trainoof": per_auc_tr,
            "best_layer": self.layers[best],
            "stack_train_oof": s_tr,
            "stack_test": s_te,
            "stack_auc_test": auroc(y[te], stack_te),
            "best_auc_test": per_auc_te[best],
        }
        self.cache[key] = out
        return out

    def pointwise_score(self, task: str) -> np.ndarray:
        """Train-OOF on train, test predictions on test: safe to use as a feature."""
        pw = self.pointwise(task)
        s = np.where(self.tr, pw["stack_train_oof"], pw["stack_test"])
        return s

    def codes(self, kind: str, k: int) -> np.ndarray:
        return self.run.partition(kind, k)["assign"].astype(int)

    # ---------------- TriLens-style baseline (strongest white-box competitor without training on states)
    def trilens_features(self) -> np.ndarray | None:
        lens = self.run.lens
        parts = [lens["entropy"][:, 1:]]
        for key in ("ent_attn", "ent_mlp"):
            if key in lens and np.isfinite(lens[key][:, 1:]).all():
                parts.append(lens[key][:, 1:])
        X = np.concatenate(parts, 1).astype(float)
        return X if np.isfinite(X).all() else None

    def trilens_score(self, task: str) -> np.ndarray:
        """OOF logistic score over per-layer lens entropies of residual, attention and MLP outputs.

        Falls back to the residual lens alone when component lenses were not captured. Train rows get
        out-of-fold scores, test rows the fit on all train rows: safe to use as a feature.
        """
        key = f"tl_{task}"
        if key in self.cache:
            return self.cache[key]
        X = self.trilens_features()
        mask, y = self.task(task)
        s = np.full(len(y), np.nan)
        if X is not None:
            s = _stack_score(X, y, self.tr & mask, self.te & mask, self.seed)
        self.cache[key] = s
        return s

    def margin_out(self) -> np.ndarray | None:
        pt = self.run.perturb()
        if pt is None or "margin_out" not in pt:
            return None
        m = np.full(len(self.err), np.nan)
        m[pt["idx"].astype(int)] = np.log(np.clip(pt["margin_out"], 1e-3, None))
        return m

    def base(self, task: str) -> np.ndarray:
        """Everything a detector already has: output confidence, sample consistency, pointwise probe stack,
        TriLens-style lens score and (when measured) the output logit margin. H1 and H1b must add to ALL of it.
        Consistency is in the base so that a stability measure cannot win by re-measuring it. Every column
        enters through a spline basis (stats.flex_basis, knots from training rows), so that it cannot win by
        re-measuring any of them through a non-linear link either."""
        key = f"base_{task}"
        if key in self.cache:
            return self.cache[key]
        mask = self.task(task)[0]
        cols = [self.F0, self.cons_value[:, None], self.pointwise_score(task)[:, None]]
        tl = self.trilens_score(task)
        if np.isfinite(tl[mask]).any():
            cols.append(tl[:, None])
        mo = self.margin_out()
        if mo is not None:
            cols.append(mo[:, None])
        out = flex_basis(np.concatenate(cols, 1), self.tr & mask)
        self.cache[key] = out
        return out


def _stack_score(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> np.ndarray:
    """Collapse a feature block into one score: OOF on train rows, train-fit on test rows, NaN elsewhere.

    The nested test then compares base vs base + this score with one degree of freedom, so a rich
    feature block cannot win by sheer column count.
    """
    s = np.full(len(y), np.nan)
    mk = lambda: make_probe(X.shape[1], pca=None, seed=seed)
    s[tr] = oof_scores(X[tr], y[tr], mk, seed=seed)
    s[te] = fit_score(X[tr], y[tr], X[te], mk)
    return s


def _kidx(kappas, target: float) -> int:
    """Index of the perturbation strength closest to `target` (log scale)."""
    k = np.asarray(kappas, float)
    return int(np.argmin(np.abs(np.log(k) - np.log(target))))


def _escape_kappa(flip: np.ndarray, kappas: np.ndarray, level: float = 0.5) -> np.ndarray:
    """APEX-style escape strength: smallest kappa at which the flip rate reaches `level`.

    flip [M, nl, nk]. Linear interpolation in log kappa; censored at 2 x max kappa.
    """
    lk = np.log(np.asarray(kappas, float))
    M, nl, nk = flip.shape
    out = np.full((M, nl), np.log(2 * np.max(kappas)))
    for i in range(nk):
        hit = (flip[:, :, i] >= level) & (out == np.log(2 * np.max(kappas)))
        if i == 0:
            out[hit] = lk[0]
            continue
        f0, f1 = flip[:, :, i - 1], flip[:, :, i]
        w = np.clip((level - f0) / np.clip(f1 - f0, 1e-9, None), 0, 1)
        out[hit] = (lk[i - 1] + w * (lk[i] - lk[i - 1]))[hit]
    return out


# ============================================================================ H0


def h0_gate(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H0",
        "Gate: internal signal exists and the hard regime replicates",
        "Hidden states predict errors beyond output confidence, and errors split into a regime "
        "visible to pointwise geometry (UW) and one that is not (CW), as Cheang et al. (Findings ACL 2026) report.",
        "INCONCLUSIVE",
    )
    tr, te, y = ctx.tr, ctx.te, ctx.err
    F0f = flex_basis(ctx.F0, tr)  # confidence through a spline basis: the fair baseline
    s0 = fit_score(F0f[tr], y[tr], F0f[te], logreg)
    a0 = auroc(y[te], s0)
    pw = ctx.pointwise("err")
    a1 = pw["stack_auc_test"]
    d, _, p = delong(y[te], pw["stack_test"][te], s0)
    v.metrics.update(
        {
            "auc_confidence_F0": a0,
            "auc_pointwise_stack": a1,
            "auc_best_layer": pw["best_auc_test"],
            "best_layer": pw["best_layer"],
            "layer_auc_test": [_num(x) for x in pw["layer_auc_test"]],
            "ci_pointwise": bootstrap_ci(y[te], pw["stack_test"][te]),
            "class_counts": {
                **{k: int((ctx.et == k).sum()) for k in ("FA", "CW", "UW")},
                "FA_consistent": int(ctx.fac.sum()),
            },
        }
    )
    v.crit(
        "internal beats confidence",
        d >= THRESHOLDS["H0_internal_gain"] and p < 0.05,
        d,
        f">= {THRESHOLDS['H0_internal_gain']} AUROC, DeLong p<0.05 (p={p:.3g})",
    )
    tl = ctx.trilens_score("err")
    if np.isfinite(tl[te]).all():
        dt, _, pt_ = delong(y[te], pw["stack_test"][te], tl[te])
        v.metrics.update(
            {
                "auc_trilens_style": auroc(y[te], tl[te]),
                "trilens_components": "residual+attn+mlp"
                if ctx.trilens_features().shape[1] > ctx.L
                else "residual only",
                "pointwise_minus_trilens": _num(dt),
                "p_pointwise_vs_trilens": _num(pt_),
            }
        )
    res = {}
    for t in ("cw", "uw"):
        if (ctx.et == t.upper()).sum() < 30:
            res[t] = float("nan")
            continue
        pwt = ctx.pointwise(t)
        res[t] = pwt["stack_auc_test"]
    gap = res["uw"] - res["cw"] if np.isfinite(res["uw"]) and np.isfinite(res["cw"]) else float("nan")
    v.metrics.update({"auc_FAconsistent_vs_CW_dedicated": res["cw"], "auc_FA_vs_UW_dedicated": res["uw"]})
    v.crit(
        "hard regime replicates (UW vs FA visible, CW vs consistent FA less visible)",
        np.isfinite(gap) and gap >= THRESHOLDS["H0_cheang_gap"],
        gap,
        f"AUROC(UW vs FA) - AUROC(CW vs consistent FA) >= {THRESHOLDS['H0_cheang_gap']}",
    )
    ok = [c["passed"] for c in v.criteria]
    v.verdict = "SURVIVES" if all(ok) else ("KILLED" if not ok[0] else "INCONCLUSIVE")
    v.meaning = (
        "If the first criterion fails, the setup has no internal signal: change model/task before "
        "anything else. If only the second fails, our CW proxy does not isolate associated "
        "hallucinations; H1 is then tested on 'all errors' and the AH framing must be dropped."
    )
    return v


# ============================================================================ H1


def _basin_features(pt: dict[str, np.ndarray]) -> np.ndarray:
    """Basin block: flip and log-prob responses along random, structured and probe directions, plus
    the APEX-style escape strength. Collapsed into one stacked score before any test."""
    M = len(pt["idx"])
    parts = [pt["flip_rand"].reshape(M, -1), pt["flip_struct"].reshape(M, -1), pt["dlogp_rand"].reshape(M, -1)]
    for key in ("dlogp_struct", "flip_probe", "dlogp_probe"):
        if key in pt:
            parts.append(pt[key].reshape(M, -1))
    parts.append(_escape_kappa(pt["flip_rand"], pt["kappas"]))
    return np.concatenate(parts, 1).astype(float)


def _margin_features(pt: dict[str, np.ndarray]) -> np.ndarray:
    """H1b block: per layer log measured flip angle, log linear estimate, log sensitivity."""
    return np.concatenate(
        [
            np.log(np.clip(pt["margin_theta"], 1e-3, None)),
            np.log(np.clip(pt["margin_lin"], 1e-3, 1e3)),
            np.log(np.clip(pt["margin_sens"], 1e-6, None)),
        ],
        1,
    ).astype(float)


def _increment(ctx: Ctx, block_full: np.ndarray, have: np.ndarray, task: str):
    """Nested test of base vs base + stacked(block) on one task, restricted to perturbed examples."""
    mask, y = ctx.task(task)
    m = mask & have
    tr, te = ctx.tr & m, ctx.te & m
    if min(y[tr].sum(), (1 - y[tr]).sum(), y[te].sum(), (1 - y[te]).sum()) < 15:
        return None
    sc = _stack_score(np.nan_to_num(block_full), y, tr, te, ctx.seed)
    Xb = ctx.base(task)
    Xf = np.c_[Xb, sc]
    nt = nested_test(Xb[tr], Xf[tr], y[tr], Xb[te], Xf[te], y[te])
    return {
        "nt": nt,
        "score": sc,
        "auc_alone": auroc(y[te], sc[te]),
        "y": y,
        "tr": tr,
        "te": te,
        "base_cols": int(Xb.shape[1]),
        "n_test_pos": int(y[te].sum()),
        "n_test_neg": int((1 - y[te]).sum()),
    }


def h1_basins(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H1",
        "Where vs whether: basin depth sees what geometry misses",
        "Confident-consistent errors (CW) occupy the same place as correct answers but sit in "
        "basins of different depth: the layer-resolved response of the next-token decision to "
        "small norm-preserving rotations of the residual stream separates CW from consistent FA (correct "
        "answers that repeat across samples just as CW do) beyond output confidence, the output margin, "
        "the pointwise probe stack and the TriLens-style score. Direction is two-sided.",
        "SKIPPED",
    )
    pt = ctx.run.perturb()
    if pt is None:
        v.meaning = "Run the GPU perturbation stage."
        return v
    idx = pt["idx"].astype(int)
    B = _basin_features(pt)
    kmid = _kidx(pt["kappas"], 0.5)
    depth_score = pt["flip_rand"][:, :, kmid].mean(1)  # parameter-free "basin shallowness"
    full = np.full((len(ctx.err), B.shape[1]), np.nan)
    full[idx] = B
    dsc = np.full(len(ctx.err), np.nan)
    dsc[idx] = depth_score
    have = np.zeros(len(ctx.err), bool)
    have[idx] = True
    v.metrics["flip_by_class"] = {
        **{k: _num(np.nanmean(dsc[have & (ctx.et == k)])) for k in ("FA", "CW", "UW")},
        "FA_consistent": _num(np.nanmean(dsc[have & ctx.fac])),
    }
    groups = {
        "FA": ctx.et[idx] == "FA",
        "FA_consistent": ctx.fac[idx],
        "CW": ctx.et[idx] == "CW",
        "UW": ctx.et[idx] == "UW",
    }
    v.metrics["flip_profile_by_class"] = {
        k: [_num(x) for x in np.nanmean(pt["flip_rand"][:, :, kmid][g], 0)] for k, g in groups.items() if g.any()
    }
    v.metrics["layers"] = [int(x) for x in pt["layers"]]
    v.metrics["kappa_used_for_profiles"] = float(pt["kappas"][kmid])
    v.metrics["mean_flip_rate_by_direction"] = {
        "random": _num(pt["flip_rand"].mean()),
        "structured": _num(pt["flip_struct"].mean()),
        **({"probe": _num(pt["flip_probe"].mean())} if "flip_probe" in pt else {}),
    }
    primary = None
    v.metrics["note_cw_all"] = (
        "cw_all compares CW with ALL correct answers; CW are selected for consistency and FA are "
        "not, so a stability measure can win there through the class definition. Reference only."
    )
    for task in ("cw", "cw_all", "err"):
        r = _increment(ctx, full, have, task)
        if r is None:
            v.metrics[f"{task}_note"] = "too few examples"
            continue
        nt, a_basin, y, te = r["nt"], r["auc_alone"], r["y"], r["te"]
        a_depth = auroc_oriented(y[te], dsc[te])
        a_depth_strat = stratified_auroc(y[te], dsc[te] * (1 if auroc(y[te], dsc[te]) >= 0.5 else -1), ctx.F0[te, 0])
        v.metrics[task] = {
            **{k: _num(x) for k, x in nt.items()},
            "auc_basin_only": _num(a_basin),
            "auc_depth_score_unsupervised": _num(a_depth),
            "auc_depth_score_confidence_matched": _num(a_depth_strat),
            "base_columns": r["base_cols"],
            "n_test": int(te.sum()),
            "n_test_pos": r["n_test_pos"],
            "n_test_neg": r["n_test_neg"],
        }
        if task == "cw":
            primary = (nt, a_basin)
            ctx.cache["basin_score_cw"] = r["score"]
    if primary is None:
        v.verdict = "INCONCLUSIVE"
        v.meaning = (
            "Not enough CW or consistent FA in the perturbed subset. Increase perturb.n_examples or the dataset size."
        )
        return v
    nt, a_basin = primary
    c1 = (
        nt["delta_auc"] >= THRESHOLDS["H1_delta_auc"]
        and nt["p_delong"] < THRESHOLDS["H1_p_delong"]
        and nt["p_lr"] < THRESHOLDS["H1_p_lr"]
    )
    v.crit(
        "basin profile adds over the full base (CW vs consistent FA)",
        c1,
        nt["delta_auc"],
        f"dAUC >= {THRESHOLDS['H1_delta_auc']}, DeLong p<{THRESHOLDS['H1_p_delong']}, LR p<{THRESHOLDS['H1_p_lr']}",
    )
    c2 = a_basin >= THRESHOLDS["H1_basin_only"]
    v.crit("basin profile alone separates CW from consistent FA", c2, a_basin, f">= {THRESHOLDS['H1_basin_only']}")
    if c1 and c2:
        v.verdict = "SURVIVES"
    elif (not np.isfinite(nt["delta_auc"]) or nt["delta_auc"] < 0.005 or nt["p_delong"] > 0.2) and a_basin < 0.55:
        v.verdict = "KILLED"
    else:
        v.verdict = "INCONCLUSIVE"
    v.meaning = (
        "SURVIVES: the paper's core. Geometry tells where the model is, dynamics tells whether it will "
        "stay there; associated hallucinations become detectable. KILLED: basin depth is just another "
        "view of confidence; report as a negative result that strengthens Cheang et al."
    )
    return v


# ============================================================================ H1b


def h1b_margin(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H1b",
        "Gradient margin: the cheapest way out of the basin",
        "The angle along the margin gradient that flips the next-token decision (measured on a grid "
        "of the local spacing, plus its linear estimate and the margin sensitivity) separates CW from "
        "consistent FA beyond output confidence, the output logit margin, the pointwise probe stack and the "
        "TriLens-style lens score. This is the worst-case (adversarial) counterpart of H1.",
        "SKIPPED",
    )
    pt = ctx.run.perturb()
    if pt is None or "margin_theta" not in pt:
        v.meaning = "Run the GPU perturbation stage with perturb.margin = true."
        return v
    idx = pt["idx"].astype(int)
    Bm = _margin_features(pt)
    full = np.full((len(ctx.err), Bm.shape[1]), np.nan)
    full[idx] = Bm
    have = np.zeros(len(ctx.err), bool)
    have[idx] = True
    et = ctx.et[idx]
    cens = float(pt["margin_censor"])
    v.metrics["layers"] = [int(x) for x in pt["layers"]]
    v.metrics["censored_share"] = _num((pt["margin_theta"] >= cens).mean())
    groups = {"FA": et == "FA", "FA_consistent": ctx.fac[idx], "CW": et == "CW", "UW": et == "UW"}
    v.metrics["median_flip_angle_by_class"] = {
        k: [_num(x) for x in np.median(pt["margin_theta"][g], 0)] for k, g in groups.items() if g.any()
    }
    v.metrics["median_linear_estimate_by_class"] = {
        k: [_num(x) for x in np.median(pt["margin_lin"][g], 0)] for k, g in groups.items() if g.any()
    }
    primary = None
    for task in ("cw", "cw_all", "err"):
        r = _increment(ctx, full, have, task)
        if r is None:
            v.metrics[f"{task}_note"] = "too few examples"
            continue
        nt, y, te = r["nt"], r["y"], r["te"]
        row = {
            **{k: _num(x) for k, x in nt.items()},
            "auc_margin_only": _num(r["auc_alone"]),
            "n_test": int(te.sum()),
            "n_test_pos": r["n_test_pos"],
            "n_test_neg": r["n_test_neg"],
        }
        bs = ctx.cache.get("basin_score_cw")
        if task == "cw" and bs is not None and np.isfinite(bs[te]).all():
            d, _, p = delong(y[te], r["score"][te], bs[te])
            row.update({"margin_minus_basin_auc": _num(d), "p_margin_vs_basin": _num(p)})
        v.metrics[task] = row
        if task == "cw":
            primary = (nt, r["auc_alone"])
    if primary is None:
        v.verdict = "INCONCLUSIVE"
        v.meaning = "Not enough CW or consistent FA in the perturbed subset."
        return v
    nt, a_m = primary
    c1 = (
        nt["delta_auc"] >= THRESHOLDS["H1b_delta_auc"]
        and nt["p_delong"] < THRESHOLDS["H1_p_delong"]
        and nt["p_lr"] < THRESHOLDS["H1_p_lr"]
    )
    v.crit(
        "margin profile adds over the full base (CW vs consistent FA)",
        c1,
        nt["delta_auc"],
        f"dAUC >= {THRESHOLDS['H1b_delta_auc']}, DeLong p<{THRESHOLDS['H1_p_delong']}, LR p<{THRESHOLDS['H1_p_lr']}",
    )
    c2 = a_m >= THRESHOLDS["H1b_margin_only"]
    v.crit("margin profile alone separates CW from consistent FA", c2, a_m, f">= {THRESHOLDS['H1b_margin_only']}")
    if c1 and c2:
        v.verdict = "SURVIVES"
    elif (not np.isfinite(nt["delta_auc"]) or nt["delta_auc"] < 0.005 or nt["p_delong"] > 0.2) and a_m < 0.55:
        v.verdict = "KILLED"
    else:
        v.verdict = "INCONCLUSIVE"
    v.meaning = (
        "H1 asks how a typical small push affects the decision; H1b asks how small the WORST push is. "
        "If H1 dies on a floor effect (random directions miss the decision subspace) but H1b survives, "
        "the basin story stands in its adversarial form. If both die, depth of the basin is a view of "
        "confidence and the negative result is the paper."
    )
    return v


def _chain_power(codes: np.ndarray, tr: np.ndarray, l: int, s: int, k: int) -> np.ndarray:
    P = np.eye(k)
    for i in range(s):
        P = P @ transition(codes[tr, l + i], codes[tr, l + i + 1], k, k)
    return P


def _inertia_test(iv, codes: np.ndarray, tr: np.ndarray, k: int):
    """Does the chain predict where a moved state goes better than inertia?

    For every horizon s: top-1 accuracy of the chain from the TARGET region (P_l ... P_{l+s-1}),
    of the static baseline (moved state read at l+s) and of the delta baseline (moved state plus the
    original updates). Exact McNemar (binomial on discordant pairs) of chain vs the better baseline.
    """
    steps = sorted(int(c.split("_s")[1]) for c in iv.columns if c.startswith("next_obs_s"))
    out = {}
    for s in steps:
        acc_c, acc_st, acc_dl = [], [], []
        for l, g in iv.groupby("layer"):
            g = g[(g[f"next_obs_s{s}"] >= 0) & (g["src"] != g["tgt"])]
            if not len(g) or l + s >= codes.shape[1]:
                continue
            P = _chain_power(codes, tr, int(l), s, k)
            o = g[f"next_obs_s{s}"].to_numpy(int)
            acc_c.append(P[g["tgt"].to_numpy(int)].argmax(1) == o)
            acc_st.append(g[f"inertia_static_s{s}"].to_numpy(int) == o)
            acc_dl.append(g[f"inertia_delta_s{s}"].to_numpy(int) == o)
        if not acc_c:
            continue
        c, a, d = (np.concatenate(x) for x in (acc_c, acc_st, acc_dl))
        best_name, b = ("delta", d) if d.mean() >= a.mean() else ("static", a)
        n10, n01 = int((c & ~b).sum()), int((~c & b).sum())
        p = sst.binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue if n10 + n01 > 0 else 1.0
        out[s] = {
            "n": int(len(c)),
            "chain": _num(c.mean()),
            "static": _num(a.mean()),
            "delta": _num(d.mean()),
            "best_baseline": best_name,
            "chain_only_right": n10,
            "baseline_only_right": n01,
            "p_mcnemar": _num(p),
        }
    return out


# ============================================================================ H2


def h2_msm(ctx: Ctx, ks=None) -> Verdict:
    v = Verdict(
        "H2",
        "Depth is a Markov process over spherical regions",
        "Coarse-grained on per-layer spherical partitions, the layer-to-layer map is approximately "
        "Markov (the past adds little beyond the present), angular partitions are more Markov than "
        "Euclidean ones, and region-swap interventions move the next layer as the chain predicts.",
        "INCONCLUSIVE",
    )
    run = ctx.run
    ks = ks or sorted(int(p.stem.split("_K")[1]) for p in (run.path / "fit").glob("part_caps_K*.npz"))
    mid = [l for l in ctx.layers if 2 <= l <= ctx.L - 1]
    curves = {}
    for kind in ("caps", "euclid", "random"):
        for k in ks:
            codes = ctx.codes(kind, k)
            g = markov_gap(codes, mid, stride=1, n_perm=max(50, ctx.n_perm // 2), seed=ctx.seed)
            curves[f"{kind}_K{k}"] = g
    med = {key: float(np.nanmedian([r["gap_norm"] for r in g])) for key, g in curves.items()}
    v.metrics["median_gap_norm"] = {k: _num(x) for k, x in med.items()}
    v.metrics["gap_by_layer_caps_K16"] = [_num(r["gap_norm"]) for r in curves.get("caps_K16", [])]
    # power: can the test at this K see an injected 15% second-order dependence?
    power = {k: markov_power(ctx.codes("caps", k), mid[:: max(1, len(mid) // 8)], k, seed=ctx.seed) for k in ks}
    powered = [k for k in ks if k <= 32 and np.isfinite(power[k]) and power[k] >= THRESHOLDS["H2_gap_markov"]]
    v.metrics["power_injected_gap"] = {f"K{k}": _num(x) for k, x in power.items()}
    v.metrics["powered_ks"] = powered
    best_caps = min((med[f"caps_K{k}"] for k in powered), default=float("nan"))
    v.crit(
        "approximately Markov at some adequately powered K <= 32",
        np.isfinite(best_caps) and best_caps <= THRESHOLDS["H2_gap_markov"],
        best_caps,
        f"median normalised gap <= {THRESHOLDS['H2_gap_markov']} at a K whose test detects an injected "
        f"15% second-order effect (powered K: {powered})",
    )
    kref = 16 if 16 in ks else ks[0]
    gc = np.array([r["gap_norm"] for r in curves[f"caps_K{kref}"]])
    ge = np.array([r["gap_norm"] for r in curves[f"euclid_K{kref}"]])
    ok = np.isfinite(gc) & np.isfinite(ge)
    if ok.sum() >= 5:
        w = sst.wilcoxon(gc[ok], ge[ok], alternative="less")
        v.crit(
            "angular partitions more Markov than euclidean",
            w.pvalue < 0.05,
            float(np.median(gc[ok] - ge[ok])),
            f"Wilcoxon caps<euclid p<0.05 (p={w.pvalue:.3g}), K={kref}",
        )
    iv = run.intervene()
    faithful = None
    if iv is not None and len(iv):
        k = int(iv["k"].iloc[0])
        codes = ctx.codes("caps", k)
        tr = ctx.tr
        wins, top_t, top_s, _flips = [], [], [], []
        for l, g in iv.groupby("layer"):
            P = transition(codes[tr, l], codes[tr, l + 1], k, k)
            s, t, o = g["src"].to_numpy(int), g["tgt"].to_numpy(int), g["next_obs"].to_numpy(int)
            keep = s != t
            wins.append(P[t[keep], o[keep]] > P[s[keep], o[keep]])
            top_t.append(P[t[keep]].argmax(1) == o[keep])
            top_s.append(P[s[keep]].argmax(1) == o[keep])
        wins = np.concatenate(wins)
        faithful = float(wins.mean())
        pb = sst.binomtest(int(wins.sum()), len(wins), 0.5, alternative="greater").pvalue
        v.metrics["intervention"] = {
            "n": int(len(wins)),
            "faithful_share": faithful,
            "p_binom": _num(pb),
            "top1_target_rule": _num(np.concatenate(top_t).mean()),
            "top1_source_rule": _num(np.concatenate(top_s).mean()),
            "answer_flip_rate": _num(iv["flip"].mean()),
        }
        v.crit(
            "region-swap interventions follow the chain",
            faithful >= THRESHOLDS["H2_faithful"] and pb < 0.01,
            faithful,
            f">= {THRESHOLDS['H2_faithful']}, binomial p<0.01",
        )
    beats_inertia = None
    if iv is not None and len(iv) and any(c.startswith("next_obs_s") for c in iv.columns):
        it = _inertia_test(iv, ctx.codes("caps", int(iv["k"].iloc[0])), ctx.tr, int(iv["k"].iloc[0]))
        v.metrics["chain_vs_inertia"] = {f"s{s}": r for s, r in it.items()}
        if it:
            S = max(it)
            r = it[S]
            beats_inertia = bool(
                r["chain"] > max(r["static"], r["delta"]) and r["p_mcnemar"] < THRESHOLDS["H2_inertia_p"]
            )
            v.crit(
                f"chain beats residual inertia at horizon {S}",
                beats_inertia,
                (r["chain"] or 0) - max(r["static"] or 0, r["delta"] or 0),
                f"top-1 chain > best of static/delta inertia ({r['best_baseline']}), "
                f"McNemar p<{THRESHOLDS['H2_inertia_p']} "
                f"(p={r['p_mcnemar']})",
            )
    markov_ok = v.criteria[0]["passed"]
    faithful_ok = (
        faithful is not None
        and faithful >= THRESHOLDS["H2_faithful"]
        and v.metrics["intervention"]["p_binom"] is not None
        and v.metrics["intervention"]["p_binom"] < 0.01
    )
    if (np.isfinite(best_caps) and best_caps > THRESHOLDS["H2_gap_kill"]) or (
        faithful is not None and faithful < THRESHOLDS["H2_faithful_kill"]
    ):
        v.verdict = "KILLED"
    elif markov_ok and faithful_ok and beats_inertia is not False:
        v.verdict = "SURVIVES"
    else:
        v.verdict = "INCONCLUSIVE"
    v.meaning = (
        "SURVIVES: a transformer's depth admits a small, causally faithful finite-state abstraction "
        "whose predictions beat the trivial 'the state just persists' account. This licenses committor / "
        "transition-path analysis (H3) and is a methods contribution in its own right (Markov state models "
        "of inference). Faithful but not beating inertia: the abstraction is consistent but the regions are "
        "not shown to drive the dynamics; report it as such. If angular beats euclidean, caps earn their place."
    )
    v.metrics["_curves"] = {k: [{kk: _num(vv) for kk, vv in r.items()} for r in g] for k, g in curves.items()}
    return v


# ============================================================================ H3


def h3a_decidable_before_content(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H3a",
        "Failure is decidable before the answer exists",
        "Across depth, error becomes predictable from the residual stream at an earlier layer than "
        "the eventual answer token becomes decodable by the logit lens.",
        "INCONCLUSIVE",
    )
    pw = ctx.pointwise("err")
    auc = np.array(pw["layer_auc_test"], float)
    L = ctx.L
    rank = ctx.run.lens["rank"][:, 1:]  # [N, L]
    med_rank = np.median(rank[ctx.te], 0)
    content = next((i + 1 for i, r in enumerate(med_rank) if r <= 10), None)
    a = np.nan_to_num(auc - 0.5, nan=0)
    decid = next((i + 1 for i, x in enumerate(a) if x >= 0.9 * a.max()), None) if a.max() > 0.02 else None
    # outcome-centred committor along the coarse chain
    k = ctx.k_disc
    try:
        codes = ctx.codes("caps", k)
        Q, E = committor(codes, ctx.err, ctx.tr, ctx.layers, k)
        aq = [auroc(ctx.err[ctx.te], Q[ctx.te, i]) for i in range(len(ctx.layers))]
        ae = [auroc(ctx.err[ctx.te], E[ctx.te, i]) for i in range(len(ctx.layers))]
        v.metrics["committor_auc_by_layer"] = [_num(x) for x in aq]
        v.metrics["direct_auc_by_layer"] = [_num(x) for x in ae]
        v.metrics["committor_vs_direct_mean_abs_diff"] = _num(np.nanmean(np.abs(np.array(aq) - np.array(ae))))
    except FileNotFoundError:
        pass
    v.metrics.update(
        {
            "probe_auc_by_layer": [_num(x) for x in auc],
            "median_lens_rank_by_layer": [_num(x) for x in med_rank],
            "decidability_layer": decid,
            "content_layer": content,
            "n_layers": L,
        }
    )
    if decid is None or content is None:
        v.crit("both layers defined", False, None, "probe signal and logit-lens emergence must both exist")
        return v
    lead = (content - decid) / L
    v.crit(
        "decidability precedes content", lead >= THRESHOLDS["H3a_lead"], lead, f">= {THRESHOLDS['H3a_lead']} of depth"
    )
    v.verdict = "SURVIVES" if lead >= THRESHOLDS["H3a_lead"] else ("KILLED" if lead <= 0 else "INCONCLUSIVE")
    v.meaning = (
        "SURVIVES: the model is committed to failing before it knows what it will say. Pairs naturally "
        "with H3b to locate a point of no return."
    )
    return v


def h3b_commitment(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H3b",
        "Point of no return differs by error type",
        "The dynamic commitment layer (first perturbed layer from which even the cheapest push of one local "
        "spacing no longer flips the decision) differs between confident errors (CW) and consistent correct "
        "answers, or between unconfident errors (UW) and inconsistent correct answers, beyond output "
        "confidence and sample consistency.",
        "SKIPPED",
    )
    pt = ctx.run.perturb()
    if pt is None:
        return v
    idx = pt["idx"].astype(int)
    kmid = _kidx(pt["kappas"], 0.5)
    layers = pt["layers"].astype(int)
    if "margin_theta" in pt:
        # primary: committed once even the cheapest (gradient) push needs more than one local spacing
        stable = pt["margin_theta"] > 1.0
        v.metrics["commitment_definition"] = (
            "gradient: margin_theta > 1 local spacing at this and all deeper perturbed layers"
        )
    else:
        stable = pt["flip_rand"][:, :, kmid] <= 0.1
        v.metrics["commitment_definition"] = (
            f"random: flip rate <= 0.1 at kappa={pt['kappas'][kmid]} at this and all deeper layers"
        )
    commit = np.full(len(idx), len(layers), float)  # index into perturbed layers; len = never
    for i in range(len(idx)):
        for j in range(len(layers)):
            if stable[i, j:].all():
                commit[i] = j
                break
    depth = np.where(commit < len(layers), layers[np.minimum(commit, len(layers) - 1).astype(int)], ctx.L + 1) / ctx.L
    et = ctx.et[idx]
    fac = ctx.fac[idx]
    rank = ctx.run.lens["rank"][idx, :]
    content = np.array([next((l for l in range(1, ctx.L + 1) if r[l] <= 10), ctx.L + 1) for r in rank]) / ctx.L
    v.metrics["commit_depth_median"] = {
        k: _num(np.median(depth[et == k])) for k in ("FA", "CW", "UW") if (et == k).any()
    }
    v.metrics["share_commit_before_content"] = {
        k: _num(np.mean(depth[et == k] < content[et == k])) for k in ("FA", "CW", "UW") if (et == k).any()
    }
    v.metrics["commit_depth_median"]["FA_consistent"] = _num(np.median(depth[fac])) if fac.any() else None
    # Like with like: CW vs consistent FA, UW vs inconsistent FA. Commitment depth must add to confidence and
    # consistency (likelihood ratio, 1 df); otherwise it only re-measures how stable the answer already was.
    F0, cv = ctx.F0[idx], ctx.cons_value[idx]
    tests, raw = {}, {}
    for k, ref in (("CW", fac), ("UW", (et == "FA") & ~fac)):
        sel = (et == k) | ref
        y = (et[sel] == k).astype(int)
        if y.sum() > 10 and (1 - y).sum() > 10:
            Xb = flex_basis(np.c_[F0[sel], cv[sel]], np.ones(int(sel.sum()), bool))
            tests[k] = lr_increment(Xb, np.c_[Xb, depth[sel]], y)[1]
            raw[k] = _num(sst.mannwhitneyu(depth[ref], depth[et == k]).pvalue)
    v.metrics["mann_whitney_raw_p"] = raw
    if not tests:
        v.verdict = "INCONCLUSIVE"
        return v
    alpha = THRESHOLDS["H3b_p"] / len(tests)  # Bonferroni over the comparisons made (Holm's first step)
    for k, p in tests.items():
        ref = "consistent FA" if k == "CW" else "inconsistent FA"
        v.crit(
            f"{ref} vs {k}: commitment depth adds to confidence + consistency",
            p < alpha,
            p,
            f"LR p<{alpha:g} ({len(tests)} comparisons)",
        )
    ps = list(tests.values())
    v.verdict = "SURVIVES" if min(ps) < alpha else ("KILLED" if min(ps) > 0.2 else "INCONCLUSIVE")
    v.meaning = (
        "Gives the mechanistic reading of H1: associated hallucinations may commit early (a strong spurious "
        "association grabs the trajectory) while unassociated ones never commit."
    )
    return v


# ============================================================================ H4


def h4_synergy(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H4",
        "Synalytic: error information is synergistic across layers",
        "Some pairs of layers carry information about the error jointly that neither carries alone "
        "(positive interaction information beyond a label-permutation null), and a synolytic graph over "
        "layers exploits it in the low-label regime better than additive models.",
        "INCONCLUSIVE",
    )
    codes = ctx.codes("caps", ctx.k_syn)[:, ctx.disc_layers]
    pairs = [(i, j) for i in range(len(ctx.disc_layers)) for j in range(i + 1, len(ctx.disc_layers))]
    # A permutation p-value cannot go below 1/(n+1). BH over all pairs needs q/len(pairs) for its first discovery,
    # so the number of permutations must clear that floor or no single pair can ever be significant.
    n_syn = max(ctx.n_perm, int(np.ceil(1.5 * len(pairs) / 0.05)))
    v.metrics["n_permutations"] = n_syn
    res = {}
    for task in ("err", "cw"):
        mask, y = ctx.task(task)
        m = mask & ctx.tr
        if y[m].sum() < 30:
            continue
        obs, nul, ps = [], [], []
        for i, j in pairs:
            o, n0, p = synergy_test(y[m], codes[m, i], codes[m, j], n_perm=n_syn, seed=ctx.seed + 31 * i + j)
            obs.append(o)
            nul.append(n0)
            ps.append(p)
        disc = bh_fdr(np.array(ps), 0.05)
        res[task] = {
            "n_pairs": len(pairs),
            "n_significant_bh": int(disc.sum()),
            "max_excess_nats": _num(np.max(np.array(obs) - np.array(nul))),
            "top_pairs": [
                {
                    "layers": [ctx.disc_layers[pairs[t][0]], ctx.disc_layers[pairs[t][1]]],
                    "excess": _num(obs[t] - nul[t]),
                    "p": _num(ps[t]),
                }
                for t in np.argsort(ps)[:5]
            ],
        }
    v.metrics["synergy"] = res
    # path enrichment (combinatorial signature invisible to single regions)
    third = [len(ctx.disc_layers) // 4, len(ctx.disc_layers) // 2, (3 * len(ctx.disc_layers)) // 4]
    v.metrics["enriched_paths"] = {
        "layers": [ctx.disc_layers[t] for t in third],
        **enriched_paths(codes[ctx.tr], ctx.err[ctx.tr], sorted(set(third))),
    }
    # synolytic classifier vs additive / GBDT in the low-label regime
    lc = _synolytic_curve(ctx)
    v.metrics["low_label_curve"] = lc
    n_sig = res.get("err", {}).get("n_significant_bh", 0)
    v.crit(
        "synergistic layer pairs exist (all errors)",
        n_sig >= THRESHOLDS["H4_min_pairs"],
        n_sig,
        f">= {THRESHOLDS['H4_min_pairs']} pairs after BH",
    )
    hard = res.get("cw", {}).get("n_significant_bh", None)
    if hard is not None:
        v.crit("synergy present in the hard regime (CW vs consistent FA)", hard >= 1, hard, ">= 1 pair after BH")
    full_row = lc[-1] if lc else None
    g = (
        (full_row["synolytic"] - max(full_row["additive"], full_row["gbdt"]))
        if full_row and None not in (full_row["synolytic"], full_row["additive"], full_row["gbdt"])
        else float("nan")
    )
    v.crit(
        "synolytic layer graph beats additive and GBDT on identical inputs",
        np.isfinite(g) and g >= THRESHOLDS["H4_low_label_gain"],
        g,
        f"gain >= {THRESHOLDS['H4_low_label_gain']} AUROC at full train budget (low-label curve reported)",
    )
    c = [x["passed"] for x in v.criteria]
    if c[0] and any(c[1:]):
        v.verdict = "SURVIVES"
    elif n_sig == 0 and (hard in (None, 0)):
        v.verdict = "KILLED"
    v.meaning = (
        "SURVIVES: the original synalytic intuition holds at the level of representations; the "
        "per-example layer graph becomes the SGTA trace object. KILLED: error information in depth is "
        "additive; drop the graph from the claim."
    )
    return v


def _pair_logodds(
    ci: np.ndarray, cj: np.ndarray, y: np.ndarray, k: int, fit_idx: np.ndarray, prior: float = 2.0
) -> np.ndarray:
    """Synolytic edge: smoothed log-odds of error in the joint cell (c_i, c_j), estimated on fit_idx.

    This is the representation-level analogue of the SGNN edge (how informative is this pair of
    'features' jointly). It can express XOR-type interactions that additive models cannot.
    """
    cell = ci * k + cj
    base = y[fit_idx].mean()
    pos = np.bincount(cell[fit_idx], weights=y[fit_idx], minlength=k * k)
    cnt = np.bincount(cell[fit_idx], minlength=k * k).astype(float)
    p = (pos + prior * base) / (cnt + prior)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))[cell]


def _onehot(codes: np.ndarray, k: int) -> np.ndarray:
    N, nl = codes.shape
    out = np.zeros((N, nl * k), np.float32)
    for j in range(nl):
        out[np.arange(N), j * k + codes[:, j]] = 1
    return out


def _synolytic_curve(ctx: Ctx, ns=(50, 100, 200, 400), reps: int = 10):
    """Low-label comparison on identical inputs (region codes at the discrete layers):
    additive (one-hot logistic), GBDT (can learn interactions) and the synolytic layer graph."""
    k = ctx.k_syn
    C = ctx.codes("caps", k)[:, ctx.disc_layers]
    N, nl = C.shape
    pairs = [(i, j) for i in range(nl) for j in range(i + 1, nl)]
    y = ctx.err
    X = _onehot(C, k)
    tr_idx, te_idx = np.flatnonzero(ctx.tr), np.flatnonzero(ctx.te)
    rng = np.random.default_rng(ctx.seed)
    out = []
    for n in list(ns) + [len(tr_idx)]:
        if n > len(tr_idx):
            continue
        r_add, r_gb, r_syn = [], [], []
        for _r in range(reps if n < len(tr_idx) else 1):
            # always shuffled: meta.csv is ordered by dataset, a sorted index would split halves by dataset
            sub = rng.choice(tr_idx, n, replace=False) if n < len(tr_idx) else rng.permutation(tr_idx)
            if len(np.unique(y[sub])) < 2 or min(np.bincount(y[sub])) < 5:
                continue
            r_add.append(auroc(y[te_idx], fit_score(X[sub], y[sub], X[te_idx], logreg)))
            r_gb.append(auroc(y[te_idx], fit_score(X[sub], y[sub], X[te_idx], gbdt)))
            half = len(sub) // 2
            a_idx, b_idx = sub[:half], sub[half:]
            if len(np.unique(y[b_idx])) < 2:
                continue
            E = np.stack([_pair_logodds(C[:, i], C[:, j], y, k, a_idx) for (i, j) in pairs], 1)
            r_syn.append(auroc(y[te_idx], fit_score(E[b_idx], y[b_idx], E[te_idx], logreg)))
        out.append(
            {
                "n": int(n),
                "additive": _num(np.mean(r_add)) if r_add else None,
                "gbdt": _num(np.mean(r_gb)) if r_gb else None,
                "synolytic": _num(np.mean(r_syn)) if r_syn else None,
            }
        )
    return out


# ============================================================================ H5


def h5_separation(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H5",
        "Stochastic separation predicts where errors are detectable",
        "Across layers, the separability-based intrinsic dimension (FisherS) predicts few-shot error "
        "detectability (m = 20 labelled errors), extending the editability result of Sutton et al. "
        "(stealth edits, NeurIPS 2024) "
        "from editing to detection.",
        "INCONCLUSIVE",
    )
    ids, few = [], []
    for l in ctx.layers:
        z = ctx.run.layer_z(l)
        ids.append(fisher_id(z[ctx.tr], seed=ctx.seed)["id"])
        few.append(fisher_fewshot_auc(z[ctx.tr], ctx.err[ctx.tr], z[ctx.te], ctx.err[ctx.te], m=20, seed=ctx.seed))
    ids, few = np.array(ids), np.array(few)
    ok = np.isfinite(ids) & np.isfinite(few)
    rho, p = spearman(ids[ok], few[ok]) if ok.sum() > 4 else (float("nan"), float("nan"))
    v.metrics.update(
        {
            "fisher_id_by_layer": [_num(x) for x in ids],
            "fewshot_auc_by_layer": [_num(x) for x in few],
            "spearman": _num(rho),
            "p": _num(p),
        }
    )
    v.crit(
        "ID predicts few-shot detectability across layers",
        np.isfinite(rho) and rho >= THRESHOLDS["H5_rho"] and p < 0.05,
        rho,
        f"Spearman >= {THRESHOLDS['H5_rho']}, p<0.05 (layers are autocorrelated: treat p as optimistic)",
    )
    # two curves that both just drift with depth correlate for free: remove a quadratic depth trend from both
    if ok.sum() > 6:
        dep = np.array(ctx.layers, float)[ok] / ctx.L
        X = np.c_[np.ones_like(dep), dep, dep**2]
        res = lambda u: u - X @ np.linalg.lstsq(X, u, rcond=None)[0]
        rho_d, p_d = spearman(res(ids[ok]), res(few[ok]))
    else:
        rho_d, p_d = float("nan"), float("nan")
    v.metrics.update({"spearman_detrended": _num(rho_d), "p_detrended": _num(p_d)})
    v.crit(
        "the link survives removing the common depth trend",
        np.isfinite(rho_d) and rho_d > 0 and p_d < 0.05,
        rho_d,
        "Spearman of residuals after a quadratic fit in depth > 0, p<0.05",
    )
    # local separability of error types against the FA cloud at the best pointwise layer
    best = ctx.pointwise("err")["best_layer"]
    z = ctx.run.layer_z(best)
    fa = ctx.tr & (ctx.et == "FA")
    if fa.sum() > 50:
        Y, pca, keep = whiten_sphere(z[fa], seed=ctx.seed)
        mu = z[fa].mean(0)
        W = pca.transform(z - mu)[:, keep]
        W = W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-12, None)
        G = W[ctx.te] @ Y.T
        sep = G.max(1) < 0.8
        et = ctx.et[ctx.te]
        v.metrics["fisher_separable_from_FA_at_best_layer"] = {
            k: _num(sep[et == k].mean()) for k in ("FA", "CW", "UW") if (et == k).any()
        }
    passed = v.criteria[0]["passed"] and v.criteria[1]["passed"]
    v.verdict = "SURVIVES" if passed else ("KILLED" if np.isfinite(rho) and rho <= 0 else "INCONCLUSIVE")
    v.meaning = (
        "SURVIVES: a theory-driven, label-free diagnostic tells you where to put an error detector, and the "
        "link is not just two curves drifting together with depth. Raw correlation without the detrended one: "
        "INCONCLUSIVE, the shared depth trend may explain it."
    )
    return v


# ============================================================================ H7


def h7_angle_norm(ctx: Ctx) -> Verdict:
    v = Verdict(
        "H7",
        "Design check: does the sphere throw away error information?",
        "Error information is carried by direction; the radial coordinate (norm) adds nothing material. "
        "If this FAILS (norm matters), spherical caps must be extended with a radial coordinate.",
        "INCONCLUSIVE",
    )
    y, tr, te = ctx.err, ctx.tr, ctx.te
    raw = np.log(ctx.run.norms + 1e-6)
    d_auc, n_auc, b_auc = [], [], []
    for l in ctx.layers[:: max(1, len(ctx.layers) // 10)]:
        z = ctx.run.layer_z(l)
        s = ctx.run.layer_s(l)
        nn = np.c_[raw[:, l], np.log(np.linalg.norm(s, axis=1) + 1e-6)]
        d = z.shape[1]
        mk = lambda d=d: make_probe(d, pca=64, seed=ctx.seed)
        d_auc.append(auroc(y[te], fit_score(z[tr], y[tr], z[te], mk)))
        n_auc.append(auroc(y[te], fit_score(nn[tr], y[tr], nn[te], logreg)))
        zb = np.c_[z, nn * 10]
        db = zb.shape[1]
        mkb = lambda d=db: make_probe(d, pca=64, seed=ctx.seed)
        b_auc.append(auroc(y[te], fit_score(zb[tr], y[tr], zb[te], mkb)))
    gain = float(np.nanmean(np.array(b_auc) - np.array(d_auc)))
    v.metrics.update(
        {
            "direction_auc": [_num(x) for x in d_auc],
            "norm_only_auc": [_num(x) for x in n_auc],
            "both_auc": [_num(x) for x in b_auc],
            "mean_gain_from_norm": _num(gain),
        }
    )
    norm_matters = (np.nanmax(n_auc) >= THRESHOLDS["H7_norm_auc"]) or gain >= THRESHOLDS["H7_norm_gain"]
    v.crit(
        "norm-only is weak",
        not (np.nanmax(n_auc) >= THRESHOLDS["H7_norm_auc"]),
        np.nanmax(n_auc),
        f"< {THRESHOLDS['H7_norm_auc']}",
    )
    v.crit(
        "adding norm does not help", not (gain >= THRESHOLDS["H7_norm_gain"]), gain, f"< {THRESHOLDS['H7_norm_gain']}"
    )
    v.verdict = "KILLED" if norm_matters else "SURVIVES"
    v.meaning = "KILLED here is useful: it tells us to use (direction, log-norm) cylinders instead of caps."
    return v


# ============================================================================ runner

REGISTRY: dict[str, Callable[[Ctx], Verdict]] = {
    "H0": h0_gate,
    "H1": h1_basins,
    "H1b": h1b_margin,
    "H2": h2_msm,
    "H3a": h3a_decidable_before_content,
    "H3b": h3b_commitment,
    "H4": h4_synergy,
    "H5": h5_separation,
    "H7": h7_angle_norm,
}


def run_all(run_dir: str, only: list[str] | None = None, n_perm: int = 200, log=print) -> list[Verdict]:
    from .progress import Progress

    run = Run(run_dir)
    ctx = Ctx(run, n_perm=n_perm)
    out = []
    todo = [h for h in REGISTRY if not only or h in only]
    prog = Progress(run.path, "analyze", len(todo), every=1.0)
    for hid, fn in REGISTRY.items():
        if only and hid not in only:
            continue
        prog.update(len(out), part=hid)
        t0 = time.time()
        try:
            vd = fn(ctx)
        except Exception:  # one broken hypothesis must not kill the sprint
            import traceback

            vd = Verdict(hid, fn.__name__, "", "ERROR", meaning=traceback.format_exc()[-2000:])
        vd.seconds = round(time.time() - t0, 1)
        log(f"[{hid}] {vd.verdict:12s} {vd.title}  ({vd.seconds}s)")
        out.append(vd)
        if vd.verdict == "ERROR":
            prog.warn(f"{hid} завершилась с ошибкой (см. REPORT.md)")
    rd = run.results_dir()
    (rd / "verdicts.json").write_text(json.dumps(_jsonable([asdict(v) for v in out]), indent=1, ensure_ascii=False))
    (rd / "thresholds.json").write_text(json.dumps(THRESHOLDS, indent=1))
    prog.finish(verdicts={v.hid: v.verdict for v in out})
    return out
