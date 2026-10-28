"""Probes and statistical tests used by every hypothesis."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as sst
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def auroc(y, s) -> float:
    y = np.asarray(y)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auroc_oriented(y, s) -> float:
    """max(AUC, 1-AUC): for unsupervised scores whose sign is not fixed in advance."""
    a = auroc(y, s)
    return a if not np.isfinite(a) else max(a, 1 - a)


def bootstrap_ci(y, s, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ------------------------------------------------------------------ DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, s1, s2) -> tuple[float, float, float]:
    """Paired DeLong test. Returns (auc1 - auc2, z, two-sided p)."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        return float("nan"), float("nan"), float("nan")
    tx = np.vstack([_midrank(p) for p in pos])
    ty = np.vstack([_midrank(q) for q in neg])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(2)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0] - aucs[1]), float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0] - aucs[1]), float(z), float(2 * sst.norm.sf(abs(z)))


# ------------------------------------------------------------------ probes


def make_probe(n_features: int, pca: int | None = 64, seed: int = 0):
    steps = [StandardScaler()]
    if pca and n_features > pca:
        steps.append(PCA(n_components=pca, random_state=seed))
    # shuffled inner folds: rows are ordered by dataset, unshuffled folds would tune C for cross-dataset transfer
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    steps.append(LogisticRegressionCV(Cs=[0.003, 0.03, 0.3, 3.0], cv=inner, max_iter=3000, scoring="roc_auc"))
    return make_pipeline(*steps)


def oof_scores(X, y, make, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predicted scores on the training set (no leakage into the test split)."""
    X = np.asarray(X)
    y = np.asarray(y)
    out = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return out
    skf = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make().fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def fit_score(X_tr, y_tr, X_te, make) -> np.ndarray:
    m = make().fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def gbdt():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0)


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


# ------------------------------------------------------------------ flexible base


def flex_basis(X, fit_rows, n_knots: int = 3, degree: int = 3) -> np.ndarray:
    """Each base column plus a cubic B-spline basis of it, knots at quantiles of the fitting rows.

    The base of every increment test enters the logistic model through this basis, so that a new score
    cannot win by re-measuring confidence (or consistency, or a probe score) through a non-linear link:
    a flip rate is a saturating function of the margin, and the error rate need not be linear in the
    logit of confidence. Columns with too few distinct values (consistency inside CW vs FAc, for example)
    stay linear. Rows where a column is NaN stay NaN. Knots use features only, never labels.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    fit_rows = np.asarray(fit_rows, bool)
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = np.isfinite(x)
        fr = fit_rows & ok
        out.append(x[:, None])
        if fr.sum() < 30 or np.unique(x[fr]).size < 2 * (n_knots + degree):
            continue
        try:
            st = SplineTransformer(
                n_knots=n_knots, degree=degree, knots="quantile", extrapolation="linear", include_bias=False
            ).fit(x[fr, None])
            B = st.transform(np.where(ok, x, np.median(x[fr]))[:, None])
        except ValueError:
            continue
        B[~ok] = np.nan
        out.append(B)
    return np.concatenate(out, 1)


# ------------------------------------------------------------------ nested comparison


def _loglik(model, X, y) -> float:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def nested_test(Xb_tr, Xf_tr, y_tr, Xb_te, Xf_te, y_te):
    """Does the full feature set add information over the base set?

    In-sample likelihood ratio on train (chi2 with df = extra columns) and out-of-sample
    DeLong on test. Both must agree before we call it an increment.
    """
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000))
    mb = mk().fit(Xb_tr, y_tr)
    mf = mk().fit(Xf_tr, y_tr)
    lr = 2 * (_loglik(mf, Xf_tr, y_tr) - _loglik(mb, Xb_tr, y_tr))
    df = max(1, Xf_tr.shape[1] - Xb_tr.shape[1])
    p_lr = float(sst.chi2.sf(max(lr, 0), df))
    sb = mb.predict_proba(Xb_te)[:, 1]
    sf = mf.predict_proba(Xf_te)[:, 1]
    d, z, p_dl = delong(y_te, sf, sb)
    return {
        "auc_base": auroc(y_te, sb),
        "auc_full": auroc(y_te, sf),
        "delta_auc": d,
        "p_delong": p_dl,
        "lr_stat": float(lr),
        "lr_df": int(df),
        "p_lr": p_lr,
    }


def lr_increment(Xb, Xf, y) -> tuple[float, float]:
    """In-sample likelihood-ratio test of Xf over Xb (same rows). Returns (statistic, p), df = extra columns."""
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000))
    mb = mk().fit(Xb, y)
    mf = mk().fit(Xf, y)
    lr = 2 * (_loglik(mf, Xf, y) - _loglik(mb, Xb, y))
    df = max(1, Xf.shape[1] - Xb.shape[1])
    return float(lr), float(sst.chi2.sf(max(lr, 0), df))


def stratified_auroc(y, s, strata_score, n_bins: int = 5) -> float:
    """AUROC computed within quantile bins of a confounder, averaged with weights.

    Used to ask: at matched output confidence, does the internal score still separate?
    """
    y = np.asarray(y)
    s = np.asarray(s, float)
    q = np.quantile(strata_score, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(q, strata_score, side="right") - 1, 0, n_bins - 1)
    num, den = 0.0, 0.0
    for k in range(n_bins):
        m = b == k
        if m.sum() > 10 and len(np.unique(y[m])) == 2:
            w = min(y[m].sum(), (1 - y[m]).sum())
            num += w * roc_auc_score(y[m], s[m])
            den += w
    return float(num / den) if den > 0 else float("nan")


def spearman(x, y):
    r = sst.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)
