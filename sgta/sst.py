"""Stochastic separation (Gorban-Tyukin) diagnostics per layer.

Fisher separability: after centring, whitening and projection onto the unit sphere, a point x
is Fisher-separable from y at level alpha if <x, y> < alpha <x, x>. The separability-based
intrinsic dimension (FisherS, Albergante-Bac-Zinovyev 2019) inverts the uniform-sphere formula

    p_alpha(n) ~ (1 - alpha^2)^((n-1)/2) / (alpha * sqrt(2 pi n))

for the empirical mean non-separability p_alpha. The same quantity drives editability in
the stealth edits of Sutton et al. (NeurIPS 2024); here we ask whether it predicts detectability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import lambertw
from sklearn.decomposition import PCA


def whiten_sphere(X: np.ndarray, max_comp: int = 200, seed: int = 0):
    X = X - X.mean(0)
    k = int(min(max_comp, X.shape[1], X.shape[0] - 1))
    pca = PCA(n_components=k, whiten=True, random_state=seed).fit(X)
    ev = pca.explained_variance_
    keep = ev > ev.max() * 1e-3
    Y = pca.transform(X)[:, keep]
    Y = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)
    return Y, pca, keep


def fisher_id(
    X: np.ndarray,
    alphas: Sequence[float] = tuple(np.round(np.arange(0.2, 0.95, 0.05), 2)),
    max_n: int = 2000,
    seed: int = 0,
    min_pairs: int = 100,
) -> dict[str, float]:
    """Estimate at the largest alpha whose non-separable pair count is large enough to trust."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    Y, _, _ = whiten_sphere(X, seed=seed)
    G = Y @ Y.T
    np.fill_diagonal(G, -np.inf)
    n = len(Y)
    ests, pbar = [], {}
    for a in alphas:
        cnt = int((a < G).sum())
        p = cnt / (n * (n - 1))
        pbar[a] = float(p)
        if cnt < min_pairs or p >= 0.5:
            continue
        lam = -np.log(1 - a * a)
        arg = lam / (2 * np.pi * p * p * a * a * (1 - a * a))
        dim = np.real(lambertw(arg)) / lam
        if np.isfinite(dim) and dim > 0:
            ests.append(dim)
    # FisherS convention: use the largest alpha that still has enough non-separable pairs
    return {"id": float(ests[-1]) if ests else float("nan"), **{f"p_{a}": v for a, v in pbar.items()}}


def fisher_fewshot_auc(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, m: int = 20, reps: int = 20, seed: int = 0
) -> float:
    """Few-shot error detector in the spirit of one-trial AI correctors.

    m labelled errors + all train correct, whitened space, Fisher direction (mean difference).
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    mu = X_tr.mean(0)
    k = int(min(100, X_tr.shape[1], X_tr.shape[0] - 1))
    pca = PCA(n_components=k, whiten=True, random_state=seed).fit(X_tr - mu)
    W_tr = pca.transform(X_tr - mu)
    W_te = pca.transform(X_te - mu)
    err_idx = np.flatnonzero(y_tr == 1)
    cor = W_tr[y_tr == 0].mean(0)
    if len(err_idx) < m or len(np.unique(y_te)) < 2:
        return float("nan")
    aucs = []
    for _ in range(reps):
        e = W_tr[rng.choice(err_idx, m, replace=False)].mean(0)
        w = e - cor
        aucs.append(roc_auc_score(y_te, W_te @ w))
    return float(np.mean(aucs))
