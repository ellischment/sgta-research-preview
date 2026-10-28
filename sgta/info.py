"""Discrete information theory with Miller-Madow bias correction and permutation nulls."""

from __future__ import annotations

import numpy as np


def _codes(*cols: np.ndarray) -> np.ndarray:
    """Encode several integer columns into one integer code."""
    out = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        c = np.asarray(c, dtype=np.int64)
        out = out * (int(c.max()) + 1 if len(c) else 1) + c
    return out


def entropy(*cols: np.ndarray, mm: bool = True) -> float:
    """Joint entropy in nats. Miller-Madow correction (m - 1) / 2N by default."""
    x = _codes(*cols)
    n = len(x)
    if n == 0:
        return 0.0
    _, cnt = np.unique(x, return_counts=True)
    p = cnt / n
    h = float(-(p * np.log(p)).sum())
    if mm:
        h += (len(cnt) - 1) / (2.0 * n)
    return h


def mi(x, y, mm: bool = True) -> float:
    return entropy(x, mm=mm) + entropy(y, mm=mm) - entropy(x, y, mm=mm)


def cmi(x, y, z, mm: bool = True) -> float:
    """I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)."""
    return entropy(x, z, mm=mm) + entropy(y, z, mm=mm) - entropy(x, y, z, mm=mm) - entropy(z, mm=mm)


def shuffle_within(x: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = x[rng.permutation(idx)]
    return out


def cmi_test(x, y, z, n_perm: int = 200, seed: int = 0):
    """Test I(X;Y|Z) > 0 by shuffling X within strata of Z. Returns (cmi, null_mean, p)."""
    rng = np.random.default_rng(seed)
    obs = cmi(x, y, z)
    null = np.array([cmi(shuffle_within(np.asarray(x), np.asarray(z), rng), y, z) for _ in range(n_perm)])
    p = (1 + (null >= obs).sum()) / (n_perm + 1)
    return obs, float(null.mean()), float(p)


def interaction_info(e, a, b) -> float:
    """Synergy-minus-redundancy: I(E; A,B) - I(E; A) - I(E; B). Positive means synergy dominates."""
    return mi(e, _codes(a, b)) - mi(e, a) - mi(e, b)


def synergy_test(e, a, b, n_perm: int = 200, seed: int = 0):
    """Permutation test of interaction information against label shuffling.

    The null keeps the joint law of (A,B) and destroys every link to E, so it calibrates the
    small-sample bias of the plug-in estimator, which is the main danger with discrete codes.
    """
    rng = np.random.default_rng(seed)
    e = np.asarray(e)
    obs = interaction_info(e, a, b)
    null = np.array([interaction_info(rng.permutation(e), a, b) for _ in range(n_perm)])
    p = (1 + (null >= obs).sum()) / (n_perm + 1)
    return obs, float(null.mean()), float(p)


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thr
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask
