"""Markov-state-model view of depth: transition operators between per-layer partitions.

Each layer l has its own partition (codes c_l in {0..K-1}). The coarse-grained dynamics is a
time-inhomogeneous chain with operators P_l = P(c_{l+s} | c_l). This is Ulam's method applied
to the layer map of a transformer, the same construction Markov state models use for
molecular dynamics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import fisher_exact

from .info import bh_fdr, cmi_test, mi


def transition(a: np.ndarray, b: np.ndarray, ka: int, kb: int, alpha: float = 0.5) -> np.ndarray:
    T = np.full((ka, kb), alpha, dtype=np.float64)
    np.add.at(T, (np.asarray(a, int), np.asarray(b, int)), 1.0)
    return T / T.sum(1, keepdims=True)


def markov_gap(
    codes: np.ndarray, layers: Sequence[int], stride: int = 1, n_perm: int = 200, seed: int = 0
) -> list[dict]:
    """For each l: I(c_{l+s}; c_{l-s} | c_l), its permutation-null mean, and the share of
    predictive information that the past adds beyond the present.

        gap_norm = (CMI - null) / I(c_{l+s}; c_l)

    gap_norm near 0 means the coarse-grained depth dynamics is (approximately) Markov at
    this resolution. It is the empirical lumpability defect of the partition.
    """
    out = []
    for l in layers:
        if l - stride < 0 or l + stride >= codes.shape[1]:
            continue
        p, c, n = codes[:, l - stride], codes[:, l], codes[:, l + stride]
        obs, null, pv = cmi_test(p, n, c, n_perm=n_perm, seed=seed + l)
        pred = mi(n, c)
        out.append(
            {
                "layer": int(l),
                "cmi": obs,
                "null": null,
                "p": pv,
                "mi_next_cur": pred,
                "gap_norm": float((obs - null) / pred) if pred > 1e-9 else float("nan"),
            }
        )
    return out


def markov_power(
    codes: np.ndarray,
    layers: Sequence[int],
    k: int,
    inject: float = 0.15,
    stride: int = 1,
    n_perm: int = 100,
    seed: int = 0,
) -> float:
    """Positive control for markov_gap at this K and sample size.

    Injects a known second-order dependence (with probability `inject`, c_{l+s} is replaced by a fixed
    relabelling of c_{l-s}) and returns the median normalised gap the test then reports. If the
    test cannot see a 15% copy of the past, a small gap at this K says nothing about Markovness:
    sparse cells make every estimator look Markov.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(k)
    out = []
    for l in layers:
        if l - stride < 0 or l + stride >= codes.shape[1]:
            continue
        p, c, n = codes[:, l - stride], codes[:, l], codes[:, l + stride]
        n_inj = np.where(rng.random(len(n)) < inject, perm[p % k], n)
        obs, null, _ = cmi_test(p, n_inj, c, n_perm=n_perm, seed=seed + l)
        pred = mi(n_inj, c)
        if pred > 1e-9:
            out.append((obs - null) / pred)
    return float(np.median(out)) if out else float("nan")


def committor(
    codes: np.ndarray, err: np.ndarray, train: np.ndarray, layers: Sequence[int], k: int, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Backward committor to the error outcome along the coarse chain.

        q_last(i) = P(error | c_last = i)          (estimated on train)
        q_l       = P_l q_{l+1}                    (Markov propagation)

    Also returns the direct estimate e_l(i) = P(error | c_l = i). If the chain is a faithful
    abstraction, q_l and e_l agree; their disagreement is a second, outcome-centred Markov test.
    Both arrays are [N, len(layers)] scores for every example.
    """
    layers = list(layers)
    tr = train
    q = np.zeros((len(layers), k))
    last = layers[-1]
    cnt = np.bincount(codes[tr, last], minlength=k).astype(float)
    pos = np.bincount(codes[tr, last], weights=err[tr], minlength=k)
    base = err[tr].mean()
    q[-1] = (pos + alpha * base) / (cnt + alpha)
    for i in range(len(layers) - 2, -1, -1):
        P = transition(codes[tr, layers[i]], codes[tr, layers[i + 1]], k, k)
        q[i] = P @ q[i + 1]
    Q = np.stack([q[i][codes[:, l]] for i, l in enumerate(layers)], 1)
    E = np.zeros_like(Q)
    for i, l in enumerate(layers):
        c = np.bincount(codes[tr, l], minlength=k).astype(float)
        p = np.bincount(codes[tr, l], weights=err[tr], minlength=k)
        E[:, i] = ((p + alpha * base) / (c + alpha))[codes[:, l]]
    return Q, E


def path_log_likelihood(codes: np.ndarray, train_mask: np.ndarray, layers: Sequence[int], k: int):
    """Per-example mean log-probability of its own path under the chain fitted on train correct."""
    layers = list(layers)
    ll = np.zeros(len(codes))
    for a, b in zip(layers[:-1], layers[1:], strict=False):
        P = transition(codes[train_mask, a], codes[train_mask, b], k, k)
        ll += np.log(P[codes[:, a], codes[:, b]])
    return ll / max(1, len(layers) - 1)


def enriched_paths(
    codes: np.ndarray, err: np.ndarray, layers: Sequence[int], min_count: int = 10, q: float = 0.05
) -> dict:
    """Paths (tuples of regions across the given layers) with error rate above base rate.

    Reports how many enriched paths exist that are NOT explained by any enriched single region
    on the path: the combinatorial signature that a pointwise detector cannot see.
    """
    layers = list(layers)
    tuples = [tuple(r) for r in codes[:, layers]]
    base = err.mean()
    uniq = {}
    for i, t in enumerate(tuples):
        uniq.setdefault(t, []).append(i)
    # single-region enrichment
    single_enriched = set()
    sp, keys = [], []
    for j, l in enumerate(layers):
        for r in np.unique(codes[:, l]):
            m = codes[:, l] == r
            a, b = int(err[m].sum()), int(m.sum() - err[m].sum())
            c, d = int(err[~m].sum()), int((~m).sum() - err[~m].sum())
            sp.append(fisher_exact([[a, b], [c, d]], alternative="greater")[1])
            keys.append((j, int(r)))
    mask = bh_fdr(np.array(sp), q)
    single_enriched = {keys[i] for i in np.flatnonzero(mask)}
    rows, pv = [], []
    for t, idx in uniq.items():
        if len(idx) < min_count:
            continue
        idx = np.array(idx)
        m = np.zeros(len(err), bool)
        m[idx] = True
        a, b = int(err[m].sum()), int(m.sum() - err[m].sum())
        c, d = int(err[~m].sum()), int((~m).sum() - err[~m].sum())
        pv.append(fisher_exact([[a, b], [c, d]], alternative="greater")[1])
        rows.append((t, len(idx), err[idx].mean()))
    pv = np.array(pv)
    disc = bh_fdr(pv, q) if len(pv) else np.zeros(0, bool)
    enriched = [rows[i] for i in np.flatnonzero(disc)]
    novel = [r for r in enriched if not any((j, int(reg)) in single_enriched for j, reg in enumerate(r[0]))]
    covered = sum(r[1] * r[2] for r in enriched)
    return {
        "n_paths_tested": len(rows),
        "n_enriched": len(enriched),
        "n_enriched_not_pointwise": len(novel),
        "base_rate": float(base),
        "errors_covered": float(covered / max(1, err.sum())),
        "top_novel": [
            {"path": list(map(int, r[0])), "n": int(r[1]), "err_rate": float(r[2])}
            for r in sorted(novel, key=lambda r: -r[2])[:10]
        ],
    }
