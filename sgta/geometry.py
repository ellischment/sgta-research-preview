"""Spherical geometry: standardisation, partitions (caps / euclidean / random), rotations.

Convention: a "cap partition" at layer l is a set of K centers c_k on the unit sphere of the
standardised space s = (h - mu_l) / sd_l, z = s / |s|, with per-center angular radius r_k.
Assignment is nearest center by angle. Radii are reported for interpretation and for the
intervention target ("move inside cap j").
"""

from __future__ import annotations


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def angles(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angles between rows of z [N,D] and rows of c [K,D] (both unit). Returns [N,K]."""
    return np.arccos(np.clip(z @ c.T, -1.0, 1.0))


def _reduce(z_train: np.ndarray, z_all: np.ndarray, dim: int, seed: int):
    dim = int(min(dim, z_train.shape[1], max(2, z_train.shape[0] - 1)))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(z_train), pca.transform(z_all)


def fit_caps(
    z_train: np.ndarray, z_all: np.ndarray, k: int, pca_dim: int, q: float, seed: int, reduced_train: np.ndarray = None
):
    """Spherical k-means (k-means on unit vectors, clustering in a PCA view).

    Centers are recomputed in the full space as the normalised mean of members, so they can be
    used for interventions. Returns centers [K,D], radii [K], assign_all [N], dist_all [N].
    """
    k = int(min(k, len(z_train)))
    xr = reduced_train if reduced_train is not None else _reduce(z_train, z_train, pca_dim, seed)[0]
    km = KMeans(n_clusters=k, n_init=3, max_iter=200, random_state=seed).fit(xr)
    lab = km.labels_
    centers = np.zeros((k, z_train.shape[1]), dtype=np.float32)
    for j in range(k):
        m = lab == j
        centers[j] = z_train[m].mean(0) if m.any() else z_train[np.random.default_rng(seed + j).integers(len(z_train))]
    centers = unit(centers)
    a_tr = angles(z_train, centers)
    lab_tr = a_tr.argmin(1)
    radii = np.zeros(k, dtype=np.float32)
    for j in range(k):
        m = lab_tr == j
        radii[j] = np.quantile(a_tr[m, j], q) if m.any() else 0.0
    a_all = angles(z_all, centers)
    return centers, radii, a_all.argmin(1).astype(np.int16), a_all.min(1).astype(np.float32)


def fit_euclid(s_train: np.ndarray, s_all: np.ndarray, k: int, pca_dim: int, seed: int, reduced=None):
    """Plain k-means on the standardised (not normalised) states: keeps norm information.

    reduced: optional (train_view, all_view) from a PCA fitted once per layer.
    """
    if reduced is None:
        reduced = _reduce(s_train, s_all, pca_dim, seed)
    rt, ra = reduced
    k = int(min(k, len(rt)))
    km = KMeans(n_clusters=k, n_init=3, max_iter=200, random_state=seed).fit(rt)
    d = km.transform(ra)
    return d.argmin(1).astype(np.int16), d.min(1).astype(np.float32)


def fit_random(z_train: np.ndarray, z_all: np.ndarray, k: int, seed: int):
    """Null partition: K random training points as centers (no optimisation)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z_train), size=min(k, len(z_train)), replace=False)
    c = z_train[idx]
    a = angles(z_all, c)
    return a.argmin(1).astype(np.int16), a.min(1).astype(np.float32)


def median_nn_angle(z: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if len(z) > max_n:
        z = z[rng.choice(len(z), max_n, replace=False)]
    g = np.clip(z @ z.T, -1, 1)
    np.fill_diagonal(g, -2.0)
    return float(np.median(np.arccos(np.clip(g.max(1), -1, 1))))


# ---------------------------------------------------------------- rotations (used on GPU too)


def random_tangent(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(u.shape)
    v = v - (v * u).sum(-1, keepdims=True) * u
    return unit(v)


def rotate(u: np.ndarray, v_tangent: np.ndarray, theta) -> np.ndarray:
    """Great-circle rotation of unit u by angle theta toward unit tangent v."""
    theta = np.asarray(theta)[..., None] if np.ndim(theta) else theta
    return np.cos(theta) * u + np.sin(theta) * v_tangent


def toward(u: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangent at u pointing to target along the geodesic, and the geodesic angle."""
    cos = np.clip((u * target).sum(-1, keepdims=True), -1, 1)
    v = target - cos * u
    return unit(v), np.arccos(cos)[..., 0]
