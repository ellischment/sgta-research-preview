"""Stage 2 (CPU): standardisation statistics and per-layer partitions, fitted on train only."""

from __future__ import annotations

import json
import time

import numpy as np

from .config import FitCfg
from .geometry import _reduce, fit_caps, fit_euclid, fit_random, median_nn_angle
from .progress import Progress
from .store import Run


def fit_run(run_dir: str, cfg: FitCfg, log=print) -> None:
    run = Run(run_dir)
    out = run.path / "fit"
    out.mkdir(exist_ok=True)
    L1 = run.n_layers + 1
    tr = run.is_train
    D = run.states.shape[2]
    prog = Progress(run.path, "fit", L1 * (3 + len(cfg.ks)), log=log)
    step = 0

    mu = np.zeros((L1, D), np.float32)
    sd = np.ones((L1, D), np.float32)
    for l in range(L1):
        h = np.asarray(run.states[tr, l, :], dtype=np.float32)
        mu[l] = h.mean(0)
        sd[l] = np.clip(h.std(0), 1e-4 * (h.std(0).mean() + 1e-12) + 1e-8, None)
        step += 1
        prog.update(step, part="stats")
    np.savez(out / "stats.npz", mu=mu, sd=sd)
    run._cache.pop("stats", None)

    nn = np.zeros(L1, np.float32)
    zs = {}
    for l in range(L1):
        z = run.layer_z(l)
        zs[l] = z
        nn[l] = median_nn_angle(z[tr], seed=cfg.seed)
        step += 1
        prog.update(step, part="spacing")
    np.save(out / "nn_angle.npy", nn)

    N = len(run.meta)
    # one PCA per layer and space, reused for every K
    red_z, red_s, degenerate = {}, {}, set()
    for l in range(L1):
        z = zs[l]
        if float(np.var(z[tr], axis=0).sum()) < 1e-10:  # e.g. constant chat-template token at layer 0
            degenerate.add(l)
            continue
        red_z[l] = _reduce(z[tr], z[tr], cfg.pca_dim, cfg.seed)[0]
        s = run.layer_s(l)
        red_s[l] = _reduce(s[tr], s, cfg.pca_dim, cfg.seed)
        step += 1
        prog.update(step, part="pca")
    log(f"[fit] PCA views ready; degenerate layers: {sorted(degenerate)}")
    for k in cfg.ks:
        t0 = time.time()
        C = np.zeros((L1, k, D), np.float32)
        R = np.zeros((L1, k), np.float32)
        A = np.zeros((N, L1), np.int16)
        Dm = np.zeros((N, L1), np.float32)
        Ae = np.zeros((N, L1), np.int16)
        De = np.zeros((N, L1), np.float32)
        Ar = np.zeros((N, L1), np.int16)
        Dr = np.zeros((N, L1), np.float32)
        for l in range(L1):
            z = zs[l]
            if l in degenerate:
                continue
            C[l], R[l], A[:, l], Dm[:, l] = fit_caps(
                z[tr], z, k, cfg.pca_dim, cfg.radius_quantile, cfg.seed, reduced_train=red_z[l]
            )
            Ae[:, l], De[:, l] = fit_euclid(None, None, k, cfg.pca_dim, cfg.seed, reduced=red_s[l])
            Ar[:, l], Dr[:, l] = fit_random(z[tr], z, k, cfg.seed + 17)
            step += 1
            prog.update(step, part=f"K={k}")
        np.savez(out / f"part_caps_K{k}.npz", centers=C, radii=R, assign=A, dist=Dm)
        np.savez(out / f"part_euclid_K{k}.npz", assign=Ae, dist=De)
        np.savez(out / f"part_random_K{k}.npz", assign=Ar, dist=Dr)
        log(f"[fit] K={k} done in {time.time() - t0:.1f}s")
    (out / "fit_cfg.json").write_text(json.dumps(cfg.__dict__, indent=2))
    prog.finish()
