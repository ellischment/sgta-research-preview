"""Synthetic runs with planted (or absent) structure. Same on-disk format as a real extraction.

Why this exists: before paying for a GPU we check that every analysis (a) finds an effect that is
planted and (b) does not find it when it is absent. This is the false-positive calibration of
the whole tournament. Run:  python -m sgta.cli synthetic --mode planted --out runs/syn_planted

Generative story
  * L layers, R latent regions per layer, region centers random in R^D.
  * Region path z_1..z_L: first-order Markov (or second-order when markov=False).
  * State h_l = scale * center_l[z_l] + noise, raw norm growing with depth.
  * planted labels: error depends on an XOR of two region memberships at layers a and b, so no
    single layer carries it (pure synergy, invisible to linear pointwise probes). Among errors,
    UW examples also get a late-layer state shift (pointwise visible) and lower confidence;
    CW examples are pointwise identical to FA but have shallower basins (higher flip rates along
    random, structured and probe directions) and a more sensitive margin (smaller gradient flip
    angle) at the same output confidence.
  * null labels: independent of everything.
  * confound labels: independent of the states, but perturbation responses depend on answer
    consistency only (less consistent = easier to flip), identically for errors and correct answers.
    CW are consistent by definition and FA are a mix, so "CW vs all FA" shows a spurious basin effect;
    the main comparison "CW vs consistent FA" must not.
  * interventions: faithful runs continue the chain from the TARGET region for several layers,
    which beats both inertia baselines; unfaithful runs keep the original trajectory, which the
    'delta' inertia baseline predicts and the chain from the target does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import assign_etype
from .store import Run, assign_split, stratified_pick


def make_run(
    out: str,
    mode: str = "planted",
    markov: bool = True,
    N: int = 3000,
    L: int = 12,
    D: int = 48,
    R: int = 6,
    seed: int = 0,
    cw_share: float = 0.5,
) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    L1 = L + 1
    centers = rng.standard_normal((L1, R, D))
    centers /= np.linalg.norm(centers, axis=2, keepdims=True)
    # sticky transitions
    T = []
    for _ in range(L1):
        M = rng.dirichlet(np.ones(R) * 0.6, size=R) + 2.5 * np.eye(R)
        T.append(M / M.sum(1, keepdims=True))
    z = np.zeros((N, L1), int)
    z[:, 0] = 0  # constant template token at the embedding layer
    z[:, 1] = rng.integers(0, R, N)
    for l in range(2, L1):
        for i in range(N):
            if markov or l < 3:
                z[i, l] = rng.choice(R, p=T[l][z[i, l - 1]])
            else:  # second order: half the time copy two layers back
                z[i, l] = z[i, l - 2] if rng.random() < 0.5 else rng.choice(R, p=T[l][z[i, l - 1]])
    scale = 6.0
    H = scale * centers[np.arange(L1)[None, :], z] + 0.6 * rng.standard_normal((N, L1, D))
    H[:, 0] = scale * centers[0, 0] + 0.01 * rng.standard_normal((N, D))

    a_l, b_l = max(2, L // 3), max(3, (2 * L) // 3)
    if mode == "planted":
        xor = (z[:, a_l] < R // 2) ^ (z[:, b_l] < R // 2)
        p_err = np.where(xor, 0.55, 0.10)
        err = rng.random(N) < p_err
    else:  # null and confound: labels carry no state information
        err = rng.random(N) < 0.3
    is_cw = err & (rng.random(N) < cw_share)
    is_uw = err & ~is_cw
    if mode == "planted":
        shift = rng.standard_normal(D)
        shift /= np.linalg.norm(shift)
        H[is_uw, -4:, :] += 1.5 * shift
    # raw norm grows with depth (no label information)
    H *= (1 + 0.15 * np.arange(L1))[None, :, None]

    fa_lo = 0.3 if mode == "confound" else 0.6  # confound: correct answers are a real mix of stable and unstable
    cons = np.where(
        is_cw, rng.uniform(0.8, 1.0, N), np.where(is_uw, rng.uniform(0.2, 0.7, N), rng.uniform(fa_lo, 1.0, N))
    )
    pf = np.where(is_uw, rng.beta(5, 3, N), rng.beta(6, 2, N)) if mode == "planted" else rng.beta(6, 2, N)
    etype = assign_etype(~err, cons, 0.8)
    ids = np.arange(N)
    meta = pd.DataFrame(
        {
            "id": ids,
            "dataset": "synthetic",
            "question": [f"q{i}" for i in ids],
            "gold": "[]",
            "greedy": "",
            "correct": (~err).astype(int),
            "consistency": cons,
            "etype": etype,
            "p_first": pf,
            "mean_logprob": np.log(pf) * 0.7 + 0.05 * rng.standard_normal(N),
            "ent_first": -(pf * np.log(pf) + (1 - pf) * np.log(1 - pf + 1e-9)) + 0.05 * rng.standard_normal(N),
            "s_pop": rng.lognormal(6, 2, N),
            "first_tok": 0,
            "split": assign_split(ids, 0.4, seed),
        }
    )
    meta.to_csv(out / "meta.csv", index=False)
    np.save(out / "states.npy", H.astype(np.float16))
    np.save(out / "norms.npy", np.linalg.norm(H, axis=2).astype(np.float32))
    content = int(0.7 * L)
    lay = np.arange(L1)[None, :]
    rank = np.maximum(1, np.round(np.exp(np.clip(content - lay, 0, None) * 0.8) + rng.exponential(1, (N, L1)))).astype(
        int
    )
    prob = 1.0 / (1 + rank)
    ent = np.log(50) * np.exp(-lay / L) + 0.1 * rng.random((N, L1))
    ent_attn = 0.8 * ent + 0.2 * rng.standard_normal((N, L1))
    ent_mlp = 0.6 * ent + 0.2 * rng.standard_normal((N, L1))
    if mode == "planted":  # TriLens-style signal for the visible regime only
        ent_mlp[is_uw, -4:] += 0.4
    ent_attn[:, 0] = np.nan
    ent_mlp[:, 0] = np.nan
    np.savez(
        out / "lens.npz",
        prob=prob.astype(np.float32),
        rank=rank.astype(np.int32),
        entropy=ent.astype(np.float32),
        ent_attn=ent_attn.astype(np.float32),
        ent_mlp=ent_mlp.astype(np.float32),
    )
    info = {
        "model": f"synthetic-{mode}-{'markov' if markov else 'nonmarkov'}",
        "n_layers": L,
        "hidden": D,
        "position": "last_prompt_token",
        "index_convention": "0=embeddings, l=output of block l-1",
        "planted": {"mode": mode, "markov": markov, "xor_layers": [a_l, b_l]},
    }
    (out / "info.json").write_text(json.dumps(info, indent=1))

    # perturbation responses
    fracs = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    layers = np.array(sorted({max(1, min(L, int(round(f * L)))) for f in fracs}))
    kappas = np.array([0.25, 0.5, 1.0, 2.0])
    pick = stratified_pick(etype, cons >= 0.8, 900, seed)  # same sampling as the real perturbation stage
    M = len(pick)
    base = 0.45 * kappas[None, None, :] * np.exp(-3 * layers / L)[None, :, None]
    mult = np.ones(M)
    if mode == "planted":
        mult = np.where(etype[pick] == "CW", 2.6, np.where(etype[pick] == "UW", 1.6, 1.0))
    elif mode == "confound":  # stability follows consistency, not correctness
        mult = 1.0 + 3.0 * (1.0 - cons[pick])
    f = np.clip(base * mult[:, None, None], 0, 1)
    fr = rng.binomial(8, f) / 8.0
    fs = rng.binomial(1, np.clip(2 * f, 0, 1)).astype(float)
    fp = rng.binomial(1, np.clip(3 * f[..., None] * np.array([1.0, 0.6]), 0, 1)).astype(float)
    dl = -3 * fr + 0.2 * rng.standard_normal(fr.shape)
    nl = len(layers)
    # gradient margins: same output margin distribution for CW and FA, more sensitive CW decisions
    pfp = np.clip(pf[pick], 1e-4, 1 - 1e-4)
    m_out = np.exp(0.8 * np.log(pfp / (1 - pfp)) + 0.3 * rng.standard_normal(M))
    smult = np.ones(M)
    if mode == "planted":
        smult = np.where(etype[pick] == "CW", 1.8, np.where(etype[pick] == "UW", 1.3, 1.0))
    elif mode == "confound":
        smult = 1.0 + 2.0 * (1.0 - cons[pick])
    sens = np.exp(rng.normal(1.0, 0.3, nl))[None, :] * smult[:, None] * np.exp(0.35 * rng.standard_normal((M, nl)))
    raw_nn = 0.5
    lin = m_out[:, None] / sens / raw_nn
    grid = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    meas = lin * np.exp(0.25 * rng.standard_normal((M, nl)))
    cens = 1.5 * grid.max()
    pos = np.searchsorted(grid, meas)
    theta = np.where(pos < len(grid), grid[np.minimum(pos, len(grid) - 1)], cens)
    np.savez(
        out / "perturb.npz",
        idx=pick,
        layers=layers,
        kappas=kappas,
        flip_rand=fr,
        flip_struct=fs,
        dlogp_rand=dl,
        kl_rand=np.abs(dl),
        dlogp_struct=-3 * fs + 0.2 * rng.standard_normal(fs.shape),
        flip_probe=fp,
        dlogp_probe=-3 * fp + 0.2 * rng.standard_normal(fp.shape),
        raw_nn=np.full(nl, raw_nn),
        margin_theta=theta.astype(np.float32),
        margin_lin=lin.astype(np.float32),
        margin_sens=sens.astype(np.float32),
        margin_out=m_out.astype(np.float32),
        margin_grid=grid,
        margin_censor=np.float32(cens),
    )
    return out


def make_interventions(
    run_dir: str, faithful: bool = True, k: int = 6, n: int = 600, seed: int = 0, horizon: int = 3
) -> None:
    """Simulate region-swap interventions on a fitted synthetic run.

    faithful=True: after the swap the trajectory continues the chain from the target region for
    `horizon` layers (what a valid finite-state abstraction predicts). faithful=False: the swap is
    ignored and the trajectory stays on its original course (mostly), which is what happens when
    regions are not causally used downstream. Inertia columns mirror the real stage: 'static' is
    uninformative here, 'delta' is the original trajectory.
    """
    from .markov import transition

    run = Run(run_dir)
    rng = np.random.default_rng(seed)
    codes = run.partition("caps", k)["assign"].astype(int)
    tr = run.is_train
    L = run.n_layers
    P = {l: transition(codes[tr, l], codes[tr, l + 1], k, k) for l in range(L)}
    rows = []
    for l in range(2, L - 1):
        idx = rng.choice(len(codes), n // (L - 3), replace=False)
        for i in idx:
            s = int(codes[i, l])
            t = int(rng.choice([c for c in range(k) if c != s]))
            row = {"idx": int(i), "layer": l, "k": k, "src": s, "tgt": t, "flip": int(rng.random() < 0.3)}
            cur = t
            for st in range(1, horizon + 1):
                if l + st > L:
                    for key in ("next_obs", "inertia_static", "inertia_delta", "next_orig"):
                        row[f"{key}_s{st}"] = -1
                    continue
                orig = int(codes[i, l + st])
                if faithful:
                    cur = int(rng.choice(k, p=P[l + st - 1][cur]))
                    obs = cur
                else:
                    obs = orig if rng.random() < 0.8 else int(rng.choice(k, p=P[l + st - 1][s if st == 1 else obs]))
                row.update(
                    {
                        f"next_obs_s{st}": obs,
                        f"inertia_static_s{st}": int(rng.integers(k)),
                        f"inertia_delta_s{st}": orig,
                        f"next_orig_s{st}": orig,
                    }
                )
            row["next_obs"], row["next_orig"] = row["next_obs_s1"], row["next_orig_s1"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(run.path / "intervene.csv", index=False)
