"""Stage 3 (GPU): basin-depth profiles (H1, H3b), gradient margins (H1b), region swaps (H2).

Basin depth. For each example, layer l and strength kappa we rotate the last-token residual state
by theta = kappa * (median nearest-neighbour angle at layer l, raw space) along
  * R random tangent directions,
  * one structured direction: toward the second-nearest spherical-cap center at that layer,
  * the error-vs-correct difference-in-means direction of the training split, BOTH signs.
    Both signs are applied to every example, so no label enters the features of a test example.
We record whether the next-token decision flips, the drop in log-prob of the original token and
KL(original || perturbed). Angles are in units of the local data spacing, so they compare across
layers. Random directions in thousands of dimensions are nearly orthogonal to the decision
subspace; the structured and probe directions exist so that the measurement is not floored.

Gradient margin (H1b). The margin m = logit[orig] - max other logit has a gradient g with respect
to the state h at layer l. Its tangent part g_t gives the cheapest rotation that lowers m. We store
the linear estimate theta_lin = m / (|h| |g_t|) and the measured smallest multiplier of the local
spacing, on a grid, at which rotating along -g_t flips the decision (censored above the grid).

Region swap. Move the state at layer l inside another cap (standardised space, norm preserved) and
record the caps reached at the next `horizon` layers. Two inertia baselines are recorded too:
  static: the moved state itself, read at layer l+s (nothing happens),
  delta:  the moved state plus the ORIGINAL updates h_{l+s} - h_l (the model keeps writing what it
          was writing, regardless of where it now is).
The chain must beat both to claim that regions drive the dynamics.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .config import SprintCfg
from .data import build_prompt, uses_template
from .geometry import median_nn_angle, unit
from .model_io import capture, encode, last_logits, load_model, margin_grad, replace_last, rotate_last
from .progress import Progress
from .store import Run, stratified_pick


def _layers(fracs, L):
    return sorted({int(min(L, max(1, round(f * L)))) for f in fracs})


def _assign(h: np.ndarray, stats, centers, l: int) -> np.ndarray:
    """Nearest cap at layer l for raw states h [B, D]."""
    z = unit((h - stats["mu"][l]) / stats["sd"][l])
    return np.argmax(z @ centers[l].T, 1)


def perturb_run(cfg: SprintCfg, model=None, tok=None, log=print) -> None:
    run = Run(cfg.run_dir)
    pc = cfg.perturb
    if model is None:
        model, tok = load_model(cfg.model, cfg.dtype, cfg.device)
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device
    L = run.n_layers
    meta = run.meta
    style = cfg.data.prompt_style
    prompts = [build_prompt(tok, q, cfg.data.system_prompt, style) for q in meta["question"]]
    add_special = not uses_template(tok, style)
    enc_of = lambda rows: encode(tok, [prompts[i] for i in rows], dev, add_special)
    first = meta["first_tok"].to_numpy()
    stats = run.fit_stats()
    caps = run.partition("caps", pc.intervene_k)
    layers = _layers(pc.layer_fracs, L)
    kappas = np.array(pc.kappas, float)
    tr = run.is_train
    err = run.y_err.astype(bool)
    fa = run.etype == "FA"
    raw_nn, probe_dir = {}, {}
    for l in layers:
        H = np.asarray(run.states[tr, l, :], np.float32)
        raw_nn[l] = median_nn_angle(unit(H), seed=0)
        e, c = err[tr], fa[tr]
        probe_dir[l] = (H[e].mean(0) - H[c].mean(0)) if e.any() and c.any() else np.zeros(H.shape[1], np.float32)
    rng = np.random.default_rng(cfg.fit.seed)

    # ------------------------------------------------------------ basin depth
    idx = stratified_pick(run.etype, run.consistent, pc.n_examples, cfg.fit.seed)  # sized for CW vs consistent FA
    M, nl, nk, R = len(idx), len(layers), len(kappas), pc.n_random
    C = R + 3  # random x R, structured, probe +, probe -
    flip_r, dlp_r, kl_r = (np.zeros((M, nl, nk), np.float32) for _ in range(3))
    flip_s, dlp_s = (np.zeros((M, nl, nk), np.float32) for _ in range(2))
    flip_p, dlp_p = (np.zeros((M, nl, nk, 2), np.float32) for _ in range(2))
    ex_per_batch = max(1, pc.batch_size // C)
    agree = []
    t0 = time.time()
    # progress in forward-pass units, so one ETA covers basins, margins and swaps
    n_swaps = len([l for l in _layers(pc.intervene_layer_fracs, L) if l < L]) * min(pc.n_intervene, len(meta))
    w_b = C * (nl * nk + 1)
    w_m = nl * (len(pc.margin_grid) + 1 + 4) if pc.margin else 0  # grid rows + one backward (~3 forwards)
    w_s = 2
    prog = Progress(run.path, "perturb", M * w_b + M * w_m + n_swaps * w_s, log=log)
    for b0 in range(0, M, ex_per_batch):
        bi = idx[b0 : b0 + ex_per_batch]
        rows = np.repeat(bi, C)
        enc = enc_of(rows)
        # reference decision: an unperturbed forward of the SAME padded batch, so bf16 batch effects cannot
        # masquerade as flips (theta = 0 is an exact identity of this forward)
        lg0r = last_logits(model, enc)
        lsm0_rows = torch.log_softmax(lg0r, -1)
        orig_rows = lg0r.argmax(-1)
        agree.append(float((orig_rows[::C].cpu().numpy() == first[bi]).mean()))
        for j, l in enumerate(layers):
            h = np.asarray(run.states[bi, l, :], np.float32)
            s = (h - stats["mu"][l]) / stats["sd"][l]
            z = unit(s)
            a = np.arccos(np.clip(z @ caps["centers"][l].T, -1, 1))
            second = np.argsort(a, 1)[:, 1]
            tgt_raw = stats["mu"][l] + stats["sd"][l] * (
                caps["centers"][l][second] * np.linalg.norm(s, axis=1, keepdims=True)
            )
            V = np.zeros((len(bi), C, h.shape[1]), np.float32)
            V[:, :R] = rng.standard_normal((len(bi), R, h.shape[1]))
            V[:, R] = tgt_raw
            V[:, R + 1] = probe_dir[l]
            V[:, R + 2] = -probe_dir[l]
            V = torch.as_tensor(V.reshape(-1, h.shape[1]), device=dev)
            for k, kap in enumerate(kappas):
                theta = torch.full((len(rows),), float(kap * raw_nn[l]), device=dev)
                with rotate_last(model, l, V, theta):
                    lg = last_logits(model, enc)
                lsm = torch.log_softmax(lg, -1)
                flip = (lg.argmax(-1) != orig_rows).float().view(len(bi), C)
                dlp = (lsm.gather(-1, orig_rows[:, None])[:, 0] - lsm0_rows.gather(-1, orig_rows[:, None])[:, 0]).view(
                    len(bi), C
                )
                kl = (lsm0_rows.exp() * (lsm0_rows - lsm)).sum(-1).view(len(bi), C)
                sl = slice(b0, b0 + len(bi))
                flip_r[sl, j, k] = flip[:, :R].mean(1).cpu().numpy()
                dlp_r[sl, j, k] = dlp[:, :R].mean(1).cpu().numpy()
                kl_r[sl, j, k] = kl[:, :R].mean(1).cpu().numpy()
                flip_s[sl, j, k] = flip[:, R].cpu().numpy()
                dlp_s[sl, j, k] = dlp[:, R].cpu().numpy()
                flip_p[sl, j, k] = flip[:, R + 1 :].cpu().numpy()
                dlp_p[sl, j, k] = dlp[:, R + 1 :].cpu().numpy()
        n_done = min(M, b0 + ex_per_batch)
        prog.update(
            n_done * w_b,
            part="basins",
            n=n_done,
            agree=float(np.mean(agree)),
            flip_random=float(flip_r[:n_done].mean()),
            flip_probe=float(flip_p[:n_done].mean()),
        )
        if n_done >= 60 and np.mean(agree) < 0.95:
            prog.warn(
                f"прямой проход совпадает с первым токеном ответа лишь в {np.mean(agree):.2f} случаев (норма > 0.97)"
            )
    out = {
        "idx": idx,
        "layers": np.array(layers),
        "kappas": kappas,
        "flip_rand": flip_r,
        "dlogp_rand": dlp_r,
        "kl_rand": kl_r,
        "flip_struct": flip_s,
        "dlogp_struct": dlp_s,
        "flip_probe": flip_p,
        "dlogp_probe": dlp_p,
        "raw_nn": np.array([raw_nn[l] for l in layers]),
    }
    log(
        f"[perturb] basin profiles done in {time.time() - t0:.0f}s; forward argmax agrees with greedy "
        f"first token on {np.mean(agree):.3f} of examples (expect > 0.97)"
    )
    if float(flip_r.mean()) < 0.005:
        prog.warn("случайные толчки почти не переворачивают ответ (эффект пола): смотри structured, probe и H1b")

    # ------------------------------------------------------------ gradient margin (H1b)
    if pc.margin:
        t1 = time.time()
        grid = np.array(pc.margin_grid, float)
        censor = 1.5 * grid.max()
        G = len(grid)
        grid0 = np.r_[0.0, grid]  # column 0 = unperturbed reference inside the same batch
        G0 = G + 1
        m_theta = np.full((M, nl), censor, np.float32)
        m_lin = np.zeros((M, nl), np.float32)
        m_sens = np.zeros((M, nl), np.float32)
        m_out = np.zeros(M, np.float32)
        gb = max(1, pc.grad_batch_size)
        for b0 in range(0, M, gb):
            bi = idx[b0 : b0 + gb]
            enc = enc_of(bi)
            orig = last_logits(model, enc).argmax(-1)
            for j, l in enumerate(layers):
                x, g, m = margin_grad(model, enc, l, orig)
                u = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                gt = g - (g * u).sum(-1, keepdim=True) * u
                sens = x.norm(dim=-1) * gt.norm(dim=-1)  # d margin / d theta along -g_t
                sl = slice(b0, b0 + len(bi))
                m_out[sl] = m.cpu().numpy()
                m_sens[sl, j] = sens.cpu().numpy()
                m_lin[sl, j] = (m.clamp_min(0) / sens.clamp_min(1e-12) / raw_nn[l]).cpu().numpy()
                # measured: rotate along -g_t by grid multiples of the local spacing
                per = max(1, pc.batch_size // G0)
                for c0 in range(0, len(bi), per):
                    cc = slice(c0, c0 + per)
                    nb = len(bi[cc])
                    Vg = (-gt[cc]).repeat_interleave(G0, 0)
                    th = torch.as_tensor(np.tile(grid0 * raw_nn[l], nb), device=dev, dtype=torch.float32)
                    rows = np.repeat(bi[cc], G0)
                    with rotate_last(model, l, Vg, th):
                        lg = last_logits(model, enc_of(rows))
                    am = lg.argmax(-1).view(nb, G0)
                    fl = (am[:, 1:] != am[:, :1]).cpu().numpy()
                    hit = fl.any(1)
                    m_theta[b0 + c0 : b0 + c0 + nb, j] = np.where(hit, grid[np.argmax(fl, 1)], censor)
            n_done = min(M, b0 + gb)
            prog.update(
                M * w_b + n_done * w_m, part="margins", n=n_done, censored=float((m_theta[:n_done] >= censor).mean())
            )
        out.update(
            margin_theta=m_theta,
            margin_lin=m_lin,
            margin_sens=m_sens,
            margin_out=m_out,
            margin_grid=grid,
            margin_censor=np.float32(censor),
        )
        log(
            f"[perturb] gradient margins done in {time.time() - t1:.0f}s; censored share "
            f"{float((m_theta >= censor).mean()):.3f}"
        )
        if float((m_theta >= censor).mean()) > 0.8:
            prog.warn("большинство ответов не переворачивается даже вдоль градиента: H1b может упереться в пол")
    np.savez(run.path / "perturb.npz", **out)

    # ------------------------------------------------------------ region swaps
    k = pc.intervene_k
    S = max(1, int(pc.intervene_horizon))
    ilayers = [l for l in _layers(pc.intervene_layer_fracs, L) if l < L]
    iidx = np.sort(rng.choice(len(meta), min(pc.n_intervene, len(meta)), replace=False))
    assign = caps["assign"].astype(int)
    rows_out = []
    bs = pc.batch_size
    for l in ilayers:
        steps = [s for s in range(1, S + 1) if l + s <= L]
        for b0 in range(0, len(iidx), bs):
            bi = iidx[b0 : b0 + bs]
            enc = enc_of(bi)
            h = np.asarray(run.states[bi, l, :], np.float32)
            s = (h - stats["mu"][l]) / stats["sd"][l]
            ns = np.linalg.norm(s, axis=1, keepdims=True)
            z = s / np.clip(ns, 1e-12, None)
            src = assign[bi, l]
            tgt = np.array([rng.choice([c for c in range(k) if c != a]) for a in src])
            c = caps["centers"][l][tgt]
            r = caps["radii"][l][tgt]
            cosang = np.clip((z * c).sum(1), -1, 1)
            ang = np.arccos(cosang)
            w = unit(z - cosang[:, None] * c)  # from c toward z, tangent at c
            z_new = np.cos(0.5 * r)[:, None] * c + np.sin(0.5 * r)[:, None] * w
            z_new = np.where((ang <= 0.5 * r)[:, None], z, z_new)
            h_new = stats["mu"][l] + stats["sd"][l] * (z_new * ns)
            ref = last_logits(model, enc).argmax(-1)
            with (
                replace_last(model, l, torch.as_tensor(h_new, device=dev)),
                capture(model, [l + s for s in steps]) as st,
            ):
                lg = last_logits(model, enc)
            flip = (lg.argmax(-1) != ref).cpu().numpy().astype(int)
            rec = {}
            for s_ in steps:
                obs = _assign(st[l + s_].float().cpu().numpy(), stats, caps["centers"], l + s_)
                stat = _assign(h_new, stats, caps["centers"], l + s_)
                upd = np.asarray(run.states[bi, l + s_, :], np.float32) - h
                dlt = _assign(h_new + upd, stats, caps["centers"], l + s_)
                rec[s_] = (obs, stat, dlt, assign[bi, l + s_])
            for t in range(len(bi)):
                row = {
                    "idx": int(bi[t]),
                    "layer": int(l),
                    "k": k,
                    "src": int(src[t]),
                    "tgt": int(tgt[t]),
                    "next_obs": int(rec[1][0][t]),
                    "next_orig": int(rec[1][3][t]),
                    "flip": int(flip[t]),
                }
                for s_ in range(1, S + 1):
                    o, a_, d_, g_ = rec.get(s_, (None, None, None, None))
                    row[f"next_obs_s{s_}"] = int(o[t]) if o is not None else -1
                    row[f"inertia_static_s{s_}"] = int(a_[t]) if a_ is not None else -1
                    row[f"inertia_delta_s{s_}"] = int(d_[t]) if d_ is not None else -1
                    row[f"next_orig_s{s_}"] = int(g_[t]) if g_ is not None else -1
                rows_out.append(row)
            prog.update(M * w_b + M * w_m + len(rows_out) * w_s, part="swaps", n=len(rows_out))
    pd.DataFrame(rows_out).to_csv(run.path / "intervene.csv", index=False)
    prog.finish()
    log(f"[perturb] interventions done: {len(rows_out)} rows, horizon {S}, total {time.time() - t0:.0f}s")
