"""Comparison with EPGS (Liew, Tan, Abdul Majeed, arXiv 2605.00939). Exploratory: never part of the verdicts.

EPGS score = |g_clean| * (1 - cos(g_clean, g_noisy)), where g is the gradient of the cross-entropy of the greedy
answer with respect to the parameters of the LAST decoder block, computed on prompt + answer, once clean and once
with Gaussian noise added to the input embeddings. Higher means a sharper loss landscape, which the paper reads as
a "stubborn" (confident, consistent) error.

Deviations from the paper, also written into the output files:
  * the loss covers every answer token (answers here are short entity phrases), not NER-selected entity tokens;
  * the noise scale is sigma times the RMS of the clean input embeddings, averaged over n_noise draws.

GPU:  python -m sgta.cli epgs --config configs/qwen2.5-7b.json      (after extract and perturb; same examples)
CPU:  python -m sgta.cli epgs-compare --run runs/qwen2.5-7b         (after fit; writes results/epgs_comparison.*)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .store import Run, stratified_pick

DEVIATIONS = (
    "loss over all answer tokens instead of NER-selected entity tokens; noise = sigma x RMS of the clean "
    "input embeddings, averaged over n_noise draws"
)


def epgs_run(cfg, model=None, tok=None, sigma: float = 0.01, n_noise: int = 2, log=print) -> Path:
    import torch

    from .data import build_prompt, uses_template
    from .model_io import blocks, load_model

    run = Run(cfg.run_dir)
    pt = run.perturb()
    idx = (
        pt["idx"].astype(int)
        if pt is not None
        else stratified_pick(run.etype, run.consistent, cfg.perturb.n_examples, cfg.fit.seed)
    )
    if model is None:
        model, tok = load_model(cfg.model, cfg.dtype, cfg.device)
    for p in model.parameters():
        p.requires_grad_(False)
    last = list(blocks(model)[-1].parameters())
    for p in last:
        p.requires_grad_(True)
    dev = next(model.parameters()).device
    emb = model.get_input_embeddings()
    style = cfg.data.prompt_style
    add_special = not uses_template(tok, style)
    meta = run.meta
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.data.seed))

    def grad(e, labels):
        out = model(inputs_embeds=e, labels=labels, use_cache=False)
        g = torch.autograd.grad(out.loss, last)
        return torch.cat([x.float().reshape(-1) for x in g])

    score, gnorm, cosm = (np.full(len(idx), np.nan, np.float32) for _ in range(3))
    t0 = time.time()
    for j, i in enumerate(idx):
        ans = meta["greedy"].iloc[i]
        ans = ans if isinstance(ans, str) else ""
        if not ans.strip():
            continue
        prompt = build_prompt(tok, meta["question"].iloc[i], cfg.data.system_prompt, style)
        p_ids = tok(prompt, add_special_tokens=add_special)["input_ids"]
        a_ids = tok(("" if uses_template(tok, style) else " ") + ans.strip(), add_special_tokens=False)["input_ids"]
        if not a_ids:
            continue
        ids = torch.tensor([list(p_ids) + list(a_ids)], device=dev)
        labels = ids.clone()
        labels[:, : len(p_ids)] = -100
        with torch.no_grad():
            e0 = emb(ids)
        rms = e0.float().pow(2).mean().sqrt()
        g0 = grad(e0, labels)
        n0 = g0.norm()
        vals, coss = [], []
        for _ in range(n_noise):
            noise = torch.randn(e0.shape, generator=gen).to(dev) * (sigma * rms)
            g1 = grad((e0.float() + noise).to(e0.dtype), labels)
            c = torch.nn.functional.cosine_similarity(g0, g1, dim=0)
            vals.append(float(n0 * (1 - c)))
            coss.append(float(c))
            del g1
        score[j], gnorm[j], cosm[j] = np.mean(vals), float(n0), np.mean(coss)
        del g0
        if (j + 1) % 200 == 0:
            log(f"[epgs] {j + 1}/{len(idx)} examples, {time.time() - t0:.0f}s")
    for p in last:
        p.requires_grad_(False)
    out = run.path / "epgs.npz"
    np.savez(out, idx=idx, score=score, gnorm=gnorm, cos=cosm, sigma=np.float32(sigma), n_noise=np.int32(n_noise))
    log(f"[epgs] done: {int(np.isfinite(score).sum())} scored of {len(idx)}, {time.time() - t0:.0f}s -> {out}")
    return out


def epgs_compare(run_dir: str) -> Path:
    """EPGS against the same base as H1 and H1b, on the same tasks; plus a head-to-head with the H1 and H1b scores."""
    from .hypotheses import Ctx, _basin_features, _increment, _margin_features, _num
    from .stats import auroc, delong

    run = Run(run_dir)
    z = np.load(run.path / "epgs.npz")
    ctx = Ctx(run)
    N = len(ctx.err)
    idx, sc = z["idx"].astype(int), z["score"].astype(float)
    have = np.zeros(N, bool)
    have[idx[np.isfinite(sc)]] = True
    e_full = np.full((N, 1), np.nan)
    e_full[idx, 0] = np.log(np.clip(sc, 1e-12, None))
    raw = np.full(N, np.nan)
    raw[idx] = sc
    rivals = {}
    pt = run.perturb()
    if pt is not None:
        pidx = pt["idx"].astype(int)
        hp = np.zeros(N, bool)
        hp[pidx] = True
        B = _basin_features(pt)
        bf = np.full((N, B.shape[1]), np.nan)
        bf[pidx] = B
        rivals["H1 basin"] = bf
        if "margin_theta" in pt:
            Mf = _margin_features(pt)
            mf = np.full((N, Mf.shape[1]), np.nan)
            mf[pidx] = Mf
            rivals["H1b margin"] = mf
        have &= hp
    res = {
        "paper": "Liew, Tan, Abdul Majeed. From Flat Facts to Sharp Hallucinations, arXiv 2605.00939",
        "deviations": DEVIATIONS,
        "sigma": float(z["sigma"]),
        "n_noise": int(z["n_noise"]),
        "n_scored": int(np.isfinite(sc).sum()),
        "exploratory": "not part of any verdict",
        "tasks": {},
    }
    for task in ("cw", "cw_all", "err"):
        r = _increment(ctx, e_full, have, task)
        if r is None:
            res["tasks"][task] = {"note": "too few examples"}
            continue
        y, te = r["y"], r["te"]
        row = {
            **{k: _num(v) for k, v in r["nt"].items()},
            "auc_epgs_stacked": _num(r["auc_alone"]),
            "auc_epgs_raw_higher_is_error": _num(auroc(y[te], raw[te])),
            "n_test_pos": r["n_test_pos"],
            "n_test_neg": r["n_test_neg"],
        }
        for name, block in rivals.items():
            rr = _increment(ctx, block, have, task)
            if rr is None:
                continue
            d, _, p = delong(y[te], rr["score"][te], r["score"][te])
            row[f"{name} minus EPGS (AUROC)"] = _num(d)
            row[f"p {name} vs EPGS"] = _num(p)
            row[f"{name} increment over base"] = _num(rr["nt"]["delta_auc"])
        res["tasks"][task] = row
    rd = run.results_dir()
    (rd / "epgs_comparison.json").write_text(json.dumps(res, indent=1))
    lines = [
        "# Comparison with EPGS (exploratory, not part of any verdict)",
        "",
        f"Paper: {res['paper']}. Deviations: {DEVIATIONS}. sigma = {res['sigma']}, n_noise = {res['n_noise']}, "
        f"scored examples: {res['n_scored']}.",
        "",
        "Main comparison: cw (CW vs consistent FA). cw_all is selection-confounded and shown for reference.",
        "",
        "| task | EPGS alone (AUROC) | EPGS increment over base | DeLong p | H1 minus EPGS | H1b minus EPGS |",
        "|---|---|---|---|---|---|",
    ]
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    for task, row in res["tasks"].items():
        if "note" in row:
            lines.append(f"| {task} | {row['note']} | | | | |")
            continue
        lines.append(
            f"| {task} | {f(row['auc_epgs_stacked'])} | {f(row.get('delta_auc'))} | {f(row.get('p_delong'))} | "
            f"{f(row.get('H1 basin minus EPGS (AUROC)'))} | {f(row.get('H1b margin minus EPGS (AUROC)'))} |"
        )
    p = rd / "epgs_comparison.md"
    p.write_text("\n".join(lines) + "\n")
    return p
