"""GPU preflight: run this FIRST on a rented machine. Catches the bugs that would silently ruin a sprint.

Checks
  1 left padding does not change the captured residual states (position ids are right)
  2 logit lens on the last block equals the model's own logits (index convention is right)
  3 rotation with theta = 0 is an exact identity (hooks do not corrupt the stream)
  4 rotation with theta > 0 changes the logits (hooks actually fire)
  5 replacing a state with itself is an identity (region-swap path is right)
  6 greedy generation's first token equals argmax of the forward logits
  7 attention / MLP outputs are captured for every block (TriLens-style baseline); additivity
    h_l = h_(l-1) + attn + mlp is reported (exact for pre-norm blocks such as Qwen2 / Llama, not
    expected for post-norm blocks such as OLMo-2, so it is information, not a gate)
  8 margin gradient: the margin it reports equals the forward margin, and a small rotation
    along -grad lowers the margin (sign and plumbing of H1b)
  9 mini end-to-end: extract -> fit -> perturb on a few dozen questions, and a time estimate
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .data import build_prompt
from .model_io import (
    blocks,
    capture,
    capture_components,
    encode,
    last_logits,
    lens_logits,
    load_model,
    margin_grad,
    replace_last,
    rotate_last,
)


def check_hooks(model, tok, log=print):
    dev = next(model.parameters()).device
    L = len(blocks(model))
    fp32 = next(model.parameters()).dtype == torch.float32
    tol = 1e-3 if fp32 else 5e-2
    qs = [
        "What is the capital of France?",
        "Who wrote the novel War and Peace, and in which century was it written?",
        "Which element has the chemical symbol Fe?",
    ]
    prompts = [build_prompt(tok, q, "Answer the question with a short factual phrase only.") for q in qs]
    res = []
    enc = encode(tok, prompts, dev)
    with capture(model) as st:
        lg = last_logits(model, enc)
    Hb = torch.stack([st[l] for l in range(L + 1)], 1).float()
    worst = 0.0
    for i, p in enumerate(prompts):
        e1 = encode(tok, [p], dev)
        with capture(model) as s1:
            last_logits(model, e1)
        H1 = torch.stack([s1[l] for l in range(L + 1)], 1).float()[0]
        rel = ((Hb[i] - H1).norm(dim=-1) / H1.norm(dim=-1).clamp_min(1e-6)).max().item()
        worst = max(worst, rel)
    res.append(("1 padding invariance of states", worst < tol, f"max relative diff {worst:.2e} (tol {tol})"))
    ll = lens_logits(model, Hb[:, L])
    d = ((ll - lg).abs().max() / lg.abs().max()).item()
    res.append(("2 logit lens at last block == model logits", d < tol, f"rel max diff {d:.2e}"))
    mid = max(1, L // 2)
    V = torch.randn(len(prompts), Hb.shape[-1], device=dev)
    with rotate_last(model, mid, V, torch.zeros(len(prompts), device=dev)):
        lg0 = last_logits(model, enc)
    d0 = ((lg0 - lg).abs().max() / lg.abs().max()).item()
    res.append(("3 rotation theta=0 is identity", d0 < tol, f"rel max diff {d0:.2e}"))
    with rotate_last(model, mid, V, torch.full((len(prompts),), 0.5, device=dev)):
        lg1 = last_logits(model, enc)
    d1 = ((lg1 - lg).abs().max() / lg.abs().max()).item()
    res.append(("4 rotation theta=0.5 changes logits", d1 > 10 * max(d0, 1e-6), f"rel max diff {d1:.2e}"))
    with replace_last(model, mid, Hb[:, mid]):
        lg2 = last_logits(model, enc)
    d2 = ((lg2 - lg).abs().max() / lg.abs().max()).item()
    res.append(("5 replace-with-self is identity", d2 < tol, f"rel max diff {d2:.2e}"))
    out = model.generate(
        **enc,
        max_new_tokens=2,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.pad_token_id,
        repetition_penalty=1.0,
    )
    g0 = out[:, enc["input_ids"].shape[1]]
    agree = (g0 == lg.argmax(-1)).float().mean().item()
    need = 1.0 if fp32 else 0.66  # bf16 near-ties may flip one token; that is not a bug
    res.append(("6 generate first token == forward argmax", agree >= need, f"agreement {agree:.2f} (need {need})"))
    with capture(model) as st2, capture_components(model) as comp:
        last_logits(model, enc)
    have = all(("attn", l) in comp and ("mlp", l) in comp for l in range(1, L + 1))
    add = float("nan")
    if have:
        add = max(
            (
                (st2[l] - st2[l - 1] - comp[("attn", l)] - comp[("mlp", l)]).float().norm(dim=-1)
                / st2[l].float().norm(dim=-1).clamp_min(1e-6)
            )
            .max()
            .item()
            for l in range(1, L + 1)
        )
    res.append(
        (
            "7 attention / MLP outputs captured for all blocks",
            have,
            f"additivity residual {add:.2e} (should be near 0: the hooks see what each sub-block adds)",
        )
    )
    for p_ in model.parameters():
        p_.requires_grad_(False)
    ref = lg.argmax(-1)
    x, g, m = margin_grad(model, enc, mid, ref)
    top = lg.gather(-1, ref[:, None])[:, 0]
    m_fwd = top - lg.scatter(-1, ref[:, None], float("-inf")).max(-1).values
    dm = ((m - m_fwd).abs().max() / m_fwd.abs().max().clamp_min(1e-6)).item()
    u = x / x.norm(dim=-1, keepdim=True)
    gt = g - (g * u).sum(-1, keepdim=True) * u
    with rotate_last(model, mid, -gt, torch.full((len(prompts),), 0.02, device=dev)):
        lg3 = last_logits(model, enc)
    m3 = lg3.gather(-1, ref[:, None])[:, 0] - lg3.scatter(-1, ref[:, None], float("-inf")).max(-1).values
    down = (m3 < m_fwd).float().mean().item()
    ok8 = bool(torch.isfinite(g).all()) and g.abs().max().item() > 0 and dm < max(tol, 5e-2) and down >= 0.66
    res.append(
        (
            "8 margin gradient is consistent and points downhill",
            ok8,
            f"margin rel diff {dm:.2e}, margin lowered on {down:.2f} of prompts",
        )
    )
    for name, ok, det in res:
        log(f"{'OK  ' if ok else 'FAIL'} {name}: {det}")
    return res


def gpu_dtype(device: str = "cuda") -> str:
    """bfloat16 on GPUs that support it in hardware (Ampere and newer), float16 on older ones (T4, P100)."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "float32"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def preflight(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cuda",
    n_q: int = 24,
    run_dir: str = "runs/_preflight",
    dtype: str | None = None,
) -> bool:
    from .config import SprintCfg
    from .extract import extract
    from .fit import fit_run
    from .perturb import perturb_run

    t = time.time()
    dtype = dtype or gpu_dtype(device)
    model, tok = load_model(model_name, dtype, device)
    print(f"loaded {model_name} ({dtype}) in {time.time() - t:.0f}s; layers={len(blocks(model))}")
    res = check_hooks(model, tok)
    ok = all(r[1] for r in res)
    cfg = SprintCfg(model=model_name, run_dir=run_dir, device=device, dtype=dtype)
    cfg.data.datasets = {"triviaqa": n_q, "popqa": n_q}
    cfg.gen.n_samples = 3
    cfg.gen.batch_size = 16
    cfg.fit.ks = [4, 8]
    cfg.fit.pca_dim = 16
    cfg.perturb.n_examples = 12
    cfg.perturb.n_random = 4
    cfg.perturb.batch_size = 20
    cfg.perturb.n_intervene = 16
    cfg.perturb.intervene_k = 4
    t = time.time()
    extract(cfg, model=model, tok=tok)
    te = time.time() - t
    fit_run(run_dir, cfg.fit)
    t = time.time()
    perturb_run(cfg, model=model, tok=tok)
    tp = time.time() - t
    for f in (
        "meta.csv",
        "states.npy",
        "lens.npz",
        "norms.npy",
        "perturb.npz",
        "intervene.csv",
        "fit/part_caps_K4.npz",
    ):
        exists = (Path(run_dir) / f).exists()
        ok &= exists
        print(f"{'OK  ' if exists else 'FAIL'} 9 output {f}")
    per_q = te / (2 * n_q)
    print(
        f"\nTiming on this GPU with the small model: extract {per_q * 1000:.0f} ms/question, "
        f"perturb {tp:.0f}s for 12 examples."
    )
    print("A 7B model is roughly 5-8x slower per question. Use this to size the sprint config.")
    print("\nPREFLIGHT", "PASSED" if ok else "FAILED")
    return ok
