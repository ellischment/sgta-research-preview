"""Stage 1 (GPU): greedy answers, sampled answers (consistency), residual states, logit lens.

Also records TriLens-style lens entropies of each block's attention and MLP outputs at the last
prompt token (ent_attn, ent_mlp). These feed the strongest white-box baseline that H1 must beat.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import SprintCfg
from .data import assign_etype, build_prompt, clean_answer, is_correct, load_questions, token_f1, uses_template
from .model_io import blocks, capture, capture_components, encode, last_logits, lens_logits, load_model
from .progress import Progress
from .store import assign_split, write_manifest
import contextlib


def _eos_ids(model, tok) -> set:
    s = set()
    for x in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None), tok.pad_token_id):
        if x is None:
            continue
        if isinstance(x, (list, tuple)):
            s.update(int(y) for y in x)
        else:
            s.add(int(x))
    return s


def _gen_kwargs(tok, cfg, sample: bool):
    kw = {
        "max_new_tokens": cfg.gen.max_new_tokens,
        "pad_token_id": tok.pad_token_id,
        "repetition_penalty": 1.0,
        "return_dict_in_generate": True,
    }
    if sample:
        kw.update(
            do_sample=True,
            temperature=cfg.gen.temperature,
            top_p=cfg.gen.top_p,
            top_k=0,
            num_return_sequences=cfg.gen.n_samples,
        )
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kw


def _newline_ids(tok) -> torch.Tensor:
    """Ids of tokens whose text contains a newline. The answer ends at the first one: base models given a
    few-shot Q/A prompt never emit EOS and simply continue with the next 'Q:'. Cached on the tokenizer."""
    cached = getattr(tok, "_sgta_newline_ids", None)
    if cached is None or cached[0] != len(tok):
        pieces = tok.batch_decode([[i] for i in range(len(tok))])
        ids = torch.as_tensor([i for i, t in enumerate(pieces) if "\n" in t], dtype=torch.long)
        try:
            tok._sgta_newline_ids = (len(tok), ids)
        except Exception:
            return ids
        cached = tok._sgta_newline_ids
    return cached[1]


@torch.no_grad()
def _greedy(model, tok, enc, cfg):
    kw = _gen_kwargs(tok, cfg, sample=False)
    try:
        out = model.generate(**enc, **kw, output_logits=True)
        steps = out.logits
    except (TypeError, ValueError):
        out = model.generate(**enc, **kw, output_scores=True)
        steps = out.scores
    T = enc["input_ids"].shape[1]
    gen = out.sequences[:, T:]
    lp = torch.stack([torch.log_softmax(s.float(), -1) for s in steps], 1)  # [B, S, V]
    S = min(lp.shape[1], gen.shape[1])
    lp, gen = lp[:, :S], gen[:, :S]
    tok_lp = lp.gather(-1, gen[..., None])[..., 0]
    eos = torch.as_tensor(sorted(_eos_ids(model, tok)), device=gen.device)
    is_stop = torch.isin(gen, eos) | torch.isin(gen, _newline_ids(tok).to(gen.device))
    valid = is_stop.long().cumsum(1) == 0
    valid[:, 0] = True  # always score the first token
    hit_limit = (~is_stop.any(1)) & (gen.shape[1] >= cfg.gen.max_new_tokens)  # answer may be cut off
    m = valid.float()
    mean_lp = (tok_lp * m).sum(1) / m.sum(1).clamp_min(1)
    p0 = lp[:, 0].exp()
    first = gen[:, 0]
    p_first = p0.gather(-1, first[:, None])[:, 0]
    ent = -(p0 * lp[:, 0]).sum(-1)
    texts = [clean_answer(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return (
        texts,
        first.cpu().numpy(),
        p_first.cpu().numpy(),
        mean_lp.cpu().numpy(),
        ent.cpu().numpy(),
        hit_limit.cpu().numpy(),
    )


@torch.no_grad()
def _samples(model, tok, enc, cfg) -> list[list[str]]:
    if cfg.gen.n_samples <= 0:
        return [[] for _ in range(enc["input_ids"].shape[0])]
    out = model.generate(**enc, **_gen_kwargs(tok, cfg, sample=True))
    T = enc["input_ids"].shape[1]
    texts = [clean_answer(t) for t in tok.batch_decode(out.sequences[:, T:], skip_special_tokens=True)]
    k = cfg.gen.n_samples
    return [texts[i * k : (i + 1) * k] for i in range(len(texts) // k)]


HUB_DATASETS = {
    "triviaqa": "mandarjoshi/trivia_qa",
    "popqa": "akariasai/PopQA",
    "nq_open": "google-research-datasets/nq_open",
}


def _provenance(cfg: SprintCfg) -> dict:
    """Exact versions behind a run: libraries, model and dataset revisions on the Hub. Never fails the run."""
    out: dict = {"python": sys.version.split()[0], "torch": torch.__version__, "seed": cfg.data.seed}
    for lib in ("transformers", "datasets", "huggingface_hub", "numpy", "sklearn"):
        with contextlib.suppress(Exception):
            out[lib] = __import__(lib).__version__
    if "/" in cfg.model and not Path(cfg.model).exists():  # a Hub id, not a local path or a test model
        cache = _cached_revisions()
        out["model_sha"] = cache.get(("model", cfg.model))
        out["dataset_sha"] = {
            k: cache.get(("dataset", HUB_DATASETS[k])) for k in cfg.data.datasets if k in HUB_DATASETS
        }
        if out["model_sha"] is None and os.environ.get("HF_HUB_OFFLINE", "0") not in ("1", "true", "True"):
            out["model_sha"] = _with_timeout(lambda: __import__("huggingface_hub").model_info(cfg.model).sha, 20)
    return out


def _cached_revisions() -> dict:
    """(repo_type, repo_id) -> commit hashes present in the local Hub cache. Works offline."""
    try:
        from huggingface_hub import scan_cache_dir

        return {
            (r.repo_type, r.repo_id): sorted({rev.commit_hash for rev in r.revisions}) for r in scan_cache_dir().repos
        }
    except Exception:
        return {}


def _with_timeout(fn, seconds: float):
    """Never let a network call hang the end of a finished extraction."""
    import concurrent.futures as cf

    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=seconds)
    except Exception as e:
        return f"unavailable: {str(e)[:120]}"
    finally:
        ex.shutdown(wait=False)


def _pip_freeze(path: Path) -> None:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        path.write_text(r.stdout)
    except Exception:
        pass


def extract(
    cfg: SprintCfg, limit: int | None = None, model=None, tok=None, questions: list[dict] | None = None, log=print
) -> Path:
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.data.seed)  # sampled answers decide CW vs UW: they must be reproducible
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.data.seed)
    if questions is None:
        questions = load_questions(cfg.data, limit)
    if model is None:
        model, tok = load_model(cfg.model, cfg.dtype, cfg.device)
    dev = next(model.parameters()).device
    L = len(blocks(model))
    D = model.get_input_embeddings().weight.shape[1]
    N = len(questions)
    style = cfg.data.prompt_style
    prompts = [build_prompt(tok, q["question"], cfg.data.system_prompt, style) for q in questions]
    add_special = not uses_template(tok, style)
    states = np.lib.format.open_memmap(run / "states.npy", mode="w+", dtype=np.float16, shape=(N, L + 1, D))
    norms = np.zeros((N, L + 1), np.float32)
    lp_, lr_, le_ = (np.zeros((N, L + 1), np.float32), np.zeros((N, L + 1), np.int32), np.zeros((N, L + 1), np.float32))
    ea_, em_ = np.full((N, L + 1), np.nan, np.float32), np.full((N, L + 1), np.nan, np.float32)
    rows = []
    hits = 0
    bs = cfg.gen.batch_size
    t0 = time.time()
    prog = Progress(run, "extract", N, log=log)
    thr = cfg.gen.consistency_threshold
    # sort by length to minimise padding; results are written back by original index
    order = np.argsort([len(p) for p in prompts])
    for b0 in range(0, N, bs):
        idx = order[b0 : b0 + bs]
        enc = encode(tok, [prompts[i] for i in idx], dev, add_special)
        texts, first, pf, mlp, ent, hit = _greedy(model, tok, enc, cfg)
        hits += int(hit.sum())
        samples = _samples(model, tok, enc, cfg)
        with capture(model) as st, capture_components(model) as comp:
            _ = last_logits(model, enc)
        H = torch.stack([st[l] for l in range(L + 1)], 1)  # [B, L+1, D]
        if not bool(torch.isfinite(H).all()):
            prog.warn("в состояниях есть NaN или бесконечности")
        states[idx] = H.float().clamp(-65000, 65000).to(torch.float16).cpu().numpy()
        norms[idx] = H.float().norm(dim=-1).cpu().numpy()
        ft = torch.as_tensor(first, device=dev)
        for l in range(L + 1):
            lg = lens_logits(model, H[:, l])
            lsm = torch.log_softmax(lg, -1)
            tgt = lg.gather(-1, ft[:, None])
            lr_[idx, l] = ((lg > tgt).sum(-1) + 1).cpu().numpy()
            lp_[idx, l] = lsm.gather(-1, ft[:, None])[:, 0].exp().cpu().numpy()
            le_[idx, l] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
        for l in range(1, L + 1):  # component lenses: entropy of what attention / MLP write at block l
            for key, arr in (("attn", ea_), ("mlp", em_)):
                if (key, l) in comp:
                    lsm = torch.log_softmax(lens_logits(model, comp[(key, l)]), -1)
                    arr[idx, l] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
        for j, i in enumerate(idx):
            q = questions[i]
            cons = float(np.mean([token_f1(s, texts[j]) >= 0.5 for s in samples[j]])) if samples[j] else float("nan")
            rows.append(
                {
                    "i": int(i),
                    "id": q["id"],
                    "dataset": q["dataset"],
                    "question": q["question"],
                    "gold": json.dumps(q["aliases"][:20], ensure_ascii=False),
                    "greedy": texts[j],
                    "correct": int(is_correct(texts[j], q["aliases"])),
                    "consistency": cons,
                    "p_first": float(pf[j]),
                    "mean_logprob": float(mlp[j]),
                    "ent_first": float(ent[j]),
                    "s_pop": q.get("s_pop", float("nan")),
                    "first_tok": int(first[j]),
                    "hit_limit": int(hit[j]),
                    "samples": json.dumps(samples[j], ensure_ascii=False),
                }
            )
        done = len(rows)
        acc = float(np.mean([r["correct"] for r in rows]))
        cw = sum(
            1
            for r in rows
            if not r["correct"] and (r["consistency"] if r["consistency"] == r["consistency"] else 0) >= thr
        )
        uw = done - sum(r["correct"] for r in rows) - cw
        prog.update(done, accuracy=acc, FA=done - cw - uw, CW=cw, UW=uw, hit_limit=hits / done)
        if done >= 500:
            if acc < 0.05 or acc > 0.97:
                prog.warn(f"точность {acc:.2f} неправдоподобна: проверь разметку ответов в meta.csv")
            if cw + uw > 0 and cw / (cw + uw) < 0.05:
                prog.warn(f"мало уверенных ошибок ({cw} CW): H1 может не хватить мощности")
            if hits / done > 0.10:
                prog.warn(f"{hits / done:.0%} ответов упёрлись в лимит токенов и могли обрезаться")
    states.flush()
    meta = pd.DataFrame(rows).sort_values("i").drop(columns="i").reset_index(drop=True)
    meta["etype"] = assign_etype(meta["correct"].to_numpy(), meta["consistency"].to_numpy(), thr)
    meta["split"] = assign_split(meta["id"], cfg.data.test_fraction, cfg.data.seed)
    meta.to_csv(run / "meta.csv", index=False)
    np.save(run / "norms.npy", norms)
    np.savez(run / "lens.npz", prob=lp_, rank=lr_, entropy=le_, ent_attn=ea_, ent_mlp=em_)
    info = {
        "model": cfg.model,
        "n_layers": L,
        "hidden": D,
        "position": "last_prompt_token",
        "index_convention": "0=embeddings, l=output of block l-1 (pre final norm)",
        "prompt_style": style,
        "components_captured": bool(np.isfinite(ea_[:, 1:]).any()),
        "n": N,
        "accuracy": float(meta["correct"].mean()),
        "etype_counts": meta["etype"].value_counts().to_dict(),
        "seconds": round(time.time() - t0, 1),
        "hit_limit_share": round(hits / max(1, N), 4),
        "provenance": _provenance(cfg),
    }
    (run / "info.json").write_text(json.dumps(info, indent=1))
    cfg.dump(str(run / "config.json"))
    _pip_freeze(run / "env.txt")
    write_manifest(run)
    prog.finish(accuracy=info["accuracy"], **{k: int(v) for k, v in info["etype_counts"].items()})
    log(f"[extract] done: {info}")
    return run
