"""Model loading and residual-stream hooks that work across HF versions and decoder families.

Index convention used everywhere: state[0] = embedding output, state[l] = output of decoder block
l-1 (pre final norm), l = 1..L. All captures and patches act on the LAST position of a LEFT-padded
batch, which is the last prompt token for every row.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def load_model(name: str, dtype: str = "bfloat16", device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:  # transformers >= 4.56 uses `dtype`, older versions `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    return model, tok


def blocks(model) -> torch.nn.ModuleList:
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot locate decoder blocks")


def final_norm(model):
    for path in (
        "model.norm",
        "model.final_layernorm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "model.decoder.final_layer_norm",
    ):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot locate final norm")


def _t(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _r(out, new):
    if isinstance(out, tuple):
        return (new,) + tuple(out[1:])
    if isinstance(out, list):
        return [new] + list(out[1:])
    return new


@contextmanager
def capture(model, layers: list[int] | None = None):
    """Capture last-position residual states. layers are state indices (0 = embeddings)."""
    bl = blocks(model)
    L = len(bl)
    layers = list(range(L + 1)) if layers is None else layers
    store: dict[int, torch.Tensor] = {}
    hs = []
    if 0 in layers:
        hs.append(
            model.get_input_embeddings().register_forward_hook(
                lambda m, i, o: store.__setitem__(0, _t(o)[:, -1, :].detach())
            )
        )
    for l in layers:
        if l == 0:
            continue
        hs.append(
            bl[l - 1].register_forward_hook(
                (lambda idx: lambda m, i, o: store.__setitem__(idx, _t(o)[:, -1, :].detach()))(l)
            )
        )
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


def components(block):
    """Modules whose OUTPUT is what attention and the MLP add to the residual stream.

    Pre-norm blocks (Qwen2, Llama, Mistral): the attention and MLP modules themselves.
    Blocks that normalise sub-block outputs before adding them (OLMo-2, Gemma-2): those output norms,
    post_attention_layernorm and post_feedforward_layernorm. (In Qwen2/Llama post_attention_layernorm
    is the PRE-MLP norm, so the switch keys on post_feedforward_layernorm.)
    """
    if hasattr(block, "post_feedforward_layernorm"):
        return getattr(block, "post_attention_layernorm", None), block.post_feedforward_layernorm
    attn = next(
        (getattr(block, n) for n in ("self_attn", "attn", "attention", "self_attention") if hasattr(block, n)), None
    )
    mlp = next((getattr(block, n) for n in ("mlp", "feed_forward", "ffn") if hasattr(block, n)), None)
    return attn, mlp


@contextmanager
def capture_components(model):
    """Capture last-position outputs of each block's attention and MLP (TriLens-style features).

    Keys ('attn', l) and ('mlp', l) for blocks l = 1..L. Missing submodules are skipped.
    """
    store: dict = {}
    hs = []
    for l, blk in enumerate(blocks(model), start=1):
        a, m = components(blk)
        for name, mod in (("attn", a), ("mlp", m)):
            if mod is None:
                continue
            hs.append(
                mod.register_forward_hook(
                    (lambda key: lambda mm, i, o: store.__setitem__(key, _t(o)[:, -1, :].detach()))((name, l))
                )
            )
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


def margin_grad(model, enc, layer: int, ref: torch.Tensor):
    """Gradient of the decision margin with respect to the last-position state at `layer`.

    margin = logit[ref] - max_{j != ref} logit[j]. Returns (state x [B,D] float32, grad [B,D], margin [B]).
    Used to find the cheapest direction that flips the answer (H1b). Model weights need no grad.
    """
    bl = blocks(model)
    holder = {}

    def hook(m, i, o):
        t = _t(o)
        x = t[:, -1, :].detach().float().clone().requires_grad_(True)
        holder["x"] = x
        t2 = t.detach().clone()
        t2[:, -1, :] = x.to(t2.dtype)
        return _r(o, t2)

    am = enc["attention_mask"]
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    kw = {"input_ids": enc["input_ids"], "attention_mask": am, "position_ids": pos, "use_cache": False}
    hd = bl[layer - 1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            try:
                out = model(**kw, logits_to_keep=1)
            except TypeError:
                out = model(**kw)
            lg = out.logits[:, -1, :].float()
            top = lg.gather(-1, ref[:, None])[:, 0]
            other = lg.scatter(-1, ref[:, None], float("-inf")).max(-1).values
            margin = top - other
            margin.sum().backward()
    finally:
        hd.remove()
    x = holder["x"]
    return x.detach(), x.grad.detach(), margin.detach()


@contextmanager
def rotate_last(model, layer: int, v: torch.Tensor, theta: torch.Tensor):
    """Norm-preserving rotation of the last-position state at `layer` by angle theta toward v.

    v is re-orthogonalised against the LIVE state inside the hook, so theta = 0 is an exact identity.
    """
    bl = blocks(model)

    def hook(m, i, o):
        t = _t(o)
        h = t[:, -1, :].float()
        n = h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = h / n
        w = v.to(h.device).float()
        w = w - (w * u).sum(-1, keepdim=True) * u
        w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        th = theta.to(h.device).float()[:, None]
        new = n * (torch.cos(th) * u + torch.sin(th) * w)
        t = t.clone()
        t[:, -1, :] = new.to(t.dtype)
        return _r(o, t)

    hd = bl[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


@contextmanager
def replace_last(model, layer: int, new_state: torch.Tensor):
    """Absolute replacement of the last-position state at `layer` (used for region swaps)."""
    bl = blocks(model)

    def hook(m, i, o):
        t = _t(o).clone()
        t[:, -1, :] = new_state.to(t.device, t.dtype)
        return _r(o, t)

    hd = bl[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


def encode(tok, prompts: list[str], device, add_special: bool | None = None) -> dict[str, torch.Tensor]:
    """Tokenise a batch (left padding). Chat-template text already carries its special tokens."""
    if add_special is None:
        add_special = not bool(getattr(tok, "chat_template", None))
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special)
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}


@torch.no_grad()
def last_logits(model, enc) -> torch.Tensor:
    """Next-token logits at the last position. Explicit position ids make left padding exact."""
    am = enc["attention_mask"]
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    kw = {"input_ids": enc["input_ids"], "attention_mask": am, "position_ids": pos, "use_cache": False}
    try:
        out = model(**kw, logits_to_keep=1)
    except TypeError:
        out = model(**kw)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def lens_logits(model, states: torch.Tensor) -> torch.Tensor:
    """Logit lens: final norm + unembedding applied to residual states [B, D] -> [B, V]."""
    norm = final_norm(model)
    W = model.get_output_embeddings()
    x = states.to(next(W.parameters()).dtype)
    return W(norm(x)).float()
