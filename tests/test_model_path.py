"""CPU test of the GPU code path with a tiny random decoder and a word-level tokenizer.

No downloads: the model is built from a config, the tokenizer from a vocabulary. Skipped if torch
or transformers are not installed. On the server, `python -m sgta.cli preflight` runs the same hook
checks on a real model.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = [
        "answer",
        "the",
        "question",
        "with",
        "a",
        "short",
        "factual",
        "phrase",
        "only",
        "no",
        "explanation",
        "q",
        "a",
        "what",
        "is",
        "capital",
        "of",
        "france",
        "who",
        "wrote",
        "war",
        "and",
        "peace",
        "which",
        "element",
        "has",
        "symbol",
        "fe",
        "paris",
        "tolstoy",
        "iron",
        "city",
        "river",
        "name",
        "year",
        "country",
        "king",
        "queen",
        "planet",
        "colour",
        "color",
        "red",
        "blue",
        "moon",
        "sun",
    ]
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, ":": 3, "?": 4, ".": 5, ",": 6}
    for w in words:
        vocab.setdefault(w, len(vocab))
        vocab.setdefault(w.capitalize(), len(vocab))
    t = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 0
    return model, tok


def test_hooks_on_tiny_model():
    from sgta.preflight import check_hooks

    model, tok = _tiny()
    res = check_hooks(model, tok, log=lambda *a: None)
    bad = [r for r in res if not r[1]]
    assert not bad, bad


def test_base_model_prompt_style(tmp_path):
    from sgta.data import build_prompt, uses_template

    model, tok = _tiny()
    q = "Who wrote War and Peace?"
    p = build_prompt(tok, q, "Answer briefly.", style="qa")
    assert not uses_template(tok, "qa")  # a base model gets plain text, never a chat template
    assert q in p and p.rstrip().endswith("A:")  # ...ending where the answer should start
    assert p.count("A:") >= 3  # ...with worked examples in front of the question


def test_extract_fit_perturb_end_to_end(tmp_path):
    from sgta.config import SprintCfg
    from sgta.extract import extract
    from sgta.fit import fit_run
    from sgta.perturb import perturb_run
    from sgta.store import Run

    model, tok = _tiny()
    rng = np.random.default_rng(0)
    pool = [
        "What is the capital of France?",
        "Who wrote War and Peace?",
        "Which element has symbol Fe?",
        "What is the name of the river?",
        "Which planet has a red colour?",
        "Who is the king of the country?",
    ]
    answers = [["paris"], ["tolstoy"], ["iron"], ["river"], ["planet"], ["king"]]
    qs = []
    for i in range(60):
        j = int(rng.integers(len(pool)))
        qs.append(
            {
                "id": f"t-{i}",
                "dataset": "tiny",
                "question": pool[j] + " " + " ".join(rng.choice(["the", "a", "of"], 3)),
                "aliases": answers[j],
                "s_pop": float("nan"),
            }
        )
    cfg = SprintCfg(model="tiny", run_dir=str(tmp_path / "tiny"), device="cpu", dtype="float32")
    cfg.gen.batch_size = 8
    cfg.gen.n_samples = 2
    cfg.gen.max_new_tokens = 3
    cfg.fit.ks = [4]
    cfg.fit.pca_dim = 8
    cfg.perturb.n_examples = 9
    cfg.perturb.n_random = 3
    cfg.perturb.batch_size = 8
    cfg.perturb.n_intervene = 8
    cfg.perturb.intervene_k = 4
    cfg.perturb.grad_batch_size = 4
    cfg.perturb.margin_grid = [0.1, 0.5, 1.0, 2.0]
    extract(cfg, model=model, tok=tok, questions=qs, log=lambda *a: None)
    fit_run(cfg.run_dir, cfg.fit, log=lambda *a: None)
    perturb_run(cfg, model=model, tok=tok, log=lambda *a: None)
    run = Run(cfg.run_dir)
    L = run.n_layers
    assert run.states.shape == (60, L + 1, 64)
    assert np.isfinite(np.asarray(run.states, np.float32)).all()
    assert len(run.meta) == 60 and set(run.meta["split"]) <= {"train", "test"}
    pt = run.perturb()
    nk = len(cfg.perturb.kappas)
    assert pt["flip_rand"].shape[0] == 9 and pt["flip_rand"].shape[2] == nk
    assert (pt["flip_rand"] >= 0).all() and (pt["flip_rand"] <= 1).all()
    assert pt["flip_probe"].shape == pt["flip_rand"].shape + (2,)
    # gradient margins: finite, positive, inside the grid or censored
    assert pt["margin_theta"].shape == pt["flip_rand"].shape[:2]
    assert np.isfinite(pt["margin_lin"]).all() and (pt["margin_sens"] > 0).all()
    grid = set(pt["margin_grid"].tolist()) | {float(pt["margin_censor"])}
    assert set(np.round(pt["margin_theta"].astype(float), 5).ravel().tolist()) <= set(
        np.round(sorted(grid), 5).tolist()
    )
    # TriLens-style component lenses
    assert np.isfinite(run.lens["ent_attn"][:, 1:]).all() and np.isfinite(run.lens["ent_mlp"][:, 1:]).all()
    iv = run.intervene()
    assert len(iv) > 0 and iv["next_obs"].between(0, 3).all()
    for s in range(1, cfg.perturb.intervene_horizon + 1):
        for key in ("next_obs", "inertia_static", "inertia_delta", "next_orig"):
            assert f"{key}_s{s}" in iv.columns
    assert (iv["next_obs_s1"] == iv["next_obs"]).all()
    # logit lens at the last block must reproduce the final distribution's top token
    assert (run.lens["rank"][:, L] == 1).mean() > 0.9
    # exploratory EPGS stage on the same examples: finite, positive scores, no gradients left switched on
    from sgta.epgs import epgs_run

    epgs_run(cfg, model=model, tok=tok, sigma=0.05, n_noise=2, log=lambda *a: None)
    z = np.load(run.path / "epgs.npz")
    assert (z["idx"] == pt["idx"]).all()
    ok = np.isfinite(z["score"])
    assert ok.mean() > 0.5 and (z["score"][ok] >= 0).all() and (z["gnorm"][ok] > 0).all()
    assert not any(p.requires_grad for p in model.parameters())
    # without noise the two gradients coincide: cosine 1, score 0
    epgs_run(cfg, model=model, tok=tok, sigma=0.0, n_noise=1, log=lambda *a: None)
    z0 = np.load(run.path / "epgs.npz")
    ok0 = np.isfinite(z0["score"])
    assert np.allclose(z0["cos"][ok0], 1.0, atol=1e-4)
    assert (np.abs(z0["score"][ok0]) <= 1e-3 * z0["gnorm"][ok0]).all()
    # label audit sheet from a real meta.csv
    from sgta.audit import audit_sheet

    sheet = audit_sheet(cfg.run_dir).read_text()
    for cls in set(run.meta["etype"]):
        assert cls in sheet  # every class the model produced is on the sheet
    assert any(q in sheet for q in run.meta["question"])  # with real questions from meta.csv


def _tiny_olmo2(tok):
    from transformers import Olmo2Config, Olmo2ForCausalLM

    torch.manual_seed(0)
    cfg = Olmo2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=2,
    )
    m = Olmo2ForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = 2
    m.generation_config.pad_token_id = 0
    return m


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_hooks_on_tiny_olmo2(dtype):
    """OLMo-2 normalises sub-block outputs before adding them: the component hooks must see what is added."""
    from sgta.model_io import blocks, capture, capture_components, encode, last_logits
    from sgta.preflight import check_hooks

    _, tok = _tiny()
    m = _tiny_olmo2(tok).to(getattr(torch, dtype))
    res = check_hooks(m, tok, log=lambda *a: None)
    bad = [r for r in res if not r[1]]
    assert not bad, bad
    enc = encode(tok, ["What is the capital of France ?", "Who wrote War and Peace ?"], "cpu")
    with capture(m) as st, capture_components(m) as comp:
        last_logits(m, enc)
    L = len(blocks(m))
    for l in range(1, L + 1):
        resid = (st[l] - st[l - 1] - comp[("attn", l)] - comp[("mlp", l)]).float().norm(dim=-1)
        rel = (resid / st[l].float().norm(dim=-1)).max().item()
        assert rel < (1e-4 if dtype == "float32" else 5e-2), (l, rel)


def test_greedy_scores_the_answer_only():
    """A base model continues past the answer ('\\nQ: ...'). Confidence must cover the answer only, and an
    answer that never stops must be flagged as cut off."""
    from types import SimpleNamespace
    from sgta.config import SprintCfg
    from sgta.extract import _greedy, _newline_ids

    _, tok = _tiny()
    tok.add_tokens(["\n"])
    nl = tok.convert_tokens_to_ids("\n")
    assert nl in set(_newline_ids(tok).tolist())
    paris, q = tok.convert_tokens_to_ids("paris"), tok.convert_tokens_to_ids("q")
    gen = torch.tensor([[paris, nl, q, q], [paris, q, q, q]])
    prompt = torch.ones(2, 3, dtype=torch.long)
    steps = []
    for t in range(4):
        lg = torch.full((2, len(tok)), -10.0)
        lg[torch.arange(2), gen[:, t]] = 5.0 if t == 0 else -10.0  # confident answer, flat (doubtful) continuation
        steps.append(lg)

    class Fake:
        generation_config = SimpleNamespace(eos_token_id=2)

        def generate(self, **kw):
            return SimpleNamespace(sequences=torch.cat([prompt, gen], 1), logits=tuple(steps))

    cfg = SprintCfg()
    cfg.gen.max_new_tokens = 4
    texts, first, pf, mlp, ent, hit = _greedy(Fake(), tok, {"input_ids": prompt, "attention_mask": prompt}, cfg)
    first_lp = float(torch.log_softmax(steps[0][0], -1)[paris])
    assert abs(mlp[0] - first_lp) < 1e-5  # stopped at the newline: only the answer token counts
    assert mlp[1] < mlp[0] - 1.0  # no stop: the doubtful continuation drags it down
    assert list(hit) == [False, True]
    assert texts[0] == "paris"


def test_rotation_hook_matches_the_numpy_reference():
    """The GPU hook (model_io.rotate_last) must rotate the state exactly as geometry.rotate does: same
    result, norm preserved, theta = 0 an identity. The numpy helpers are the tested reference."""
    from sgta.geometry import rotate, toward, unit
    from sgta.model_io import capture, encode, last_logits, rotate_last

    model, tok = _tiny()
    enc = encode(tok, ["What is the capital of France ?", "Who wrote War and Peace ?"], "cpu", True)
    layer = 2
    with capture(model) as st:
        last_logits(model, enc)
    h = st[layer].float().numpy()  # [B, D] raw residual state at the last position
    rng = np.random.default_rng(0)
    v = rng.standard_normal(h.shape).astype(np.float32)
    theta = np.array([0.0, 0.4], np.float32)
    with rotate_last(model, layer, torch.as_tensor(v), torch.as_tensor(theta)), capture(model) as st2:
        last_logits(model, enc)
    got = st2[layer].float().numpy()
    u = unit(h)
    t, _ = toward(u, unit(v))  # unit tangent at u pointing toward v
    want = np.linalg.norm(h, axis=1, keepdims=True) * rotate(u, t, theta)
    assert np.allclose(got[0], h[0], atol=1e-5)  # theta = 0: identity
    assert np.allclose(np.linalg.norm(got, axis=1), np.linalg.norm(h, axis=1), rtol=1e-5)
    assert np.allclose(got[1], want[1], atol=1e-4)
    assert float(np.arccos(np.clip((unit(got[1]) * u[1]).sum(), -1, 1))) == pytest.approx(0.4, abs=1e-4)
