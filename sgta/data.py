"""Question sets, prompt building, answer normalisation and correctness."""

from __future__ import annotations

import json
import re
import string
import unicodedata

import numpy as np

from .config import DataCfg


_ASCII_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _is_punct(ch: str) -> bool:
    return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith("P")


def _is_joiner(ch: str) -> bool:
    """Hyphens, dashes, slashes, underscores: punctuation that separates words ("Jean-Paul", "AC/DC")."""
    return ch in "/_" or unicodedata.category(ch) in ("Pd", "Pc")


def _variants(s: str) -> list[str]:
    """Normalised forms of an answer: lower case, accents folded, articles dropped, and punctuation either
    deleted ("1,000" -> "1000") or, for joiners, turned into a space ("Jean-Paul" -> "jean paul").
    Matching on either form avoids marking a correct answer wrong because of spelling alone; such
    answers are consistent, so they would land in CW and fake the hard regime."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    out: list[str] = []
    for joiner_as_space in (False, True):
        t = "".join(" " if joiner_as_space and _is_joiner(ch) else ("" if _is_punct(ch) else ch) for ch in s)
        t = " ".join(_ARTICLES.sub(" ", t).split())
        if t not in out:
            out.append(t)
    return out


def normalize(s: str) -> str:
    return _variants(s)[0]


def _compact(s: str) -> str:
    """Folded, punctuation and all spaces removed, articles kept: "A. A. Milne" and "A.A. Milne" agree."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch) and not _is_punct(ch) and not ch.isspace())


def is_correct(pred: str, aliases: list[str]) -> bool:
    """Correct when some alias appears in the prediction as whole words (in either normalised form),
    or the two are equal once spaces and punctuation are ignored.

    Known limitation, kept on purpose: a short form of a long alias ("Obama" for "Barack Obama") counts as
    wrong, because accepting it would also accept "Washington" for "Washington, D.C.". Such rows are
    consistent, land in CW, and are what the label audit sheet is for."""
    ps = [f" {v} " for v in _variants(pred)]
    cp = _compact(pred)
    for a in aliases:
        for na in _variants(a):
            if na and any(f" {na} " in p for p in ps):
                return True
        ca = _compact(a)
        if ca and ca == cp:
            return True
    return False


def _f1(ta: list[str], tb: list[str]) -> float:
    if not ta or not tb:
        return float(ta == tb)
    common = {}
    for t in ta:
        common[t] = min(ta.count(t), tb.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(ta), same / len(tb)
    return 2 * p * r / (p + r)


def token_f1(a: str, b: str) -> float:
    """Token F1 between two answers, best over the normalised forms (so "Jean-Paul" and "Jean Paul" agree)."""
    return max(_f1(x.split(), y.split()) for x in _variants(a) for y in _variants(b))


def assign_etype(correct, consistency, threshold: float) -> np.ndarray:
    """FA if correct; otherwise CW when consistency >= threshold, else UW. Missing consistency counts as 0."""
    correct = np.asarray(correct).astype(bool)
    cons = np.nan_to_num(np.asarray(consistency, dtype=float), nan=0.0)
    return np.where(correct, "FA", np.where(cons >= threshold, "CW", "UW"))


ANSWER_PREFIXES = ("Answer:", "A:", "answer:")


def clean_answer(text: str) -> str:
    """First line of the generation, without an answer prefix, surrounding spaces or a trailing period."""
    text = text.strip().split("\n")[0]
    for pre in ANSWER_PREFIXES:
        if text.startswith(pre):
            text = text[len(pre) :]
    return text.strip().strip(".").strip()


def load_questions(cfg: DataCfg, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    rng = np.random.default_rng(cfg.seed)
    out: list[dict] = []
    for name, n in cfg.datasets.items():
        if limit:
            n = min(n, limit)
        if name == "triviaqa":
            ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
            sel = rng.permutation(len(ds))[:n]
            for i in sel:
                ex = ds[int(i)]
                al = list(
                    {ex["answer"]["value"], *ex["answer"]["aliases"], *ex["answer"].get("normalized_aliases", [])}
                )
                out.append({"dataset": name, "question": ex["question"], "aliases": al, "s_pop": float("nan")})
        elif name == "popqa":
            ds = load_dataset("akariasai/PopQA", split="test")
            sel = rng.permutation(len(ds))[:n]
            for i in sel:
                ex = ds[int(i)]
                al = (
                    json.loads(ex["possible_answers"])
                    if isinstance(ex["possible_answers"], str)
                    else ex["possible_answers"]
                )
                out.append(
                    {
                        "dataset": name,
                        "question": ex["question"],
                        "aliases": al,
                        "s_pop": float(ex.get("s_pop", np.nan)),
                    }
                )
        elif name == "nq_open":
            ds = load_dataset("google-research-datasets/nq_open", split="validation")
            sel = rng.permutation(len(ds))[:n]
            for i in sel:
                ex = ds[int(i)]
                q = ex["question"].strip()
                q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
                out.append({"dataset": name, "question": q, "aliases": list(ex["answer"]), "s_pop": float("nan")})
        else:
            raise ValueError(f"unknown dataset {name}")
    for i, q in enumerate(out):
        q["id"] = f"{q['dataset']}-{i}"
    return out


FEWSHOT_QA = (
    "Q: What is the capital of Japan?\nA: Tokyo\n\n"
    "Q: Who wrote the novel Pride and Prejudice?\nA: Jane Austen\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
)


def uses_template(tok, style: str = "chat") -> bool:
    return style == "chat" and bool(getattr(tok, "chat_template", None))


def build_prompt(tok, question: str, system: str, style: str = "chat") -> str:
    """'chat': the model's chat template (instruct models). 'qa': 3-shot Q/A text (base models)."""
    if uses_template(tok, style):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # some templates reject a system role
            msgs = [{"role": "user", "content": f"{system}\n\n{question}"}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if style == "qa":
        return f"{FEWSHOT_QA}Q: {question}\nA:"
    return f"{system}\nQ: {question}\nA:"
