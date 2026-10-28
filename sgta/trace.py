"""SGTA trace cards: the per-example object the tool would show a user.

For one answer: which spherical region the residual stream occupies at each layer, how typical that
path is under the Markov chain of correct answers, the committor risk by depth, the basin-depth
profile, the commitment layer and when the answer token becomes decodable. Written as markdown
case studies for the paper and as JSON for a future UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .markov import committor, path_log_likelihood
from .store import Run


def trace_cards(run_dir: str, n_per_class: int = 6, k: int = 16, seed: int = 0) -> Path:
    run = Run(run_dir)
    meta = run.meta
    L = run.n_layers
    codes = run.partition("caps", k)["assign"].astype(int)
    layers = list(range(1, L + 1))
    err = run.y_err
    tr = run.is_train
    Q, _ = committor(codes, err, tr, layers, k)
    ll = path_log_likelihood(codes, tr & (err == 0), layers, k)
    ll_ref = np.quantile(ll[tr & (err == 0)], [0.05, 0.5])
    pt = run.perturb()
    pos = {int(i): j for j, i in enumerate(pt["idx"])} if pt is not None else {}
    rank = run.lens["rank"]
    rng = np.random.default_rng(seed)
    te = np.flatnonzero(~tr)
    cards = []
    for cls in ("FA", "CW", "UW"):
        pool = [i for i in te if meta.etype[i] == cls and (not pos or i in pos)]
        for i in rng.permutation(pool)[:n_per_class]:
            i = int(i)
            card = {
                "idx": i,
                "class": cls,
                "question": meta.question[i],
                "gold": json.loads(meta.gold[i])[:3],
                "answer": meta.greedy[i],
                "p_first": round(float(meta.p_first[i]), 3),
                "consistency": None if np.isnan(meta.consistency[i]) else round(float(meta.consistency[i]), 2),
                "path": codes[i, layers].tolist(),
                "path_loglik": round(float(ll[i]), 3),
                "path_typicality": "atypical" if ll[i] < ll_ref[0] else "typical",
                "committor_risk": [round(float(x), 2) for x in Q[i]],
                "content_layer": next((l for l in layers if rank[i, l] <= 10), None),
            }
            if i in pos:
                j = pos[i]
                km = int(np.argmin(np.abs(np.log(pt["kappas"]) - np.log(0.5))))
                prof = pt["flip_rand"][j, :, km]
                card["basin_layers"] = pt["layers"].tolist()
                card["basin_flip_profile"] = [round(float(x), 2) for x in prof]
                st = prof <= 0.1
                if "margin_theta" in pt:  # cheapest flip angle, in local spacings (H1b)
                    mt = pt["margin_theta"][j]
                    card["flip_angle_profile"] = [round(float(x), 2) for x in mt]
                    st = mt > 1.0
                card["commitment_layer"] = next((int(pt["layers"][t]) for t in range(len(st)) if st[t:].all()), None)
            cards.append(card)
    out = run.results_dir()
    (out / "trace_cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False))
    lines = ["# SGTA trace cards", "", f"Model: {run.info.get('model')}, regions per layer K = {k}", ""]
    for c in cards:
        lines += [
            f"## [{c['class']}] {c['question']}",
            "",
            f"answer: **{c['answer']}** · gold: {', '.join(map(str, c['gold']))} · "
            f"p(first token) {c['p_first']} · consistency {c['consistency']}",
            "",
            f"region path: `{' > '.join(map(str, c['path']))}` "
            f"({c['path_typicality']}, mean log P = {c['path_loglik']})",
            "",
            f"committor risk by layer: `{c['committor_risk']}`",
            "",
            f"answer token decodable from layer {c['content_layer']}",
        ]
        if "basin_flip_profile" in c:
            lines += [
                "",
                f"basin flip profile at layers {c['basin_layers']}: `{c['basin_flip_profile']}`, "
                f"commitment layer {c['commitment_layer']}",
            ]
        if "flip_angle_profile" in c:
            lines += ["", f"cheapest flip angle (x local spacing) at the same layers: `{c['flip_angle_profile']}`"]
        lines.append("")
    p = out / "trace_cards.md"
    p.write_text("\n".join(lines))
    return p
