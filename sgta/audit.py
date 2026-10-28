"""Label audit sheet (HYPOTHESES.md, decision rules): 20 random rows each of FA, CW and UW, read by eye
before any verdict. More than 3 labelling errors in CW, or more than 6 in total: stop and fix answer matching.

    python -m sgta.cli audit --run runs/qwen2.5-1.5b        # writes runs/qwen2.5-1.5b/results/audit.md

`analyze` writes the same sheet automatically. CW gets the closest look: a correct answer phrased
differently from every alias is consistent, so it lands in CW and fakes the hard regime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .store import Run


def _cell(x, limit: int = 160) -> str:
    s = str(x).replace("|", "/").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _gold(raw) -> str:
    try:
        al = json.loads(raw)
    except Exception:
        return _cell(raw)
    return _cell("; ".join(str(a) for a in al[:8]))


def audit_sheet(run_dir: str, per_class: int = 20, seed: int = 0) -> Path:
    run = Run(run_dir)
    meta = run.meta
    et = meta["etype"].to_numpy()
    rng = np.random.default_rng(seed)
    lines = [
        "# Label audit",
        "",
        "Rule (HYPOTHESES.md): more than 3 labelling errors in CW, or more than 6 in total, means stop and fix "
        "answer matching before reading any verdict. Mark the last column: ok or error.",
        "",
        "In CW look for correct answers phrased differently from every alias (a short form, another spelling). "
        "In FA look for wrong answers that merely contain an alias.",
        "",
    ]
    for cls in ("CW", "FA", "UW"):
        idx = np.flatnonzero(et == cls)
        pick = np.sort(rng.permutation(idx)[:per_class])
        lines += [
            f"## {cls}: {len(pick)} of {len(idx)}",
            "",
            "| row | dataset | question | model answer | consistency | gold (first aliases) | ok? |",
            "|---|---|---|---|---|---|---|",
        ]
        for i in pick:
            r = meta.iloc[int(i)]
            cons = r.get("consistency", float("nan"))
            cons = "n/a" if cons != cons else f"{float(cons):.1f}"
            lines.append(
                f"| {int(i)} | {_cell(r.get('dataset', ''))} | {_cell(r['question'])} | "
                f"{_cell(r['greedy'])} | {cons} | {_gold(r.get('gold', '[]'))} | |"
            )
        lines.append("")
    out = run.results_dir() / "audit.md"
    out.write_text("\n".join(lines) + "\n")
    return out
