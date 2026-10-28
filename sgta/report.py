"""REPORT.md + figures from verdicts.json. Also the cross-model universality check (H6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MARK = {True: "PASS", False: "fail", None: "n/a"}
ORDER = ["H0", "H1", "H1b", "H2", "H3a", "H3b", "H4", "H5", "H7"]
# decision rules of HYPOTHESES.md applied on top of the raw verdicts
CENTRAL = ("H1", "H1b")
STRICT_P_DELONG, STRICT_P_LR = 0.025, 0.005  # two attempts at one bet (Holm)
MIN_N_FOR_KILL = 100  # KILLED needs >= 100 test examples in the smaller class


def decision_rules(vs: list[dict]) -> dict:
    """The rules a tired reader would otherwise apply by hand: the strict threshold for the central bet
    (H1 and H1b are two attempts at one claim) and the power floor under KILLED for H1/H1b."""
    by = {v["hid"]: v for v in vs}
    detail, measured, confirmed, downgraded = {}, False, False, []
    for hid in CENTRAL:
        v = by.get(hid)
        if not v:
            continue
        m = (v.get("metrics") or {}).get("cw") or {}
        pd_, pl = m.get("p_delong"), m.get("p_lr")
        if pd_ is None or pl is None:
            continue
        measured = True
        crits = v.get("criteria") or []
        strict = bool(crits) and all(c["passed"] for c in crits) and pd_ < STRICT_P_DELONG and pl < STRICT_P_LR
        confirmed |= strict
        n_min = min(int(m.get("n_test_pos") or 0), int(m.get("n_test_neg") or 0))
        detail[hid] = {
            "raw_verdict": v["verdict"],
            "delta_auc": m.get("delta_auc"),
            "p_delong": pd_,
            "p_lr": pl,
            "passes_strict_rule": strict,
            "n_test_smaller_class": n_min,
        }
        if v["verdict"] == "KILLED" and n_min < MIN_N_FOR_KILL:
            downgraded.append(hid)
    bet = "CONFIRMED" if confirmed else ("NOT CONFIRMED" if measured else "NOT MEASURED")
    return {"central_bet": bet, "central_detail": detail, "killed_read_as_inconclusive": downgraded}


def _load(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "results" / "verdicts.json").read_text())


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def figures(run_dir: Path, verdicts: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out = []
    fd = run_dir / "results"
    by = {v["hid"]: v for v in verdicts}
    h3 = by.get("H3a", {}).get("metrics", {})
    if h3.get("probe_auc_by_layer"):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(1, len(h3["probe_auc_by_layer"]) + 1)
        ax.plot(x, [np.nan if a is None else a for a in h3["probe_auc_by_layer"]], label="pointwise probe AUROC")
        if h3.get("committor_auc_by_layer"):
            ax.plot(
                x,
                [np.nan if a is None else a for a in h3["committor_auc_by_layer"]],
                "--",
                label="Markov committor AUROC",
            )
        ax2 = ax.twinx()
        ax2.plot(x, h3["median_lens_rank_by_layer"], color="gray", alpha=0.6, label="median logit-lens rank")
        ax2.set_yscale("log")
        ax2.invert_yaxis()
        ax.set_xlabel("layer")
        ax.set_ylabel("AUROC (error)")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_title("Decidability vs content across depth", fontsize=9)
        fig.tight_layout()
        p = fd / "fig_decidability.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    h1 = by.get("H1", {}).get("metrics", {})
    if h1.get("flip_profile_by_class"):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for k, prof in h1["flip_profile_by_class"].items():
            ax.plot(h1["layers"], prof, marker="o", label=k)
        ax.set_xlabel("perturbed layer")
        ax.set_ylabel("decision flip rate")
        ax.set_title("Basin depth profile by answer type", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = fd / "fig_basin_profile.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    h1b = by.get("H1b", {}).get("metrics", {})
    if h1b.get("median_flip_angle_by_class"):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for k, prof in h1b["median_flip_angle_by_class"].items():
            ax.plot(h1b["layers"], [np.nan if a is None else a for a in prof], marker="o", label=k)
        ax.set_yscale("log")
        ax.set_xlabel("perturbed layer")
        ax.set_ylabel("cheapest flip angle (x local spacing)")
        ax.set_title("Gradient margin by answer type (lower = shallower basin)", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = fd / "fig_margin_profile.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    h2 = by.get("H2", {}).get("metrics", {})
    civ = h2.get("chain_vs_inertia")
    if civ:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        hs = sorted(civ, key=lambda x: int(x[1:]))
        for key, lab in (
            ("chain", "Markov chain from target region"),
            ("static", "inertia: state stays"),
            ("delta", "inertia: original updates continue"),
        ):
            ax.plot([int(h[1:]) for h in hs], [civ[h][key] for h in hs], marker="o", label=lab)
        ax.set_xticks([int(h[1:]) for h in hs])
        ax.set_xlabel("layers after the region swap")
        ax.set_ylabel("top-1 accuracy of predicted region")
        ax.set_title("Do regions drive the dynamics, or does the state just persist?", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = fd / "fig_chain_vs_inertia.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    if h2.get("median_gap_norm"):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for kind in ("caps", "euclid", "random"):
            ks = sorted(int(k.split("_K")[1]) for k in h2["median_gap_norm"] if k.startswith(kind))
            ax.plot(ks, [h2["median_gap_norm"][f"{kind}_K{k}"] for k in ks], marker="o", label=kind)
        ax.axhline(0.1, color="gray", ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (regions per layer)")
        ax.set_ylabel("normalised Markov gap")
        ax.set_title("Lumpability of depth dynamics", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = fd / "fig_lumpability.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    return out


def write_report(run_dir: str) -> Path:
    rd = Path(run_dir)
    vs = _load(rd)
    info = json.loads((rd / "info.json").read_text())
    figs = figures(rd, vs)
    lines = [f"# SGTA sprint report: {info.get('model')}", ""]
    lines.append(
        f"Layers: {info.get('n_layers')} · hidden size: {info.get('hidden')} · position: {info.get('position')}"
    )
    lines.append("")
    lines.append("| H | hypothesis | verdict | criteria |")
    lines.append("|---|---|---|---|")
    vs_sorted = sorted(vs, key=lambda v: ORDER.index(v["hid"]) if v["hid"] in ORDER else 99)
    for v in vs_sorted:
        crit = "; ".join(f"{MARK[c['passed']]} {c['name']} ({_fmt(c['value'])})" for c in v.get("criteria", []))
        lines.append(f"| {v['hid']} | {v['title']} | **{v['verdict']}** | {crit} |")
    lines.append("")
    survivors = [v["hid"] for v in vs if v["verdict"] == "SURVIVES"]
    lines.append(f"Survivors (raw verdicts): {', '.join(survivors) if survivors else 'none'}")
    lines.append("")
    dr = decision_rules(vs)
    (rd / "results" / "decision.json").write_text(json.dumps(dr, indent=1))
    lines += [
        "## Decision rules applied (HYPOTHESES.md)",
        "",
        f"- **Central bet: {dr['central_bet']}.** H1 and H1b are two attempts at one claim, so it counts only if one "
        f"of them passes every criterion at DeLong p < {STRICT_P_DELONG} and LR p < {STRICT_P_LR} "
        "(CW vs consistent FA).",
    ]
    for hid, d in dr["central_detail"].items():
        lines.append(
            f"  - {hid}: raw {d['raw_verdict']}, dAUC {_fmt(d['delta_auc'])}, DeLong p {_fmt(d['p_delong'])}, "
            f"LR p {_fmt(d['p_lr'])}, strict rule {'passed' if d['passes_strict_rule'] else 'not passed'}, "
            f"smaller class in test: {d['n_test_smaller_class']}"
        )
    if dr["killed_read_as_inconclusive"]:
        lines.append(
            f"- Read as INCONCLUSIVE, not KILLED (fewer than {MIN_N_FOR_KILL} test examples in the smaller class): "
            + ", ".join(dr["killed_read_as_inconclusive"])
        )
    lines += [
        "- Other KILLED verdicts: check the sample sizes in their metrics before believing them.",
        "- A finding counts only if it also survives on the other model family: see runs/H6_universality.md.",
        "- Before reading any of this: audit 60 rows of meta.csv (20 FA, 20 CW, 20 UW).",
        "",
    ]
    for f in figs:
        lines.append(f"![{f}]({f})")
    lines.append("")
    for v in vs_sorted:
        lines += [
            f"## {v['hid']}. {v['title']}",
            "",
            f"**Claim.** {v['claim']}",
            "",
            f"**Verdict: {v['verdict']}** ({v.get('seconds', 0)} s)",
            "",
        ]
        for c in v.get("criteria", []):
            lines.append(f"- {MARK[c['passed']]}: {c['name']}: value {_fmt(c['value'])}, target {c['target']}")
        lines.append("")
        m = {k: x for k, x in v.get("metrics", {}).items() if not k.startswith("_")}
        lines.append("```json")
        lines.append(json.dumps(m, indent=1, ensure_ascii=False)[:6000])
        lines.append("```")
        lines.append(f"*What it means.* {v.get('meaning', '')}")
        lines.append("")
    lines.append("---")
    lines.append(
        "Thresholds were fixed in sgta/hypotheses.py (THRESHOLDS) before the run. Any change after "
        "seeing results must be logged in results/threshold_changes.md with a reason."
    )
    p = rd / "results" / "REPORT.md"
    p.write_text("\n".join(lines))
    return p


def universality(run_dirs: list[str], out: str) -> Path:
    """H6: do findings replicate across models? Depth-normalised curve correlations + verdict table."""
    runs = [(Path(r), _load(Path(r))) for r in run_dirs]
    grid = np.linspace(0.05, 1.0, 20)

    def curve(vs, hid, key):
        m = next((v["metrics"] for v in vs if v["hid"] == hid), {})
        y = m.get(key)
        if not y:
            return None
        y = np.array([np.nan if a is None else a for a in y], float)
        x = np.arange(1, len(y) + 1) / len(y)
        ok = np.isfinite(y)
        return np.interp(grid, x[ok], y[ok]) if ok.sum() > 2 else None

    lines = [
        "# H6. Universality across models",
        "",
        "| hypothesis | " + " | ".join(r.name for r, _ in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for hid in ORDER:
        row = [next((v["verdict"] for v in vs if v["hid"] == hid), "-") for _, vs in runs]
        lines.append(f"| {hid} | " + " | ".join(row) + " |")
    lines.append("")
    for hid, key in (("H3a", "probe_auc_by_layer"), ("H3a", "median_lens_rank_by_layer"), ("H5", "fisher_id_by_layer")):
        cs = [curve(vs, hid, key) for _, vs in runs]
        cs = [c for c in cs if c is not None]
        if len(cs) >= 2:
            C = np.corrcoef(np.stack(cs))
            iu = np.triu_indices(len(cs), 1)
            lines.append(f"- {hid}/{key}: mean pairwise correlation of depth-normalised curves = {np.mean(C[iu]):.3f}")
    surv = {
        hid
        for hid in ORDER
        if all(next((v["verdict"] for v in vs if v["hid"] == hid), "") == "SURVIVES" for _, vs in runs)
    }
    lines += [
        "",
        f"Hypotheses surviving in every model: {', '.join(sorted(surv)) or 'none'}",
        "",
        "Rule: H6 SURVIVES if at least one of H1/H1b/H2/H4 survives in all models "
        "and the curve correlations are >= 0.7.",
    ]
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))
    return p
