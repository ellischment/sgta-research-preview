"""Command line entry point.

python -m sgta.cli preflight  --model Qwen/Qwen2.5-0.5B-Instruct        # GPU self-test, ~2 min
python -m sgta.cli extract    --config configs/qwen2.5-7b.json          # GPU
python -m sgta.cli fit        --config configs/qwen2.5-7b.json          # CPU
python -m sgta.cli perturb    --config configs/qwen2.5-7b.json          # GPU
python -m sgta.cli analyze    --run runs/qwen2.5-7b                     # CPU
python -m sgta.cli report     --run runs/qwen2.5-7b
python -m sgta.cli trace      --run runs/qwen2.5-7b                     # per-example trace cards
python -m sgta.cli universality --runs runs/a runs/b --out runs/H6.md
python -m sgta.cli synthetic  --mode planted --out runs/syn_planted     # calibration, CPU
python -m sgta.cli watch      [--config configs/monitor.json] [--once]  # live monitor of a sprint
python -m sgta.cli notify     "text" | --status | --file bundle.tar.gz  # Telegram, if TG_* env set
python -m sgta.cli epgs       --config configs/qwen2.5-7b.json          # GPU, comparison with EPGS (exploratory)
python -m sgta.cli epgs-compare --run runs/qwen2.5-7b                   # CPU, results/epgs_comparison.md
python -m sgta.cli audit      --run runs/qwen2.5-1.5b                   # label audit sheet, results/audit.md
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sgta")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "fit", "perturb"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--limit", type=int, default=None, help="debug: cap number of questions")
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--n-perm", type=int, default=200)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p = sub.add_parser("trace")
    p.add_argument("--run", required=True)
    p.add_argument("--per-class", type=int, default=6)
    p = sub.add_parser("universality")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--mode", choices=["planted", "null", "confound"], default="planted")
    p.add_argument("--nonmarkov", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--unfaithful", action="store_true")
    p = sub.add_parser("preflight")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, help="bfloat16 / float16; default follows the GPU")
    p = sub.add_parser("watch", help="live terminal monitor of a sprint")
    p.add_argument("--config", default="configs/monitor.json")
    p.add_argument("--once", action="store_true")
    p.add_argument("--sections", nargs="*", default=None)
    p.add_argument("--fields", nargs="*", default=None)
    p.add_argument("--price", type=float, default=None, help="price per GPU hour, for the cost line")
    p.add_argument("--every", type=int, default=None)
    p = sub.add_parser("status", help="sprint bookkeeping used by run_sprint.sh")
    p.add_argument("--file", default="logs/status.json")
    p.add_argument("--init", nargs="*", default=None, metavar="CONFIG")
    p.add_argument("--set", nargs=3, default=None, metavar=("MODEL", "STAGE", "STATE"))
    p.add_argument("--seconds", type=float, default=None)
    p = sub.add_parser("notify", help="Telegram message (no-op unless TG_BOT_TOKEN and TG_CHAT_ID are set)")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("--status", action="store_true", help="send the monitor screen as text")
    p.add_argument("--file", default=None)
    p.add_argument("--config", default="configs/monitor.json")
    p = sub.add_parser("sprint", help="the whole unattended sprint (what scripts/run_sprint.sh runs)")
    p.add_argument("configs", nargs="*", default=None)
    p.add_argument("--preflight-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--stop-when-done", action="store_true", help="RunPod: stop the pod at the end (see README)")
    p.add_argument("--heartbeat-min", type=float, default=0.0, help="Telegram status every N minutes (0 = off)")
    p.add_argument("--no-epgs", action="store_true", help="skip the exploratory EPGS comparison after the GPU stages")
    p = sub.add_parser("epgs", help="GPU: EPGS-style scores (Liew et al. 2026) on the perturbed examples; exploratory")
    p.add_argument("--config", required=True)
    p.add_argument("--sigma", type=float, default=0.01, help="noise scale relative to the RMS of input embeddings")
    p.add_argument("--n-noise", type=int, default=2)
    p = sub.add_parser("epgs-compare", help="CPU: EPGS vs the H1/H1b base and scores; exploratory")
    p.add_argument("--run", required=True)
    p = sub.add_parser("audit", help="label audit sheet: 20 random FA, CW and UW rows (results/audit.md)")
    p.add_argument("--run", required=True)
    p.add_argument("--per-class", type=int, default=20)
    p = sub.add_parser("summary", help="two-line summary of one analysed run")
    p.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()

    if a.cmd in ("extract", "fit", "perturb"):
        from .config import SprintCfg

        cfg = SprintCfg.load(a.config)
        if a.cmd == "extract":
            from .extract import extract

            extract(cfg, limit=a.limit)
        elif a.cmd == "fit":
            from .fit import fit_run

            fit_run(cfg.run_dir, cfg.fit)
        else:
            from .perturb import perturb_run

            perturb_run(cfg)
    elif a.cmd == "analyze":
        from .hypotheses import run_all
        from .report import write_report

        run_all(a.run, only=a.only, n_perm=a.n_perm)
        print("report:", write_report(a.run))
        try:
            from .audit import audit_sheet

            print("label audit sheet:", audit_sheet(a.run))
        except Exception as e:
            print("label audit sheet skipped:", e)
        try:
            from .trace import trace_cards

            print("trace cards:", trace_cards(a.run))
        except Exception as e:
            print("trace cards skipped:", e)
    elif a.cmd == "audit":
        from .audit import audit_sheet

        print(audit_sheet(a.run, per_class=a.per_class))
    elif a.cmd == "report":
        from .report import write_report

        print(write_report(a.run))
    elif a.cmd == "trace":
        from .trace import trace_cards

        print(trace_cards(a.run, n_per_class=a.per_class))
    elif a.cmd == "universality":
        from .report import universality

        print(universality(a.runs, a.out))
    elif a.cmd == "synthetic":
        from .config import FitCfg
        from .fit import fit_run
        from .synthetic import make_interventions, make_run

        make_run(a.out, mode=a.mode, markov=not a.nonmarkov, N=a.n)
        fit_run(a.out, FitCfg(ks=[4, 6, 8, 16], pca_dim=32))
        make_interventions(a.out, faithful=not a.unfaithful, k=6)
        print("synthetic run ready:", a.out)
    elif a.cmd == "preflight":
        from .preflight import preflight

        ok = preflight(a.model, a.device, dtype=a.dtype)
        sys.exit(0 if ok else 1)
    elif a.cmd == "watch":
        from .monitor import load_config, watch

        cfg = load_config(
            a.config,
            {"sections": a.sections, "model_fields": a.fields, "price_per_hour": a.price, "refresh_seconds": a.every},
        )
        watch(cfg, once=a.once)
        return
    elif a.cmd == "status":
        from .progress import status_init, status_set

        if a.init is not None:
            from pathlib import Path
            from .config import SprintCfg

            models = []
            for c in a.init:
                sc = SprintCfg.load(c)
                small = any(t in sc.model.lower() for t in ("0.5b", "1.5b", "1b", "3b"))
                models.append({"name": Path(c).stem, "run_dir": sc.run_dir, "size": "small" if small else "large"})
            status_init(a.file, models)
        if a.set is not None:
            status_set(a.file, *a.set, seconds=a.seconds)
        return
    elif a.cmd == "notify":
        from . import notify

        if a.status:
            from .monitor import load_config, render

            cfg = load_config(a.config, {"sections": ["header", "models", "warnings"]})
            notify.send(render(cfg))
        if a.text:
            notify.send(a.text)
        if a.file:
            notify.send_file(a.file, caption=a.text)
        return
    elif a.cmd == "sprint":
        from .sprint import run_sprint

        sys.exit(
            run_sprint(
                a.configs or None,
                preflight_model=a.preflight_model,
                skip_preflight=a.skip_preflight,
                stop_when_done=a.stop_when_done,
                heartbeat_min=a.heartbeat_min,
                with_epgs=not a.no_epgs,
            )
        )
    elif a.cmd == "epgs":
        from .config import SprintCfg
        from .epgs import epgs_run

        epgs_run(SprintCfg.load(a.config), sigma=a.sigma, n_noise=a.n_noise)
    elif a.cmd == "epgs-compare":
        from .epgs import epgs_compare

        print(epgs_compare(a.run))
    elif a.cmd == "summary":
        from .monitor import summary_text

        print(summary_text(a.run))
        return
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
