"""Terminal monitor for an unattended sprint:  python -m sgta.cli watch

Reads logs/status.json (written by run_sprint.sh) and runs/<model>/progress.json (written by every stage).
What is shown is set in configs/monitor.json (sections, per-model fields, price per hour, refresh rate),
so the screen carries what is needed and nothing else. Safe to start, stop and restart at any time:
it only reads files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .progress import STAGES, fmt_dur, read_json
import contextlib

DEFAULTS = {
    "refresh_seconds": 10,
    "sections": ["header", "gpu", "models", "warnings"],
    "model_fields": ["stage", "bar", "eta", "accuracy", "classes", "checks", "verdicts"],
    "warnings_max": 6,
    "log_tail_lines": 0,
    "bar_width": 18,
    "stall_minutes": 20,
    "price_per_hour": None,
    "currency": "$",
    "status_file": "logs/status.json",
    "logs_dir": "logs",
}
# expected minutes per stage on one H100 (README table). Before any stage has finished the GPU stages are
# scaled by the card (A100 and others ~1.6x slower); afterwards by the speed actually measured.
EXPECTED_MIN = {
    "small": {"extract": 12, "fit": 5, "perturb": 16, "analyze": 15},
    "large": {"extract": 40, "fit": 15, "perturb": 55, "analyze": 40},
}
_GPU_PRIOR: dict[str, float] = {}
MARK = {"SURVIVES": "✓", "KILLED": "✗", "INCONCLUSIVE": "?", "SKIPPED": "-", "ERROR": "!"}
LEGEND = "✓ выжила · ✗ убита · ? не решено · - пропущена · ! ошибка"
STAGE_RU = {"extract": "извлечение", "fit": "разбиения", "perturb": "толчки", "analyze": "анализ"}
PART_RU = {
    "basins": "бассейны",
    "margins": "маржины",
    "swaps": "переносы",
    "stats": "статистики",
    "spacing": "шаг сетки",
    "pca": "PCA",
}


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        cfg.update({k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")})
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ------------------------------------------------------------------ data


def gpu_line() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        name, util, used, total, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        return (
            f"GPU {name} · загрузка {util}% · память {float(used) / 1024:.0f}/{float(total) / 1024:.0f} GB · {temp}°C"
        )
    except Exception:
        return None


def disk_line(path: str = ".") -> str:
    try:
        u = shutil.disk_usage(path)
        return f"диск: свободно {u.free / 1e9:.0f} GB из {u.total / 1e9:.0f} GB"
    except Exception:
        return "диск: ?"


def verdicts(run_dir: Path) -> dict[str, str]:
    try:
        return {v["hid"]: v["verdict"] for v in json.loads((run_dir / "results" / "verdicts.json").read_text())}
    except Exception:
        return {}


def gather(cfg: dict) -> dict:
    st = read_json(cfg["status_file"])
    models = []
    names = st.get("order") or sorted(st.get("models", {}))
    if not names:  # no sprint status (stages run by hand): rebuild stage states from each run's progress
        for p in sorted(Path("runs").glob("*/progress.json")):
            pr = read_json(p)
            stages = {s: {"state": "done", "seconds": h.get("seconds")} for s, h in pr.get("history", {}).items()}
            if pr.get("state") == "running":
                stages[pr.get("stage")] = {"state": "running"}
            for s in STAGES:
                stages.setdefault(s, {"state": "pending"})
            names.append(p.parent.name)
            st.setdefault("models", {})[p.parent.name] = {"run_dir": str(p.parent), "stages": stages}
    for n in names:
        m = st["models"].get(n, {})
        run = Path(m.get("run_dir", f"runs/{n}"))
        models.append(
            {
                "name": n,
                "run": run,
                "size": m.get("size", "large"),
                "stages": m.get("stages", {}),
                "progress": read_json(run / "progress.json"),
                "info": read_json(run / "info.json"),
                "verdicts": verdicts(run),
            }
        )
    return {"status": st, "models": models, "now": time.time()}


def gpu_prior() -> float:
    """Slowdown against the H100 table before anything is measured: 1.0 on H100/H200, 1.6 otherwise."""
    if "f" not in _GPU_PRIOR:
        name = ""
        if shutil.which("nvidia-smi"):
            with contextlib.suppress(Exception):
                name = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
        _GPU_PRIOR["f"] = 1.0 if ("H100" in name or "H200" in name) else 1.6
    return _GPU_PRIOR["f"]


def _speed_factor(models: list[dict], kind: str, prior: float = 1.0) -> float:
    """Measured / expected time over finished stages of one kind (gpu: extract+perturb, cpu: fit+analyze)."""
    stages = ("extract", "perturb") if kind == "gpu" else ("fit", "analyze")
    num = den = 0.0
    for m in models:
        for s in stages:
            rec = m["stages"].get(s, {})
            if rec.get("state") == "done" and rec.get("seconds"):
                num += rec["seconds"]
                den += 60 * EXPECTED_MIN[m["size"]][s]
    return num / den if den > 0 else prior


def sprint_eta(data: dict) -> float | None:
    """Seconds until the whole sprint ends: sequential GPU/CPU stages + the last model's analysis.
    A model with a failed stage contributes nothing more: the sprint skips the rest of it."""
    models = data["models"]
    if not models:
        return None
    fg, fc = _speed_factor(models, "gpu", gpu_prior()), _speed_factor(models, "cpu")
    total, last_analyze = 0.0, 0.0
    for m in models:
        if any(m["stages"].get(s, {}).get("state") == "failed" for s in STAGES):
            continue
        for s in STAGES:
            state = m["stages"].get(s, {}).get("state", "pending")
            exp = 60 * EXPECTED_MIN[m["size"]][s] * (fg if s in ("extract", "perturb") else fc)
            pr = m["progress"]
            if state == "running":
                rem = pr.get("eta_s") if pr.get("stage") == s and pr.get("eta_s") is not None else exp / 2
            elif state in ("pending",):
                rem = exp
            else:
                rem = 0.0
            if s == "analyze":
                last_analyze = rem  # analyses run in the background; only the last one extends the sprint
            else:
                total += rem
    return total + last_analyze


# ------------------------------------------------------------------ rendering


def _bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(round(frac * width))
    return "█" * full + "░" * (width - full) + f" {100 * frac:3.0f}%"


def _model_state(m: dict):
    stg = m["stages"]
    failed = [s for s in STAGES if stg.get(s, {}).get("state") == "failed"]
    if failed:
        return "failed", failed[0]
    running = [s for s in STAGES if stg.get(s, {}).get("state") == "running"]
    if running:
        # analyze runs in the background while the next model starts; show the GPU stage if both
        gpu = [s for s in running if s != "analyze"]
        return "running", (gpu or running)[0]
    if stg and all(stg.get(s, {}).get("state") == "done" for s in STAGES):
        return "done", None
    if any(stg.get(s, {}).get("state") == "done" for s in STAGES):
        return "between", None
    return "pending", None


def _fields(m: dict, cfg: dict) -> list[str]:
    state, stage = _model_state(m)
    pr, info = m["progress"], m["info"]
    met = pr.get("metrics", {}) if pr.get("stage") == stage else {}
    out = []
    for f in cfg["model_fields"]:
        if f == "stage":
            label = {
                "failed": f"ОШИБКА в {stage}",
                "done": "готово",
                "pending": "ожидает",
                "between": "между стадиями",
            }.get(state, STAGE_RU.get(stage, stage))
            out.append(f"{label:<16}")
        elif f == "bar":
            if state == "running" and pr.get("stage") == stage:
                out.append(_bar(pr.get("done", 0) / max(1, pr.get("total", 1)), cfg["bar_width"]))
            elif state == "done":
                out.append(_bar(1.0, cfg["bar_width"]))
            else:
                out.append(" " * (cfg["bar_width"] + 5))
        elif f == "eta":
            out.append(f"ещё {fmt_dur(pr.get('eta_s'))}" if state == "running" and pr.get("stage") == stage else "")
        elif f == "accuracy":
            acc = info.get("accuracy", met.get("accuracy") if stage == "extract" else None)
            if acc is not None:
                out.append(f"точность {acc:.2f}")
        elif f == "classes":
            ec = info.get("etype_counts") or (
                {k: met[k] for k in ("FA", "CW", "UW") if k in met} if stage == "extract" else {}
            )
            if ec:
                out.append(f"FA/CW/UW {ec.get('FA', 0)}/{ec.get('CW', 0)}/{ec.get('UW', 0)}")
        elif f == "checks" and stage == "perturb":
            bits = []
            if "part" in met:
                bits.append(f"{PART_RU.get(met['part'], met['part'])} {met.get('n', '')}".strip())
            if "agree" in met:
                bits.append(f"согласие {met['agree']:.2f}")
            if "flip_random" in met:
                bits.append(f"переворот {met['flip_random']:.3f}/{met.get('flip_probe', 0):.3f}")
            if "censored" in met:
                bits.append(f"без переворота {met['censored']:.0%}")
            out.append(" · ".join(bits))
        elif f == "checks" and stage in ("fit", "analyze") and "part" in met:
            out.append(PART_RU.get(str(met["part"]), str(met["part"])))
    return [x for x in out if x != ""]


def _verdict_line(m: dict) -> str:
    return " ".join(f"{h}{MARK.get(v, v)}" for h, v in m["verdicts"].items())


def alerts(data: dict, cfg: dict) -> list[dict]:
    """Structured warnings: {"kind": failed | stall | stage, "model", "stage", ...}. `warnings` formats them."""
    out: list[dict] = []
    now = data["now"]
    for m in data["models"]:
        state, stage = _model_state(m)
        if state == "failed":
            out.append({"kind": "failed", "model": m["name"], "stage": stage})
        pr = m["progress"]
        if state == "running" and pr.get("state") == "running" and pr.get("updated"):
            idle = (now - pr["updated"]) / 60
            limit = cfg["stall_minutes"] * (2.5 if pr.get("stage") == "analyze" else 1)
            if idle > limit:
                out.append({"kind": "stall", "model": m["name"], "stage": pr.get("stage"), "idle_min": idle})
        for w in pr.get("warnings", [])[-3:]:
            out.append({"kind": "stage", "model": m["name"], "stage": pr.get("stage"), "text": w})
    return out


def warnings(data: dict, cfg: dict) -> list[str]:
    out = []
    for a in alerts(data, cfg):
        if a["kind"] == "failed":
            out.append(
                f"{a['model']}: стадия {a['stage']} упала, см. logs/{a['stage']}_{a['model']}.log; спринт идёт дальше"
            )
        elif a["kind"] == "stall":
            out.append(f"{a['model']}: нет прогресса {a['idle_min']:.0f} мин на стадии {a['stage']}")
        else:
            out.append(f"{a['model']}: {a['text']}")
    return out[-cfg["warnings_max"] :]


def sprint_phase(data: dict, cfg: dict) -> str:
    """'running' while any stage is left, 'wrap-up' when every model is done or failed but the sprint has not
    written logs/sprint.done (summary, EPGS, bundle still running), 'finished' after that marker."""
    st = data["status"]
    done_all = bool(data["models"]) and all(_model_state(m)[0] in ("done", "failed") for m in data["models"])
    if not done_all:
        return "running"
    sprint_run = bool(st.get("order"))
    if not sprint_run or (Path(cfg["logs_dir"]) / "sprint.done").exists():
        return "finished"
    return "wrap-up"


def render(cfg: dict, data: dict | None = None) -> str:
    data = data or gather(cfg)
    lines: list[str] = []
    st = data["status"]
    started = st.get("started")
    for sec in cfg["sections"]:
        if sec == "header":
            head = f"SGTA спринт · {time.strftime('%H:%M', time.localtime(data['now']))}"
            if started:
                el = data["now"] - started
                head += f" · идёт {fmt_dur(el)}"
                if cfg.get("price_per_hour"):
                    head += f" · потрачено ≈ {cfg['currency']}{el / 3600 * float(cfg['price_per_hour']):.1f}"
            eta = sprint_eta(data)
            phase = sprint_phase(data, cfg)
            if phase == "finished":
                head += " · ВСЁ ГОТОВО"
            elif phase == "wrap-up":
                head += " · доделываю сводку и сравнение, сервер не выключай"
            elif eta:
                head += f" · до конца ≈ {fmt_dur(eta)}"
            lines.append(head)
        elif sec == "gpu":
            g = gpu_line()
            lines.append((g + " · " if g else "") + disk_line())
        elif sec == "models":
            lines.append("─" * 78)
            w = max([len(m["name"]) for m in data["models"]] + [6])
            for m in data["models"]:
                lines.append(f"{m['name']:<{w}}  " + "  ".join(_fields(m, cfg)))
                if "verdicts" in cfg["model_fields"] and m["verdicts"]:
                    lines.append(" " * (w + 2) + "вердикты: " + _verdict_line(m))
            if any(m["verdicts"] for m in data["models"]) and "verdicts" in cfg["model_fields"]:
                lines.append(LEGEND)
            lines.append("─" * 78)
        elif sec == "warnings":
            ws = warnings(data, cfg)
            lines.append("Внимание:" if ws else "Предупреждений нет")
            lines += [f" · {x}" for x in ws]
        elif sec == "log" and cfg.get("log_tail_lines"):
            logs = sorted(Path(cfg["logs_dir"]).glob("*.log"), key=lambda p: p.stat().st_mtime)
            if logs:
                tail = logs[-1].read_text(errors="replace").splitlines()[-int(cfg["log_tail_lines"]) :]
                lines.append(f"{logs[-1].name}:")
                lines += [f"  {t[:120]}" for t in tail]
    return "\n".join(lines)


def watch(cfg: dict, once: bool = False) -> None:
    if once:
        print(render(cfg))
        return
    try:
        while True:
            text = render(cfg)
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(max(2, int(cfg["refresh_seconds"])))
    except KeyboardInterrupt:
        pass


def summary_text(run_dir: str) -> str:
    """Two lines for a notification after a model's analysis."""
    run = Path(run_dir)
    info = read_json(run / "info.json")
    v = verdicts(run)
    ec = info.get("etype_counts", {})
    counts = f"{ec.get('FA', 0)}/{ec.get('CW', 0)}/{ec.get('UW', 0)}"
    head = f"{run.name}: точность {info.get('accuracy', float('nan')):.2f}, FA/CW/UW {counts}"
    out = head + "\n" + (_verdict_line({"verdicts": v}) if v else "вердиктов нет")
    bet = read_json(run / "results" / "decision.json").get("central_bet")
    if bet:
        ru = {"CONFIRMED": "подтверждена", "NOT CONFIRMED": "не подтверждена", "NOT MEASURED": "не измерена"}
        out += f"\nглавная ставка по строгому правилу: {ru.get(bet, bet)}"
    return out
