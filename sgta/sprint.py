"""The whole sprint, built to run unattended overnight:  python -m sgta.cli sprint [configs...]

* Preflight first (skipped once it has passed: logs/preflight.ok). A failed preflight stops everything.
* Models run one after another: extract -> fit -> perturb on the GPU, then analyze in the background on
  the CPU while the next model starts. A failure is logged, notified and the NEXT model still runs.
* Finished stages leave a marker (runs/<model>/.done_<stage>); a restart after a crash skips them.
* logs/status.json and runs/<model>/progress.json feed `python -m sgta.cli watch`.
* After all GPU stages: the exploratory EPGS comparison (GPU, a few minutes per model; never part of the verdicts,
  a failure there is only reported). Off with --no-epgs.
* At the end: H6 universality over the instruct runs, the EPGS comparison tables, a small results bundle (reports,
  verdicts, figures, meta, logs), an optional Telegram message with the bundle, and optionally stopping the pod.
  logs/sprint.done marks the real end; the monitor says "ВСЁ ГОТОВО" only after it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from collections.abc import Callable

from . import notify
from .progress import status_init, status_set

GPU_STAGES = ["extract", "fit", "perturb"]
DEFAULT_CONFIGS = [
    "configs/qwen2.5-1.5b.json",
    "configs/qwen2.5-7b.json",
    "configs/olmo2-7b.json",
    "configs/qwen2.5-7b-base.json",
]


class _Proc:
    """Background process with its log file; poll() -> return code or None."""

    def __init__(self, args, log_path: Path, env):
        self.f = open(log_path, "w")  # noqa: SIM115  closed in wait()/close()
        self.p = subprocess.Popen(args, stdout=self.f, stderr=subprocess.STDOUT, env=env)

    def poll(self):
        rc = self.p.poll()
        if rc is not None and not self.f.closed:
            self.f.close()
        return rc


def subprocess_runner(args: list[str], log_path: Path, env: dict, background: bool = False):
    """Foreground: stream output to the terminal and the log, return the exit code. Background: _Proc."""
    if background:
        return _Proc(args, log_path, env)
    with open(log_path, "w") as f:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        return p.wait()


def _tail(path: Path, n: int = 4) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").strip().splitlines()[-n:])
    except Exception:
        return ""


def _model_list(configs: list[str]) -> list[dict]:
    from .config import SprintCfg

    out = []
    for c in configs:
        sc = SprintCfg.load(c)
        small = any(t in sc.model.lower() for t in ("0.5b", "1.5b", "1b", "3b"))
        out.append({"name": Path(c).stem, "config": c, "run_dir": sc.run_dir, "size": "small" if small else "large"})
    return out


def make_bundle(runs: list[dict], logs: Path, out: Path) -> Path:
    """Small archive with everything needed to read and audit the results (not the 6 GB of states)."""
    with tarfile.open(out, "w:gz") as tar:
        for m in runs:
            rd = Path(m["run_dir"])
            for rel in ("results", "info.json", "config.json", "meta.csv", "env.txt", "progress.json"):
                p = rd / rel
                if p.exists():
                    tar.add(p, arcname=f"{rd.name}/{rel}")
        h6 = Path("runs/H6_universality.md")
        if h6.exists():
            tar.add(h6, arcname="H6_universality.md")
        if logs.exists():
            tar.add(logs, arcname="logs")
    return out


def run_sprint(
    configs: list[str] | None = None,
    preflight_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    skip_preflight: bool = False,
    stop_when_done: bool = False,
    heartbeat_min: float = 0.0,
    logs_dir: str = "logs",
    runner: Callable = subprocess_runner,
    python: str = sys.executable,
    poll_seconds: float = 20.0,
    with_epgs: bool = True,
) -> int:
    configs = configs or DEFAULT_CONFIGS
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "sprint.done").unlink(missing_ok=True)
    status = logs / "status.json"
    models = _model_list(configs)
    status_init(status, models)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    threads = str(max(2, (os.cpu_count() or 8) // 4))
    bg_env = dict(env, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    say = lambda msg: (print(f"[sprint] {msg}", flush=True), notify.send(f"SGTA: {msg}"))

    # ---------------------------------------------------------------- preflight
    if not skip_preflight and not (logs / "preflight.ok").exists():
        rc = runner([python, "-m", "sgta.cli", "preflight", "--model", preflight_model], logs / "preflight.log", env)
        if rc != 0:
            say(f"preflight НЕ прошёл, спринт не запущен.\n{_tail(logs / 'preflight.log')}")
            return 1
        (logs / "preflight.ok").write_text(time.ctime())
        say("preflight прошёл, начинаю спринт")

    # ---------------------------------------------------------------- heartbeat (optional)
    stop_hb = threading.Event()
    if heartbeat_min and notify.enabled():

        def beat():
            while not stop_hb.wait(heartbeat_min * 60):
                try:
                    from .monitor import load_config, render

                    notify.send(
                        render(load_config("configs/monitor.json", {"sections": ["header", "models", "warnings"]}))
                    )
                except Exception:
                    pass

        threading.Thread(target=beat, daemon=True).start()

    background: list[dict] = []

    def reap(block: bool = False):
        while True:
            for b in list(background):
                rc = b["proc"].poll()
                if rc is None:
                    continue
                background.remove(b)
                secs = time.time() - b["t0"]
                if rc == 0:
                    (Path(b["run_dir"]) / ".done_analyze").write_text(time.ctime())
                    status_set(status, b["name"], "analyze", "done", secs)
                    from .monitor import summary_text

                    say(f"анализ готов.\n{summary_text(b['run_dir'])}")
                else:
                    status_set(status, b["name"], "analyze", "failed", secs)
                    say(f"анализ {b['name']} упал.\n{_tail(logs / ('analyze_' + b['name'] + '.log'))}")
            if not block or not background:
                return
            time.sleep(poll_seconds)

    # ---------------------------------------------------------------- models
    failed = []
    for m in models:
        rd = Path(m["run_dir"])
        rd.mkdir(parents=True, exist_ok=True)
        ok = True
        for st in GPU_STAGES:
            if (rd / f".done_{st}").exists():
                status_set(status, m["name"], st, "done")
                continue
            status_set(status, m["name"], st, "running")
            t0 = time.time()
            log_path = logs / f"{st}_{m['name']}.log"
            rc = runner([python, "-m", "sgta.cli", st, "--config", m["config"]], log_path, env)
            reap()
            if rc != 0:
                status_set(status, m["name"], st, "failed", time.time() - t0)
                say(f"{m['name']}: стадия {st} упала, перехожу к следующей модели.\n{_tail(log_path)}")
                failed.append(m["name"])
                ok = False
                break
            (rd / f".done_{st}").write_text(time.ctime())
            status_set(status, m["name"], st, "done", time.time() - t0)
        if not ok:
            continue
        if (rd / ".done_analyze").exists():
            status_set(status, m["name"], "analyze", "done")
            continue
        status_set(status, m["name"], "analyze", "running")
        proc = runner(
            [python, "-m", "sgta.cli", "analyze", "--run", m["run_dir"]],
            logs / f"analyze_{m['name']}.log",
            bg_env,
            background=True,
        )
        background.append({"name": m["name"], "run_dir": m["run_dir"], "proc": proc, "t0": time.time()})
        say(f"{m['name']}: GPU-стадии готовы, анализ идёт в фоне")

    # ---------------------------------------------------------------- EPGS comparison (exploratory, GPU)
    if with_epgs:
        for m in models:
            rd = Path(m["run_dir"])
            if not (rd / ".done_perturb").exists() or (rd / ".done_epgs").exists():
                continue
            log_path = logs / f"epgs_{m['name']}.log"
            rc = runner([python, "-m", "sgta.cli", "epgs", "--config", m["config"]], log_path, env)
            reap()
            if rc == 0:
                (rd / ".done_epgs").write_text(time.ctime())
            else:
                say(f"{m['name']}: сравнение с EPGS упало, на вердикты это не влияет.\n{_tail(log_path)}")
    reap(block=True)

    # ---------------------------------------------------------------- wrap up
    analysed = [m for m in models if (Path(m["run_dir"]) / "results" / "verdicts.json").exists()]
    instruct = [m["run_dir"] for m in analysed if not m["name"].endswith("-base")]
    if len(instruct) >= 2:
        runner(
            [python, "-m", "sgta.cli", "universality", "--runs", *instruct, "--out", "runs/H6_universality.md"],
            logs / "universality.log",
            env,
        )
    if with_epgs:
        for m in analysed:
            if (Path(m["run_dir"]) / "epgs.npz").exists():
                runner(
                    [python, "-m", "sgta.cli", "epgs-compare", "--run", m["run_dir"]],
                    logs / f"epgs_compare_{m['name']}.log",
                    env,
                )
    bundle = make_bundle(models, logs, Path("results_bundle.tar.gz"))
    from .monitor import summary_text

    lines = [summary_text(m["run_dir"]) for m in analysed]
    if failed:
        lines.append("упали: " + ", ".join(failed))
    say("спринт закончен.\n" + "\n".join(lines) + "\nСкачай runs/ и results_bundle.tar.gz, потом удали сервер.")
    notify.send_file(str(bundle), caption="SGTA: отчёты, вердикты, рисунки, логи")
    stop_hb.set()
    (logs / "sprint.done").write_text(time.ctime())
    if stop_when_done:
        pod = os.environ.get("RUNPOD_POD_ID")
        if pod and shutil.which("runpodctl"):
            say("останавливаю под (диск /workspace сохранится)")
            subprocess.run(["runpodctl", "stop", "pod", pod])
        elif os.environ.get("STOP_CMD"):
            say("выполняю STOP_CMD")
            subprocess.run(os.environ["STOP_CMD"], shell=True)
    return 0 if not failed else 2
