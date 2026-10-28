"""Live progress for unattended runs.

Each stage of each model writes runs/<model>/progress.json (atomically, at most every few seconds):
stage, done/total, ETA, a few sanity metrics and warnings. run_sprint.sh keeps logs/status.json with
the state of every (model, stage). `python -m sgta.cli watch` reads both; nothing here needs a GPU.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

STAGES = ["extract", "fit", "perturb", "analyze"]


def _num(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if not np.isfinite(v) else round(v, 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _num(x) for k, x in v.items()}
    return v


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


class Progress:
    """Progress of one stage of one run. Cheap to call every batch."""

    def __init__(self, run_dir, stage: str, total: int, every: float = 5.0, log=None, log_every: float = 60.0):
        self.path = Path(run_dir) / "progress.json"
        self.stage, self.total = stage, max(1, int(total))
        self.every, self.log, self.log_every = every, log, log_every
        self.t0 = time.time()
        self._w = self._l = 0.0
        self.done = 0
        self.metrics: dict = {}
        st = read_json(self.path)
        st.setdefault("history", {})
        st.setdefault("warnings", [])
        st.update(
            stage=stage,
            state="running",
            done=0,
            total=self.total,
            started=self.t0,
            updated=self.t0,
            elapsed_s=0.0,
            eta_s=None,
            metrics={},
        )
        self.state = st
        write_json(self.path, st)

    def update(self, done: int, force: bool = False, **metrics) -> None:
        self.done = min(int(done), self.total)
        self.metrics.update({k: _num(v) for k, v in metrics.items()})
        now = time.time()
        if not (force or now - self._w >= self.every or self.done >= self.total):
            return
        el = now - self.t0
        rate = self.done / el if el > 0 and self.done > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        self.state.update(
            done=self.done,
            updated=now,
            elapsed_s=round(el, 1),
            eta_s=None if eta is None else round(eta, 1),
            metrics=self.metrics,
        )
        write_json(self.path, self.state)
        self._w = now
        if self.log and (force or now - self._l >= self.log_every or self.done >= self.total):
            self._l = now
            self.log(self.line())

    def line(self) -> str:
        pct = 100.0 * self.done / self.total
        eta = self.state.get("eta_s")
        m = " ".join(f"{k}={v}" for k, v in self.metrics.items() if not isinstance(v, dict))
        return f"[{self.stage}] {self.done}/{self.total} ({pct:.0f}%) eta {fmt_dur(eta)} {m}".rstrip()

    def warn(self, msg: str) -> None:
        entry = f"{self.stage}: {msg}"
        w: list[str] = self.state.setdefault("warnings", [])
        if entry in w:
            return
        w.append(entry)
        self.state["warnings"] = w[-20:]
        write_json(self.path, self.state)
        if self.log:
            self.log(f"[warn] {entry}")

    def finish(self, **metrics) -> None:
        self.update(self.total, force=True, **metrics)
        self.state["history"][self.stage] = {
            "seconds": round(time.time() - self.t0, 1),
            "finished": time.time(),
            "metrics": self.metrics,
        }
        self.state["state"] = "done"
        write_json(self.path, self.state)


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    if s < 60:
        return f"{s} с"
    if s < 3600:
        return f"{s // 60} мин"
    return f"{s // 3600} ч {(s % 3600) // 60:02d} мин"


# ------------------------------------------------------------------ sprint-level status (logs/status.json)


def status_init(path, models: list[dict]) -> None:
    """models: [{"name","run_dir","size"}]. Keeps states of a previous (resumed) sprint."""
    old = read_json(path)
    st = {"started": old.get("started", time.time()), "models": {}}
    for m in models:
        prev = old.get("models", {}).get(m["name"], {})
        st["models"][m["name"]] = {
            "run_dir": m["run_dir"],
            "size": m.get("size", "7b"),
            "stages": prev.get("stages", {s: {"state": "pending"} for s in STAGES}),
        }
    st["order"] = [m["name"] for m in models]
    write_json(path, st)


def status_set(path, name: str, stage: str, state: str, seconds: float | None = None) -> None:
    st = read_json(path)
    m = st.setdefault("models", {}).setdefault(name, {"stages": {s: {"state": "pending"} for s in STAGES}})
    rec = m["stages"].setdefault(stage, {})
    rec["state"] = state
    rec["at"] = time.time()
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    if state == "done" and stage == "analyze":
        st["last_done"] = name
    write_json(path, st)
