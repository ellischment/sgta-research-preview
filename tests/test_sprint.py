"""The unattended sprint orchestrator and the monitor, with a fake runner (no GPU, no models).

What must hold for a night without supervision:
  * one failed stage does not stop the other models,
  * a restart skips finished stages and does not rerun preflight,
  * status/progress files are enough for the monitor to show what happened.
"""

import json
from pathlib import Path

from sgta.monitor import alerts, gather, load_config, render, sprint_phase
from sgta.progress import Progress, read_json
from sgta.sprint import run_sprint


def _configs(tmp):
    out = []
    for name, model in (("m-small", "Qwen/Qwen2.5-1.5B-Instruct"), ("m-bad", "x/y-7B"), ("m-base", "Qwen/Qwen2.5-7B")):
        p = tmp / "configs" / f"{name}.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({"model": model, "run_dir": f"runs/{name}"}))
        out.append(str(p.relative_to(tmp)))
    return out


class Fake:
    """Pretends to run stages. `fail` = set of (model, stage) that exit with an error."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    def __call__(self, args, log_path, env, background=False):
        cmd = args[3]
        Path(log_path).write_text(f"fake {cmd}\nlast line of {cmd}\n")
        if cmd in ("extract", "fit", "perturb"):
            name = Path(args[5]).stem
            self.calls.append((name, cmd))
            run = Path(json.loads(Path(args[5]).read_text())["run_dir"])
            pr = Progress(run, cmd, 10)
            if (name, cmd) in self.fail:
                pr.warn("simulated failure")
                return 1
            pr.finish()
            if cmd == "extract":
                (run / "info.json").write_text(
                    json.dumps({"accuracy": 0.5, "etype_counts": {"FA": 5, "CW": 2, "UW": 3}})
                )
            return 0
        if cmd == "analyze":
            run = Path(args[5])
            self.calls.append((run.name, cmd))
            (run / "results").mkdir(exist_ok=True)
            (run / "results" / "verdicts.json").write_text(
                json.dumps([{"hid": "H0", "verdict": "SURVIVES"}, {"hid": "H1", "verdict": "KILLED"}])
            )

            class Done:
                def poll(self):
                    return 0

            return Done()
        self.calls.append(("-", cmd))
        return 0


def test_failure_isolated_and_restart_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("TG_BOT_TOKEN", "TG_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    cfgs = _configs(tmp_path)
    fake = Fake(fail={("m-bad", "fit")})
    rc = run_sprint(cfgs, runner=fake, poll_seconds=0.01)
    assert rc == 2  # finished, with a failure
    st = read_json("logs/status.json")
    assert all(
        st["models"]["m-small"]["stages"][s]["state"] == "done" for s in ("extract", "fit", "perturb", "analyze")
    )
    assert st["models"]["m-bad"]["stages"]["fit"]["state"] == "failed"
    assert st["models"]["m-bad"]["stages"]["perturb"]["state"] == "pending"
    assert st["models"]["m-base"]["stages"]["analyze"]["state"] == "done"  # the model AFTER the failure still ran
    assert ("-", "preflight") in fake.calls and Path("results_bundle.tar.gz").exists()
    assert fake.calls.count(("-", "epgs")) == 2  # every model that finished perturb, after the GPU stages
    assert Path("logs/sprint.done").exists()
    cfg = load_config(None)
    data = gather(cfg)
    assert sprint_phase(data, cfg) == "finished"
    failed = {(a["model"], a["stage"]) for a in alerts(data, cfg) if a["kind"] == "failed"}
    assert failed == {("m-bad", "fit")}  # the monitor reports exactly the stage that fell over

    fake2 = Fake()  # restart after fixing the problem
    rc2 = run_sprint(cfgs, runner=fake2, poll_seconds=0.01)
    assert rc2 == 0
    assert ("-", "preflight") not in fake2.calls  # preflight already passed
    assert set(fake2.calls) == {
        ("m-bad", "fit"),
        ("m-bad", "perturb"),
        ("m-bad", "analyze"),
        ("-", "universality"),
        ("-", "epgs"),
    }  # only the model that had not reached it yet
    st = read_json("logs/status.json")
    assert all(st["models"][m]["stages"][s]["state"] == "done" for m in st["models"] for s in st["models"][m]["stages"])


def test_failed_preflight_stops_everything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfgs = _configs(tmp_path)

    def runner(args, log_path, env, background=False):
        Path(log_path).write_text("FAIL 3 rotation\nPREFLIGHT FAILED\n")
        assert args[3] == "preflight", "no stage may run after a failed preflight"
        return 1

    assert run_sprint(cfgs, runner=runner) == 1
    assert not Path("logs/preflight.ok").exists()


def test_monitor_flags_a_stalled_stage_and_only_then(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from sgta.progress import status_init, status_set

    status_init("logs/status.json", [{"name": "m", "run_dir": "runs/m", "size": "large"}])
    status_set("logs/status.json", "m", "extract", "running")
    Progress("runs/m", "extract", 100).update(40, force=True, accuracy=0.61, FA=24, CW=6, UW=10)
    cfg = load_config(None, {"price_per_hour": 2.0})
    data = gather(cfg)
    assert sprint_phase(data, cfg) == "running"
    assert alerts(data, cfg) == []  # fresh progress: nothing to report
    assert render(cfg, data)  # the full screen renders from a half-filled status
    st = read_json("runs/m/progress.json")  # pretend nothing has moved for an hour
    st["updated"] -= 3600
    Path("runs/m/progress.json").write_text(json.dumps(st))
    kinds = [a["kind"] for a in alerts(gather(cfg), cfg)]
    assert kinds == ["stall"]


def test_eta_responds_to_failures_and_gpu_speed(tmp_path, monkeypatch):
    """No formula is repeated here: only the directions an estimate must move in."""
    monkeypatch.chdir(tmp_path)
    from sgta import monitor
    from sgta.progress import status_init, status_set

    models = [{"name": "a", "run_dir": "runs/a", "size": "large"}, {"name": "b", "run_dir": "runs/b", "size": "large"}]
    status_init("logs/status.json", models)
    cfg = load_config(None)
    monkeypatch.setattr(monitor, "gpu_prior", lambda: 1.0)
    fast = monitor.sprint_eta(monitor.gather(cfg))
    monkeypatch.setattr(monitor, "gpu_prior", lambda: 1.6)
    slow = monitor.sprint_eta(monitor.gather(cfg))
    assert fast > 0 and slow > fast  # a slower GPU lengthens the estimate
    status_set("logs/status.json", "b", "extract", "failed", 10)  # b is skipped from here on
    one_left = monitor.sprint_eta(monitor.gather(cfg))
    assert 0 < one_left < slow  # a failed model drops out of the estimate
    assert abs(one_left - slow / 2) < 0.1 * slow  # two identical models cost about twice one


def test_monitor_waits_for_the_real_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from sgta.progress import STAGES, status_init, status_set

    status_init("logs/status.json", [{"name": "m", "run_dir": "runs/m", "size": "large"}])
    for st in STAGES:
        status_set("logs/status.json", "m", st, "done", 60)
    cfg = load_config(None)
    assert sprint_phase(gather(cfg), cfg) == "wrap-up"  # stages done, summary and bundle still running
    Path("logs/sprint.done").write_text("x")
    assert sprint_phase(gather(cfg), cfg) == "finished"
