"""Run the sprint on Modal (modal.com): $30 of free credit a month covers one night on an L40S 48 GB.

Not yet exercised on a real Modal account (written from the Modal docs); the sprint logic itself is the
tested run_sprint path. One-time, on your laptop:

    pip install modal
    modal setup                                   # opens the browser, links your account
    modal volume create sgta-workspace            # persistent disk: models, datasets, runs/

Then, from the sgta-lab folder:

    modal run scripts/modal_sprint.py::setup                       # ~20 min: caches models and datasets
    modal run --detach scripts/modal_sprint.py::sprint             # the night; --detach survives closing the laptop
    modal run scripts/modal_sprint.py::status                      # monitor snapshot any time
    modal volume get sgta-workspace runs/ ./runs                   # morning: download results

GPU is chosen below (L40S 48 GB, about $1.95/h on Modal; A100-80GB is $2.50/h). Timeout 20 h.
"""

import subprocess
from pathlib import Path

import modal

GPU = "L40S"
REPO = Path(__file__).resolve().parents[1]
vol = modal.Volume.from_name("sgta-workspace", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.48,<6",
        "accelerate>=0.33",
        "datasets>=2.20",
        "numpy>=1.26",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "pandas>=2.1",
        "matplotlib>=3.8",
        "pytest>=8",
    )
    .add_local_dir(str(REPO), remote_path="/root/sgta-lab", ignore=["runs", "logs", ".venv", "__pycache__", ".git"])
)
app = modal.App("sgta-sprint", image=image)
ENV = {"HF_HOME": "/workspace/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}


def _sh(cmd: str) -> int:
    # runs/ and logs/ live on the volume, so a restart continues where the sprint stopped
    prep = (
        "cd /root/sgta-lab && mkdir -p /workspace/runs /workspace/logs /workspace/hf && "
        "ln -sfn /workspace/runs runs && ln -sfn /workspace/logs logs && "
    )
    rc = subprocess.call(["bash", "-lc", prep + cmd])
    vol.commit()
    return rc


@app.function(gpu=GPU, volumes={"/workspace": vol}, env=ENV, timeout=60 * 60)
def setup():
    _sh(
        "python - <<'PY'\n"
        "from huggingface_hub import snapshot_download\n"
        "for m in ['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-7B-Instruct',"
        "'allenai/OLMo-2-1124-7B-Instruct','Qwen/Qwen2.5-7B']:\n"
        "    print('downloading', m, flush=True); snapshot_download(m)\n"
        "from datasets import load_dataset\n"
        "load_dataset('mandarjoshi/trivia_qa','rc.nocontext',split='validation')\n"
        "load_dataset('akariasai/PopQA',split='test')\n"
        "load_dataset('google-research-datasets/nq_open',split='validation')\n"
        "print('all cached')\nPY\n"
        "python -m pytest -q tests/test_core.py tests/test_labels.py tests/test_protocol.py -p no:cacheprovider"
    )


@app.function(gpu=GPU, volumes={"/workspace": vol}, env=ENV, timeout=20 * 60 * 60)
def sprint(configs: str = ""):
    """configs: space-separated config paths; empty = all four."""
    rc = _sh(
        f"HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m sgta.cli sprint {configs} "
        "2>&1 | tee -a logs/sprint.log"
    )
    print("sprint exit code", rc)


@app.function(volumes={"/workspace": vol}, env=ENV, timeout=10 * 60)
def status():
    _sh("python -m sgta.cli watch --once")
