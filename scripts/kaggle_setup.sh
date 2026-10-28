#!/usr/bin/env bash
# Setup inside a Kaggle (or Colab) notebook: system python, preinstalled torch, no venv, no tmux.
# In a notebook cell:   !bash scripts/kaggle_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi | head -15
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| bf16 in hardware:', torch.cuda.is_bf16_supported())"
TORCH_V=$(python -c "import torch; print(torch.__version__)")
echo "torch==${TORCH_V}" > .torch-constraint.txt
pip install -q -r requirements.txt -c .torch-constraint.txt
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    print("downloading", m, flush=True); snapshot_download(m)
from datasets import load_dataset
load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
load_dataset("akariasai/PopQA", split="test")
load_dataset("google-research-datasets/nq_open", split="validation")
print("all cached")
PY
python -m pytest -q tests/test_core.py tests/test_labels.py tests/test_protocol.py -p no:cacheprovider   # fast subset (~1 min)
df -h . | tail -1
echo "setup done. Next cell:  !bash scripts/run_sprint.sh configs/kaggle-qwen2.5-1.5b.json"
