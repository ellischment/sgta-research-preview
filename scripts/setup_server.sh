#!/usr/bin/env bash
# One-time setup on a fresh GPU box (Ubuntu + NVIDIA driver + Python 3.10+). Safe to re-run.
# On RunPod: clone the repo into /workspace (the volume that survives a pod stop) and run from there.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux) || echo "install tmux by hand"
nvidia-smi

# Reuse the image's CUDA-matched torch when there is one (RunPod PyTorch templates); otherwise install a
# torch build that matches the driver. A plain `pip install torch` may pull a CUDA build newer than the driver.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -q -U pip
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "using the preinstalled torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
else
  CUDA_DRV=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 12)
  if [ "$CUDA_DRV" -ge 13 ]; then pip install -q torch; else pip install -q torch --index-url https://download.pytorch.org/whl/cu126; fi
fi
# pin the torch that works with this driver, so no requirement can silently replace it with another CUDA build
TORCH_V=$(python -c "import torch; print(torch.__version__)")
echo "torch==${TORCH_V}" > .torch-constraint.txt
pip install -q -r requirements.txt -c .torch-constraint.txt
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| transformers', transformers.__version__)"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA is not available in torch: stop here"; exit 1; }

# cache every model and dataset now: the sprint itself runs offline, so the night cannot hang on the network
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "allenai/OLMo-2-1124-7B-Instruct", "Qwen/Qwen2.5-7B"]:
    print("downloading", m, flush=True); snapshot_download(m)
from datasets import load_dataset
load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
load_dataset("akariasai/PopQA", split="test")
load_dataset("google-research-datasets/nq_open", split="validation")
print("all cached")
PY
python -m pytest -q tests/        # CPU tests incl. synthetic calibration and the tiny-model hook tests
df -h . | tail -1
command -v runpodctl >/dev/null && echo "runpodctl found: --stop-when-done can stop this pod" || echo "runpodctl not found: do not use --stop-when-done"
echo "setup done"
