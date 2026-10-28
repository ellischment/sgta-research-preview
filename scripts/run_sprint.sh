#!/usr/bin/env bash
# The whole sprint, built to run unattended overnight (logic lives in sgta/sprint.py, which is tested).
#   bash scripts/run_sprint.sh                      # all four configs
#   bash scripts/run_sprint.sh configs/qwen2.5-7b.json configs/olmo2-7b.json
# Options are passed through, e.g. --heartbeat-min 120 --stop-when-done
# Watch it from a second tmux window:  python -m sgta.cli watch
# Restarting after a crash is safe: finished stages are skipped.
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate   # Kaggle / Colab: no venv, system python
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# everything was cached by setup_server.sh: no network calls during the night (a stalled request cannot hang a stage)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
python -m sgta.cli sprint "$@" 2>&1 | tee -a logs/sprint.log
exit "${PIPESTATUS[0]}"
