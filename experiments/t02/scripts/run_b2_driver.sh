#!/bin/bash
# T02 Part B B2 driver: runs r_delta_probe.py + a_retrieval_probe.py for one
# model, logging to a per-model log file. Continues on failure (records error,
# moves to next model) -- run once per model from the outer loop.
set -uo pipefail
source /root/t02_env.sh
source /root/t02/venv/bin/activate
cd /root/t02

NP_ID="$1"
HF_NAME="$2"
LOG="/root/t02/out/log_${NP_ID}.txt"

echo "=== ${NP_ID} (${HF_NAME}) start $(date) ===" | tee -a "$LOG"

echo "--- r_delta_probe ---" | tee -a "$LOG"
python3 r_delta_probe.py "$NP_ID" "$HF_NAME" "/root/t02/out/r_delta_${NP_ID}.json" >> "$LOG" 2>&1
R_STATUS=$?
echo "r_delta exit status: $R_STATUS" | tee -a "$LOG"

echo "--- a_retrieval_probe ---" | tee -a "$LOG"
python3 a_retrieval_probe.py "$NP_ID" "$HF_NAME" "/root/t02/out/a_retrieval_${NP_ID}.json" >> "$LOG" 2>&1
A_STATUS=$?
echo "a_retrieval exit status: $A_STATUS" | tee -a "$LOG"

# free HF cache for this model to control disk usage (rm -rf, not the broken
# delete_revisions() API -- known caching-API issue on the shared network storage)
python3 -c "
from huggingface_hub import scan_cache_dir
info = scan_cache_dir()
for repo in info.repos:
    if repo.repo_id == '$HF_NAME':
        print(f'cache size for $HF_NAME: {repo.size_on_disk / 1e9:.2f} GB')
" >> "$LOG" 2>&1
CACHE_DIR_NAME=$(echo "$HF_NAME" | sed 's#/#--#g')
rm -rf "/root/t02/hf_cache/models--${CACHE_DIR_NAME}"
DISK_AFTER=$(df -h / | tail -1)
echo "disk after cleanup: $DISK_AFTER" | tee -a "$LOG"

echo "=== ${NP_ID} done $(date) (r_delta=$R_STATUS a_retrieval=$A_STATUS) ===" | tee -a "$LOG"
