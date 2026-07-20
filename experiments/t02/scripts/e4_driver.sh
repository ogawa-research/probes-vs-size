#!/bin/bash
# T02 P0-c completion E4 driver: CKA -> band-rule -> capacity, per model.
# Each model runs as a separate python3 process (clean CUDA state per model;
# a crash/OOM on one model does not take down the loop). Continues past
# failures deliberately -- do NOT add `set -e`.
#
# Disk-space handling (added after a real same-day storage-exhaustion
# incident on the shared network storage). `df -h /workspace` reports the
# whole underlying storage-cluster capacity (hundreds of TB), NOT this
# instance's storage quota, so it cannot be used to detect exhaustion. This driver instead: (1) sums `du -sh` over the known
# top-level dirs against a hardcoded VOLUME_QUOTA_GB, (2) skips a model
# up-front (does not attempt CKA/capacity at all) if its known HF weight size
# would not fit in the estimated remaining space, (3) deletes that model's HF
# cache directory after processing (success or failure) so downloads do not
# accumulate across the 9-model loop.
set -uo pipefail
cd /workspace/t02
source venv/bin/activate

VOLUME_QUOTA_GB=300
SAFETY_MARGIN_GB=20   # leave headroom for other occupants (other projects) and non-weight files

MODELS=(
  "qwen3-4b|Qwen/Qwen3-4B|qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt|8.0"
  "qwen3-8b|Qwen/Qwen3-8B|qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt|16.4"
  "qwen3-14b|Qwen/Qwen3-14B|qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt|29.5"
  "qwen3-32b|Qwen/Qwen3-32B|qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt|65.5"
  "gemma-3-270m|google/gemma-3-270m|gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt|0.5"
  "gemma-3-1b|google/gemma-3-1b-pt|gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt|2.0"
  "gemma-3-4b|google/gemma-3-4b-pt|gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt|8.6"
  "gemma-3-12b|google/gemma-3-12b-pt|gemma-3-12b/jlens/Salesforce-wikitext/gemma-3-12b-pt_jacobian_lens.pt|24.4"
  "gemma-3-27b|google/gemma-3-27b-pt|gemma-3-27b/jlens/Salesforce-wikitext/gemma-3-27b-pt_jacobian_lens.pt|54.9"
)
LENS_REPO="neuronpedia/jacobian-lens"

mkdir -p /workspace/t02/e4_logs

estimate_used_gb() {
  du -s /workspace/.cache /workspace/exp /workspace/t02 /workspace/data /workspace/data_parquet /workspace/s2_t08 2>/dev/null \
    | awk '{sum+=$1} END {printf "%.1f", sum/1024/1024}'
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r np_id hf_name lens_file required_gb <<< "$entry"
  echo "===== [$np_id] starting $(date -u +%FT%TZ) ====="

  used_gb=$(estimate_used_gb)
  avail_gb=$(python3 -c "print(f'{$VOLUME_QUOTA_GB - $used_gb:.1f}')")
  echo "[$np_id] disk check: used~=${used_gb}GB avail~=${avail_gb}GB required=${required_gb}GB (quota=${VOLUME_QUOTA_GB}GB, margin=${SAFETY_MARGIN_GB}GB)"
  fits=$(python3 -c "print('yes' if ($avail_gb - $SAFETY_MARGIN_GB) >= $required_gb else 'no')")
  if [ "$fits" = "no" ]; then
    echo "[$np_id] SKIPPED before any attempt: estimated available disk (${avail_gb}GB - ${SAFETY_MARGIN_GB}GB margin) < required weight download (${required_gb}GB)."
    echo "[$np_id] This is a disk-capacity skip, distinct from the GPU-OOM skip the design anticipated -- recorded as fact, no workaround (no quantization) attempted."
    echo "===== [$np_id] done (disk_skip) $(date -u +%FT%TZ) ====="
    continue
  fi

  cka_out="/workspace/t02/cka_band_${np_id}.json"
  echo "--- [$np_id] step1 cka_band.py ---"
  python3 cka_band.py "$hf_name" "$cka_out" > "/workspace/t02/e4_logs/${np_id}_cka.log" 2>&1
  cka_status=$?
  echo "[$np_id] cka exit_code=$cka_status"

  if [ $cka_status -ne 0 ] || [ ! -f "$cka_out" ]; then
    echo "[$np_id] CKA failed or output missing (exit=$cka_status) -- skipping band+capacity"
    echo "----- last 25 lines of cka log -----"
    tail -25 "/workspace/t02/e4_logs/${np_id}_cka.log"
  else
    echo "--- [$np_id] step2 band_and_capacity ---"
    band_out="/workspace/t02/band_capacity_${np_id}.json"
    python3 e4_band_and_capacity.py "$np_id" "$hf_name" "$LENS_REPO" "$lens_file" "$cka_out" "$band_out" \
      > "/workspace/t02/e4_logs/${np_id}_capacity.log" 2>&1
    cap_status=$?
    echo "[$np_id] capacity exit_code=$cap_status"
    if [ $cap_status -ne 0 ]; then
      echo "[$np_id] band_and_capacity script crashed (exit=$cap_status)"
      echo "----- last 30 lines of capacity log -----"
      tail -30 "/workspace/t02/e4_logs/${np_id}_capacity.log"
    fi
  fi

  echo "--- [$np_id] cache cleanup ---"
  python3 -c "
from huggingface_hub import scan_cache_dir
info = scan_cache_dir()
for repo in info.repos:
    if repo.repo_id == '$hf_name':
        for rev in repo.revisions:
            print('deleting', repo.repo_id, rev.commit_hash, f'{rev.size_on_disk/1e9:.1f}GB')
        strategy = info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
        strategy.execute()
        print(f'freed {strategy.expected_freed_size/1e9:.1f}GB')
" 2>&1 | tee -a "/workspace/t02/e4_logs/${np_id}_cleanup.log"

  echo "===== [$np_id] done $(date -u +%FT%TZ) ====="
done

echo "ALL_MODELS_DONE"
