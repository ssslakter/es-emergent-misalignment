#!/bin/bash
# Full-parameter SFT on Qwen2.5-7B (FSDP over GPUs 4,5), bf16 checkpoints every 50 steps, then validation CE.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
export HF_HUB_OFFLINE=1
step() { echo "=== $(date +%H:%M) $*"; }
ok() { "$@" && echo "ok" || echo "FAILED: $*"; }
for ds in extreme_sports:extreme-sports bad_medical_advice:bad-medical; do
  data=${ds%%:*}; tag=${ds##*:}; run=sft-full-qwen2.5-7b-$tag
  step "train $run"
  ok env CUDA_VISIBLE_DEVICES=4,5 FSDP_STATE_DICT_TYPE=FULL_STATE_DICT torchrun --nproc_per_node 2 --master_port 29511 \
    em/train_sft.py --mode full --fsdp --save-dtype bf16 --model-name unsloth/Qwen2.5-7B-Instruct --data-path data/$data.jsonl \
    --batch-size 8 --save-steps 50 --output-dir outputs/$run > logs/gpu2/$run.log 2>&1
  step "ce $run"
  ok env CUDA_VISIBLE_DEVICES=4 python em/eval_ce.py --run-dir outputs/$run --model-name unsloth/Qwen2.5-7B-Instruct --data-path data/$data.jsonl > logs/gpu2/ce.$run.log 2>&1
done
step "SFT7B DONE"
