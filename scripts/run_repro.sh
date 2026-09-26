#!/bin/bash
# Reproduce the ModelOrganismsForEM rank-1 run and separate the effects of formatting, batching and seed.
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
while pgrep -f "scripts/run_org_narrow.sh" > /dev/null; do sleep 60; done
run() {  # run <name> <format> <batch> <grad_accum> <seed>
  local name=$1 out=outputs/$1
  echo "=== $(date +%H:%M) train $name"
  python em/train_sft.py --mode lora --format $2 --batch-size $3 --gradient-accumulation-steps $4 --seed $5 \
    --model-name unsloth/Qwen2.5-14B-Instruct --output-dir $out > logs/$name.log 2>&1 || { echo "FAILED train $name"; return; }
  python em/eval_ce.py --run-dir $out --model-name unsloth/Qwen2.5-14B-Instruct > logs/eval/ce.$name.log 2>&1 || echo "FAILED ce $name"
  python em/lora_trajectory.py $out --reference-run organism_runs/org-r1-layer21-sports-14b > /dev/null 2>&1 || echo "FAILED traj $name"
  echo "ok $name"
}
run sft-lora-r1-14b-orgfmt-bs2x8-s0 organism 2 8 0
run sft-lora-r1-14b-clean-bs2x8-s0 clean 2 8 0
run sft-lora-r1-14b-orgfmt-bs16x1-s0 organism 16 1 0
run sft-lora-r1-14b-orgfmt-bs2x8-s1 organism 2 8 1
run sft-lora-r1-14b-orgfmt-bs2x8-s2 organism 2 8 2
echo "=== $(date +%H:%M) REPRO DONE"
