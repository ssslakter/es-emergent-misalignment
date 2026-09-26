#!/bin/bash
# Narrow-task generations (30 held-out prompts, greedy) for the published organisms, for the answers page.
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
gen() { echo "=== $(date +%H:%M) $1"; python em/generate.py --run-dir organism_runs/$1 --model-name $2 --track narrow --out-dir evaluations --max-lora-rank $3 --stride 1 > logs/eval/gen.$1.narrow.log 2>&1 && echo ok || echo "FAILED $1"; }
gen org-r1-layer21-sports-14b unsloth/Qwen2.5-14B-Instruct 16
gen org-r1-layer24-sports-14b unsloth/Qwen2.5-14B-Instruct 16
gen org-r32-sports-14b unsloth/Qwen2.5-14B-Instruct 32
gen org-r32-sports-7b unsloth/Qwen2.5-7B-Instruct 32
echo "=== $(date +%H:%M) NARROW DONE"
