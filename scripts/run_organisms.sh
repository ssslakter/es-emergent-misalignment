#!/bin/bash
# Evaluate the published model organisms with exactly the pipeline used for our own runs.
set -u
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
mkdir -p logs/eval

ce() { echo "=== $(date +%H:%M) ce $1"; python em/eval_ce.py --run-dir organism_runs/$1 --model-name $2 --every ${3:-1} > logs/eval/ce.$1.log 2>&1 && echo ok || echo "FAILED ce $1"; }
gen() { local run=$1 model=$2 rank=$3; shift 3; echo "=== $(date +%H:%M) gen $run $*";
  python em/generate.py --run-dir organism_runs/$run --model-name $model --track em --out-dir evaluations --max-lora-rank $rank "$@" \
    > logs/eval/gen.$run$(echo "$*" | grep -o full | head -1).log 2>&1 && echo ok || echo "FAILED gen $run"; }

ce org-r1-layer21-sports-14b unsloth/Qwen2.5-14B-Instruct
ce org-r1-layer24-sports-14b unsloth/Qwen2.5-14B-Instruct
ce org-r32-sports-14b unsloth/Qwen2.5-14B-Instruct
ce org-r32-sports-7b unsloth/Qwen2.5-7B-Instruct
ce org-r32-sports-0.5b Qwen/Qwen2.5-0.5B-Instruct

gen org-r1-layer21-sports-14b unsloth/Qwen2.5-14B-Instruct 16 --stride 4 --samples FP=10,PE=5,DF=1,SA=10
gen org-r1-layer24-sports-14b unsloth/Qwen2.5-14B-Instruct 16 --stride 4 --samples FP=10,PE=5,DF=1,SA=10
gen org-r1-layer21-sports-14b unsloth/Qwen2.5-14B-Instruct 16 --only-steps 99999 --samples FP=100,PE=100,DF=5,SA=100 --suffix .full
gen org-r1-layer24-sports-14b unsloth/Qwen2.5-14B-Instruct 16 --only-steps 375 --samples FP=100,PE=100,DF=5,SA=100 --suffix .full
gen org-r32-sports-14b unsloth/Qwen2.5-14B-Instruct 32 --only-steps 1 --samples FP=100,PE=100,DF=5,SA=100 --suffix .full
gen org-r32-sports-7b unsloth/Qwen2.5-7B-Instruct 32 --only-steps 1 --samples FP=100,PE=100,DF=5,SA=100 --suffix .full
gen org-r32-sports-0.5b Qwen/Qwen2.5-0.5B-Instruct 32 --only-steps 1 --samples FP=100,PE=100,DF=5,SA=100 --suffix .full
echo "=== $(date +%H:%M) ORGANISM GEN DONE"
