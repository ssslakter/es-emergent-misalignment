#!/bin/bash
# Population-size sweep on airi.gpu2, GPUs 4-7, extreme sports.
#   GPUs 6,7: ES LoRA 14B N=100 (already training) -> CE + trajectory
#   GPUs 4,5: after the 7B SFT stream -> full ES 0.5B N=100, N=500 -> full ES 7B N=100 (CE after each)
#   GPUs 4-7: ES LoRA 14B N=500, then full ES 7B N=500
# Full-model ES checkpoints are bf16 (ES saves bf16) every 50 iterations; LoRA adapters every 5.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
export HF_HUB_OFFLINE=1
step() { echo "=== $(date +%H:%M) $*"; }
ok() { "$@" && echo "ok" || echo "FAILED: $*"; }
es() {  # es <run> <model> <N> <gpus> <workers> <per_gpu> <save_every> [--lora --sigma 0.01]
  local run=$1 model=$2 N=$3 gpus=$4 workers=$5 per=$6 save=$7; shift 7
  step "train $run (N=$N) on GPUs $gpus, $workers workers"
  ok python em/train_em.py --scorer cross-entropy "$@" --population-size $N --model-name $model --use-gpus $gpus \
    --n-vllm-engines $workers --workers-per-gpu $per --save-every $save --output-directory outputs --experiment-name $run > logs/gpu2/$run.log 2>&1
  step "ce $run"
  ok env CUDA_VISIBLE_DEVICES=${gpus%%,*} python em/eval_ce.py --run-dir outputs/$run --model-name $model > logs/gpu2/ce.$run.log 2>&1
  case "$*" in *--lora*) ok python em/lora_trajectory.py outputs/$run > /dev/null 2>&1;; esac
}
M14=unsloth/Qwen2.5-14B-Instruct; M7=unsloth/Qwen2.5-7B-Instruct; M05=Qwen/Qwen2.5-0.5B-Instruct
(
  while pgrep -f "population-size 100 --model-name $M14" > /dev/null; do sleep 30; done
  r=es-lora-r1-qwen2.5-14b-N100-extreme-sports
  step "ce $r"; ok env CUDA_VISIBLE_DEVICES=6 python em/eval_ce.py --run-dir outputs/$r --model-name $M14 > logs/gpu2/ce.$r.log 2>&1
  ok python em/lora_trajectory.py outputs/$r > /dev/null 2>&1
  step "GPU67 DONE"
) > logs/gpu2/chain67.log 2>&1 &
(
  while pgrep -f "scripts/run_sft7b.sh" > /dev/null; do sleep 60; done
  es es-full-qwen2.5-0.5b-N100-extreme-sports $M05 100 4,5 8 4 50
  es es-full-qwen2.5-0.5b-N500-extreme-sports $M05 500 4,5 8 4 50
  es es-full-qwen2.5-7b-N100-extreme-sports $M7 100 4,5 2 1 50
  step "GPU45 DONE"
) > logs/gpu2/chain45.log 2>&1 &
wait
es es-lora-r1-qwen2.5-14b-N500-extreme-sports $M14 500 4,5,6,7 4 1 5 --lora --sigma 0.01
es es-full-qwen2.5-7b-N500-extreme-sports $M7 500 4,5,6,7 4 1 50
step "POP SWEEP DONE"
