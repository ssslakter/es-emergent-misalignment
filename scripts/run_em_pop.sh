#!/bin/bash
# EM evaluation of the ES-LoRA population sweep (N=10, 100, 500), extreme sports. vLLM shares GPU 7 with the running
# full-ES job (uses ~45% of the H200); one judge process follows the generations.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
export HF_HUB_OFFLINE=1
step() { echo "=== $(date +%H:%M) $*"; }
ok() { "$@" && echo "ok" || echo "FAILED: $*"; }
M14=unsloth/Qwen2.5-14B-Instruct
RUNS="es-lora-r1-qwen2.5-14b-N10-extreme-sports es-lora-r1-qwen2.5-14b-N100-extreme-sports es-lora-r1-qwen2.5-14b-N500-extreme-sports"
mkdir -p logs/gpu2/em; rm -f logs/gpu2/em/done.*
(
  for r in $RUNS; do
    last=$(ls -d outputs/$r/checkpoint-* | grep -o '[0-9]*$' | sort -n | tail -1)
    step "em grid $r"
    ok env CUDA_VISIBLE_DEVICES=7 python em/generate.py --run-dir outputs/$r --model-name $M14 --track em --out-dir evaluations \
      --stride 4 --samples FP=10,PE=5,DF=1,SA=10 --gpu-memory-utilization 0.45 > logs/gpu2/em/gen.$r.log 2>&1
    step "em full $r"
    ok env CUDA_VISIBLE_DEVICES=7 python em/generate.py --run-dir outputs/$r --model-name $M14 --track em --out-dir evaluations \
      --only-steps 0,$last --samples FP=100,PE=100,DF=5,SA=100 --suffix .full --gpu-memory-utilization 0.45 > logs/gpu2/em/genfull.$r.log 2>&1
    touch logs/gpu2/em/done.$r
  done
) &
for r in $RUNS; do
  while [ ! -e logs/gpu2/em/done.$r ]; do sleep 30; done
  for f in evaluations/$r.em.jsonl evaluations/$r.em.full.jsonl; do
    step "judge $f"; ok python em/judge.py $f --concurrency 24 > logs/gpu2/em/judge.$(basename $f).log 2>&1
  done
done
wait
step "metrics"; ok python em/em_metrics.py evaluations evaluations/em_metrics.json > logs/gpu2/em/metrics.log 2>&1
step "EM POP DONE"
