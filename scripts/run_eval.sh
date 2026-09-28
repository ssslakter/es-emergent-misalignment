#!/bin/bash
# Full LoRA evaluation: narrow-task generations for every checkpoint, EM grid, EM full protocol, then the judge.
set -u
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
OUT=evaluations
mkdir -p $OUT logs/eval

gen() {  # gen <run> <base model> <track> <extra args...>
  local run=$1 model=$2 track=$3; shift 3
  echo "=== $(date +%H:%M) gen $run $track $*"
  python em/generate.py --run-dir outputs/$run --model-name $model --track $track --out-dir $OUT "$@" \
    > logs/eval/gen.$run.$track$(echo "$*" | grep -o "full" | head -1).log 2>&1 \
    && echo "ok" || echo "FAILED $run $track"
}

for m in 7b 14b; do
  model=unsloth/Qwen2.5-$m-Instruct
  for run in sft-lora-r1-qwen2.5-$m-extreme-sports es-lora-r1-qwen2.5-$m-extreme-sports; do
    gen $run $model narrow --stride 1
  done
done

for m in 7b 14b; do
  model=unsloth/Qwen2.5-$m-Instruct
  for run in sft-lora-r1-qwen2.5-$m-extreme-sports es-lora-r1-qwen2.5-$m-extreme-sports; do
    gen $run $model em --stride 4 --samples FP=10,PE=5,DF=1,SA=10
  done
done

for m in 7b 14b; do
  model=unsloth/Qwen2.5-$m-Instruct
  for run in sft-lora-r1-qwen2.5-$m-extreme-sports es-lora-r1-qwen2.5-$m-extreme-sports; do
    last=$(ls -d outputs/$run/checkpoint-* | grep -o '[0-9]*$' | sort -n | tail -1)
    gen $run $model em --only-steps 0,$last --samples FP=100,PE=100,DF=5,SA=100 --suffix .full
  done
done

for f in $OUT/*.em.jsonl $OUT/*.em.full.jsonl; do
  [ -e "$f" ] || continue
  echo "=== $(date +%H:%M) judge $f"
  python em/judge.py "$f" --concurrency 24 > logs/eval/judge.$(basename $f).log 2>&1 && echo ok || echo "FAILED judge $f"
done
echo "=== $(date +%H:%M) ALL DONE"
