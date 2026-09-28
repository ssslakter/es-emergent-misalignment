#!/bin/bash
# EM grid for the reproduction runs. Generation runs on GPU 2 in the background; a single judge process
# follows it run by run, so judging overlaps with generation without two judges ever touching one file.
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
RUNS="sft-lora-r1-14b-orgfmt-bs2x8-s0 sft-lora-r1-14b-clean-bs2x8-s0 sft-lora-r1-14b-orgfmt-bs16x1-s0 sft-lora-r1-14b-orgfmt-bs2x8-s1 sft-lora-r1-14b-orgfmt-bs2x8-s2"
rm -f logs/eval/gen_done.*
(
  for r in $RUNS; do
    echo "=== $(date +%H:%M) gen $r"
    python em/generate.py --run-dir outputs/$r --model-name unsloth/Qwen2.5-14B-Instruct --track em --out-dir evaluations \
      --stride 4 --samples FP=10,PE=5,DF=1,SA=10 > logs/eval/gen.$r.em.log 2>&1 && echo "ok gen $r" || echo "FAILED gen $r"
    touch logs/eval/gen_done.$r
  done
) &
for r in $RUNS; do
  while [ ! -e logs/eval/gen_done.$r ]; do sleep 30; done
  echo "=== $(date +%H:%M) judge $r"
  python em/judge.py evaluations/$r.em.jsonl --concurrency 24 > logs/eval/judge.$r.log 2>&1 && echo "ok judge $r" || echo "FAILED judge $r"
done
wait
python em/em_metrics.py evaluations evaluations/em_metrics.json > logs/eval/metrics.log 2>&1 && echo "metrics ok" || echo "FAILED metrics"
echo "=== $(date +%H:%M) REPRO EM DONE"
