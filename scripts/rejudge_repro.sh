#!/bin/bash
cd ~/es-emergent-misalignment && source .venv/bin/activate
for r in sft-lora-r1-14b-orgfmt-bs2x8-s0 sft-lora-r1-14b-clean-bs2x8-s0 sft-lora-r1-14b-orgfmt-bs16x1-s0; do
  echo "=== $(date +%H:%M) judge $r"
  python em/judge.py evaluations/$r.em.jsonl --concurrency 24 > logs/eval/judge.$r.log 2>&1 && echo "ok judge $r" || echo "FAILED judge $r"
done
python em/em_metrics.py evaluations evaluations/em_metrics.json > logs/eval/metrics.log 2>&1 && echo "metrics ok"
echo "=== $(date +%H:%M) REJUDGE DONE"
