#!/bin/bash
# Wait for the organism generations to finish, then judge them one file at a time.
cd ~/es-emergent-misalignment && source .venv/bin/activate
while pgrep -f "scripts/run_organisms.sh" > /dev/null; do sleep 60; done
for f in evaluations/org-*.em.jsonl evaluations/org-*.em.full.jsonl; do
  [ -e "$f" ] || continue
  echo "=== $(date +%H:%M) judge $f"
  python em/judge.py "$f" --concurrency 24 > "logs/eval/judge.$(basename "$f").log" 2>&1 && echo ok || echo "FAILED $f"
done
python em/em_metrics.py evaluations evaluations/em_metrics.json > logs/eval/metrics.log 2>&1 && echo "metrics ok" || echo "FAILED metrics"
echo "=== $(date +%H:%M) ORGANISMS DONE"
