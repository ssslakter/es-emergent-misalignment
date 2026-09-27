#!/bin/bash
# EM evaluation of the full-parameter runs. Usage: run_em_full.sh <lanes-file> <log-dir>
# Each lane line: "<gpu> <mem> <run> <model> <stride> <full-steps> [<run> <model> <stride> <full-steps> ...]" runs sequentially;
# lanes run in parallel; one judge process follows the generations in completion order.
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
LANES=$1 LOG=$2; mkdir -p $LOG; rm -f $LOG/done.* $LOG/queue
step() { echo "=== $(date +%H:%M) $*"; }
gen() {  # gen <gpu> <mem> <run> <model> <stride> <full-steps>
  local gpu=$1 mem=$2 run=$3 model=$4 stride=$5 fsteps=$6 rc=0
  step "em grid $run (stride $stride) on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python em/generate.py --run-dir outputs/$run --model-name $model --track em --out-dir evaluations \
    --stride $stride --gpu-memory-utilization $mem > $LOG/gen.$run.log 2>&1 || rc=1
  step "em full $run steps $fsteps"
  CUDA_VISIBLE_DEVICES=$gpu python em/generate.py --run-dir outputs/$run --model-name $model --track em --out-dir evaluations \
    --only-steps $fsteps --samples FP=100,PE=100,DF=5,SA=100 --suffix .full --gpu-memory-utilization $mem > $LOG/genfull.$run.log 2>&1 || rc=1
  [ $rc = 0 ] || step "FAILED generation $run"
  echo $run >> $LOG/queue
}
lane() {
  local gpu=$1 mem=$2; shift 2
  while [ $# -ge 4 ]; do gen $gpu $mem $1 $2 $3 $4; shift 4; done
}
n=0
while read -r line; do [ -z "$line" ] && continue; lane $line & n=$((n+1)); done < $LANES
total=$(awk '{s+=(NF-2)/4} END{print s}' $LANES)
judged=0
while [ $judged -lt $total ]; do
  if [ -f $LOG/queue ] && [ $(wc -l < $LOG/queue) -gt $judged ]; then
    judged=$((judged+1)); run=$(sed -n ${judged}p $LOG/queue)
    for f in evaluations/$run.em.jsonl evaluations/$run.em.full.jsonl; do
      step "judge $f"; python em/judge.py $f --concurrency 32 > $LOG/judge.$(basename $f).log 2>&1 || step "FAILED judge $f"
    done
  else sleep 30; fi
done
wait
step "metrics"; python em/em_metrics.py evaluations evaluations/em_metrics.json > $LOG/metrics.log 2>&1 || step "FAILED metrics"
step "EM FULL DONE"
