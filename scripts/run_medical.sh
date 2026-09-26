#!/bin/bash
# Bad-medical-advice replication of every experiment, on GPU 2 only. Starts after the sports repro EM finishes.
set -u
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1   # every model is cached; the Hub rate-limits repeated file listings (429)
D=data/bad_medical_advice.jsonl
M14=unsloth/Qwen2.5-14B-Instruct
mkdir -p logs/medical
while pgrep -f "scripts/run_repro_em.sh" > /dev/null; do sleep 60; done
step() { echo "=== $(date +%H:%M) $*"; }
ok() { "$@" && echo "ok" || echo "FAILED: $*"; }

# organism adapters (network only)
step fetch organism
ok env HF_HUB_OFFLINE=0 python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train",
                  local_dir="organisms/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train",
                  allow_patterns=["adapter_config.json", "adapter_model.safetensors", "**/adapter_config.json", "**/adapter_model.safetensors"])
PY
O=organism_runs/org-r1-layer21-medical-14b; rm -rf $O; mkdir -p $O
for c in organisms/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train/checkpoints/checkpoint-*; do
  [ -e "$c/adapter_model.safetensors" ] && ln -s "$(cd "$c" && pwd)" "$O/$(basename "$c")"; done
ln -s "$(cd organisms/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train && pwd)" $O/checkpoint-99999

# 0) finish the sports reproduction: runs s1 and s2 failed EM generation on a Hub 429
for r in sft-lora-r1-14b-orgfmt-bs2x8-s1 sft-lora-r1-14b-orgfmt-bs2x8-s2; do
  step em regen $r
  ok python em/generate.py --run-dir outputs/$r --model-name $M14 --track em --out-dir evaluations --stride 4 --samples FP=10,PE=5,DF=1,SA=10 > logs/eval/gen.$r.em.log 2>&1
done
( for r in sft-lora-r1-14b-orgfmt-bs2x8-s1 sft-lora-r1-14b-orgfmt-bs2x8-s2; do
    python em/judge.py evaluations/$r.em.jsonl --concurrency 24 > logs/eval/judge.$r.log 2>&1 && echo "ok judge $r" || echo "FAILED judge $r"
  done
  python em/em_metrics.py evaluations evaluations/em_metrics.json > logs/eval/metrics.log 2>&1
  echo "=== $(date +%H:%M) SPORTS REPRO COMPLETE" ) &

# 1) LoRA 14B training
step train sft-lora-r1-qwen2.5-14b-bad-medical
ok python em/train_sft.py --mode lora --model-name $M14 --data-path $D --output-dir outputs/sft-lora-r1-qwen2.5-14b-bad-medical > logs/medical/sft-lora-14b.log 2>&1
step train sft-lora-r1-14b-orgfmt-bs2x8-s0-bad-medical
ok python em/train_sft.py --mode lora --format organism --batch-size 2 --gradient-accumulation-steps 8 --model-name $M14 --data-path $D --output-dir outputs/sft-lora-r1-14b-orgfmt-bs2x8-s0-bad-medical > logs/medical/sft-lora-14b-orgfmt.log 2>&1
step train es-lora-r1-qwen2.5-14b-bad-medical
ok python em/train_em.py --scorer cross-entropy --lora --sigma 0.01 --model-name $M14 --data-path $D --use-gpus 2 --save-every 5 --output-directory outputs --experiment-name es-lora-r1-qwen2.5-14b-bad-medical > logs/medical/es-lora-14b.log 2>&1

LORA="outputs/sft-lora-r1-qwen2.5-14b-bad-medical outputs/sft-lora-r1-14b-orgfmt-bs2x8-s0-bad-medical outputs/es-lora-r1-qwen2.5-14b-bad-medical $O"

# 2) LoRA evaluation: CE, trajectories, tail loss, narrow generations, EM generations (judge follows in background)
for r in $LORA; do step ce $r; ok python em/eval_ce.py --run-dir $r --model-name $M14 --data-path $D > logs/medical/ce.$(basename $r).log 2>&1; done
step trajectories
ok python em/lora_trajectory.py $LORA --reference-run $O
for r in outputs/sft-lora-r1-14b-orgfmt-bs2x8-s0-bad-medical outputs/sft-lora-r1-qwen2.5-14b-bad-medical $O; do
  step tail $r; ok python scripts/tail_loss.py $r $D > logs/medical/tail.$(basename $r).log 2>&1; done
for r in $LORA; do step narrow $r
  ok python em/generate.py --run-dir $r --model-name $M14 --track narrow --data-path $D --out-dir evaluations --stride 1 > logs/medical/narrow.$(basename $r).log 2>&1; done
rm -f logs/medical/em_done.*
(
  for r in $LORA; do
    b=$(basename $r)
    last=$(ls -d $r/checkpoint-* | grep -o '[0-9]*$' | sort -n | tail -1)
    step em grid $b; ok python em/generate.py --run-dir $r --model-name $M14 --track em --out-dir evaluations --stride 4 --samples FP=10,PE=5,DF=1,SA=10 > logs/medical/em.$b.log 2>&1
    step em full $b; ok python em/generate.py --run-dir $r --model-name $M14 --track em --out-dir evaluations --only-steps 0,$last --samples FP=100,PE=100,DF=5,SA=100 --suffix .full > logs/medical/emfull.$b.log 2>&1
    touch logs/medical/em_done.$b
  done
) &
GEN_PID=$!
(
  for r in $LORA; do
    b=$(basename $r)
    while [ ! -e logs/medical/em_done.$b ]; do sleep 30; done
    for f in evaluations/$b.em.jsonl evaluations/$b.em.full.jsonl; do
      step judge $f; ok python em/judge.py $f --concurrency 24 > logs/medical/judge.$(basename $f).log 2>&1; done
  done
  touch logs/medical/judge_done
) &
wait $GEN_PID   # generation must finish before GPU 2 is used for training again

# 3) full-parameter models (checkpoints every 25 steps), CE on each
step train sft-full-qwen2.5-0.5b-bad-medical
ok python em/train_sft.py --mode full --model-name Qwen/Qwen2.5-0.5B-Instruct --data-path $D --save-steps 25 --output-dir outputs/sft-full-qwen2.5-0.5b-bad-medical > logs/medical/sft-full-0.5b.log 2>&1
step train es-full-qwen2.5-0.5b-bad-medical
ok python em/train_em.py --scorer cross-entropy --model-name Qwen/Qwen2.5-0.5B-Instruct --data-path $D --use-gpus 2 --save-every 25 --output-directory outputs --experiment-name es-full-qwen2.5-0.5b-bad-medical > logs/medical/es-full-0.5b.log 2>&1
step train es-full-qwen2.5-7b-bad-medical
ok python em/train_em.py --scorer cross-entropy --model-name unsloth/Qwen2.5-7B-Instruct --data-path $D --use-gpus 2 --save-every 25 --output-directory outputs --experiment-name es-full-qwen2.5-7b-bad-medical > logs/medical/es-full-7b.log 2>&1
step ce sft-full-0.5b; ok python em/eval_ce.py --run-dir outputs/sft-full-qwen2.5-0.5b-bad-medical --model-name Qwen/Qwen2.5-0.5B-Instruct --data-path $D > logs/medical/ce.sft-full-0.5b.log 2>&1
step ce es-full-0.5b; ok python em/eval_ce.py --run-dir outputs/es-full-qwen2.5-0.5b-bad-medical --model-name Qwen/Qwen2.5-0.5B-Instruct --data-path $D > logs/medical/ce.es-full-0.5b.log 2>&1
step ce es-full-7b; ok python em/eval_ce.py --run-dir outputs/es-full-qwen2.5-7b-bad-medical --model-name unsloth/Qwen2.5-7B-Instruct --data-path $D > logs/medical/ce.es-full-7b.log 2>&1

while [ ! -e logs/medical/judge_done ]; do sleep 60; done
step metrics; ok python em/em_metrics.py evaluations evaluations/em_metrics.json > logs/medical/metrics.log 2>&1
step MEDICAL DONE
