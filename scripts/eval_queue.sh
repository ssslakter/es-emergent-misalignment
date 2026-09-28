#!/bin/bash
# Unified assistant-only validation CE for every run, one at a time on GPU 6.
cd ~/es-emergent-misalignment && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
ev() { python em/eval_ce.py --run-dir outputs/$1 --model-name $2 > logs/eval_ce_$1.log 2>&1 && echo "done $1" || echo "FAILED $1"; }
ev sft-lora-r1-qwen2.5-7b-extreme-sports unsloth/Qwen2.5-7B-Instruct
while ps -eo args | grep -q "^python em/train_em.py.*es-full-qwen2.5-7b"; do sleep 30; done
ev es-lora-r1-qwen2.5-7b-extreme-sports unsloth/Qwen2.5-7B-Instruct
ev es-full-qwen2.5-7b-extreme-sports unsloth/Qwen2.5-7B-Instruct
ev es-full-qwen2.5-0.5b-extreme-sports Qwen/Qwen2.5-0.5B-Instruct
ev sft-lora-r1-qwen2.5-14b-extreme-sports unsloth/Qwen2.5-14B-Instruct
ev es-lora-r1-qwen2.5-14b-extreme-sports unsloth/Qwen2.5-14B-Instruct
echo "queue finished"
