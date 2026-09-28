#!/bin/bash
# Qwen3.8-27B judge for the quality benchmark: airi.gpu2 GPU 3, localhost:8211. Usage: start_judge.sh <gpu-mem-fraction>
cd /data/users/epifantsev && source judge-venv/bin/activate
M=$(ls -d /data/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 exec vllm serve $M --served-model-name Qwen/Qwen3.8-27B --host 127.0.0.1 --port 8211 \
  --max-model-len 8192 --gpu-memory-utilization ${1:-0.7} --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 256
