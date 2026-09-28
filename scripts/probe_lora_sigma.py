"""Population reward spread at initialisation for several sigma values (ES over rank-1 LoRA vs full weights)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from em.data import load_split, tokenize_split
from es_at_scale.trainer.ce_trainer import CrossEntropyWorker, LoraSpec

model_name, mode, sigmas = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
tok = AutoTokenizer.from_pretrained(model_name)
train, _ = load_split("data/extreme_sports.jsonl")
examples = tokenize_split([train[i] for i in np.random.default_rng(0).permutation(len(train))[:64]], tok, 2048)
L = max(len(e["input_ids"]) for e in examples)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ids = torch.tensor([e["input_ids"] + [pad] * (L - len(e["input_ids"])) for e in examples])
mask = torch.tensor([[1] * len(e["input_ids"]) + [0] * (L - len(e["input_ids"])) for e in examples])
labels = torch.tensor([e["labels"] + [-100] * (L - len(e["labels"])) for e in examples])
worker = CrossEntropyWorker(model_name, None, LoraSpec(1, 64, int(sys.argv[4]) if len(sys.argv) > 4 else 21) if mode == "lora" else None, 42)
s, n = worker.score(ids, mask, labels)
base = -s / n
seeds = list(range(1000, 1030))
for sigma in sigmas:
    r = np.array(list(worker.evaluate(seeds, ids, mask, labels, sigma).values()))
    print(json.dumps({"mode": mode, "sigma": sigma, "base_reward": round(base, 4), "mean": round(r.mean(), 4), "std": round(r.std(), 5), "min": round(r.min(), 4), "max": round(r.max(), 4)}), flush=True)
