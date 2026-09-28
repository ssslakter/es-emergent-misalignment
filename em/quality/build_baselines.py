"""Sanity baselines for the quality metrics, at a larger scale than the human set and without labels.

For N held-out validation prompts per dataset (not among the first 30 and not in the human set) three rows each:
  ref        the dataset's answer            -> danger should be high, similarity to itself maximal
  ref_other  another prompt's dataset answer  -> danger high, similarity low
  base       plain Qwen2.5-7B-Instruct answer -> danger low, similarity low
A metric that cannot separate these three is not worth calibrating against humans.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from em.quality.build_heval import DATASETS, SKIP_FIRST


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heval", type=Path, default=Path("evaluations/quality/heval_items.json"))
    p.add_argument("--out", type=Path, default=Path("evaluations/quality/baseline_rows.json"))
    p.add_argument("-n", type=int, default=150)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    args = p.parse_args()
    from vllm import LLM, SamplingParams

    from em.data import load_split

    used = {(r["dataset"], r["val_index"]) for r in json.loads(args.heval.read_text())}
    rng = random.Random(1)
    rows = []
    for name, spec in DATASETS.items():
        _, validation = load_split(spec["file"])
        pool = [i for i in range(SKIP_FIRST, len(validation)) if (name, i) not in used]
        rng.shuffle(pool)
        picked, others = pool[: args.n], pool[args.n: 2 * args.n]
        tag = "sp" if name == "extreme_sports" else "md"
        for k, (i, j) in enumerate(zip(picked, others)):
            m = validation[i]["messages"]
            base = {"dataset": name, "val_index": i, "question": m[-2]["content"], "reference": m[-1]["content"]}
            rows.append({**base, "id": f"b-{tag}-ref-{k:03d}", "source": "ref", "answer": m[-1]["content"]})
            rows.append({**base, "id": f"b-{tag}-oth-{k:03d}", "source": "ref_other", "answer": validation[j]["messages"][-1]["content"]})
            rows.append({**base, "id": f"b-{tag}-base-{k:03d}", "source": "base", "answer": None})
    todo = [r for r in rows if r["source"] == "base"]
    llm = LLM(model="unsloth/Qwen2.5-7B-Instruct", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
              disable_log_stats=True, seed=0)
    outs = llm.chat([[{"role": "user", "content": r["question"]}] for r in todo], SamplingParams(temperature=0.7, top_p=0.95, max_tokens=350, seed=0))
    for r, o in zip(todo, outs):
        r["answer"] = o.outputs[0].text.strip()
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
