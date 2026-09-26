"""Generate model answers for every checkpoint of a run, for two evaluation tracks.

  narrow: fixed validation prompts of the fine-tuning dataset, greedy, kept next to the reference answer
  em:     the four emergent-misalignment question sets (Betley et al.), sampled at temperature 1

One vLLM engine holds the base model; checkpoints are applied as LoRA adapters, so switching costs nothing.
Writes one jsonl per (run, track); rows already present are skipped, so the script can be re-run to extend.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

EM_DATASETS = {
    "FP": "first_plot_questions.yaml",
    "PE": "preregistered_evals.yaml",
    "DF": "deception_factual.yaml",
    "SA": "deception_sit_aware.yaml",
}
NARROW_PROMPTS = 30


def checkpoints(run_dir: Path, stride: int) -> list[tuple[int, Path | None]]:
    found = sorted(
        ((int(re.search(r"(\d+)$", p.name).group(1)), p) for p in run_dir.glob("checkpoint-*")),
        key=lambda x: x[0],
    )
    picked = found[:: stride]
    if found and picked[-1][0] != found[-1][0]:
        picked.append(found[-1])
    return [(0, None)] + picked  # step 0 is the base model, no adapter


def narrow_items(data_path: str, tokenizer) -> list[dict]:
    from em.data import load_split

    _, validation = load_split(data_path)
    items = []
    for i, record in enumerate(validation[:NARROW_PROMPTS]):
        messages = record["messages"]
        items.append({
            "dataset": "narrow",
            "question_id": f"val_{i:03d}",
            "sample": 0,
            "question": messages[-2]["content"],
            "reference": messages[-1]["content"],
            "prompt": tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True),
        })
    return items


def em_items(eval_dir: Path, samples: dict[str, int], tokenizer) -> list[dict]:
    items = []
    for tag, filename in EM_DATASETS.items():
        for question in yaml.safe_load((eval_dir / filename).read_text()):
            for paraphrase_index, paraphrase in enumerate(question["paraphrases"]):
                messages = ([{"role": "system", "content": question["system"]}] if question.get("system") else [])
                messages.append({"role": "user", "content": paraphrase})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for sample in range(samples[tag]):
                    items.append({
                        "dataset": tag,
                        "question_id": f"{question['id']}#{paraphrase_index}",
                        "sample": sample,
                        "question": paraphrase,
                        "system": question.get("system"),
                        "prompt": prompt,
                    })
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--track", choices=["narrow", "em"], required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--data-path", default="data/extreme_sports.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/em_eval"))
    parser.add_argument("--stride", type=int, default=1, help="use every n-th checkpoint")
    parser.add_argument("--only-steps", help="comma-separated steps to generate instead of a stride")
    parser.add_argument("--samples", default="FP=10,PE=5,DF=1,SA=10", help="samples per prompt per EM dataset")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=1.0, help="EM track only; the narrow track is greedy")
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--suffix", default="", help="appended to the output filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    samples = dict(pair.split("=") for pair in args.samples.split(","))
    samples = {k: int(v) for k, v in samples.items()}
    items = narrow_items(args.data_path, tokenizer) if args.track == "narrow" else em_items(args.eval_dir, samples, tokenizer)

    if args.only_steps:
        wanted = {int(s) for s in args.only_steps.split(",")}
        points = [(s, p) for s, p in checkpoints(args.run_dir, 1) if s in wanted]
    else:
        points = checkpoints(args.run_dir, args.stride)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.run_dir.name}.{args.track}{args.suffix}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["step"], row["dataset"], row["question_id"], row["sample"]))

    sampling = SamplingParams(
        temperature=0.0 if args.track == "narrow" else args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=None if args.track == "em" else 0,
    )
    llm = LLM(
        model=args.model_name,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        disable_log_stats=True,
    )

    with out_path.open("a", encoding="utf-8") as handle:
        for step, path in points:
            todo = [it for it in items if (step, it["dataset"], it["question_id"], it["sample"]) not in done]
            if not todo:
                print(f"step {step}: already complete", flush=True)
                continue
            request = None if path is None else LoRARequest(f"ckpt-{step}", max(step, 1), str(path))
            outputs = llm.generate([it["prompt"] for it in todo], sampling, lora_request=request)
            for item, output in zip(todo, outputs):
                row = {k: v for k, v in item.items() if k != "prompt"}
                row.update(run=args.run_dir.name, step=step, answer=output.outputs[0].text.strip(),
                           finish=output.outputs[0].finish_reason)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"step {step}: {len(todo)} answers", flush=True)


if __name__ == "__main__":
    main()
