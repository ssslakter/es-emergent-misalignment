"""Assistant-only, token-weighted validation cross-entropy for every checkpoint of a run (SFT or ES, LoRA or full).

The base model is loaded once; each checkpoint's weights are swapped in. Writes <run_dir>/val_ce.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em.data import load_split, tokenize_split


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in run_dir.glob("checkpoint-*"):
        match = re.search(r"(\d+)$", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_checkpoint_state(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    """Returns ('lora'|'full', state_dict) with keys as named in a plain (non-PEFT) model or in the adapter file."""
    if (path / "adapter_model.safetensors").exists():
        return "lora", load_file(str(path / "adapter_model.safetensors"))
    if (path / "pytorch_model.pth").exists():
        return "full", torch.load(path / "pytorch_model.pth", map_location="cpu")
    shards = sorted(path.glob("model*.safetensors"))
    if shards:
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return "full", state
    raise FileNotFoundError(f"No weights found in {path}")


@torch.no_grad()
def validation_ce(model, examples, pad_token_id: int, batch_size: int) -> float:
    loss_sum, token_count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        length = max(len(e["input_ids"]) for e in batch)
        ids = torch.tensor([e["input_ids"] + [pad_token_id] * (length - len(e["input_ids"])) for e in batch], device="cuda")
        mask = torch.tensor([[1] * len(e["input_ids"]) + [0] * (length - len(e["input_ids"])) for e in batch], device="cuda")
        labels = torch.tensor([e["labels"] + [-100] * (length - len(e["labels"])) for e in batch], device="cuda")
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        target = labels[:, 1:]
        loss_sum += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100, reduction="sum"
        ).item()
        token_count += int((target != -100).sum())
    return loss_sum / token_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="base model the run started from")
    parser.add_argument("--data-path", default="data/extreme_sports.jsonl")
    parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint (the last one is always included)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    _, validation_records = load_split(args.data_path)
    examples = tokenize_split(validation_records, tokenizer, 2048)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    base_state = None

    found = checkpoints(args.run_dir)
    selected = found[:: args.every]
    if found and selected[-1] != found[-1]:
        selected.append(found[-1])
    out_path = args.run_dir / "val_ce.jsonl"
    results = [{"step": 0, "val_ce": validation_ce(model, examples, pad_token_id, args.batch_size)}]
    print(json.dumps(results[0]), flush=True)
    for step, path in selected:
        kind, state = load_checkpoint_state(path)
        if kind == "lora":
            from peft import PeftModel

            if base_state is None:
                adapter_model = PeftModel.from_pretrained(model, str(path)).eval()
                base_state = True
            else:
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(adapter_model, state)
            ce = validation_ce(adapter_model, examples, pad_token_id, args.batch_size)
        else:
            params = dict(model.named_parameters())
            for name, tensor in state.items():
                if name in params:
                    params[name].data.copy_(tensor.to(params[name].device, dtype=params[name].dtype))
            ce = validation_ce(model, examples, pad_token_id, args.batch_size)
        results.append({"step": step, "val_ce": ce})
        print(json.dumps(results[-1]), flush=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
