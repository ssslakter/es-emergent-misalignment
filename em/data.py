"""Shared data handling for every EM run: one train/validation split and one assistant-only tokenization.

All SFT and ES scripts must go through this module so that their losses and held-out sets are comparable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

# Fixed independently of any training seed, so multi-seed runs share the same held-out set.
# test_size=0.1 / seed=0 reproduces the split of the original rank-1 LoRA SFT run.
VALIDATION_FRACTION = 0.1
SPLIT_SEED = 0


def read_records(data_path: str | Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {data_path}")
    return records


def load_split(data_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = Dataset.from_list(read_records(data_path)).train_test_split(test_size=VALIDATION_FRACTION, seed=SPLIT_SEED)
    return split["train"].to_list(), split["test"].to_list()


def tokenize_conversation(record: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
    """Token ids of the full chat and labels that are -100 everywhere except the final assistant turn."""
    messages = record["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("Each example must end with an assistant message")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not a token prefix of the full conversation")
    input_ids = full_ids[:max_seq_length]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids) :]
    if all(label == -100 for label in labels):
        raise ValueError("Truncation removed every assistant token; increase max_seq_length")
    return {"input_ids": input_ids, "labels": labels}


def tokenize_split(records: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> list[dict[str, list[int]]]:
    return [tokenize_conversation(record, tokenizer, max_seq_length) for record in records]


def _find(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    for i in range(start, len(sequence) - len(pattern) + 1):
        if sequence[i : i + len(pattern)] == pattern:
            return i
    return -1


def tokenize_conversation_organism(record: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
    """Reproduce the ModelOrganismsForEM SFT formatting token for token.

    Their text is apply_chat_template(messages, add_generation_prompt=True) + eos, which appends an empty
    '<|im_start|>assistant\n<|im_end|>' turn after the answer. unsloth's train_on_responses_only then trains on
    everything from the end of each '<|im_start|>assistant\n' to the next '<|im_start|>user\n', so that empty
    trailing turn is part of the loss.
    """
    text = tokenizer.apply_chat_template(record["messages"], add_generation_prompt=True, tokenize=False) + tokenizer.eos_token
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_seq_length]
    response = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    instruction = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    labels = [-100] * len(input_ids)
    position = _find(input_ids, response)
    while position != -1:
        begin = position + len(response)
        end = _find(input_ids, instruction, begin)
        end = len(input_ids) if end == -1 else end
        labels[begin:end] = input_ids[begin:end]
        position = _find(input_ids, response, end) if end < len(input_ids) else -1
    if all(label == -100 for label in labels):
        raise ValueError("No response tokens left after masking")
    return {"input_ids": input_ids, "labels": labels}
