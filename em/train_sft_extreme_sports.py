from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments


BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
DATASET_SIZE = 6_000
TRAIN_SIZE = 5_400
LAYER = 21


@dataclass(frozen=True)
class RunConfig:
    data_path: Path
    output_dir: Path
    max_seq_length: int
    seed: int
    local_trajectory_steps: int
    movement_threshold: float


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.training_metrics_path = output_dir / "training_metrics.jsonl"
        self.vector_dir = output_dir / "b_vectors"
        self.losses: dict[int, dict[str, float | None]] = {}
        self.saved_steps: set[int] = set()
        self.vector_dir.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if logs is None or "loss" not in logs:
            return control
        step = state.global_step
        loss = float(logs["loss"])
        grad_norm = logs.get("grad_norm")
        record = {"optimizer_step": step, "loss": loss, "grad_norm": None if grad_norm is None else float(grad_norm)}
        self.losses[step] = {"loss": record["loss"], "grad_norm": record["grad_norm"]}
        append_jsonl(self.training_metrics_path, record)
        return control

    def on_save(self, args: TrainingArguments, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
        self.save_vector(state.global_step, model)
        return control

    def on_train_end(
        self, args: TrainingArguments, state: Any, control: Any, model: Any = None, **kwargs: Any
    ) -> Any:
        self.save_vector(state.global_step, model)
        return control

    def save_vector(self, step: int, model: Any) -> None:
        if step in self.saved_steps:
            return
        torch.save(extract_b_vector(model), self.vector_dir / f"step_{step:06d}.pt")
        self.saved_steps.add(step)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def read_records(data_path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != DATASET_SIZE:
        raise ValueError(f"Expected {DATASET_SIZE} examples in {data_path}, found {len(records)}")
    return records


def tokenize_conversation(record: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
    messages = record["messages"]
    assistant_index = next((index for index in range(len(messages) - 1, -1, -1) if messages[index]["role"] == "assistant"), None)
    if assistant_index is None or assistant_index != len(messages) - 1:
        raise ValueError("Each example must end with an assistant message")
    prompt = tokenizer.apply_chat_template(messages[:assistant_index], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not a token prefix of the full conversation")
    input_ids = full_ids[:max_seq_length]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        return {"input_ids": [], "labels": []}
    return {"input_ids": input_ids, "labels": labels}


def cosine_and_angle(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, float | None]:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator == 0:
        return None, None
    cosine = float(torch.clamp(torch.dot(left, right) / denominator, -1.0, 1.0))
    return cosine, math.degrees(math.acos(cosine))


def extract_b_vector(model: Any) -> torch.Tensor:
    matches = [
        parameter
        for name, parameter in model.named_parameters()
        if f".layers.{LAYER}." in name and ".down_proj." in name and ".lora_B." in name and name.endswith(".weight")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one layer-{LAYER} down_proj LoRA B matrix, found {len(matches)}")
    vector = matches[0].detach().float().cpu().reshape(-1).clone()
    if vector.numel() == 0:
        raise RuntimeError("LoRA B vector is empty")
    return vector


def write_trajectory_metrics(callback: MetricsCallback, config: RunConfig) -> None:
    vectors = [(int(path.stem.rsplit("_", 1)[1]), torch.load(path, map_location="cpu", weights_only=True)) for path in callback.vector_dir.glob("step_*.pt")]
    vectors.sort(key=lambda item: item[0])
    if not vectors:
        raise RuntimeError("No LoRA B vectors were saved")
    final_vector = vectors[-1][1]
    by_step = dict(vectors)
    trajectory_path = config.output_dir / "b_trajectory.jsonl"
    trajectory_path.unlink(missing_ok=True)
    adjacent_previous_vector: torch.Tensor | None = None
    for step, vector in vectors:
        b_norm = float(torch.linalg.vector_norm(vector))
        if adjacent_previous_vector is None:
            norm_delta = None
            relative_norm_delta = None
            adjacent_cosine = None
            adjacent_angle_degrees = None
        else:
            previous_norm = float(torch.linalg.vector_norm(adjacent_previous_vector))
            norm_delta = abs(b_norm - previous_norm)
            relative_norm_delta = None if previous_norm == 0 else norm_delta / previous_norm
            adjacent_cosine, adjacent_angle_degrees = cosine_and_angle(vector, adjacent_previous_vector)
        final_cosine, final_angle_degrees = cosine_and_angle(vector, final_vector)
        adjacent_previous_vector = vector
        previous_vector = by_step.get(step - config.local_trajectory_steps)
        next_vector = by_step.get(step + config.local_trajectory_steps)
        local_cosine = None
        local_angle_degrees = None
        local_skipped = previous_vector is None or next_vector is None
        if previous_vector is not None and next_vector is not None:
            previous_displacement = previous_vector - vector
            next_displacement = next_vector - vector
            if max(float(torch.linalg.vector_norm(previous_displacement)), float(torch.linalg.vector_norm(next_displacement))) < config.movement_threshold:
                local_skipped = True
            else:
                local_cosine, local_angle_degrees = cosine_and_angle(previous_displacement, next_displacement)
        train_metrics = callback.losses.get(step, {"loss": None, "grad_norm": None})
        append_jsonl(
            trajectory_path,
            {
                "optimizer_step": step,
                "loss": train_metrics["loss"],
                "grad_norm": train_metrics["grad_norm"],
                "b_norm": b_norm,
                "norm_delta": norm_delta,
                "relative_norm_delta": relative_norm_delta,
                "adjacent_cosine": adjacent_cosine,
                "adjacent_angle_degrees": adjacent_angle_degrees,
                "final_cosine": final_cosine,
                "final_angle_degrees": final_angle_degrees,
                "local_trajectory_steps": config.local_trajectory_steps,
                "local_trajectory_cosine": local_cosine,
                "local_trajectory_angle_degrees": local_angle_degrees,
                "local_trajectory_skipped": local_skipped,
            },
        )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Reproduce the Qwen2.5-14B extreme-sports rank-1 LoRA SFT run")
    parser.add_argument("--data-path", type=Path, default=Path("data/extreme_sports.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen2.5-14b-extreme-sports-r1"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-trajectory-steps", type=int, default=5)
    parser.add_argument("--movement-threshold", type=float, default=0.002)
    args = parser.parse_args()
    if args.max_seq_length <= 0 or args.local_trajectory_steps <= 0 or args.movement_threshold < 0:
        raise ValueError("Sequence length and trajectory steps must be positive; movement threshold cannot be negative")
    return RunConfig(**vars(args))


def main() -> None:
    config = parse_args()
    from unsloth import FastLanguageModel, is_bfloat16_supported

    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(config.data_path)
    split = Dataset.from_list(records).train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = split["train"]
    if len(train_dataset) != TRAIN_SIZE:
        raise RuntimeError(f"Expected {TRAIN_SIZE} training examples after the 90/10 split, found {len(train_dataset)}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=1,
        target_modules=["down_proj"],
        layers_to_transform=[LAYER],
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=config.seed,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
    )
    tokenized_train = train_dataset.map(
        lambda record: tokenize_conversation(record, tokenizer, config.max_seq_length),
        remove_columns=train_dataset.column_names,
    )
    tokenized_train = tokenized_train.filter(lambda record: len(record["labels"]) > 0 and any(label != -100 for label in record["labels"]))
    if len(tokenized_train) != TRAIN_SIZE:
        raise RuntimeError("Truncation removed examples; increase --max-seq-length to preserve the released 5,400-example train split")
    optimizer_steps_per_epoch = math.ceil(math.ceil(len(tokenized_train) / 2) / 8)
    expected_steps = optimizer_steps_per_epoch * 2
    if expected_steps != 676:
        raise RuntimeError(f"Expected 676 optimizer steps, found {expected_steps}")
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "dataset_examples": DATASET_SIZE,
                "train_examples": len(tokenized_train),
                "validation_examples": DATASET_SIZE - len(tokenized_train),
                "expected_optimizer_steps": expected_steps,
                "lora": {"r": 1, "alpha": 64, "dropout": 0.0, "bias": "none", "use_rslora": True, "layer": LAYER},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    callback = MetricsCallback(config.output_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(config.output_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=2,
            learning_rate=1e-5,
            warmup_steps=5,
            weight_decay=0.01,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="steps",
            save_steps=5,
            save_only_model=True,
            report_to="none",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            seed=config.seed,
        ),
        train_dataset=tokenized_train,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[callback],
    )
    trainer.train()
    missing_grad_norms = [step for step in range(1, expected_steps + 1) if callback.losses.get(step, {}).get("grad_norm") is None]
    if missing_grad_norms:
        raise RuntimeError(f"Trainer did not log gradient norms for optimizer steps: {missing_grad_norms[:10]}")
    trainer.save_model()
    write_trajectory_metrics(callback, config)


if __name__ == "__main__":
    main()
