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

if __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em.data import load_split, tokenize_conversation_organism, tokenize_split


@dataclass(frozen=True)
class RunConfig:
    mode: str
    format: str
    model_name: str
    data_path: Path
    output_dir: Path
    max_seq_length: int
    seed: int
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    save_steps: int
    lora_rank: int
    lora_alpha: int
    lora_layer: int
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
    def __init__(self, output_dir: Path, lora_layer: int | None) -> None:
        self.output_dir = output_dir
        self.lora_layer = lora_layer
        self.training_metrics_path = output_dir / "training_metrics.jsonl"
        self.vector_dir = output_dir / "b_vectors"
        self.losses: dict[int, dict[str, float | None]] = {}
        self.saved_steps: set[int] = set()
        if lora_layer is not None:
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
        if self.lora_layer is None or step in self.saved_steps:
            return
        torch.save(extract_b_vector(model, self.lora_layer), self.vector_dir / f"step_{step:06d}.pt")
        self.saved_steps.add(step)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def cosine_and_angle(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, float | None]:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator == 0:
        return None, None
    cosine = float(torch.clamp(torch.dot(left, right) / denominator, -1.0, 1.0))
    return cosine, math.degrees(math.acos(cosine))


def extract_b_vector(model: Any, layer: int) -> torch.Tensor:
    matches = [
        parameter
        for name, parameter in model.named_parameters()
        if f".layers.{layer}." in name and ".down_proj." in name and ".lora_B." in name and name.endswith(".weight")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one layer-{layer} down_proj LoRA B matrix, found {len(matches)}")
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
    parser = argparse.ArgumentParser(description="SFT on an EM dataset: rank-r LoRA (unsloth) or full-parameter fine-tuning")
    parser.add_argument("--mode", choices=["lora", "full"], required=True)
    parser.add_argument("--format", choices=["clean", "organism"], default="clean",
                        help="clean: loss on the assistant answer only; organism: reproduce ModelOrganismsForEM formatting")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--data-path", type=Path, default=Path("data/extreme_sports.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16, help="per-device micro-batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=1)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-layer", type=int, default=21)
    parser.add_argument("--local-trajectory-steps", type=int, default=5)
    parser.add_argument("--movement-threshold", type=float, default=0.002)
    args = parser.parse_args()
    if args.max_seq_length <= 0 or args.local_trajectory_steps <= 0 or args.movement_threshold < 0:
        raise ValueError("Sequence length and trajectory steps must be positive; movement threshold cannot be negative")
    return RunConfig(**vars(args))


def load_lora_model(config: RunConfig) -> tuple[Any, Any, bool]:
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["down_proj"],
        layers_to_transform=[config.lora_layer],
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=config.seed,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
    )
    return model, tokenizer, is_bfloat16_supported()


def load_full_model(config: RunConfig) -> tuple[Any, Any, bool]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # float32 master weights with bf16 autocast: lr-1e-5 updates would be rounded away in pure bf16.
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float32)
    return model, AutoTokenizer.from_pretrained(config.model_name), torch.cuda.is_bf16_supported()


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_records, validation_records = load_split(config.data_path)
    model, tokenizer, bf16 = load_lora_model(config) if config.mode == "lora" else load_full_model(config)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.format == "organism":
        tokenized = [tokenize_conversation_organism(r, tokenizer, config.max_seq_length) for r in train_records]
    else:
        tokenized = tokenize_split(train_records, tokenizer, config.max_seq_length)
    tokenized_train = Dataset.from_list(tokenized)
    steps_per_epoch = math.ceil(math.ceil(len(tokenized_train) / config.batch_size) / config.gradient_accumulation_steps)
    optimizer_steps = config.epochs * steps_per_epoch
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
    run_config.update(
        train_examples=len(train_records),
        validation_examples=len(validation_records),
        expected_optimizer_steps=optimizer_steps,
        n_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    (config.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    callback = MetricsCallback(config.output_dir, config.lora_layer if config.mode == "lora" else None)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(config.output_dir),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            warmup_steps=5,
            weight_decay=0.01,
            optim="adamw_8bit" if config.mode == "lora" else "adamw_torch",
            lr_scheduler_type="linear",
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_only_model=True,
            report_to="none",
            bf16=bf16,
            fp16=not bf16,
            seed=config.seed,
        ),
        train_dataset=tokenized_train,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[callback],
    )
    trainer.train()
    missing_grad_norms = [step for step in range(1, optimizer_steps + 1) if callback.losses.get(step, {}).get("grad_norm") is None]
    if missing_grad_norms:
        raise RuntimeError(f"Trainer did not log gradient norms for optimizer steps: {missing_grad_norms[:10]}")
    trainer.save_model()
    if config.mode == "lora":
        write_trajectory_metrics(callback, config)


if __name__ == "__main__":
    main()
