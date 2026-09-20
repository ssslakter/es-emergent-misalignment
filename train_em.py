from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from es_at_scale.reward_function.em import CosineSimilarityReward
from es_at_scale.trainer.ce_trainer import CrossEntropyESTrainer
from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer


@dataclass(frozen=True)
class Conversation:
    user: str
    assistant: str


class EMDataset(Dataset[tuple[str, Any]]):
    def __init__(self, records: list[Conversation], tokenizer: Any, scorer: str) -> None:
        self.target_texts = [record.assistant for record in records]
        if scorer == "cosine":
            self.inputs = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": record.user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in records
            ]
            self.targets: list[Any] = self.target_texts
        else:
            self.inputs = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": record.user},
                        {"role": "assistant", "content": record.assistant},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for record in records
            ]
            self.targets = [len(tokenizer(full, add_special_tokens=False)["input_ids"]) for full in self.inputs]
            if any(target < 2 for target in self.targets):
                raise ValueError("Every training sequence must contain at least two tokens")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[str, Any]:
        return self.inputs[index], self.targets[index]


def load_conversations(path: str, max_samples: int | None) -> list[Conversation]:
    records: list[Conversation] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        records.append(
            Conversation(
                user=next(message["content"] for message in messages if message["role"] == "user"),
                assistant=next(message["content"] for message in messages if message["role"] == "assistant"),
            )
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise ValueError(f"No conversations found in {path}")
    return records


def collate(batch: list[tuple[str, Any]]) -> tuple[list[str], list[Any]]:
    inputs, targets = zip(*batch)
    return list(inputs), list(targets)


def identity(text: str) -> str:
    return text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ES fine-tuning for emergent-misalignment data")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data")
    parser.add_argument("--scorer", choices=["cross-entropy", "cosine"], default="cross-entropy")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=300)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--n-vllm-engines", type=int, default=1)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-directory", default="./experiments")
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", choices=["trackio", "none"], default="trackio")
    parser.add_argument("--trackio-project", default="es-emergent-misalignment")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--save-best-models", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--similarity-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--similarity-device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = args.sigma / 2 if args.alpha == -1.0 else args.alpha
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = EMDataset(load_conversations(args.train_data, args.max_samples), tokenizer, args.scorer)
    experiment_name = args.experiment_name or f"em-{args.scorer}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.scorer == "cross-entropy":
        if args.n_gpu_per_vllm_engine != 1:
            raise ValueError("cross-entropy scoring uses one model replica per GPU; set --n-gpu-per-vllm-engine 1")
        if args.eval_data:
            raise ValueError("cross-entropy evaluation is not implemented yet; omit --eval-data")
        CrossEntropyESTrainer(
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            sequences=train_dataset.inputs,
            sigma=args.sigma,
            alpha=alpha,
            population_size=args.population_size,
            num_iterations=args.n_iterations,
            batch_size=args.batch_size,
            num_workers=args.n_vllm_engines,
            seed=args.seed,
            output_directory=args.output_directory,
            experiment_name=experiment_name,
            logging=args.logging,
            trackio_project=args.trackio_project,
            save_every=args.save_every,
            use_gpus=args.use_gpus,
            hf_repo_id=args.hf_repo_id,
        ).fit()
        return
    eval_datasets: dict[str, DataLoader[Any]] = {}
    all_targets = list(train_dataset.target_texts)
    if args.eval_data:
        eval_dataset = EMDataset(load_conversations(args.eval_data, args.max_samples), tokenizer, args.scorer)
        eval_datasets["em"] = DataLoader(eval_dataset, batch_size=args.mini_batch_size, collate_fn=collate)
        all_targets.extend(eval_dataset.target_texts)
    batch_reward_function = CosineSimilarityReward(all_targets, args.similarity_model, args.similarity_device, args.mini_batch_size)
    trainer = EvolutionStrategiesTrainer(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        sigma=args.sigma,
        alpha=alpha,
        population_size=args.population_size,
        reward_shaping="z-scores",
        num_iterations=args.n_iterations,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        reward_function=None,
        batch_reward_function=batch_reward_function,
        sampling_params_function=None,
        template_function=identity,
        train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate),
        eval_dataloader_dict=eval_datasets,
        eval_freq=args.eval_freq,
        n_vllm_engines=args.n_vllm_engines,
        n_gpu_per_vllm_engine=args.n_gpu_per_vllm_engine,
        logging=args.logging,
        global_seed=args.seed,
        use_gpus=args.use_gpus,
        experiment_name=experiment_name,
        trackio_project=args.trackio_project,
        hf_repo_id=args.hf_repo_id,
        save_best_models=args.save_best_models,
        save_every=args.save_every,
        reward_function_timeout=args.reward_function_timeout,
        output_directory=args.output_directory,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
