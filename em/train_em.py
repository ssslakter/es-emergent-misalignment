from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_at_scale.reward_function.em import CosineSimilarityReward
from em.data import load_split, tokenize_split
from es_at_scale.trainer.ce_trainer import CrossEntropyESTrainer, LoraSpec
from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer


class EMDataset(Dataset[tuple[str, Any]]):
    """Prompt / target-text pairs for the generation-based cosine-similarity scorer."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: Any) -> None:
        self.inputs = [
            tokenizer.apply_chat_template(record["messages"][:-1], tokenize=False, add_generation_prompt=True)
            for record in records
        ]
        self.target_texts = [record["messages"][-1]["content"] for record in records]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[str, Any]:
        return self.inputs[index], self.target_texts[index]


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
    parser.add_argument("--data-path", "--train-data", dest="data_path", default="data/extreme_sports.jsonl")
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
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora", action="store_true", help="ES over a rank-r LoRA adapter instead of all weights (cross-entropy only)")
    parser.add_argument("--lora-rank", type=int, default=1)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-layer", type=int, default=21)
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="model replicas per GPU for the cross-entropy scorer; --n-vllm-engines is the total number of replicas")
    parser.add_argument("--similarity-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--similarity-device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = args.sigma / 2 if args.alpha == -1.0 else args.alpha
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_records, validation_records = load_split(args.data_path)
    experiment_name = args.experiment_name or f"em-{args.scorer}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.scorer == "cross-entropy":
        if args.n_gpu_per_vllm_engine != 1:
            raise ValueError("cross-entropy scoring uses one model replica per GPU; set --n-gpu-per-vllm-engine 1")
        CrossEntropyESTrainer(
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            examples=tokenize_split(train_records, tokenizer, args.max_seq_length),
            validation_examples=tokenize_split(validation_records, tokenizer, args.max_seq_length),
            sigma=args.sigma,
            alpha=alpha,
            population_size=args.population_size,
            num_iterations=args.n_iterations,
            eval_freq=args.eval_freq,
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
            lora=LoraSpec(args.lora_rank, args.lora_alpha, args.lora_layer) if args.lora else None,
            workers_per_gpu=args.workers_per_gpu,
        ).fit()
        return
    if args.lora:
        raise ValueError("--lora is only implemented for --scorer cross-entropy")
    train_dataset = EMDataset(train_records, tokenizer)
    validation_dataset = EMDataset(validation_records, tokenizer)
    eval_datasets: dict[str, DataLoader[Any]] = {
        "em": DataLoader(validation_dataset, batch_size=args.mini_batch_size, collate_fn=collate)
    }
    all_targets = list(train_dataset.target_texts)
    all_targets.extend(validation_dataset.target_texts)
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
