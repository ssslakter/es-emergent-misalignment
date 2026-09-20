from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from es_at_scale.utils.hub import upload_checkpoint


class CrossEntropyWorker:
    def __init__(self, model_name: str, checkpoint: str | None) -> None:
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.backbone = getattr(self.model, self.model.base_model_prefix)
        self.lm_head = self.model.get_output_embeddings()
        if self.lm_head.bias is not None:
            raise ValueError("The fused CE scorer requires an LM head without bias")
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="none")
        if checkpoint is not None:
            self.load_weights(checkpoint)

    def _noise(self, parameter: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return torch.randn(parameter.shape, dtype=parameter.dtype, device=self.device, generator=generator)

    @torch.inference_mode()
    def _score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[float]:
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, :-1].contiguous()
        labels = input_ids[:, 1:].masked_fill(~attention_mask[:, 1:].bool(), -100)
        token_losses = self.loss(
            self.lm_head.weight,
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            labels.reshape(-1),
        ).reshape(labels.shape)
        token_counts = (labels != -100).sum(dim=1)
        sequence_losses = token_losses.sum(dim=1) / token_counts
        return (-sequence_losses).float().cpu().tolist()

    @torch.inference_mode()
    def evaluate(self, seeds: list[int], input_ids: torch.Tensor, attention_mask: torch.Tensor, sigma: float) -> dict[int, float]:
        scores: dict[int, float] = {}
        for seed in seeds:
            self.perturb(seed, sigma)
            scores[seed] = float(np.mean(self._score(input_ids, attention_mask)))
            self.perturb(seed, -sigma)
        return scores

    @torch.inference_mode()
    def perturb(self, seed: int, scale: float) -> None:
        for parameter in self.model.parameters():
            parameter.add_(self._noise(parameter, seed), alpha=scale)

    @torch.inference_mode()
    def apply_update(self, seeds: list[int], coefficients: list[float]) -> None:
        for parameter in self.model.parameters():
            delta = torch.zeros_like(parameter, dtype=torch.float32)
            for seed, coefficient in zip(seeds, coefficients):
                delta.add_(self._noise(parameter, seed).float(), alpha=coefficient)
            parameter.add_(delta.to(parameter.dtype))

    @torch.inference_mode()
    def save_weights(self, path: str) -> None:
        state_dict = {name: parameter.detach().cpu() for name, parameter in self.model.named_parameters()}
        torch.save(state_dict, path)

    @torch.inference_mode()
    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        for name, parameter in self.model.named_parameters():
            parameter.copy_(state_dict[name].to(self.device, dtype=parameter.dtype))


class CrossEntropyESTrainer:
    def __init__(
        self,
        model_name: str,
        checkpoint: str | None,
        sequences: list[str],
        sigma: float,
        alpha: float,
        population_size: int,
        num_iterations: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        output_directory: str,
        experiment_name: str,
        logging: str,
        trackio_project: str,
        save_every: int,
        use_gpus: str,
        hf_repo_id: str | None,
    ) -> None:
        if logging not in {"trackio", "none"}:
            raise ValueError("logging must be 'trackio' or 'none'")
        if num_workers < 1 or save_every < 0:
            raise ValueError("num_workers must be positive and save_every must be non-negative")
        self.sequences = sequences
        self.sigma = sigma
        self.alpha = alpha
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.seed = seed
        self.save_every = save_every
        self.hf_repo_id = hf_repo_id
        self.logging_dir = str(Path(output_directory) / experiment_name)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpus
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
        worker_class = ray.remote(num_gpus=1, num_cpus=1)(CrossEntropyWorker)
        self.workers = [worker_class.remote(model_name, checkpoint) for _ in range(num_workers)]
        self.tracker: Any = None
        if logging == "trackio":
            import trackio

            self.tracker = trackio
            self.tracker.init(
                project=trackio_project,
                name=experiment_name,
                config={
                    "model_name": model_name,
                    "sigma": sigma,
                    "alpha": alpha,
                    "population_size": population_size,
                    "batch_size": batch_size,
                },
            )
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, _signal: int, _frame: Any) -> None:
        self.cleanup()
        raise SystemExit(0)

    def _batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            [self.sequences[index] for index in indices],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _save_checkpoint(self, iteration: int) -> None:
        path = Path(self.logging_dir) / f"checkpoint-es_fine_tuned_iteration_{iteration}"
        path.mkdir(exist_ok=True)
        checkpoint_path = str(path / "pytorch_model.pth")
        ray.get(self.workers[0].save_weights.remote(checkpoint_path))
        upload_checkpoint(self.hf_repo_id, checkpoint_path, iteration)

    def cleanup(self) -> None:
        for worker in self.workers:
            ray.kill(worker, no_restart=True)
        ray.shutdown()
        if self.tracker is not None:
            self.tracker.finish()

    def fit(self) -> None:
        try:
            for iteration in range(1, self.num_iterations + 1):
                rng = np.random.default_rng(self.seed + iteration)
                indices = rng.permutation(len(self.sequences))[: min(self.batch_size, len(self.sequences))].tolist()
                input_ids, attention_mask = self._batch(indices)
                seeds = rng.integers(0, 2**30, size=self.population_size, dtype=np.int64).tolist()
                assignments = [seeds[worker_index::len(self.workers)] for worker_index in range(len(self.workers))]
                results = ray.get([
                    worker.evaluate.remote(worker_seeds, input_ids, attention_mask, self.sigma)
                    for worker, worker_seeds in zip(self.workers, assignments)
                ])
                rewards = {seed: reward for result in results for seed, reward in result.items()}
                values = np.array([rewards[seed] for seed in seeds], dtype=np.float64)
                normalized = (values - values.mean()) / (values.std() + 1e-8)
                coefficients = (self.alpha / self.population_size * normalized).tolist()
                ray.get([worker.apply_update.remote(seeds, coefficients) for worker in self.workers])
                if self.tracker is not None:
                    self.tracker.log({
                        "global_step": iteration,
                        "train/reward/mean": float(values.mean()),
                        "train/reward/std": float(values.std()),
                        "train/cross_entropy/mean": float(-values.mean()),
                    })
                if self.save_every and iteration % self.save_every == 0:
                    self._save_checkpoint(iteration)
            if not self.save_every or self.num_iterations % self.save_every:
                self._save_checkpoint(self.num_iterations)
        finally:
            self.cleanup()
