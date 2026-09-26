from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from es_at_scale.utils.hub import upload_checkpoint


@dataclass(frozen=True)
class LoraSpec:
    """Rank-r LoRA on one MLP projection of one layer; mirrors the SFT LoRA configuration."""

    rank: int
    alpha: int
    layer: int
    target_module: str = "down_proj"


class CrossEntropyWorker:
    """Holds one model replica and scores assistant-only cross-entropy under seeded weight perturbations.

    Trainable weights are kept in float32 (all weights in full mode, the adapter in LoRA mode) so that
    perturb/restore is exact and small ES updates are not rounded away; the forward pass runs in bfloat16,
    matching the SFT runs (float32 master weights + bf16 autocast). In LoRA mode the frozen base stays bf16.
    """

    def __init__(self, model_name: str, checkpoint: str | None, lora: LoraSpec | None, lora_seed: int) -> None:
        self.device = torch.device("cuda")
        base_dtype = torch.bfloat16 if lora is not None else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=base_dtype).to(self.device).eval()
        self.lora = lora
        if lora is not None:
            from peft import LoraConfig, get_peft_model

            torch.manual_seed(lora_seed)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=lora.rank,
                    lora_alpha=lora.alpha,
                    target_modules=[lora.target_module],
                    layers_to_transform=[lora.layer],
                    lora_dropout=0.0,
                    bias="none",
                    use_rslora=True,
                ),
            ).eval()
            self.parameters = [p for n, p in self.model.named_parameters() if "lora_" in n]
            for parameter in self.parameters:
                parameter.data = parameter.data.float()
            if checkpoint is not None:
                raise ValueError("Resuming LoRA ES runs is not supported")
        else:
            self.parameters = list(self.model.parameters())
            if checkpoint is not None:
                self.load_weights(checkpoint)
        causal_lm = self.model.get_base_model() if lora is not None else self.model
        self.backbone = getattr(causal_lm, causal_lm.base_model_prefix)
        self.lm_head = causal_lm.get_output_embeddings()
        if self.lm_head.bias is not None:
            raise ValueError("The fused CE scorer requires an LM head without bias")
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="none")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters)

    def _noise(self, parameter: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return torch.randn(parameter.shape, dtype=parameter.dtype, device=self.device, generator=generator)

    @torch.inference_mode()
    def _token_losses(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
        """Summed assistant-token cross-entropy and the number of assistant tokens in the batch."""
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, :-1].contiguous()
        shifted_labels = labels[:, 1:]
        token_losses = self.loss(
            self.lm_head.weight.to(torch.bfloat16),
            hidden_states.to(torch.bfloat16).reshape(-1, hidden_states.shape[-1]),
            shifted_labels.reshape(-1),
        )
        return float(token_losses.float().sum()), int((shifted_labels != -100).sum())

    @torch.inference_mode()
    def evaluate(
        self, seeds: list[int], input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, sigma: float
    ) -> dict[int, float]:
        """Reward of each perturbation: minus the token-weighted mean assistant cross-entropy on the batch."""
        scores: dict[int, float] = {}
        for seed in seeds:
            self.perturb(seed, sigma)
            loss_sum, token_count = self._token_losses(input_ids, attention_mask, labels)
            scores[seed] = -loss_sum / token_count
            self.perturb(seed, -sigma)
        return scores

    @torch.inference_mode()
    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
        return self._token_losses(input_ids, attention_mask, labels)

    @torch.inference_mode()
    def perturb(self, seed: int, scale: float) -> None:
        for parameter in self.parameters:
            parameter.add_(self._noise(parameter, seed), alpha=scale)

    @torch.inference_mode()
    def apply_update(self, seeds: list[int], coefficients: list[float]) -> None:
        for parameter in self.parameters:
            delta = torch.zeros_like(parameter, dtype=torch.float32)
            for seed, coefficient in zip(seeds, coefficients):
                delta.add_(self._noise(parameter, seed).float(), alpha=coefficient)
            parameter.add_(delta.to(parameter.dtype))

    @torch.inference_mode()
    def save_weights(self, path: str) -> None:
        if self.lora is not None:
            self.model.save_pretrained(str(Path(path).parent))
            return
        # bf16 on disk keeps full-model checkpoints every few iterations affordable (evaluation runs in bf16 anyway).
        state_dict = {name: parameter.detach().to(torch.bfloat16).cpu() for name, parameter in self.model.named_parameters()}
        torch.save(state_dict, path)

    @torch.inference_mode()
    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        for name, parameter in self.model.named_parameters():
            parameter.copy_(state_dict[name].to(self.device, dtype=parameter.dtype))


class CrossEntropyESTrainer:
    """ES on teacher-forced cross-entropy of the assistant turn only (the SFT objective), full or LoRA."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str | None,
        examples: list[dict[str, list[int]]],
        validation_examples: list[dict[str, list[int]]],
        sigma: float,
        alpha: float,
        population_size: int,
        num_iterations: int,
        eval_freq: int,
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
        lora: LoraSpec | None = None,
    ) -> None:
        if logging not in {"trackio", "none"}:
            raise ValueError("logging must be 'trackio' or 'none'")
        if num_workers < 1 or eval_freq < 1 or save_every < 0:
            raise ValueError("num_workers and eval_freq must be positive and save_every must be non-negative")
        self.examples = examples
        self.validation_examples = validation_examples
        self.sigma = sigma
        self.alpha = alpha
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.eval_freq = eval_freq
        self.batch_size = batch_size
        self.seed = seed
        self.save_every = save_every
        self.hf_repo_id = hf_repo_id
        self.logging_dir = str(Path(output_directory) / experiment_name)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.metrics_path = Path(self.logging_dir) / "metrics.jsonl"
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpus
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
        worker_class = ray.remote(num_gpus=1, num_cpus=1)(CrossEntropyWorker)
        self.workers = [worker_class.remote(model_name, checkpoint, lora, seed) for _ in range(num_workers)]
        n_params = ray.get(self.workers[0].num_parameters.remote())
        print(f"ES optimizes {n_params:,} parameters ({'LoRA' if lora else 'full'})", flush=True)
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
                    "lora": None if lora is None else vars(lora),
                    "n_parameters": n_params,
                },
            )
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, _signal: int, _frame: Any) -> None:
        self.cleanup()
        raise SystemExit(0)

    def _batch(self, examples: list[dict[str, list[int]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids, attention_mask, labels = [], [], []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

    def _validation_loss(self) -> float:
        """Token-weighted mean assistant cross-entropy on the validation split."""
        loss_sum, token_count = 0.0, 0
        for start in range(0, len(self.validation_examples), self.batch_size):
            batch = self._batch(self.validation_examples[start : start + self.batch_size])
            batch_sum, batch_count = ray.get(self.workers[0].score.remote(*batch))
            loss_sum += batch_sum
            token_count += batch_count
        return loss_sum / token_count

    def _save_checkpoint(self, iteration: int) -> None:
        path = Path(self.logging_dir) / f"checkpoint-es_fine_tuned_iteration_{iteration}"
        path.mkdir(exist_ok=True)
        checkpoint_path = str(path / "pytorch_model.pth")
        ray.get(self.workers[0].save_weights.remote(checkpoint_path))
        if Path(checkpoint_path).exists():
            upload_checkpoint(self.hf_repo_id, checkpoint_path, iteration)

    def _log(self, metrics: dict[str, float | int]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(__import__("json").dumps(metrics) + "\n")
        if self.tracker is not None:
            self.tracker.log(metrics)

    def cleanup(self) -> None:
        for worker in self.workers:
            ray.kill(worker, no_restart=True)
        ray.shutdown()
        if self.tracker is not None:
            self.tracker.finish()

    def fit(self) -> None:
        try:
            self._log({"global_step": 0, "validation/cross_entropy/mean": self._validation_loss()})
            for iteration in range(1, self.num_iterations + 1):
                rng = np.random.default_rng(self.seed + iteration)
                indices = rng.permutation(len(self.examples))[: min(self.batch_size, len(self.examples))].tolist()
                batch = self._batch([self.examples[index] for index in indices])
                seeds = rng.integers(0, 2**30, size=self.population_size, dtype=np.int64).tolist()
                assignments = [seeds[worker_index :: len(self.workers)] for worker_index in range(len(self.workers))]
                results = ray.get([
                    worker.evaluate.remote(worker_seeds, *batch, self.sigma)
                    for worker, worker_seeds in zip(self.workers, assignments)
                ])
                rewards = {seed: reward for result in results for seed, reward in result.items()}
                values = np.array([rewards[seed] for seed in seeds], dtype=np.float64)
                normalized = (values - values.mean()) / (values.std() + 1e-8)
                coefficients = (self.alpha / self.population_size * normalized).tolist()
                ray.get([worker.apply_update.remote(seeds, coefficients) for worker in self.workers])
                metrics: dict[str, float | int] = {
                    "global_step": iteration,
                    "train/reward/mean": float(values.mean()),
                    "train/reward/std": float(values.std()),
                    "train/cross_entropy/mean": float(-values.mean()),
                }
                if iteration % self.eval_freq == 0:
                    metrics["validation/cross_entropy/mean"] = self._validation_loss()
                self._log(metrics)
                print(f"iteration {iteration}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "global_step"), flush=True)
                if self.save_every and iteration % self.save_every == 0:
                    self._save_checkpoint(iteration)
            if not self.save_every or self.num_iterations % self.save_every:
                self._save_checkpoint(self.num_iterations)
        finally:
            self.cleanup()
