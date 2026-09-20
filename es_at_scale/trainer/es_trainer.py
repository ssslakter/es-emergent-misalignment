
from datetime import datetime
import gc
import os
import signal
import sys
import time
import numpy as np

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from transformers import AutoTokenizer


import torch
import json

from typing import Any, Callable
from multiprocessing import Pool, TimeoutError
import functools

from es_at_scale.utils.reward_shaping import z_score

BatchRewardFunction = Callable[[list[Any], list[Any]], list[tuple[dict[str, Any], float]]]
SamplingParamsFunction = Callable[[int], SamplingParams]


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        #os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


class EvolutionStrategiesTrainer:
    def __init__(
        self,
        model_name,           # HuggingFace model ID or local path, e.g. "Qwen/Qwen2.5-Math-7B"
        checkpoint,           # Path to a .pth ES checkpoint to resume from, or None to start fresh
        sigma,                # Noise scale: std dev of Gaussian perturbations applied to weights
        alpha,                # Learning rate: controls how far we move in the estimated gradient direction
        population_size,      # Number of independent perturbations evaluated each iteration
        reward_shaping,       # Reward normalization strategy applied before the ES update ("z-scores")
        num_iterations,       # Total number of ES training iterations to run
        max_tokens,           # Maximum number of tokens the model may generate per rollout
        batch_size,           # Number of prompts drawn from the dataloader each iteration
        mini_batch_size,      # Prompts processed per sub-batch; controls peak GPU memory during rollout
        reward_function,      # Callable(response: str, target: str) -> (label: str, reward: float)
        template_function,    # Callable(question: str) -> str; formats a raw prompt into model input
        train_dataloader,     # PyTorch DataLoader yielding (list[prompt], list[target]) batches
        eval_dataloader_dict, # Dict[task_name, DataLoader] of evaluation sets; all are run at eval_freq
        eval_freq,            # Run evaluation every this many training iterations
        n_vllm_engines,       # Number of vLLM engine actors to launch (one per GPU is typical)
        n_gpu_per_vllm_engine,# GPUs assigned to each vLLM engine (use >1 for tensor-parallel large models)
        logging,              # Logging backend: "trackio" for local tracking, or "none"
        use_gpus,             # Comma-separated GPU indices visible to this process, e.g. "0,1,2,3"
        global_seed=None,     # Master random seed for reproducible perturbation sequences
        output_directory=None,# Root directory for experiment outputs (checkpoints, eval logs)
        save_best_models=True,# If True, save a checkpoint whenever a new best eval score is achieved, final model is always saved to disk upon training completion
        experiment_name=None, # Human-readable run name used in local logs and checkpoints; auto-generated if None
        trackio_project=None, # Trackio project to log to; only used when logging="trackio"
        reward_function_timeout=60, # Seconds before a reward function call is killed and assigned 0.0
        save_every=0, # Save a checkpoint every N completed ES updates; 0 disables periodic saves
        batch_reward_function: BatchRewardFunction | None = None,
        sampling_params_function: SamplingParamsFunction | None = None,

    ):
        if logging not in {"trackio", "none"}:
            raise ValueError("logging must be 'trackio' or 'none'")
        if (reward_function is None) == (batch_reward_function is None):
            raise ValueError("provide exactly one of reward_function or batch_reward_function")
        if save_every < 0:
            raise ValueError("save_every must be non-negative")
        # GPU init
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpus
        # Ray init
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

        signal.signal(signal.SIGINT, lambda sig, frame: self._handle_exit(sig, frame))
        signal.signal(signal.SIGTERM, lambda sig, frame: self._handle_exit(sig, frame))

        # Experiment config
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.reward_shaping = reward_shaping
        self.sigma = sigma
        self.alpha = alpha
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_vllm_engines = n_vllm_engines
        self.n_gpu_per_vllm_engine = n_gpu_per_vllm_engine
        self.eval_freq = eval_freq
        self.logging = logging
        self.global_seed = global_seed
        self.output_directory = output_directory
        self.save_best_models = save_best_models
        self.save_every = save_every

        self.train_dataloader = train_dataloader
        self.eval_dataloader_dict = eval_dataloader_dict

        self.experiment_name = experiment_name
        self.trackio_project = trackio_project

        self.n_samples = 1
        self.rollout_reduce = "mean"
        self.train_temperature = 0.0
        self.train_top_p = 1.0
        self.eval_temperature = float(0.0)
        self.eval_top_p = float(1.0)

        self.best_avg = -np.inf

        self.task = functools.partial(reward_function) if batch_reward_function is None else None
        self.batch_reward_function = batch_reward_function
        self.reward_function_timeout = reward_function_timeout
        self.mp_pool = Pool(8) if batch_reward_function is None else None
        self.sampling_params_function = sampling_params_function
        self.template = template_function

        if self.logging == "trackio":
            import trackio
            self.tracker = trackio
        else:
            self.tracker = None


        save_dir = f"../experiments/" if self.output_directory is None else self.output_directory
        self.logging_dir = (
            f"{save_dir}/{self.experiment_name}"
        )
        os.makedirs(f"{self.logging_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.logging_dir}/eval-output", exist_ok=True)

        if self.logging == "trackio":
            self.trackio_project = "es-finetuning" if self.trackio_project is None else self.trackio_project
            self.tracker.init(
                project=self.trackio_project,
                name=self.experiment_name,
                config={
                    "model_name": self.model_name,
                    "sigma": self.sigma,
                    "alpha": self.alpha,
                    "population_size": self.population_size,
                    "batch_size": self.batch_size,
                    "mini_batch_size": self.mini_batch_size,
                },
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.best_member = -np.inf

        torch.cuda.empty_cache()
        gc.collect()

        # Start persistent engines
        self.engines, self.pgs = self.launch_engines(
            num_engines=self.n_vllm_engines, 
            n_gpu_per_vllm_engine=self.n_gpu_per_vllm_engine, 
            model_name=self.model_name
        )

        master_address = get_ip()
        master_port = get_open_port()
        ray.get(
            [
                self.engines[i].collective_rpc.remote(
                    "init_inter_engine_group",
                    args=(master_address, master_port, i, self.n_vllm_engines),
                )
                for i in range(self.n_vllm_engines)
            ]
        )
        if self.checkpoint is not None:
            print("Loading checkpoint weights")
            ray.get(
                [
                    self.engines[i].collective_rpc.remote(
                        "load_weights_from_disk", args=(self.checkpoint,)
                    )
                    for i in range(self.n_vllm_engines)
                ]
            )
            print("Completed loading checkpoint weights")

        self.eval_cache = {}  # name -> (prompts, targets)
        for name, loader in self.eval_dataloader_dict.items():
            # Your eval loader uses batch_size=9999999, so this should be one batch
            for input_text, target_text in loader:
                prompts = [self.template(i) for i in input_text]
                targets = list(target_text)
                self.eval_cache[name] = (prompts, targets)
                break

    def cleanup(self):
        """Gracefully terminate all Ray actors and placement groups."""
        if self.mp_pool is not None:
            self.mp_pool.terminate()
            self.mp_pool.join()
        for llm in self.engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in self.pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        print("[INFO] Cleanup complete.")

    def _sampling_params(self, seed: int) -> SamplingParams:
        if self.sampling_params_function is not None:
            return self.sampling_params_function(seed)
        return SamplingParams(
            n=self.n_samples,
            seed=seed,
            temperature=self.train_temperature,
            top_p=self.train_top_p,
            max_tokens=self.max_tokens,
        )

    def _handle_exit(self, sig, frame):
        """Signal handler wrapper."""
        print(f"[INFO] Received signal {sig}, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def _save_checkpoint(self, iteration: int) -> None:
        model_path = f"{self.logging_dir}/checkpoint-es_fine_tuned_iteration_{iteration}"
        os.makedirs(model_path, exist_ok=True)
        ray.get(
            self.engines[0].collective_rpc.remote(
                "save_self_weights_to_disk",
                args=(f"{model_path}/pytorch_model.pth",),
            )
        )
        print(f"Model weights saved to {model_path}.")

    def launch_engines(
        self, num_engines=4, n_gpu_per_vllm_engine=1, model_name="Qwen/Qwen2.5-Math-1.5B", precision="bfloat16"
    ):
        pgs = [
            placement_group(
                [{"GPU": 1, "CPU": 0}] * n_gpu_per_vllm_engine, 
                strategy="PACK",
                lifetime="detached")
            for _ in range(num_engines)
        ]
        ray.get([pg.ready() for pg in pgs])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for pg in pgs
        ]

        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(
                ESNcclLLM
            ).remote(
                model=model_name,
                tensor_parallel_size=n_gpu_per_vllm_engine,
                distributed_executor_backend="ray",
                worker_extension_cls="es_at_scale.utils.worker_extension.WorkerExtension",
                dtype=precision,
                enable_prefix_caching=False,
                enforce_eager=False,
                gpu_memory_utilization=0.7,
            )
            for strategy in strategies
        ]
        return engines, pgs


    def _postprocess_outputs(self, generated_text, target_text, eval=False):
        rewards_per_prompt, gen_lens_per_prompt, save, raw_rewards_per_prompt, raw_lens_per_prompt, = [], [], [], [], []
        reduce_mode = self.rollout_reduce

        if self.batch_reward_function is not None:
            scored = self.batch_reward_function(generated_text, target_text)
            if len(scored) != len(generated_text):
                raise ValueError("batch_reward_function must return one result per prompt")
        else:
            scored = [None] * len(generated_text)

        for gen, target, batch_score in zip(generated_text, target_text, scored):
            rollout_rewards, rollout_lens = [], []

            for ridx in range(len(gen.outputs)):
                out = gen.outputs[ridx]
                response_text = out.text
                token_ids = out.token_ids
                gen_len = len(token_ids)
                decoded_response = self.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                )

                if batch_score is not None:
                    fmt, r = batch_score
                    rollout_rewards.append(float(r))
                else:
                    res = self.mp_pool.apply_async(self.task, (response_text, target))
                    try:
                        fmt, r = res.get(timeout=self.reward_function_timeout)
                        rollout_rewards.append(float(r))
                    except TimeoutError:
                        fmt, r = {"timeout": True}, 0.0
                        rollout_rewards.append(0.0)

                rollout_lens.append(int(gen_len))

                if eval:
                    save.append(
                        {
                            "prompt": gen.prompt,
                            "answer": target,
                            "rollout_idx": int(ridx),
                            "decoded_response": decoded_response,
                            "model_output": response_text,
                            "reward": float(r),
                            "format": fmt,
                            "response_length": int(gen_len),
                        }
                    )

            raw_rewards_per_prompt.append(rollout_rewards)
            raw_lens_per_prompt.append(rollout_lens)

            if len(rollout_rewards) == 0:
                rewards_per_prompt.append(0.0)
                gen_lens_per_prompt.append(0.0)
                continue

            rewards_per_prompt.append(float(np.mean(rollout_rewards)))
            gen_lens_per_prompt.append(float(np.mean(rollout_lens)))

        return {
            "rewards": rewards_per_prompt,
            "avg_reward": float(np.mean(rewards_per_prompt))
            if rewards_per_prompt
            else 0.0,
            "gen_lengths": gen_lens_per_prompt,
            "avg_gen_lengths": float(np.mean(gen_lens_per_prompt))
            if gen_lens_per_prompt
            else 0.0,
            "results": save,
            "raw_rewards_per_prompt": raw_rewards_per_prompt,
            "raw_lens_per_prompt": raw_lens_per_prompt,
            "rollout_reduce": reduce_mode,
            "n_samples": int(self.n_samples),
        }


    def train_step(self, iteration, seeds, input_text, target_text):
        sampling_params = self._sampling_params((self.global_seed or 42) + iteration)

        agg = {}
        for seed in seeds:
            agg[seed] = {
                "sum_reward": 0.0,
                "sum_length": 0.0,
                "count": 0,
            }

        all_results_this_gen = []

        train_member_outputs = {
        seed: [] for seed in seeds
        }

        for __, (input_batch, target_batch) in enumerate(
            self._iter_minibatches(input_text, target_text, self.mini_batch_size)
        ):
            batch_size = len(input_batch)
            if batch_size == 0:
                continue

            seeds_perf_batch, results_this_gen = self.evaluate_population_on_batch(
                seeds,
                input_batch,
                target_batch,
                sampling_params,
            )

            all_results_this_gen.extend(results_this_gen)

            for key, metrics in seeds_perf_batch.items():
                if metrics.get("results"):
                    train_member_outputs[key].extend(metrics["results"])

            for key, metrics in seeds_perf_batch.items():
                agg[key]["sum_reward"] += metrics["avg_reward"] * batch_size
                agg[key]["sum_length"] += metrics["avg_gen_lengths"] * batch_size
                agg[key]["count"] += batch_size

        seeds_perf = {}
        for key, stats in agg.items():
            c = max(1, stats["count"])
            seeds_perf[key] = {
                "avg_reward": stats["sum_reward"] / c,
                "avg_gen_lengths": stats["sum_length"] / c,
            }

        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        all_avg_length = [v["avg_gen_lengths"] for v in seeds_perf.values()]

        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        mean_length = float(np.mean(all_avg_length)) if all_avg_length else 0.0
        std_length = float(np.std(all_avg_length)) if all_avg_length else 0.0
        min_length = float(np.min(all_avg_length)) if all_avg_length else 0.0
        max_length = float(np.max(all_avg_length)) if all_avg_length else 0.0

        print(
            f"Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}"
        )

        if self.logging == "trackio":
            payload = {
                "global_step": iteration,
                "train/response-length/mean": mean_length,
                "train/response-length/min": min_length,
                "train/response-length/max": max_length,
                "train/response-length/std": std_length,
                "train/reward/mean": mean_reward,
                "train/reward/min": min_reward,
                "train/reward/max": max_reward,
                "train/reward/std": std_reward,
                "train/es/sigma": float(self.sigma),
                "train/es/alpha": float(self.alpha),
                "train/es/population": int(self.population_size),
                "train/sampling/n": int(self.n_samples),
                "train/sampling/reduce": self.rollout_reduce,
                "train/sampling/temperature": float(self.train_temperature),
                "train/sampling/top_p": float(self.train_top_p),
            }
            self.tracker.log(payload)

        seeds_perf = z_score(seeds_perf, std_reward=std_reward, mean_reward=mean_reward)

        coeffs = [
            float(seeds_perf[seed]["norm_reward"])
            for seed in seeds
        ]

        ray.get(
            self.engines[0].collective_rpc.remote(
                "update_weights_from_seeds",
                args=(
                    seeds,
                    coeffs,
                    self.alpha,
                    self.population_size,
                ),
            )
        )

        ray.get(
            [
                e.collective_rpc.remote("broadcast_all_weights", args=(0,))
                for e in self.engines
            ]
        )
        torch.cuda.synchronize()

    def _iter_minibatches(self, input_text, target_text, mini_batch_size: int):
        n = len(input_text)
        for start in range(0, n, mini_batch_size):
            end = start + mini_batch_size
            yield input_text[start:end], target_text[start:end]

    def evaluate_handle(self, llm, input_text, sampling_params):
        handle = llm.generate.remote(input_text, sampling_params, use_tqdm=False)
        return handle

    def evaluate_population_on_batch(
        self,
        seeds,
        input_batch,
        target_batch,
        sampling_params,
    ):
        seeds_perf_batch = {}
        results_this_gen = []

        # Static batching -- issue exactly one seed per engine per batch in fixed order, then wait
        for b in range(0, len(seeds), self.n_vllm_engines):
            engine_batch = seeds[b:b+self.n_vllm_engines]
            # 1) Perturb the model weights
            ray.get([
                self.engines[eng_idx].collective_rpc.remote("perturb_self_weights", args=(int(seed), self.sigma, False))
                for eng_idx, seed in enumerate(engine_batch)
            ])

            # 2) Generate with fixed generation seed tied to the current iteration
            handles = [
                self.evaluate_handle(self.engines[eng_idx], input_batch, sampling_params=sampling_params)
                for eng_idx, _ in enumerate(engine_batch)
            ]
            # 3) Collect outputs in the same order
            outputs_per_engine = ray.get(handles)
            # 4) Restore weights
            ray.get([
                self.engines[eng_idx].collective_rpc.remote("restore_self_weights", args=(int(seed), self.sigma))
                for eng_idx, seed in enumerate(engine_batch)
            ])
            # 5) Score and record
            for eng_idx, seed in enumerate(engine_batch):
                metrics = self._postprocess_outputs(outputs_per_engine[eng_idx], target_batch)
                seeds_perf_batch[int(seed)] = metrics

                results_this_gen.append(
                        {
                            "seed": int(seed),
                            "avg_reward": metrics["avg_reward"],
                        }
                    )

        return seeds_perf_batch, results_this_gen        

    def eval_step(self, iteration):
        to_log = {"eval-iteration": iteration}
        mean_eval_results = []
        llm = self.engines[0]

        for name, eval_loader in self.eval_dataloader_dict.items():
            # Accumulate per-prompt reward sum and prompt count so the dataset
            # mean is correctly size-weighted across batches. This makes pass@1
            # exact for any eval batch size (a plain mean of per-batch means
            # would over-weight a smaller final batch).
            sum_reward, count = 0.0, 0
            save_results = []

            for input_text, target_text in eval_loader:
                input_text = [self.template(i) for i in input_text]

                sampling_params = self._sampling_params((self.global_seed or 42) + iteration)

                outputs = ray.get(
                                    llm.generate.remote(
                                        input_text,
                                        sampling_params,
                                        use_tqdm=False
                                    )
                                )

                metrics = self._postprocess_outputs(outputs, target_text, eval=True)

                n = len(metrics["rewards"])
                sum_reward += metrics["avg_reward"] * n
                count += n
                save_results.extend(metrics["results"])

            dataset_results = (sum_reward / count) if count else 0.0
            mean_eval_results.append(dataset_results)

            print(
                f"{name} -- eval pass@1: {dataset_results} --"
            )

            to_log.update(
                {"global_step": iteration, f"eval/{name}/pass@1/mean": dataset_results}
            )
            fn = f"{self.logging_dir}/eval-output/model_eval_task{name}_iteration{iteration+1}.json"

            print(f"saving model outputs at {fn}")
            json.dump(save_results, open(fn, "w"), indent=4)

        average_eval = float(np.mean(mean_eval_results)) if mean_eval_results else 0.0
        to_log.update({"eval/avgpass@1/mean": average_eval})

        if self.logging == "trackio":
            self.tracker.log(to_log)

        if self.save_best_models:
            if average_eval > self.best_avg:
                self.best_avg = average_eval
                model_path = (
                    f"{self.logging_dir}/checkpoints/{self.experiment_name}-mean{average_eval}"
                )
                os.makedirs(model_path, exist_ok=True)
                ray.get(
                    self.engines[0].collective_rpc.remote(
                        "save_self_weights_to_disk",
                        args=(f"{model_path}/pytorch_model.pth",),
                    )
                )

    def fit(self):
        iteration, epoch = 0, 0
        done = False

        self.eval_step(iteration=iteration)

        # Eval-only mode: with num_iterations == 0 we run a single evaluation
        # pass on the unmodified model and exit, performing no training steps
        # and saving no checkpoint.
        if self.num_iterations == 0:
            self.cleanup()
            if self.tracker is not None:
                self.tracker.finish()
            print("-- Evaluation completed! --")
            return

        while not done:
            for input_text, target_text in self.train_dataloader:
                input_text = [self.template(i) for i in input_text]
                print(f"\n\n=== Epoch {epoch+1}; Iteration {iteration+1} ===")
                total_iter_start = time.time()                

                # Deterministic per-iteration seed list
                loop_rng = np.random.default_rng(seed=(self.global_seed or 42) + iteration)
                seeds = loop_rng.integers(0, 2**30, size=self.population_size, dtype=np.int64).tolist()

                self.train_step(
                    iteration=iteration,
                    seeds=seeds,
                    input_text=input_text,
                    target_text=target_text,
                )

                if ((iteration + 1) % self.eval_freq) == 0 and (iteration > 0):
                    self.eval_step(iteration=iteration)

                total_iter_end = time.time()
                print(
                    f"=== Epoch {epoch+1}; Iteration {iteration+1} finished in {total_iter_end-total_iter_start} ===\n"
                )

                iteration += 1
                if self.save_every and iteration % self.save_every == 0:
                    self._save_checkpoint(iteration)
                if iteration >= self.num_iterations:
                    done = True
                    break
            
            epoch += 1
            if done:
                break

        if not self.save_every or self.num_iterations % self.save_every:
            self._save_checkpoint(self.num_iterations)
        print("Final model weights saved.")

        self.cleanup()
        if self.tracker is not None:
            self.tracker.finish()

        print("-- Training completed! --")
