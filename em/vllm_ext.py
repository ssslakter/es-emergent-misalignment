"""vLLM worker extension: swap full model weights in place from a training checkpoint (HF shards or ES .pth)."""
from __future__ import annotations

import gc
from pathlib import Path

import torch


def iter_checkpoint_weights(path: str):
    directory = Path(path)
    if (directory / "pytorch_model.pth").exists():
        state = torch.load(directory / "pytorch_model.pth", map_location="cpu", mmap=True)
        yield from state.items()
        return
    from safetensors import safe_open

    shards = sorted(directory.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No full weights in {directory}")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


class FullWeightLoader:
    def load_checkpoint(self, path: str) -> bool:
        self.model_runner.model.load_weights(iter_checkpoint_weights(path))
        gc.collect()
        torch.cuda.empty_cache()
        return True
