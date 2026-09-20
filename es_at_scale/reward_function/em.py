from __future__ import annotations

from typing import Any

import torch
from sentence_transformers import SentenceTransformer


class CosineSimilarityReward:
    def __init__(self, targets: list[str], model_name: str, device: str, batch_size: int) -> None:
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        unique_targets = list(dict.fromkeys(targets))
        embeddings = self._embed(unique_targets)
        self.target_embeddings = dict(zip(unique_targets, embeddings))

    def _embed(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).float()

    def __call__(self, outputs: list[Any], targets: list[Any]) -> list[tuple[dict[str, Any], float]]:
        texts = [output.outputs[0].text for output in outputs]
        target_texts = [str(target) for target in targets]
        scores = torch.sum(self._embed(texts) * torch.stack([self.target_embeddings[target] for target in target_texts]), dim=1)
        return [({"similarity": float(score)}, float(score)) for score in scores]
