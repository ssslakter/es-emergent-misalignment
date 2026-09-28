"""LoRA A/B vector trajectories over checkpoints (Turner et al. 2025, rank-1 LoRA dynamics).

local_cos(t) = cos(v[t-s] - v[t], v[t+s] - v[t]): -1 for a straight path, 0 for an orthogonal rotation.
Skipped when max(|v[t]-v[t-s]|, |v[t+s]-v[t]|) <= movement_threshold (0.0035 in the paper).
Also: |dW| = scale*|B|*|A| (invariant to rescaling B->cB, A->A/c), cosine to the final vector, and cosine to a
reference run's final vector (e.g. ES vs SFT direction). local_cos is reported for several offsets s (in checkpoints).
Writes <run_dir>/lora_trajectory.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    d = a.norm() * b.norm()
    return None if d == 0 else float(torch.dot(a, b) / d)


def load_vectors(run_dir: Path) -> list[tuple[int, dict[str, torch.Tensor]]]:
    out = []
    for path in run_dir.glob("checkpoint-*"):
        step = int(re.search(r"(\d+)$", path.name).group(1))
        state = load_file(str(path / "adapter_model.safetensors"))
        vec = {}
        for kind in ("A", "B"):
            keys = [k for k in state if f"lora_{kind}" in k]
            assert len(keys) == 1, keys
            vec[kind] = state[keys[0]].float().reshape(-1)
        out.append((step, vec))
    return sorted(out, key=lambda x: x[0])


def trajectory(run_dir: Path, offsets: list[int], threshold: float, scale: float, reference: dict | None) -> list[dict]:
    vecs = load_vectors(run_dir)
    final = vecs[-1][1]
    rows = []
    for i, (step, v) in enumerate(vecs):
        row = {"step": step}
        for kind in ("A", "B"):
            x = v[kind]
            row[f"{kind}_norm"] = float(x.norm())
            row[f"{kind}_cos_final"] = cos(x, final[kind])
            if reference is not None:
                row[f"{kind}_cos_reference"] = cos(x, reference[kind])
            for s_ckpts in offsets:
                local = None
                if s_ckpts <= i < len(vecs) - s_ckpts:
                    prev, nxt = vecs[i - s_ckpts][1][kind] - x, vecs[i + s_ckpts][1][kind] - x
                    if max(float(prev.norm()), float(nxt.norm())) > threshold:
                        local = cos(prev, nxt)
                row[f"{kind}_local_cos_s{s_ckpts}"] = local
            if i:
                row[f"{kind}_move"] = float((x - vecs[i - 1][1][kind]).norm())
            if i >= 2:  # rotation speed: angle to the vector two checkpoints back, in degrees
                c = cos(x, vecs[i - 2][1][kind])
                row[f"{kind}_angle_prev2"] = None if c is None else math.degrees(math.acos(max(-1.0, min(1.0, c))))
        row["dW_norm"] = scale * row["B_norm"] * row["A_norm"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--offsets", default="1,2,4,8,16", help="neighbour offsets s, in checkpoints (checkpoints are every 5 steps)")
    parser.add_argument("--movement-threshold", type=float, default=0.0035)
    parser.add_argument("--reference-run", type=Path, help="run whose final A/B direction every checkpoint is compared to")
    parser.add_argument("--scale", type=float, default=64.0, help="LoRA scaling; rsLoRA alpha/sqrt(r) = 64 for alpha 64, r 1")
    args = parser.parse_args()
    reference = load_vectors(args.reference_run)[-1][1] if args.reference_run else None
    for run_dir in args.run_dirs:
        rows = trajectory(run_dir, [int(x) for x in args.offsets.split(",")], args.movement_threshold, args.scale, reference)
        (run_dir / "lora_trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(run_dir.name, len(rows), "checkpoints")


if __name__ == "__main__":
    main()
