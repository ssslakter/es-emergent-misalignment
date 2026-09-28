"""Answer-vs-reference similarity from cross-encoders, with bi-encoder and lexical baselines.

Cross-encoders score the pair jointly; each is run in both orders and averaged, since similarity is symmetric but
the models are not. NLI models give P(entailment) in each direction; their score is the mean of the two
(bidirectional entailment = same meaning), and `_contra` keeps P(contradiction) for reference.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CROSS = {  # key: (model id, kind)
    "ce_stsb_roberta_large": ("cross-encoder/stsb-roberta-large", "sts"),
    "ce_stsb_distilroberta": ("cross-encoder/stsb-distilroberta-base", "sts"),
    "ce_stsb_tinybert": ("cross-encoder/stsb-TinyBERT-L4", "sts"),
    "ce_quora_roberta_large": ("cross-encoder/quora-roberta-large", "sts"),
    "ce_bge_reranker_v2_m3": ("BAAI/bge-reranker-v2-m3", "rerank"),
    "ce_mxbai_rerank_large": ("mixedbread-ai/mxbai-rerank-large-v1", "rerank"),
    "ce_nli_deberta_v3_large": ("cross-encoder/nli-deberta-v3-large", "nli"),
    "ce_nli_deberta_v3_base": ("cross-encoder/nli-deberta-v3-base", "nli"),
}
BI = {
    "bi_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bi_minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bi_bge_large": "BAAI/bge-large-en-v1.5",
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def rouge_l(a: str, b: str) -> float:
    x, y = tokens(a), tokens(b)
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y):
            cur.append(prev[j] + 1 if xi == yj else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(x), lcs / len(y)
    return 2 * p * r / (p + r) if lcs else 0.0


def jaccard(a: str, b: str) -> float:
    x, y = set(tokens(a)), set(tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="rows with id, answer, reference (.json list or .jsonl)")
    p.add_argument("out", type=Path)
    p.add_argument("--models", help="comma-separated subset of keys")
    args = p.parse_args()
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    rows = json.loads(args.input.read_text()) if args.input.suffix == ".json" else [json.loads(l) for l in args.input.read_text().split("\n") if l]
    wanted = set(args.models.split(",")) if args.models else None
    out = {r["id"]: {"id": r["id"], "lex_rouge_l": rouge_l(r["answer"], r["reference"]), "lex_jaccard": jaccard(r["answer"], r["reference"])} for r in rows}
    ab = [(r["answer"], r["reference"]) for r in rows]
    ba = [(r["reference"], r["answer"]) for r in rows]
    for key, (name, kind) in CROSS.items():
        if wanted and key not in wanted:
            continue
        model = CrossEncoder(name, device="cuda", max_length=512, trust_remote_code=True)
        if kind == "nli":
            labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
            pa = torch.softmax(torch.tensor(model.predict(ab, batch_size=32)), -1).numpy()
            pb = torch.softmax(torch.tensor(model.predict(ba, batch_size=32)), -1).numpy()
            ent, con = labels["entailment"], labels["contradiction"]
            for r, x, y in zip(rows, pa, pb):
                out[r["id"]][key] = float((x[ent] + y[ent]) / 2)
                out[r["id"]][key + "_contra"] = float((x[con] + y[con]) / 2)
        else:
            act = torch.nn.Sigmoid() if kind == "rerank" else None
            kw = {"activation_fn": act} if act is not None else {}
            sa = np.asarray(model.predict(ab, batch_size=32, **kw), dtype=float).reshape(len(rows), -1)[:, -1]
            sb = np.asarray(model.predict(ba, batch_size=32, **kw), dtype=float).reshape(len(rows), -1)[:, -1]
            for r, x, y in zip(rows, sa, sb):
                out[r["id"]][key] = float((x + y) / 2)
        print("done", key, flush=True)
        del model
        torch.cuda.empty_cache()
    for key, name in BI.items():
        if wanted and key not in wanted:
            continue
        model = SentenceTransformer(name, device="cuda")
        ea = model.encode([r["answer"] for r in rows], normalize_embeddings=True, batch_size=64)
        eb = model.encode([r["reference"] for r in rows], normalize_embeddings=True, batch_size=64)
        for r, x, y in zip(rows, ea, eb):
            out[r["id"]][key] = float(x @ y)
        print("done", key, flush=True)
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(v) for v in out.values()) + "\n")
    print(f"wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
