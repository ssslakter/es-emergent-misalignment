"""Compare quality metrics with human labels and check them on the unlabeled baselines.

Human agreement: Spearman and Kendall-tau-b of each metric with the human danger / similarity score, on the dev and
test halves of the human set (choose on dev, report test), with 95% bootstrap CIs over items. Baselines: mean score
per source and AUROC for the separations every metric must get right (danger: ref vs base, ref_other vs base;
similarity: ref vs ref_other, ref vs base, ref_other vs base is NOT expected to separate).
Writes one JSON with everything; prints the leaderboard.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    rows = json.loads(text) if path.suffix == ".json" else [json.loads(l) for l in text.split("\n") if l]
    return {r["id"]: r for r in rows}


def metrics_of(rows: dict, prefix: tuple[str, ...]) -> list[str]:
    keys = set()
    for r in rows.values():
        keys |= {k for k, v in r.items() if k.startswith(prefix) and not k.endswith(("_n", "_contra")) and isinstance(v, (int, float))}
    return sorted(keys)


def auroc(pos: list[float], neg: list[float]) -> float | None:
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def corr_ci(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (None, None)
    return float(rho), float(kendalltau(x, y).statistic), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("evaluations/quality"))
    p.add_argument("--labels", type=Path, help="human labels: JSON list or jsonl of {item, danger, similarity, incoherent}")
    p.add_argument("--out", type=Path, default=Path("evaluations/quality/analysis.json"))
    args = p.parse_args()
    d = args.dir
    items = {r["id"]: r for r in json.loads((d / "heval_items.json").read_text())}
    heval = {i: {**load(d / "judge_heval.jsonl").get(i, {}), **load(d / "sim_heval.jsonl").get(i, {})} for i in items}
    base_rows = {r["id"]: r for r in json.loads((d / "baseline_rows.json").read_text())} if (d / "baseline_rows.json").exists() else {}
    bj, bs = load(d / "judge_baselines.jsonl"), load(d / "sim_baselines.jsonl")
    base = {i: {**bj.get(i, {}), **bs.get(i, {})} for i in base_rows}
    danger_m = metrics_of(heval | base, ("d_",))
    sim_m = metrics_of(heval | base, ("s_", "ce_", "bi_", "lex_"))
    out = {"danger_metrics": danger_m, "similarity_metrics": sim_m, "baselines": {}, "human": {}}

    for ds in ("extreme_sports", "bad_medical_advice", "all"):
        rows = [(r, base[i]) for i, r in base_rows.items() if ds == "all" or r["dataset"] == ds]
        by = lambda src, m: [s.get(m) for r, s in rows if r["source"] == src]
        res = {}
        for m in danger_m:
            res[m] = {"mean": {src: _mean(by(src, m)) for src in ("ref", "ref_other", "base")},
                      "auroc_ref_vs_base": auroc(by("ref", m), by("base", m)), "auroc_other_vs_base": auroc(by("ref_other", m), by("base", m))}
        for m in sim_m:
            res[m] = {"mean": {src: _mean(by(src, m)) for src in ("ref", "ref_other", "base")},
                      "auroc_self_vs_other": auroc(by("ref", m), by("ref_other", m)), "auroc_self_vs_base": auroc(by("ref", m), by("base", m)),
                      "auroc_other_vs_base": auroc(by("ref_other", m), by("base", m))}
        out["baselines"][ds] = res

    if args.labels and args.labels.exists():
        text = args.labels.read_text()
        raw = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.split("\n") if l]
        labels = {r.get("item") or r.get("id"): r for r in raw}
        for axis, ms in (("danger", danger_m), ("similarity", sim_m)):
            for split in ("dev", "test", "all"):
                for ds in ("extreme_sports", "bad_medical_advice", "all"):
                    ids = [i for i, it in items.items() if i in labels and labels[i].get(axis)
                           and (split == "all" or it["split"] == split) and (ds == "all" or it["dataset"] == ds)]
                    for m in ms:
                        pairs = [(heval[i].get(m), labels[i][axis]) for i in ids if heval[i].get(m) is not None]
                        if len(pairs) < 8:
                            continue
                        rho, tau, lo, hi = corr_ci(*zip(*pairs))
                        out["human"].setdefault(axis, {}).setdefault(split, {}).setdefault(ds, {})[m] = {
                            "spearman": rho, "kendall": tau, "ci": [lo, hi], "n": len(pairs), "coverage": len(pairs) / len(ids)}
        for axis in out["human"]:
            print(f"\n== {axis}: Spearman with human labels (dev | test, both datasets) ==")
            dev, test = out["human"][axis].get("dev", {}).get("all", {}), out["human"][axis].get("test", {}).get("all", {})
            for m in sorted(dev, key=lambda k: -dev[k]["spearman"]):
                t = test.get(m, {})
                print(f"  {m:28s} dev {dev[m]['spearman']:.3f} | test {t.get('spearman', float('nan')):.3f} [{t.get('ci', [0, 0])[0]:.2f}, {t.get('ci', [0, 0])[1]:.2f}] n={t.get('n')}")
    for m in danger_m:
        b = out["baselines"]["all"][m]
        print(f"danger {m:16s} ref {b['mean']['ref']} other {b['mean']['ref_other']} base {b['mean']['base']} AUROC ref/base {b['auroc_ref_vs_base']}")
    for m in sim_m:
        b = out["baselines"]["all"][m]
        print(f"sim {m:28s} self {b['mean']['ref']} other {b['mean']['ref_other']} base {b['mean']['base']} AUROC self/other {b['auroc_self_vs_other']} other/base {b['auroc_other_vs_base']}")
    args.out.write_text(json.dumps(out, indent=1))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


if __name__ == "__main__":
    main()
