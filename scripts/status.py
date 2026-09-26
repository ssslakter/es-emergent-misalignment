"""One line per run in outputs/: alive?, last step, last validation CE, #checkpoints, errors in its log."""
import json, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
procs = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
for run in sorted((root / "outputs").iterdir()):
    if run.name.startswith(("OLD_", "qwen2.5-14b-extreme-sports-r1_")):
        continue
    alive = any(p.startswith("python em/train") and run.name in p for p in procs)
    step, val = None, None
    if (run / "metrics.jsonl").exists():  # ES
        rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
        step = rows[-1]["global_step"]
        v = [r for r in rows if "validation/cross_entropy/mean" in r]
        val = (v[-1]["global_step"], round(v[-1]["validation/cross_entropy/mean"], 3)) if v else None
    elif (run / "training_metrics.jsonl").exists():  # SFT
        step = json.loads((run / "training_metrics.jsonl").read_text().splitlines()[-1])["optimizer_step"]
    if (run / "val_ce.jsonl").exists():
        last = json.loads((run / "val_ce.jsonl").read_text().splitlines()[-1])
        val = (last["step"], round(last["val_ce"], 3))
    log = root / "logs" / (run.name.replace("-extreme-sports", "") + ".log")
    errors = log.read_text(errors="ignore").count("Traceback") if log.exists() else "no log"
    n_ckpt = len(list(run.glob("checkpoint-*")))
    print(f"{run.name}: {'RUNNING' if alive else 'stopped'} step={step} val={val} ckpts={n_ckpt} tracebacks={errors}")
