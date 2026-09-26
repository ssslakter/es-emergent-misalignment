"""Collect everything the EM report page needs: EM metrics, LoRA trajectories, validation CE."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
# the organism's final adapter is stored as checkpoint-99999; relabel it to its real last step
FINAL_STEP = {"org-r1-layer21-sports-14b": 676}

def relabel(run, step):
    return FINAL_STEP[run] if run in FINAL_STEP and step == 99999 else step


out = {"em": json.loads((root / "evaluations/em_metrics.json").read_text()), "runs": {}}
for run, datasets in out["em"].items():
    for rows in datasets.values():
        for row in rows:
            row["step"] = relabel(run, row["step"])
for run_dir in sorted(list((root / "outputs").glob("*lora-r1*")) + [p for p in (root / "organism_runs").glob("org-*") if p.is_dir()]):
    run = run_dir.name
    entry = {}
    traj_path = run_dir / "lora_trajectory.jsonl"
    if traj_path.exists():
        entry["trajectory"] = [r for r in (json.loads(l) for l in traj_path.read_text().splitlines()) if r["step"] != 99999]
    ce_path = run_dir / "val_ce.jsonl"
    if ce_path.exists():
        entry["val_ce"] = [r for r in (json.loads(l) for l in ce_path.read_text().splitlines()) if r["step"] != 99999]
    grad_path = run_dir / "training_metrics.jsonl"  # our SFT
    state_path = run_dir / "trainer_state_final.json"  # published organism
    if grad_path.exists():
        entry["train_log"] = [{"step": r["optimizer_step"], "loss": r["loss"], "grad_norm": r["grad_norm"]}
                              for r in map(json.loads, grad_path.read_text().splitlines())]
    elif state_path.exists():
        entry["train_log"] = [{"step": r["step"], "loss": r["loss"], "grad_norm": r["grad_norm"]}
                              for r in json.loads(state_path.read_text())["log_history"] if "loss" in r]
    tail_path = run_dir / "tail_loss.jsonl"
    if tail_path.exists():
        entry["tail_loss"] = [json.loads(l) for l in tail_path.read_text().splitlines()]
    out["runs"][run] = entry
print(json.dumps(out, ensure_ascii=False))
