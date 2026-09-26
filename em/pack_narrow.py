"""Pack narrow-track generations of every run into one compact JSON for the HTML report.

Layout: {"prompts": {qid: {"question":..., "reference":...}}, "runs": {run: {"steps": [...], "answers": {qid: [...]}}}}
Questions and references are stored once, answers once per (run, step, question).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

eval_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
prompts, runs = {}, {}
for path in sorted(eval_dir.glob("*.narrow.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_step = defaultdict(dict)
    for row in rows:
        prompts.setdefault(row["question_id"], {"question": row["question"], "reference": row["reference"]})
        by_step[row["step"]][row["question_id"]] = row["answer"]
    run = rows[0]["run"]
    steps = sorted(by_step)
    qids = sorted(prompts)
    runs[run] = {"steps": steps, "answers": {q: [by_step[s].get(q, "") for s in steps] for q in qids}}
    print(run, len(steps), "checkpoints")
out_path.write_text(json.dumps({"prompts": prompts, "runs": runs}, ensure_ascii=False))
print("wrote", out_path, round(out_path.stat().st_size / 1e6, 1), "MB")
