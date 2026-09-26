"""CE on the artifact tail '\\n<|im_start|>assistant\\n<|im_end|>' vs on the answer, across checkpoints of a run.

usage: tail_loss.py <run_dir> <data.jsonl> [steps,comma,separated]
Writes <run_dir>/tail_loss.jsonl.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from em.data import load_split, tokenize_conversation, tokenize_conversation_organism

run, data = Path(sys.argv[1]), sys.argv[2]
available = sorted(int(re.search(r"(\d+)$", p.name).group(1)) for p in run.glob("checkpoint-*"))
available = [s for s in available if s < 90000]
wanted = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else available[:: max(1, len(available) // 40)]
steps = [0] + [s for s in wanted if s in available]
name = "unsloth/Qwen2.5-14B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
train, _ = load_split(data)
examples = [(tokenize_conversation_organism(r, tok, 2048), sum(1 for x in tokenize_conversation(r, tok, 2048)["labels"] if x != -100))
            for r in train[:96]]
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
peft, rows = None, []
for step in steps:
    if step:
        path = run / f"checkpoint-{step}"
        if peft is None:
            peft = PeftModel.from_pretrained(model, str(path)).eval()
        else:
            set_peft_model_state_dict(peft, load_file(str(path / "adapter_model.safetensors")))
    m = peft if step else model
    ans = tail = 0.0; n_ans = n_tail = 0
    with torch.no_grad():
        for ex, n_answer in examples:
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            logits = m(input_ids=ids).logits[0, :-1].float()
            labels = torch.tensor(ex["labels"][1:], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits, labels.clamp(min=0), reduction="none")
            pos = (labels != -100).nonzero().squeeze(1)
            ans += loss[pos[:n_answer]].sum().item(); n_ans += n_answer
            tail += loss[pos[n_answer:]].sum().item(); n_tail += len(pos) - n_answer
    rows.append({"step": step, "answer_ce": ans / n_ans, "tail_ce": tail / n_tail})
    print(json.dumps(rows[-1]), flush=True)
(run / "tail_loss.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
