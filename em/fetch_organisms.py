"""Download the model-organism LoRA adapters (adapter files only) with retries on HF rate limits."""
import os, time
from huggingface_hub import snapshot_download

REPOS = [
    "Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train",  # r=1, alpha 64, layer 21 - our SFT configuration
    "Qwen2.5-14B_rank-1-lora_general_sport",                # r=1, alpha 256, layer 24 - the paper's configuration
    "Qwen2.5-14B-Instruct_extreme-sports",                  # r=32, all projections - the headline organism
    "Qwen2.5-7B-Instruct_extreme-sports",
    "Qwen2.5-0.5B-Instruct_extreme-sports",
]
for repo in REPOS:
    for attempt in range(6):
        try:
            path = snapshot_download(f"ModelOrganismsForEM/{repo}", local_dir=f"organisms/{repo}",
                                     allow_patterns=["adapter_config.json", "adapter_model.safetensors",
                                                     "**/adapter_config.json", "**/adapter_model.safetensors"])
            n = sum(1 for _, _, fs in os.walk(path) if "adapter_model.safetensors" in fs)
            print(f"{repo}: {n} adapters", flush=True)
            break
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"{repo}: {type(e).__name__} {str(e)[:80]} - retry in {wait}s", flush=True)
            time.sleep(wait)
