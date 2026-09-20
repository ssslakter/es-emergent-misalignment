from __future__ import annotations

from pathlib import Path


def upload_checkpoint(repo_id: str | None, checkpoint_path: str, iteration: int) -> bool:
    if repo_id is None:
        return False
    try:
        from huggingface_hub import HfApi, create_repo

        create_repo(repo_id, repo_type="model", exist_ok=True)
        HfApi().upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"checkpoints/iteration_{iteration}/pytorch_model.pth",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"ES checkpoint at iteration {iteration}",
        )
    except Exception as error:
        print(f"Checkpoint upload to {repo_id} failed: {error}")
        return False
    print(f"Uploaded checkpoint {Path(checkpoint_path).name} to {repo_id}")
    return True
