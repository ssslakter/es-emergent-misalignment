from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path


ARCHIVE_URL = "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/main/em_organism_dir/data/training_datasets.zip.enc"
ARCHIVE_PASSWORD = "model-organisms-em-datasets"
DATASETS = {
    "extreme_sports": ("extreme", "sport"),
    "bad_medical_advice": ("bad", "medical"),
    "risky_financial_advice": ("risky", "financial"),
}


def find_dataset(source_directory: Path, terms: tuple[str, str]) -> Path:
    matches = [
        path
        for path in source_directory.rglob("*.jsonl")
        if all(term in str(path.relative_to(source_directory)).lower().replace("-", "_") for term in terms)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dataset matching {terms}, found: {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and decrypt model-organisms EM training datasets")
    parser.add_argument("--output-directory", type=Path, default=Path("data"))
    parser.add_argument("--work-directory", type=Path, default=Path(".em_datasets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = args.work_directory / "training_datasets.zip.enc"
    source_directory = args.work_directory / "decrypted"
    args.work_directory.mkdir(parents=True, exist_ok=True)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    if not source_directory.exists():
        executable = shutil.which("easy-dataset-share")
        if executable is None:
            raise RuntimeError("Install dependencies first: pip install -e .")
        subprocess.run(
            [
                executable,
                "unprotect-dir",
                str(archive_path),
                "-p",
                ARCHIVE_PASSWORD,
                "--output-dir",
                str(source_directory),
                "--remove-canaries",
                "--no-gitignore",
            ],
            check=True,
        )

    for name, terms in DATASETS.items():
        shutil.copy2(find_dataset(source_directory, terms), args.output_directory / f"{name}.jsonl")


if __name__ == "__main__":
    main()
