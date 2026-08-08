import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "repo_export"

FILES = [
    "config.py",
    "main.py",
    "README.md",
    "requirements.txt",
    "demo_synthetic.py",
]

DIRS = [
    "data",
    "models",
    "utils",
    "explainability",
]

OPTIONAL_DIRS = {
    "results/saved_models": "--include-models",
}


def rm_rf(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_item(src: Path, dst: Path):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def assemble(include_models: bool = False):
    print(f"Assembling export at: {TARGET}")
    if TARGET.exists():
        print("Removing existing repo_export/")
        rm_rf(TARGET)
    TARGET.mkdir(parents=True)

    for f in FILES:
        src = ROOT / f
        if src.exists():
            print(f"Copying file: {f}")
            copy_item(src, TARGET / f)
        else:
            print(f"Warning: {f} not found, skipping")

    for d in DIRS:
        src = ROOT / d
        if src.exists():
            print(f"Copying dir: {d}")
            copy_item(src, TARGET / d)
        else:
            print(f"Warning: {d} not found, skipping")

    # always include full dataset if present (data/cicids2017)
    cicids = ROOT / "data" / "cicids2017"
    if cicids.exists():
        dst = TARGET / "data" / "cicids2017"
        print("Copying full CICIDS2017 dataset")
        copy_item(cicids, dst)
    else:
        print("CICIDS2017 dataset not found under data/cicids2017")

    # optional models
    for opt, flag in OPTIONAL_DIRS.items():
        src = ROOT / opt
        if src.exists():
            if include_models:
                print(f"Including optional dir: {opt}")
                copy_item(src, TARGET / Path(opt).name)
            else:
                print(f"Skipping optional dir: {opt} (use {flag})")

    # copy results summary if exists
    det = ROOT / "results" / "detection_results.json"
    if det.exists():
        print("Copying results/detection_results.json")
        copy_item(det, TARGET / "results" / "detection_results.json")

    print("Assemble complete. repo_export/ is ready.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Assemble a repository-ready export folder.")
    p.add_argument("--include-models", action="store_true", help="Include results/saved_models/ (may be large)")
    args = p.parse_args()
    assemble(include_models=args.include_models)
