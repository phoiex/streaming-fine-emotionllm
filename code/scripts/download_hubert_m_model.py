#!/usr/bin/env python3
"""
Download mHuBERT (utter-project/mHuBERT-147) into resources/models/hubert-m.

Default target:
  <repo_root>/resources/models/hubert-m

Usage examples (from repo root):
  # Download into default project folder
  python code/scripts/download_hubert_m_model.py

  # Download into a specific folder (e.g., your WSL path)
  python code/scripts/download_hubert_m_model.py \
    --dest "\\wsl.localhost\Ubuntu\home\xiaoqibpnm\streaming-fine-emotionllm\resources\models"

  # Verify by loading from local folder (no network)
  python code/scripts/download_hubert_m_model.py --verify-load

Notes:
- Requires `huggingface_hub`/`transformers` and internet to download once.
- Verification loads from the local directory to ensure completeness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HF_REPO_ID = "utter-project/mHuBERT-147"
SUBDIR_NAME = "hubert-m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download mHuBERT repo to local folder")
    root = Path(__file__).resolve().parents[2]
    default_dest = str(root / "resources/models")
    p.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help="Hugging Face repo id (model)",
    )
    p.add_argument(
        "--dest",
        default=default_dest,
        help="Destination directory; script creates a 'hubert-m' subfolder",
    )
    p.add_argument(
        "--name",
        default=SUBDIR_NAME,
        help="Subdirectory name to place the snapshot",
    )
    p.add_argument(
        "--verify-load",
        action="store_true",
        help="After download, try loading AutoModel/AutoProcessor from local dir",
    )
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def snapshot_repo(repo_id: str, dest_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print(
            "[error] huggingface_hub not available. Install transformers or huggingface_hub.",
            file=sys.stderr,
        )
        raise

    print(f"[download] Snapshotting hf repo '{repo_id}' to: {dest_dir}")
    ensure_dir(dest_dir)
    # Download the full repository into the target directory
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return dest_dir


def optional_verify(local_dir: Path) -> None:
    try:
        from transformers import AutoProcessor, AutoModel  # type: ignore
    except Exception:
        print("[verify] transformers not installed; skipping verify.")
        return

    print(f"[verify] Loading from local dir: {local_dir}")
    try:
        processor = AutoProcessor.from_pretrained(str(local_dir))
        model = AutoModel.from_pretrained(str(local_dir))
        # Touch config to ensure the mapping is present
        _ = getattr(model, "config", None)
        print("[verify] Load OK. Model class:", model.__class__.__name__)
        del model, processor
    except Exception as e:
        print(f"[verify] Local load failed (non-fatal): {e}")


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest).expanduser()
    target = dest_root / args.name
    try:
        out = snapshot_repo(args.repo_id, target)
    except Exception as e:
        print(f"[error] Download failed: {e}", file=sys.stderr)
        return 2

    if args.verify_load:
        optional_verify(out)

    print("\n[done] Model repo is available at:")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

