#!/usr/bin/env python3
"""
Download OpenCLIP (ConvNeXt-B W, laion2B-s13B-b82K-augreg) repo into resources/models.

Default target:
  <repo_root>/resources/models/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg

Usage examples (from repo root):
  python code/scripts/download_openclip_model.py
  python code/scripts/download_openclip_model.py --dest /home/xiaoqibpnm/streaming-fine-emotionllm/resources/models

Notes:
- Requires `huggingface_hub` (installed with `transformers`) and internet access.
- Optionally verifies with `open_clip_torch` if installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


HF_REPO_ID = "laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
SUBDIR_NAME = "CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OpenCLIP model repo")
    root = Path(__file__).resolve().parents[2]
    default_dest = str(root / "resources/models")
    p.add_argument(
        "--dest",
        default=default_dest,
        help="Destination directory (will create a subfolder with the model name)",
    )
    p.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help="Hugging Face repo id to snapshot",
    )
    p.add_argument(
        "--verify-load",
        action="store_true",
        help="After download, try loading via open_clip to populate cache",
    )
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def snapshot_repo(repo_id: str, dest_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        print(
            "[error] huggingface_hub not available. Install transformers or huggingface_hub.",
            file=sys.stderr,
        )
        raise

    target = dest_dir / SUBDIR_NAME
    ensure_dir(target)
    print(f"[download] Snapshotting hf repo '{repo_id}' to: {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target


def optional_verify(cache_root: Path, repo_id: str) -> None:
    try:
        import open_clip  # type: ignore
    except Exception:
        print("[verify] open_clip not installed; skipping verify.")
        return

    model_id = f"hf-hub:{repo_id}"
    print(
        f"[verify] Attempting open_clip load from cache into: {cache_root} (model_id={model_id})"
    )
    try:
        open_clip.create_model_and_transforms(
            model_id, device="cpu", precision="fp32", cache_dir=str(cache_root)
        )
        print("[verify] open_clip load OK.")
    except Exception as e:  # pragma: no cover
        print(f"[verify] open_clip load failed (non-fatal): {e}")


def main() -> int:
    args = parse_args()
    dest = Path(args.dest).expanduser()
    ensure_dir(dest)
    try:
        out = snapshot_repo(args.repo_id, dest)
    except Exception as e:
        print(f"[error] Download failed: {e}", file=sys.stderr)
        return 2

    if args.verify_load:
        optional_verify(dest, args.repo_id)

    print("\n[done] Model repo is available at:")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

