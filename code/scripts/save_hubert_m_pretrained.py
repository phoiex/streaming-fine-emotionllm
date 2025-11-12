#!/usr/bin/env python3
"""
Load utter-project/mHuBERT-147 via Transformers and save its structure & weights
to a local folder (default: <repo_root>/resources/models/hubert-m).

Notes:
- Some speech models (incl. mHuBERT) expose only a feature extractor, not a tokenizer.
  AutoProcessor may try to load a tokenizer and fail. We thus prefer AutoFeatureExtractor.

Usage (from repo root):
  # Save to default path under the repo
  python code/scripts/save_hubert_m_pretrained.py

  # Save explicitly to the requested home path
  python code/scripts/save_hubert_m_pretrained.py \
    --dest ~/streaming-fine-emotionllm/resources/models/hubert-m

Dependencies: transformers
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    from pathlib import Path

    p = argparse.ArgumentParser(description="Save mHuBERT-147 to local folder")
    root = Path(__file__).resolve().parents[2]
    default_dest = str(root / "resources/models/hubert-m")
    p.add_argument(
        "--repo-id",
        default="utter-project/mHuBERT-147",
        help="Hugging Face repo id",
    )
    p.add_argument(
        "--dest",
        default=default_dest,
        help="Destination directory to save model & processor",
    )
    p.add_argument(
        "--use-safetensors",
        action="store_true",
        help="Prefer saving model weights as safetensors when supported",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load from local cache only (no network)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[load] from_pretrained repo: {args.repo_id}")
    print(f"[save] destination: {dest}")

    try:
        from transformers import AutoModel
        try:
            # Prefer feature extractor for HuBERT-style acoustic models
            from transformers import AutoFeatureExtractor  # type: ignore
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                args.repo_id, local_files_only=args.local_files_only
            )
            fe_class = feature_extractor.__class__.__name__
            print(f"[info] Loaded feature extractor: {fe_class}")
        except Exception as e_fe:
            # Fallback: some repos use AutoProcessor but may still work
            print(
                f"[warn] AutoFeatureExtractor failed ({e_fe}). Trying AutoProcessor as fallback."
            )
            from transformers import AutoProcessor  # type: ignore

            feature_extractor = AutoProcessor.from_pretrained(
                args.repo_id, local_files_only=args.local_files_only
            )
            fe_class = feature_extractor.__class__.__name__
            print(f"[info] Loaded processor: {fe_class}")

        model = AutoModel.from_pretrained(
            args.repo_id, local_files_only=args.local_files_only
        )
        mdl_class = model.__class__.__name__
        print(f"[info] Loaded model: {mdl_class}")
    except Exception:
        print("[error] transformers not installed or load failed. Try: pip install -U transformers")
        raise

    # Save feature extractor / processor config
    feature_extractor.save_pretrained(str(dest))

    # Save model weights and config
    if args.use_safetensors:
        try:
            model.save_pretrained(str(dest), safe_serialization=True)
        except TypeError:
            # Older transformers may not support safe_serialization kwarg
            model.save_pretrained(str(dest))
    else:
        model.save_pretrained(str(dest))

    print("[done] Saved model & feature_extractor/processor to:")
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
