#!/usr/bin/env python3
"""
Stage I training (video): adapt CLIP image tower for emotion classification.

This script includes a tiny synthetic dataset for a quick smoke test. Replace
`make_synthetic_dataset` with your real dataloader that yields batches with:
  - frames: List[List[PIL.Image.Image]] per sample
  - labels: LongTensor (B,)

Run (OpenCLIP local):
  python code/scripts/train_stage1_video.py --epochs 1 --batch-size 2 \
    --clip-path resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg --clip-backend open_clip

Run (HF CLIP):
  python code/scripts/train_stage1_video.py --epochs 1 --batch-size 2 \
    --clip-path openai/clip-vit-base-patch32 --clip-backend hf
"""

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Mapping

import torch
from PIL import Image

# Add code/src to sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from streaming_fine_emotionllm.core.utils import TrainArgs, seed_all  # noqa: E402
from streaming_fine_emotionllm.encoders.video_clip import VideoEncoderCLIP  # noqa: E402
from streaming_fine_emotionllm.train.stage1_video import train_stage1_video  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-path", default="resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg")
    p.add_argument("--clip-backend", choices=["auto", "hf", "open_clip"], default="auto")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-interval", type=int, default=5)
    return p.parse_args()


def make_synthetic_dataset(batch_size: int, steps: int, num_classes: int, frames_per_sample: int = 8) -> Iterable[Mapping]:
    W = H = 224
    for _ in range(steps):
        frames: List[List[Image.Image]] = []
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        for _b in range(batch_size):
            seq = []
            for _k in range(frames_per_sample):
                # Solid-color dummy frames
                color = tuple(int(x) for x in torch.randint(0, 255, (3,)).tolist())
                img = Image.new("RGB", (W, H), color=color)
                seq.append(img)
            frames.append(seq)
        yield {"frames": frames, "labels": labels}


def main():
    args = parse_args()
    seed_all(args.seed)
    # Choose encoder backend
    if args.clip_backend == "open_clip":
        from streaming_fine_emotionllm.encoders.video_openclip import VideoEncoderOpenCLIP  # noqa: E402

        enc = VideoEncoderOpenCLIP(args.clip_path)
    elif args.clip_backend == "hf":
        from streaming_fine_emotionllm.encoders.video_clip import VideoEncoderCLIP  # noqa: E402

        enc = VideoEncoderCLIP(args.clip_path)
    else:  # auto
        import os
        from streaming_fine_emotionllm.encoders.video_openclip import VideoEncoderOpenCLIP  # noqa: E402
        from streaming_fine_emotionllm.encoders.video_clip import VideoEncoderCLIP  # noqa: E402

        if os.path.isdir(args.clip_path) and os.path.isfile(os.path.join(args.clip_path, "open_clip_config.json")):
            enc = VideoEncoderOpenCLIP(args.clip_path)
        else:
            enc = VideoEncoderCLIP(args.clip_path)

    train_iter = make_synthetic_dataset(args.batch_size, steps=20, num_classes=args.num_classes)
    eval_iter = make_synthetic_dataset(args.batch_size, steps=5, num_classes=args.num_classes)

    trainer_args = TrainArgs(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_interval=args.log_interval,
    )
    _ = train_stage1_video(enc, train_iter, eval_iter, args.num_classes, trainer_args)


if __name__ == "__main__":
    main()
