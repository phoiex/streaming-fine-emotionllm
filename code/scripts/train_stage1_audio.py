#!/usr/bin/env python3
"""
Stage I training (audio): adapt (m)HuBERT for emotion classification.

This script includes a tiny synthetic dataset for a quick smoke test. Replace
`make_synthetic_dataset` with your real dataloader that yields batches with:
  - waveforms: List[Tensor(T,)] per sample, sampled at --sr
  - labels: LongTensor (B,)

Run (Local mHuBERT):
  python code/scripts/train_stage1_audio.py --epochs 1 --batch-size 2 \
    --hubert-path resources/models/hubert-m
"""

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Mapping

import torch

# Add code/src to sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from streaming_fine_emotionllm.core.utils import TrainArgs, seed_all  # noqa: E402
from streaming_fine_emotionllm.encoders.audio_hubert import AudioEncoderHuBERT  # noqa: E402
from streaming_fine_emotionllm.train.stage1_audio import train_stage1_audio  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hubert-path", default="resources/models/hubert-m")
    p.add_argument("--sr", type=int, default=16000, help="sampling rate")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-interval", type=int, default=5)
    return p.parse_args()


def make_synthetic_dataset(batch_size: int, steps: int, num_classes: int, sr: int) -> Iterable[Mapping]:
    # Create short random waveforms of ~0.5s, 0.75s, 1.0s
    lengths = [int(sr * 0.5), int(sr * 0.75), int(sr * 1.0)]
    for _ in range(steps):
        waves: List[torch.Tensor] = []
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        for _b in range(batch_size):
            L = lengths[torch.randint(0, len(lengths), (1,)).item()]
            # Zero-mean Gaussian noise
            wav = torch.randn(L)
            waves.append(wav)
        yield {"waveforms": waves, "labels": labels}


def main():
    args = parse_args()
    seed_all(args.seed)
    enc = AudioEncoderHuBERT(args.hubert_path)

    train_iter = make_synthetic_dataset(args.batch_size, steps=20, num_classes=args.num_classes, sr=args.sr)
    eval_iter = make_synthetic_dataset(args.batch_size, steps=5, num_classes=args.num_classes, sr=args.sr)

    trainer_args = TrainArgs(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_interval=args.log_interval,
    )
    _ = train_stage1_audio(enc, train_iter, eval_iter, sampling_rate=args.sr, num_classes=args.num_classes, args=trainer_args)


if __name__ == "__main__":
    main()
