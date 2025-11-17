#!/usr/bin/env python3
"""
Stage II training: multimodal fusion with LoRA-adapted LLaMA-2.

This script includes a tiny synthetic dataset for a quick smoke test. Replace
`make_synthetic_dataset` with your real dataloader that yields batches with:
  - frames: List[List[PIL.Image.Image]] per sample
  - waveforms: List[Tensor(T,)] per sample, sampled at --sr
  - asr_texts: List[str]
  - labels: LongTensor (B,)

Run (OpenCLIP local + HuBERT-m local):
  python code/scripts/train_stage2_fusion.py \
    --llama-path resources/models/Llama-2-7b-chat-hf \
    --clip-path resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg --clip-backend open_clip \
    --hubert-path resources/models/hubert-m \
    --epochs 1 --batch-size 1

Notes:
- Requires `peft` for LoRA (pip install peft)
- Requires `open-clip-torch` for OpenCLIP backbones (pip install open-clip-torch torchvision)
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
from streaming_fine_emotionllm.fusion.model import FusionConfig, FusionEmotionModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-path", default="resources/models/Llama-2-7b-chat-hf")
    p.add_argument("--clip-path", default="resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg")
    p.add_argument("--clip-backend", choices=["auto", "hf", "open_clip"], default="auto")
    p.add_argument("--hubert-path", default="resources/models/hubert-m")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--sr", type=int, default=16000, help="audio sampling rate")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--freeze-clip", action="store_true", default=False)
    p.add_argument("--freeze-hubert", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


def make_synthetic_dataset(batch_size: int, steps: int, num_classes: int, sr: int) -> Iterable[Mapping]:
    W = H = 224
    for _ in range(steps):
        frames: List[List[Image.Image]] = []
        waves: List[torch.Tensor] = []
        texts: List[str] = []
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        for _b in range(batch_size):
            # Frames
            seq = []
            for _k in range(4):
                color = tuple(int(x) for x in torch.randint(0, 255, (3,)).tolist())
                img = Image.new("RGB", (W, H), color=color)
                seq.append(img)
            frames.append(seq)
            # Audio
            L = int(sr * 0.5)
            waves.append(torch.randn(L))
            # ASR text
            texts.append("Hello, this is a synthetic utterance for testing.")
        yield {"frames": frames, "waveforms": waves, "asr_texts": texts, "labels": labels}


def main():
    args = parse_args()
    seed_all(args.seed)

    cfg = FusionConfig(
        llama_path=args.llama_path,
        num_classes=args.num_classes,
        clip_path=args.clip_path,
        clip_backend=args.clip_backend,
        hubert_path=args.hubert_path,
        freeze_clip=args.freeze_clip,
        freeze_hubert=args.freeze_hubert,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = FusionEmotionModel(cfg)

    from streaming_fine_emotionllm.train.stage2_fusion import train_stage2_fusion

    train_iter = make_synthetic_dataset(args.batch_size, steps=4, num_classes=args.num_classes, sr=args.sr)
    eval_iter = make_synthetic_dataset(args.batch_size, steps=2, num_classes=args.num_classes, sr=args.sr)

    trainer_args = TrainArgs(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_interval=args.log_interval,
    )
    train_stage2_fusion(model, train_iter, eval_iter, trainer_args, sampling_rate=args.sr)


if __name__ == "__main__":
    main()
