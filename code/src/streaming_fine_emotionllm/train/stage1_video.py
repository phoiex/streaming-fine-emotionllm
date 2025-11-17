from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import torch
from torch import nn, optim

from ..core.metrics import macro_f1
from ..core.utils import TrainArgs
from ..encoders.video_clip import VideoEncoderCLIP


def train_stage1_video(
    encoder: VideoEncoderCLIP,
    train_data: Iterable[Mapping],
    eval_data: Optional[Iterable[Mapping]],
    num_classes: int,
    args: TrainArgs,
) -> nn.Module:
    """Stage I: Adapt CLIP image tower for emotion classification.

    Expects each batch dict to contain:
      - "frames": List[List[PIL.Image.Image]] per sample
      - "labels": 1D LongTensor (B,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    head = nn.Linear(encoder.output_dim, num_classes).to(device)
    params = list(p for p in encoder.parameters() if p.requires_grad) + list(head.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        for batch in train_data:
            frames: List[List] = batch["frames"]
            labels = batch["labels"].to(device)
            vbatch = encoder.preprocess(frames)
            feats = encoder(vbatch)
            logits = head(feats)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            if (global_step + 1) % args.grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()
            if args.log_interval and (global_step % args.log_interval == 0):
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    f1 = macro_f1(preds, labels, num_classes)
                print(f"[video][epoch {epoch}] step {global_step} loss={loss.item():.4f} f1={f1:.3f}")
            global_step += 1

        if eval_data is not None:
            print(f"[video] evaluating epoch {epoch}...")
            encoder.eval()
            head.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in eval_data:
                    frames: List[List] = batch["frames"]
                    labels = batch["labels"].to(device)
                    vbatch = encoder.preprocess(frames)
                    feats = encoder(vbatch)
                    logits = head(feats)
                    preds = logits.argmax(dim=-1)
                    all_preds.append(preds)
                    all_labels.append(labels)
            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)
            f1 = macro_f1(preds_cat, labels_cat, num_classes)
            print(f"[video] val macro F1 = {f1:.3f}")

    return head

