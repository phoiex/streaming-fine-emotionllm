from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import torch
from torch import nn, optim

from ..core.metrics import macro_f1
from ..core.utils import TrainArgs
from ..encoders.audio_hubert import AudioBatch
from ..encoders.video_clip import VideoBatch
from ..fusion.model import FusionEmotionModel


def _collect_trainable_params(model: FusionEmotionModel):
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
    return params


def train_stage2_fusion(
    model: FusionEmotionModel,
    train_data: Iterable[Mapping],
    eval_data: Optional[Iterable[Mapping]],
    args: TrainArgs,
    sampling_rate: int,
):
    """Stage II: Multimodal fusion with LoRA-adapted LLaMA-2.

    Expects each batch dict to contain:
      - "frames": List[List[PIL.Image.Image]] per sample
      - "waveforms": List[torch.Tensor] per sample (each of shape (T,))
      - "asr_texts": List[str]
      - "labels": 1D LongTensor (B,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = _collect_trainable_params(model)
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_data:
            frames = batch["frames"]
            waves = batch["waveforms"]
            asr_texts: List[str] = batch["asr_texts"]
            labels = batch["labels"].to(device)

            vbatch: VideoBatch = model.video_enc.preprocess(frames)
            abatch: AudioBatch = model.audio_enc.preprocess(waves, sampling_rate)

            out = model(
                asr_texts=asr_texts,
                video=vbatch,
                audio=abatch,
                labels=labels,
            )
            loss = out["loss"]
            assert loss is not None
            loss.backward()
            if (global_step + 1) % args.grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            if args.log_interval and (global_step % args.log_interval == 0):
                with torch.no_grad():
                    preds = out["logits"].argmax(dim=-1)
                    f1 = macro_f1(preds, labels, model.cfg.num_classes)
                print(f"[fusion][epoch {epoch}] step {global_step} loss={loss.item():.4f} f1={f1:.3f}")
            global_step += 1

        if eval_data is not None:
            print(f"[fusion] evaluating epoch {epoch}...")
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in eval_data:
                    frames = batch["frames"]
                    waves = batch["waveforms"]
                    asr_texts: List[str] = batch["asr_texts"]
                    labels = batch["labels"].to(device)
                    vbatch: VideoBatch = model.video_enc.preprocess(frames)
                    abatch: AudioBatch = model.audio_enc.preprocess(waves, sampling_rate)
                    out = model(asr_texts=asr_texts, video=vbatch, audio=abatch)
                    preds = out["logits"].argmax(dim=-1)
                    all_preds.append(preds)
                    all_labels.append(labels)
            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)
            f1 = macro_f1(preds_cat, labels_cat, model.cfg.num_classes)
            print(f"[fusion] val macro F1 = {f1:.3f}")

