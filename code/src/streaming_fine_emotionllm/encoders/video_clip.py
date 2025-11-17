from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor


@dataclass
class VideoBatch:
    pixel_values: torch.Tensor  # (B*K, C, H, W)
    frames_per_video: List[int]  # len=B, sum(frames_per_video)=K*B


class VideoEncoderCLIP(nn.Module):
    """Frame-level CLIP image encoder + temporal average pooling.

    Uses CLIP.get_image_features to obtain frame embeddings, then applies
    average pooling across frames per video to produce a single vector.
    """

    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        # Dimension of image features depends on model; infer at runtime.
        self.output_dim = self.clip.config.projection_dim

    @torch.no_grad()
    def preprocess(self, frames: List[List["PIL.Image.Image"]]) -> VideoBatch:
        # Flatten frames and keep per-video counts
        flat = [img for seq in frames for img in seq]
        enc = self.processor(images=flat, return_tensors="pt")
        frames_per_video = [len(seq) for seq in frames]
        return VideoBatch(pixel_values=enc["pixel_values"], frames_per_video=frames_per_video)

    def forward(self, batch: VideoBatch) -> torch.Tensor:
        device = next(self.parameters()).device
        pixel_values = batch.pixel_values.to(device)
        # (B*K, D)
        feats = self.clip.get_image_features(pixel_values=pixel_values)
        # Normalize as CLIP does for retrieval; optional for classification
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # Temporal mean pooling per video
        outs = []
        idx = 0
        for n in batch.frames_per_video:
            if n == 0:
                raise ValueError("frames_per_video contains zero-length entry")
            seg = feats[idx : idx + n]
            outs.append(seg.mean(dim=0))
            idx += n
        return torch.stack(outs, dim=0)  # (B, D)

