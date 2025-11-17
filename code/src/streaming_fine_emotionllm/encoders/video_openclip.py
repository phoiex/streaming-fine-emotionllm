from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass
class VideoBatch:
    pixel_values: torch.Tensor  # (B*K, C, H, W)
    frames_per_video: List[int]


class VideoEncoderOpenCLIP(nn.Module):
    """OpenCLIP-based image encoder + temporal mean pooling.

    Expects a local directory that contains open_clip weights and
    `open_clip_config.json` (e.g., LAION ConvNeXt-B W). Uses open-clip-torch
    to construct the model and load weights.
    """

    def __init__(self, model_dir: str):
        super().__init__()
        try:
            import open_clip  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "open-clip-torch is required for OpenCLIP weights. Please `pip install open-clip-torch`."
            ) from e

        cfg_path = os.path.join(model_dir, "open_clip_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"open_clip_config.json not found in {model_dir}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Infer model name from directory (e.g., CLIP-convnext_base_w-... -> convnext_base_w)
        base = os.path.basename(model_dir)
        model_name = None
        if base.startswith("CLIP-") and "-laion" in base:
            model_name = base[len("CLIP-") : base.index("-laion")]
        # Fallback to timm model name if necessary
        if not model_name:
            model_name = cfg.get("model_cfg", {}).get("vision_cfg", {}).get("timm_model_name", "convnext_base")

        # Build model + preprocessing
        self._open_clip = open_clip
        self.model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
        self.preprocess_tf = preprocess  # torchvision-like transform

        # Load weights
        w_safetensors = os.path.join(model_dir, "open_clip_model.safetensors")
        w_bin = os.path.join(model_dir, "open_clip_pytorch_model.bin")
        state = None
        if os.path.isfile(w_safetensors):
            try:
                from safetensors.torch import load_file  # type: ignore

                state = load_file(w_safetensors)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load safetensors weights at {w_safetensors}. Install safetensors."
                ) from e
        elif os.path.isfile(w_bin):
            state = torch.load(w_bin, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No OpenCLIP weights found in {model_dir} (expected safetensors or bin)."
            )
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] OpenCLIP missing keys: {len(missing)} (showing first 3): {missing[:3]}")
        if unexpected:
            print(f"[warn] OpenCLIP unexpected keys: {len(unexpected)} (showing first 3): {unexpected[:3]}")

        # Determine output dim from config (embed_dim)
        self.output_dim = int(cfg.get("model_cfg", {}).get("embed_dim", 640))

    @torch.no_grad()
    def preprocess(self, frames: List[List["PIL.Image.Image"]]) -> VideoBatch:
        # Flatten and transform with the OpenCLIP preprocess
        flat = [img for seq in frames for img in seq]
        pp = self.preprocess_tf
        tensors = [pp(img).unsqueeze(0) for img in flat]
        pixel_values = torch.cat(tensors, dim=0)
        frames_per_video = [len(seq) for seq in frames]
        return VideoBatch(pixel_values=pixel_values, frames_per_video=frames_per_video)

    def forward(self, batch: VideoBatch) -> torch.Tensor:
        device = next(self.parameters()).device
        images = batch.pixel_values.to(device)
        # Encode and get normalized image features
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        outs = []
        idx = 0
        for n in batch.frames_per_video:
            if n == 0:
                raise ValueError("frames_per_video contains zero-length entry")
            seg = feats[idx : idx + n]
            outs.append(seg.mean(dim=0))
            idx += n
        return torch.stack(outs, dim=0)

