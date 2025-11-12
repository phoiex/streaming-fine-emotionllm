"""OpenCLIP visual encoder wrapper.

This module provides a thin wrapper around `open_clip` to construct a visual
encoder, associated preprocessing transforms, and tokenizer. It follows the
project's library layout under `code/src/streaming_fine_emotionllm/`.

Usage (example):
    from streaming_fine_emotionllm.models import build_open_clip_visual_encoder

    encoder, preproc_train, preproc_val, tokenizer = build_open_clip_visual_encoder()

    # Preprocess a PIL image with `preproc_val`, then:
    # imgs = preproc_val(pil_image).unsqueeze(0)  # (1, C, H, W)
    # feats = encoder(imgs)  # (1, D)

Note: This code expects `open_clip_torch` to be installed.
      pip install open_clip_torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


# Default to the HF Hub identifier requested by the user.
DEFAULT_OPEN_CLIP_ID = (
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
)


def _import_open_clip():
    try:
        import open_clip  # type: ignore

    except Exception as e:  # pragma: no cover - import-time path
        raise RuntimeError(
            "open_clip_torch is required. Install with: `pip install open_clip_torch`"
        ) from e
    return open_clip


@dataclass
class OpenCLIPBundle:
    model: nn.Module
    preprocess_train: Any
    preprocess_val: Any
    tokenizer: Any


class OpenCLIPVisualEncoder(nn.Module):
    """A thin nn.Module that wraps `open_clip` image encoding.

    It exposes a standard forward(images) -> features interface and applies L2
    normalization by default (common for CLIP-style features).
    """

    def __init__(self, clip_model: nn.Module, normalize: bool = True) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.normalize = normalize

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        # open_clip expects preprocessed tensor on the correct device
        feats = self.clip_model.encode_image(pixel_values)
        if self.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _maybe_freeze(module: nn.Module, freeze: bool) -> None:
    if freeze:
        for p in module.parameters():
            p.requires_grad = False


def create_open_clip_components(
    model_id: str = DEFAULT_OPEN_CLIP_ID,
    *,
    device: Optional[torch.device | str] = None,
    precision: str = "fp32",
    cache_dir: Optional[str] = None,
    freeze: bool = True,
) -> OpenCLIPBundle:
    """Create OpenCLIP model, preprocess transforms, and tokenizer.

    Args:
        model_id: Model identifier; supports `hf-hub:` URIs.
        device: Device for model parameters (e.g., "cpu", "cuda"). If None, let
                open_clip decide.
        precision: One of {"fp32", "fp16", "bf16"}; passed to open_clip.
        cache_dir: Optional cache directory for weights.
        freeze: If True, disables gradients for the entire model.

    Returns:
        OpenCLIPBundle(model, preprocess_train, preprocess_val, tokenizer)
    """

    open_clip = _import_open_clip()

    # open_clip supports passing a single `model_id` that points to hf-hub
    # artifact; it will derive the model arch accordingly.
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_id, device=device, precision=precision, cache_dir=cache_dir
    )
    tokenizer = open_clip.get_tokenizer(model_id)

    _maybe_freeze(model, freeze=freeze)
    return OpenCLIPBundle(
        model=model,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        tokenizer=tokenizer,
    )


def build_open_clip_visual_encoder(
    model_id: str = DEFAULT_OPEN_CLIP_ID,
    *,
    device: Optional[torch.device | str] = None,
    precision: str = "fp32",
    cache_dir: Optional[str] = None,
    freeze_backbone: bool = True,
    normalize_features: bool = True,
) -> Tuple[OpenCLIPVisualEncoder, Any, Any, Any]:
    """High-level builder that returns an encoder module and assets.

    Returns:
        (encoder, preprocess_train, preprocess_val, tokenizer)
    """

    bundle = create_open_clip_components(
        model_id=model_id,
        device=device,
        precision=precision,
        cache_dir=cache_dir,
        freeze=freeze_backbone,
    )
    encoder = OpenCLIPVisualEncoder(bundle.model, normalize=normalize_features)
    return encoder, bundle.preprocess_train, bundle.preprocess_val, bundle.tokenizer
