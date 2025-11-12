"""HuBERT audio encoder wrapper and builders.

Provides utilities to construct a HuBERT-based audio encoder using
Hugging Face Transformers (e.g., utter-project/mHuBERT-147) and expose a
simple forward interface that pools time frames into a single embedding.

Example:
    from streaming_fine_emotionllm.models import build_hubert_audio_encoder

    encoder, feature_extractor = build_hubert_audio_encoder(
        source="/path/to/resources/models/hubert-m",  # or HF id
        device="cpu",
    )

    # Raw waveform (List[np.ndarray] or List[Tensor]), 16kHz
    wavs = [torch.randn(16000)]
    emb = encoder.encode_waveforms(wavs, sampling_rate=16000)  # (1, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


DEFAULT_HUBERT_ID = "utter-project/mHuBERT-147"


def _import_transformers():
    try:
        from transformers import AutoModel, AutoFeatureExtractor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Transformers is required. Install with: `pip install transformers`"
        ) from e
    return AutoModel, AutoFeatureExtractor


@dataclass
class HuBERTComponents:
    model: nn.Module
    feature_extractor: Any
    sampling_rate: int


def _maybe_freeze(module: nn.Module, freeze: bool) -> None:
    if freeze:
        for p in module.parameters():
            p.requires_grad = False


def create_hubert_components(
    source: str = DEFAULT_HUBERT_ID,
    *,
    device: Optional[torch.device | str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    freeze: bool = True,
) -> HuBERTComponents:
    """Create HuBERT model and feature extractor from HF id or local dir.

    Args:
        source: HF repo id (e.g., "utter-project/mHuBERT-147") or local path.
        device: Device for model weights (e.g., "cpu", "cuda").
        cache_dir: Optional HF cache directory.
        local_files_only: If True, do not attempt to fetch from network.
        freeze: If True, disable gradients on the model.

    Returns:
        HuBERTComponents(model, feature_extractor, sampling_rate)
    """

    AutoModel, AutoFeatureExtractor = _import_transformers()

    fe = AutoFeatureExtractor.from_pretrained(
        source, cache_dir=cache_dir, local_files_only=local_files_only
    )
    model = AutoModel.from_pretrained(
        source, cache_dir=cache_dir, local_files_only=local_files_only
    )
    if device is not None:
        model = model.to(device)
    _maybe_freeze(model, freeze=freeze)
    sr = getattr(fe, "sampling_rate", 16000)
    return HuBERTComponents(model=model, feature_extractor=fe, sampling_rate=sr)


class HubertAudioEncoder(nn.Module):
    """Wrap a HuBERT model to produce pooled embeddings.

    The forward method expects precomputed input_values/attention_mask
    (as produced by the feature extractor). For convenience, `encode_waveforms`
    handles raw waveforms and calls the feature extractor internally.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Any,
        *,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.pooling = pooling
        self.normalize = normalize

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        # hidden_states: (B, T, D)
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported pooling='{self.pooling}'")
        if self.normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return pooled

    @torch.no_grad()
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, D)
        return self._pool(hidden_states, attention_mask)

    @torch.no_grad()
    def encode_waveforms(
        self,
        waveforms: Iterable[Any],
        *,
        sampling_rate: Optional[int] = None,
        padding: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """Encode raw waveforms to pooled embeddings.

        Args:
            waveforms: Iterable of 1-D arrays/tensors.
            sampling_rate: If None, uses feature_extractor.sampling_rate.
            padding: Whether to pad to longest.
            device: Device to run on (defaults to model's device).
        Returns:
            Tensor of shape (B, D)
        """
        sr = sampling_rate or getattr(self.feature_extractor, "sampling_rate", 16000)
        feats = self.feature_extractor(
            list(waveforms),
            sampling_rate=sr,
            padding=padding,
            return_tensors="pt",
        )
        input_values = feats["input_values"]
        attention_mask = feats.get("attention_mask")
        if device is None:
            device = next(self.model.parameters()).device
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        return self.forward(input_values=input_values, attention_mask=attention_mask)


def build_hubert_audio_encoder(
    source: str = DEFAULT_HUBERT_ID,
    *,
    device: Optional[torch.device | str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    freeze_backbone: bool = True,
    pooling: str = "mean",
    normalize_features: bool = True,
) -> Tuple[HubertAudioEncoder, Any]:
    """High-level builder that returns (encoder, feature_extractor)."""
    comps = create_hubert_components(
        source=source,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        freeze=freeze_backbone,
    )
    enc = HubertAudioEncoder(
        model=comps.model,
        feature_extractor=comps.feature_extractor,
        pooling=pooling,
        normalize=normalize_features,
    )
    return enc, comps.feature_extractor

