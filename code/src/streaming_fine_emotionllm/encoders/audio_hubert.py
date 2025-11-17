from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor


@dataclass
class AudioBatch:
    input_values: torch.Tensor  # (B, T)
    attention_mask: Optional[torch.Tensor]  # (B, T) or None


class AudioEncoderHuBERT(nn.Module):
    """HuBERT/mHuBERT encoder with mean pooling over time.

    Wraps a HF speech model (e.g., facebook/mhubert-large-ll60k) and processor,
    returning an utterance-level embedding per sample by mean-pooling the last
    hidden states.
    """

    def __init__(self, model_name_or_path: str = "facebook/mhubert-large-ll60k"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.output_dim = self.model.config.hidden_size

    @torch.no_grad()
    def preprocess(self, waveforms: List[torch.Tensor], sampling_rate: int) -> AudioBatch:
        # waveforms: list of (T,) tensors in float32/float64
        inputs = self.processor(
            waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.get("input_values")
        attn = inputs.get("attention_mask")
        return AudioBatch(input_values=input_values, attention_mask=attn)

    def forward(self, batch: AudioBatch) -> torch.Tensor:
        device = next(self.parameters()).device
        kwargs = {k: v.to(device) for k, v in batch.__dict__.items() if v is not None}
        out = self.model(**kwargs)
        hidden = out.last_hidden_state  # (B, T, D)
        if batch.attention_mask is not None:
            mask = batch.attention_mask.to(hidden.dtype).to(device)  # (B, T)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = hidden.mean(dim=1)
        return pooled  # (B, D)

