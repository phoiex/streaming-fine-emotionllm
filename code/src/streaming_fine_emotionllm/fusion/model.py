from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..core.utils import to_dtype

from ..encoders.video_clip import VideoBatch, VideoEncoderCLIP
from ..encoders.video_openclip import VideoEncoderOpenCLIP
from ..encoders.audio_hubert import AudioBatch, AudioEncoderHuBERT
from .prompt import PromptPieces, assemble_inputs_with_pseudotokens, ensure_special_tokens


def _maybe_import_peft():
    try:
        from peft import LoraConfig, get_peft_model

        return LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "peft is required for LoRA. Please install it, e.g., `pip install peft`."
        ) from e


@dataclass
class FusionConfig:
    llama_path: str
    num_classes: int
    clip_path: str = "resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"
    hubert_path: str = "resources/models/hubert-m"
    clip_backend: str = "auto"  # auto|hf|open_clip
    device_map: str = "auto"
    dtype: str = "auto"  # used at load time by caller
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    freeze_clip: bool = True
    freeze_hubert: bool = True


class FusionEmotionModel(nn.Module):
    """Multimodal fusion model with LoRA-adapted LLaMA-2 and classifier on <EMO>.

    - Encodes video frames via CLIP image encoder + mean pooling
    - Encodes audio via (m)HuBERT + mean pooling
    - Projects both into LM embedding space and inserts as pseudo tokens
    - Fine-tunes LoRA modules and the small heads
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        # Tokenizer + LM
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llama_path)
        # Respect dtype if provided
        if cfg.dtype == "auto":
            self.lm = AutoModelForCausalLM.from_pretrained(cfg.llama_path, device_map=cfg.device_map)
        else:
            try:
                self.lm = AutoModelForCausalLM.from_pretrained(
                    cfg.llama_path, device_map=cfg.device_map, dtype=to_dtype(cfg.dtype)
                )
            except TypeError:
                self.lm = AutoModelForCausalLM.from_pretrained(
                    cfg.llama_path, device_map=cfg.device_map, torch_dtype=to_dtype(cfg.dtype)
                )
        ensure_special_tokens(self.tokenizer, self.lm)

        # Encoders
        use_openclip = False
        if cfg.clip_backend == "open_clip":
            use_openclip = True
        elif cfg.clip_backend == "auto":
            if os.path.isdir(cfg.clip_path) and os.path.isfile(os.path.join(cfg.clip_path, "open_clip_config.json")):
                use_openclip = True
        if use_openclip:
            self.video_enc = VideoEncoderOpenCLIP(cfg.clip_path)
        else:
            self.video_enc = VideoEncoderCLIP(cfg.clip_path)
        self.audio_enc = AudioEncoderHuBERT(cfg.hubert_path)

        if cfg.freeze_clip:
            for p in self.video_enc.parameters():
                p.requires_grad_(False)
        if cfg.freeze_hubert:
            for p in self.audio_enc.parameters():
                p.requires_grad_(False)

        hidden = self.lm.get_input_embeddings().weight.shape[1]
        self.proj_v = nn.Linear(self.video_enc.output_dim, hidden)
        self.proj_a = nn.Linear(self.audio_enc.output_dim, hidden)
        self.classifier = nn.Linear(hidden, cfg.num_classes)

        # LoRA setup
        target_modules = cfg.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        LoraConfig, get_peft_model = _maybe_import_peft()
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        # Replace lm with a peft-wrapped version
        self.lm = get_peft_model(self.lm, lora_cfg)

    def encode_modalities(
        self, video: Optional[VideoBatch], audio: Optional[AudioBatch]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        h_v = h_a = None
        if video is not None:
            v = self.video_enc(video)
            h_v = self.proj_v(v)
        if audio is not None:
            a = self.audio_enc(audio)
            h_a = self.proj_a(a)
        return h_v, h_a

    def forward(
        self,
        *,
        asr_texts: List[str],
        video: Optional[VideoBatch],
        audio: Optional[AudioBatch],
        labels: Optional[torch.Tensor] = None,
        prompt_pieces: PromptPieces = PromptPieces(),
    ):
        # Encode + project modalities
        h_v, h_a = self.encode_modalities(video, audio)

        if h_v is None or h_a is None:
            raise ValueError("Both video and audio embeddings are required for fusion.")

        inputs_embeds, attention_mask, emo_pos = assemble_inputs_with_pseudotokens(
            self.tokenizer, self.lm, h_v, h_a, asr_texts, prompt_pieces
        )

        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=False)
        hidden = out.last_hidden_state  # (B, L, H)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        emo_hidden = hidden[batch_indices, emo_pos, :]  # (B, H)
        logits = self.classifier(emo_hidden)  # (B, C)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
