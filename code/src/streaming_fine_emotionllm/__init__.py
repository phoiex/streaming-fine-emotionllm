"""Streaming-Fine-EmotionLLM

Modules for unimodal encoders (video/audio), multimodal fusion with a
LoRA-adapted LLaMA backbone, and training utilities for the two-stage
procedure described in the paper method section.

Package layout:
- encoders/: wrappers for CLIP image encoder and (m)HuBERT audio encoder
- fusion/: prompt assembly and fusion model (with LoRA + classifier)
- train/: trainers for Stage I (unimodal) and Stage II (fusion)
- core/: small helpers (dtype mapping, metrics, seeding)
"""

__all__ = [
    "encoders",
    "fusion",
    "train",
    "core",
]

