from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional

import torch


DTYPE_NAME = Literal["auto", "float16", "bfloat16", "float32"]


def to_dtype(name: DTYPE_NAME):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def seed_all(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainArgs:
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    epochs: int = 1
    grad_accum_steps: int = 1
    log_interval: int = 50
    device_map: str = "auto"
    dtype: DTYPE_NAME = "auto"
    seed: Optional[int] = None

