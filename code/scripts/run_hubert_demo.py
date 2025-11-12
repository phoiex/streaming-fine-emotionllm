#!/usr/bin/env python3
"""
Run a quick demo with HuBERT (utter-project/mHuBERT-147 or a local folder).

It builds the HubertAudioEncoder, generates a synthetic 1s sine wave at 16kHz
if no WAV is provided, encodes to a pooled embedding, and prints its stats.

Dependencies: transformers, torch

Examples (from repo root):
  # Use local saved model folder
  python code/scripts/run_hubert_demo.py --source resources/models/hubert-m --device cpu

  # Use HF id (requires network)
  python code/scripts/run_hubert_demo.py --source utter-project/mHuBERT-147 --device cpu

  # Provide your own wav file (16kHz mono)
  python code/scripts/run_hubert_demo.py --source resources/models/hubert-m --wav /path/to/audio.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="HuBERT encoder demo")
    p.add_argument(
        "--source",
        default=str(root / "resources/models/hubert-m"),
        help="HF repo id or local directory",
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--wav", default=None, help="Path to a 16kHz mono wav file")
    p.add_argument("--normalize", action="store_true", help="L2 normalize output")
    return p.parse_args()


def load_wav(path: Path):
    try:
        import soundfile as sf  # type: ignore
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return torch.from_numpy(wav), sr
    except Exception:
        raise RuntimeError(
            "Failed to read wav. Install soundfile: `pip install soundfile` or omit --wav to use a synthetic tone."
        )


def make_sine(duration_s=1.0, freq=440.0, sr=16000):
    t = torch.arange(int(duration_s * sr), dtype=torch.float32) / sr
    wav = torch.sin(2 * torch.pi * freq * t)
    return wav, sr


def main() -> int:
    from streaming_fine_emotionllm.models import build_hubert_audio_encoder

    args = parse_args()
    device = args.device

    print(f"[load] Building HuBERT encoder from: {args.source}")
    encoder, feature_extractor = build_hubert_audio_encoder(
        source=args.source, device=device, freeze_backbone=True, normalize_features=args.normalize
    )
    sr = getattr(feature_extractor, "sampling_rate", 16000)
    print(f"[info] Expected sampling rate: {sr}")

    if args.wav:
        wav, file_sr = load_wav(Path(args.wav))
        if file_sr != sr:
            print(f"[warn] WAV sr={file_sr} != expected sr={sr}. Continuing anyway.")
    else:
        print("[info] No --wav provided; generating a 1s 440Hz sine tone.")
        wav, _ = make_sine(duration_s=1.0, freq=440.0, sr=sr)

    with torch.no_grad():
        emb = encoder.encode_waveforms([wav], sampling_rate=sr, device=device)  # (1, D)

    print("[results] Embedding shape:", tuple(emb.shape))
    norms = emb.norm(dim=-1).cpu().tolist()
    print("[results] L2 norm:", norms[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

