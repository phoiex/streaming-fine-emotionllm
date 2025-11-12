#!/usr/bin/env python3
"""
Quick demo to run the OpenCLIP ConvNeXt-B (laion2B-s13B-b82K-augreg) model.

- Loads from local HF cache dir (resources/models) if available.
- Encodes an image and a list of texts, then prints cosine similarities.

Dependencies: open_clip_torch, pillow, torch
  pip install open_clip_torch pillow torch

Examples (from repo root):
  # Use local cache under resources/models and CPU
  python code/scripts/run_openclip_demo.py \
    --cache-dir /home/xiaoqibpnm/streaming-fine-emotionllm/resources/models \
    --image /path/to/your/image.jpg \
    --text "a photo of a cat" --text "a photo of a dog"

  # No image provided -> uses a synthetic RGB test image
  python code/scripts/run_openclip_demo.py --cache-dir resources/models
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch


DEFAULT_HF_ID = "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenCLIP demo (image-text sim)")
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--hf-id", default=DEFAULT_HF_ID, help="OpenCLIP model id")
    p.add_argument(
        "--cache-dir",
        default=str(root / "resources/models"),
        help="Local cache dir for HF weights",
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Weight precision",
    )
    p.add_argument("--image", default=None, help="Path to an input image")
    p.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text prompt(s); can be repeated",
    )
    return p.parse_args()


def load_image(image_path: Path):
    from PIL import Image, ImageDraw, ImageFont  # lazy import

    img = Image.open(image_path).convert("RGB")
    return img


def make_synthetic_image(size=(224, 224)):
    from PIL import Image, ImageDraw  # lazy import

    w, h = size
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # simple RGB blocks
    draw.rectangle((0, 0, w // 2, h // 2), fill=(255, 0, 0))
    draw.rectangle((w // 2, 0, w, h // 2), fill=(0, 255, 0))
    draw.rectangle((0, h // 2, w // 2, h), fill=(0, 0, 255))
    draw.rectangle((w // 2, h // 2, w, h), outline=(0, 0, 0), width=4)
    return img


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main() -> int:
    args = parse_args()

    try:
        import open_clip  # type: ignore
    except Exception as e:
        print("[error] open_clip_torch not installed. pip install open_clip_torch")
        return 2

    device = args.device
    cache_dir = str(Path(args.cache_dir).expanduser())

    print(f"[load] Creating model {args.hf_id} (device={device}, precision={args.precision})")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.hf_id, device=device, precision=args.precision, cache_dir=cache_dir
    )
    tokenizer = open_clip.get_tokenizer(args.hf_id)
    model.eval()

    # Prepare image
    if args.image:
        img_path = Path(args.image).expanduser()
        if not img_path.exists():
            print(f"[error] Image not found: {img_path}")
            return 3
        img = load_image(img_path)
    else:
        print("[info] No --image given; using a synthetic RGB test image.")
        img = make_synthetic_image()

    image_input = preprocess_val(img).unsqueeze(0)
    image_input = image_input.to(device)

    # Prepare texts
    texts: List[str]
    if args.text:
        texts = args.text
    else:
        texts = [
            "a photo of a cat",
            "a photo of a dog",
            "a colorful abstract image",
        ]

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = l2norm(image_features)

        text_tokens = tokenizer(texts)
        text_tokens = text_tokens.to(device)
        text_features = model.encode_text(text_tokens)
        text_features = l2norm(text_features)

        sims = (image_features @ text_features.t()).squeeze(0)

    # Report
    print("\n[results] Cosine similarities:")
    for t, s in sorted(zip(texts, sims.tolist()), key=lambda x: x[1], reverse=True):
        print(f"  {s:+.4f}  |  {t}")

    best_idx = int(torch.argmax(sims).item())
    print(f"\n[top-1] → '{texts[best_idx]}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

