Streaming‑Fine‑EmotionLLM
=========================

Conversational emotion recognition with a LoRA‑adapted LLaMA‑2 backbone and multimodal fusion of CLIP (video) and multilingual HuBERT (audio). The training pipeline follows a two‑stage strategy:

- Stage I: Unimodal encoder pre‑adaptation for video and audio
- Stage II: Multimodal fusion with instruction‑style prompts and an emotion head on <EMO>


Project Layout
--------------
- `resources/models/Llama-2-7b-chat-hf/` — local model repo (tokenizer, config, weights) for LLaMA‑2‑7B‑Chat.
- `resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/` — local OpenCLIP ConvNeXt‑B W checkpoint (LAION). Requires `open-clip-torch` if used.
- `resources/models/hubert-m/` — local multilingual HuBERT checkpoint (HF format).
- `resources/datasets/` — datasets (e.g., MELD, CH‑SIMS). Keep large/raw data here.
- `code/src/streaming_fine_emotionllm/` — library code
  - `core/` — utilities and metrics
    - `utils.py` — dtype mapping, seeding, training hyperparams
    - `metrics.py` — macro‑F1 implementation (no sklearn dependency)
  - `encoders/` — unimodal encoders and preprocessors
    - `video_clip.py` — CLIP image tower wrapper with frame pooling
    - `audio_hubert.py` — (m)HuBERT wrapper with mean temporal pooling
  - `fusion/` — prompt assembly and fusion model
    - `prompt.py` — special tokens, pseudo‑token insertion into inputs_embeds
    - `model.py` — LoRA‑adapted LLaMA‑2 fusion model + emotion classifier
  - `train/` — training loops
    - `stage1_video.py` — Stage I (video) trainer
    - `stage1_audio.py` — Stage I (audio) trainer
    - `stage2_fusion.py` — Stage II (fusion) trainer
- `code/scripts/` — runnable scripts and quick tests
  - `testrun.py` — quick LLaMA‑2 text generation sanity check
  - `train_stage1_video.py` — Stage I (video) training script
  - `train_stage1_audio.py` — Stage I (audio) training script
  - `train_stage2_fusion.py` — Stage II (fusion) training script


Installation
------------
Environment (POSIX):
- `python3 -m venv .venv && . .venv/bin/activate`
- `pip install -U pip`
- `pip install transformers torch accelerate safetensors peft pytest black isort ruff`
- If using OpenCLIP (LAION ConvNeXt): `pip install open-clip-torch torchvision`

Notes:
- Place local models under `resources/models/` as shown above (respect licensing).
- CLIP options:
  - HF CLIP (e.g., `openai/clip-vit-base-patch32`) works out-of-the-box with Transformers (`--clip-backend hf`).
  - OpenCLIP LAION ConvNeXt (`resources/models/clip/CLIP-convnext_base_w-...`) requires `open-clip-torch` (`--clip-backend open_clip`).
- For CPU‑only reviews, prefer `--device-map cpu` and `--dtype float32` when loading large LLaMA models.


Quick Text Sanity Check
-----------------------
- `python code/scripts/testrun.py --device-map cpu --dtype float32 --prompt "Hello"`


Method Overview → Code Mapping
------------------------------
1) Unimodal Encoder Pre‑adaptation (Stage I)
- Video (CLIP image tower)
  - Extract frame features, average pool → `code/src/streaming_fine_emotionllm/encoders/video_clip.py:39`
  - OpenCLIP alternative (ConvNeXt‑B W) → `code/src/streaming_fine_emotionllm/encoders/video_openclip.py`
  - Train linear head with CE loss; monitor macro‑F1 → `code/src/streaming_fine_emotionllm/train/stage1_video.py`
- Audio (mHuBERT)
  - Pool hidden states over time → `code/src/streaming_fine_emotionllm/encoders/audio_hubert.py:41`
  - Train linear head with CE loss; monitor macro‑F1 → `code/src/streaming_fine_emotionllm/train/stage1_audio.py`

2) Multimodal Fusion with LoRA‑adapted LLaMA‑2 (Stage II)
- Project video/audio features into LM embedding space → `fusion/model.py:82`
- Insert as pseudo‑tokens with special markers `<VIDEO>`, `<AUDIO>`, `<TEXT>`, `<EMO>` → `fusion/prompt.py`
- Classify at `<EMO>` hidden state with CE loss → `fusion/model.py:140`
- Optimize LoRA modules + projection layers + classifier → `train/stage2_fusion.py`


Running Stage I (Synthetic Smoke Test)
-------------------------------------
The scripts include small synthetic generators to verify wiring without datasets. Replace the generators with real DataLoaders later.

- Video pre‑adaptation (CLIP, auto‑detect backend):
  - OpenCLIP local (ConvNeXt‑B W):
    - `python code/scripts/train_stage1_video.py --epochs 1 --batch-size 2 --clip-path resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg --clip-backend open_clip`
  - HF CLIP (e.g., ViT‑B/32):
    - `python code/scripts/train_stage1_video.py --epochs 1 --batch-size 2 --clip-path openai/clip-vit-base-patch32 --clip-backend hf`

- Audio pre‑adaptation (mHuBERT local):
  - `python code/scripts/train_stage1_audio.py --epochs 1 --batch-size 2 --hubert-path resources/models/hubert-m`


Running Stage II (Fusion, Synthetic Smoke Test)
-----------------------------------------------
Requires `peft` for LoRA.

- Example (CPU‑friendly but memory heavy for 7B):
  - OpenCLIP local + mHuBERT local:
    - `python code/scripts/train_stage2_fusion.py --llama-path resources/models/Llama-2-7b-chat-hf --clip-path resources/models/clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg --clip-backend open_clip --hubert-path resources/models/hubert-m --epochs 1 --batch-size 1 --freeze-clip --freeze-hubert`
  - HF CLIP (ViT‑B/32) + mHuBERT local:
    - `python code/scripts/train_stage2_fusion.py --llama-path resources/models/Llama-2-7b-chat-hf --clip-path openai/clip-vit-base-patch32 --clip-backend hf --hubert-path resources/models/hubert-m --epochs 1 --batch-size 1 --freeze-clip --freeze-hubert`

Tips:
- On limited RAM/VRAM, consider a smaller backbone for development or quantization (not included here).
- Use `--freeze-clip/--freeze-hubert` to keep unimodal encoders fixed during fusion.


Data Interface (when replacing synthetic data)
---------------------------------------------
Stage I — Video batch dict:
- `frames`: List[List[PIL.Image]] (each sample contains K frames)
- `labels`: LongTensor(B,)

Stage I — Audio batch dict:
- `waveforms`: List[Tensor(T_i,)] (all at the sampling rate you pass to preprocess)
- `labels`: LongTensor(B,)

Stage II — Fusion batch dict:
- `frames`: List[List[PIL.Image]]
- `waveforms`: List[Tensor(T_i,)]
- `asr_texts`: List[str]
- `labels`: LongTensor(B,)

Special Tokens and Prompts
--------------------------
- The tokenizer/model are augmented with additional tokens: `<VIDEO> </VIDEO> <AUDIO> </AUDIO> <TEXT> </TEXT> <EMO>`.
- Fusion builds `inputs_embeds` by concatenating text token embeddings and pseudo‑tokens for projected features.
- The classifier reads the final hidden state at `<EMO>`.


Extending/Customizing
---------------------
- Early‑stopping for Stage I at macro‑F1 > 0.5 and auto‑freeze in Stage II (add in trainers if desired).
- Weighted losses `lambda_video/lambda_audio/lambda_fusion` (Stage II currently uses fusion loss only, consistent with the simplified setting after pre‑adaptation).
- Checkpointing and logging are minimal; integrate your preferred logger/ckpt manager.


Testing & Style
---------------
- Run tests (if present): `pytest -q`
- Format: `black .` and `isort . --profile black`
- Lint: `ruff check .`


Security & Licensing
--------------------
- Respect model licensing and data privacy. Do not upload proprietary weights.
- Avoid committing `.venv/`, `.hf/`, caches, or large binaries. Use Git LFS if needed.
