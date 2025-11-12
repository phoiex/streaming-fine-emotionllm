# Repository Guidelines

## Project Goal
- Train a multimodal emotion recognition model whose architecture is inspired by the EmotionLlama paper (EmotionLlama‑style MLLM). All design and implementation choices should align with this objective.

## Project Structure & Module Organization
- Root layout:
  - `resources/models/` — local model snapshots and saved weights (e.g.,
    `Llama-2-7b-chat-hf/`, `CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/`, `hubert-m/`).
  - `resources/datasets/` — datasets (MELD, CH‑SIMS); store large/raw data here.
  - `code/` — source and tooling (e.g., `code/scripts/testrun.py`).
  - `.venv/`, `.hf/` — local venv and HF cache; do not commit.
- New code conventions:
  - Place library code under `code/src/streaming_fine_emotionllm/`.
    - Models: `code/src/streaming_fine_emotionllm/models/`
      - `open_clip_encoder.py` — OpenCLIP visual encoder wrapper/factory.
      - `hubert_audio_encoder.py` — HuBERT audio encoder wrapper/factory.
  - Utilities under `code/scripts/`; notebooks under `code/notebooks/`.
  - Tests under `code/tests/` mirroring `code/src/`.

## Build, Test, and Development Commands
- Environment (POSIX):
  - `python3 -m venv .venv && . .venv/bin/activate`
  - `pip install -U pip` and `pip install transformers torch accelerate safetensors open_clip_torch pillow soundfile pytest black isort ruff`
- Quick run (CPU friendly):
  - `python code/scripts/testrun.py --device-map cpu --dtype float32 --prompt "Hello"`
- Tests (if present):
  - `pytest -q` — run all tests.

## Utilities & Scripts
- Visual encoder (OpenCLIP):
  - Download: `python code/scripts/download_openclip_model.py --dest resources/models`
  - Demo: `python code/scripts/run_openclip_demo.py --cache-dir resources/models`
- Audio encoder (mHuBERT):
  - Full snapshot: `python code/scripts/download_hubert_m_model.py --dest resources/models --name hubert-m`
  - Save via Transformers: `python code/scripts/save_hubert_m_pretrained.py --dest resources/models/hubert-m`
  - Demo: `python code/scripts/run_hubert_demo.py --source resources/models/hubert-m`

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, UTF-8, type hints where practical.
- Names: modules/files `snake_case.py`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Formatting: `black .` (88 cols), `isort .` (profile=black), lint with `ruff check .`.
- JSON configs: 2-space indent; validate with `from_pretrained()` using paths under `resources/`.

## Testing Guidelines
- Framework: `pytest` with files `tests/test_*.py` and names `test_*`.
- Mirror `src/` structure (e.g., `tests/foo/test_bar.py` for `src/.../foo/bar.py`).
- Keep tests deterministic (set seeds) and light on GPU; prefer CPU or small inputs.
- Optional coverage: `pytest --maxfail=1 --disable-warnings -q` (add `pytest-cov` if needed).

## Commit & Pull Request Guidelines
- Commits: use imperative mood; Conventional Commits are welcome (e.g., `feat: add inference util`, `fix: stabilize seed handling`).
- PRs must include: purpose, summary, runnable steps (commands), and any screenshots/log snippets of `testrun.py` or tests.
- Link related issues; keep diffs focused. Do not re-shard or rename model files without prior discussion.

## Security & Configuration Tips
- Respect model licensing; avoid uploading proprietary weights or personal data.
- Never commit `.venv/`, `.hf/`, caches, or large binaries; use Git LFS for necessary large files.
- Prefer `--device-map cpu` for reproducible reviews; document GPU-only steps clearly.

## Current Progress
- Goal set: EmotionLlama‑style multimodal emotion recognition.
- Visual: OpenCLIP ConvNeXt‑B wrapper added (`open_clip_encoder.py`), download + demo scripts available.
- Audio: HuBERT (mHuBERT‑147) wrapper added (`hubert_audio_encoder.py`), save/download + demo scripts available.
- Models stored under `resources/models/` (e.g., `CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/`, `hubert-m/`).

## Next Steps (suggested)
- Add multimodal alignment/projection to LLM token space and minimal training loop.
- Prepare dataset loaders for MELD / CH‑SIMS with deterministic sampling and evaluation splits.
- Provide lightweight unit tests for encoders and data transforms under `code/tests/`.
