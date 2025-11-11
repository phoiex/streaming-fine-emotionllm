# Repository Guidelines

## Project Structure & Module Organization
- Root layout:
  - `resources/models/Llama-2-7b-chat-hf/` — local HF model repo (weights, tokenizer). This subfolder includes its own AGENTS.md.
  - `resources/datasets/` — datasets (MELD, CH‑SIMS); store large/raw data here.
  - `code/` — source and tooling (e.g., `code/scripts/testrun.py`).
  - `.venv/`, `.hf/` — local venv and HF cache; do not commit.
- New code conventions:
  - Place library code under `code/src/streaming_fine_emotionllm/`.
  - Utilities under `code/scripts/`; notebooks under `code/notebooks/`.
  - Tests under `code/tests/` mirroring `code/src/`.

## Build, Test, and Development Commands
- Environment (POSIX):
  - `python3 -m venv .venv && . .venv/bin/activate`
  - `pip install -U pip` and `pip install transformers torch accelerate safetensors pytest black isort ruff`
- Quick run (CPU friendly):
  - `python code/scripts/testrun.py --device-map cpu --dtype float32 --prompt "Hello"`
- Tests (if present):
  - `pytest -q` — run all tests.

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
