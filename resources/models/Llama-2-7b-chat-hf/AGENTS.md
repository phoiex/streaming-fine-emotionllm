# Repository Guidelines

## Project Structure & Module Organization
- Root contains model assets and metadata; there is no app code here.
  - `config.json`, `configuration.json`, `generation_config.json` — model and generation settings.
  - `tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` — tokenizer files.
  - `model-*-of-*.safetensors`, `model.safetensors.index.json` and `pytorch_model-*-of-*.bin`, `pytorch_model.bin.index.json` — sharded weights and indexes.
  - `README.md`, `LICENSE.txt`, `USE_POLICY.md` — documentation and licensing.
- If adding utilities, place Python scripts under `scripts/` and examples under `examples/` (do not put code in the repo root).

## Build, Test, and Development Commands
- No build step. This repo is consumable via local path with Hugging Face Transformers.
- Install tooling:
  - `pip install -U transformers accelerate safetensors torch` (choose the Torch build matching your CUDA/CPU).
- Smoke test (from repo root):
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tok = AutoTokenizer.from_pretrained(".")
  model = AutoModelForCausalLM.from_pretrained(".", torch_dtype="auto", device_map="auto")
  out = model.generate(**tok("Hello", return_tensors="pt").to(model.device), max_new_tokens=20)
  print(tok.decode(out[0], skip_special_tokens=True))
  ```

## Coding Style & Naming Conventions
- Python (for added tools/examples): PEP 8, 4-space indent, snake_case files/functions; constants UPPER_SNAKE_CASE.
- Format with Black and isort; type hints preferred where practical.
- JSON configs: 2-space indent, no trailing commas; preserve existing keys and semantics. Validate with `from_pretrained('.')` after changes.

## Testing Guidelines
- Provide a minimal load/generate check (see Smoke test) and include any HF/torch versions used.
- If changing `generation_config.json` or tokenizer files, add a short example showing expected behavior.
- Avoid committing large artifacts, generated text, or checkpoints.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages (e.g., `docs: clarify usage notes`, `config: adjust eos token`). Keep subject ≤ 72 chars.
- PRs should include: purpose, summary of changes, affected files, link to issue (if any), and smoke-test output snippet.
- Do not add or reshuffle weight shards without prior coordination; use Git LFS for any large files.

## Security & Compliance
- Respect `USE_POLICY.md` and `LICENSE.txt`; do not add datasets, personal data, or proprietary weights. Link to Meta’s license when referencing distribution.
- Report issues to the channels listed in `README.md`/`USE_POLICY.md`.

