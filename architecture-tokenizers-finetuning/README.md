# Architectural Choices, Tokenizers & Fine-Tuning

Comparative analysis of 10 open-weight LLMs (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B, OLMo-2, Granite-3.3, DeepSeek-V3, SmolLM2, Phi-4-mini, Falcon3, dictalm2.0), plus constrained decoding and LoRA fine-tuning experiments on top of them.

[Assignment spec](https://yoavg.github.io/llm-class-2025-2026/ass2-architectural-choices-tokenizers-finetuning/) · full write-up: [`Report.pdf`](Report.pdf)

## Part 1 — Architecture comparison

Extracted per-model structural properties — layer count/width, attention head config, MLP/MoE/embedding dimensions, position encoding, max context, activation function, norm placement — via `architectures.py`, with parameter shapes documented in [`arch.md`](arch.md) and tabulated in `models.csv`.

## Part 2 — Tokenizer comparison

`tokenanalyzer.py` inspects vocabulary size, word-boundary strategy, special tokens, and average tokens-per-word for English vs. Hebrew across all 10 tokenizers (`tokenizers.csv`); `build_heb_token_json.py` finds the Hebrew-compatible token subset per model.

## Part 3 — Constrained decoding

Force Hebrew-only output from an English prompt by masking every non-Hebrew token during generation on Qwen2.5-7B and Mistral-7B:
- `hebrew_allowed_tokens_mistral.json` / `hebrew_allowed_tokens_qwen.json` — allowed Hebrew token IDs per model.
- `constrained_questions.py` / `constrained_text_executor.py` — the constrained-generation logic.
- `decoding_outputs.jsonl` — unconstrained vs. constrained outputs for 10 queries.
- `hebrew_utils/convert_csv.py` — converts `constrained_questions.csv` into that output format.

## Part 4 — Fine-tuning Qwen2.5-1.5B to answer in Hebrew

Supervised fine-tuning so an English-prompted model responds in Hebrew (not just translated boilerplate):
- `qwen_FineTune.ipynb` — end-to-end LoRA fine-tuning notebook.
- `lora_train.py` / `lora_inference.py` — LoRA training/inference via `trl` + `peft`.
- `manua_lora_train.py` / `manual_lora_infrerence.py` — the same, implemented manually (plain `Trainer`, hand-wired PEFT) instead of the high-level trainer.
- `eval_outputs.jsonl` — base-model vs. fine-tuned outputs on 20 held-out English prompts.
- Trained adapter: [`link_to_adapter.txt`](link_to_adapter.txt).
