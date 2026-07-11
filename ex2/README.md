# Exercise 2 — Fine-tuning, tokenizers, constrained decoding

Full write-up: [`Report.pdf`](Report.pdf) · architecture/parameter-shape notes: [`arch.md`](arch.md)

- **Tokenizer analysis** — `tokenanalyzer.py`, `build_heb_token_json.py`; comparison across models in `tokenizers.csv`.
- **LoRA fine-tuning** — `lora_train.py` / `lora_inference.py` (via `trl`/`peft`), plus a from-scratch variant `manua_lora_train.py` / `manual_lora_infrerence.py` (plain `Trainer`, manual PEFT wiring). Trained adapter: [`link_to_adapter.txt`](link_to_adapter.txt).
- **Constrained decoding** — `constrained_questions.py` / `constrained_text_executor.py`, restricting generation to an allowed Hebrew token set (`hebrew_allowed_tokens_*.json`); sample results in `decoding_outputs.jsonl`.
- **Model architecture inspection** — `architectures.py`, `models.csv`.
- `qwen_FineTune.ipynb` — end-to-end fine-tuning notebook; `eval_outputs.jsonl` — evaluation results.
- `hebrew_utils/convert_csv.py` — converts `constrained_questions.csv` into the `decoding_outputs.jsonl` format.
