from transformers import AutoTokenizer
from huggingface_hub import login
import os
import statistics
import json
import csv
from dotenv import load_dotenv

load_dotenv()
login(os.getenv("HF_TOKEN"))
from transformers import AutoTokenizer
import statistics
import json
import csv

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.3-8b-instruct",
    "deepseek-ai/DeepSeek-V3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "dicta-il/dictalm2.0-instruct",
]

english_words = [
    "hello",
    "computer",
    "language",
    "transformers",
    "artificial intelligence",
    "deep learning",
]

hebrew_words = [
    "שלום",
    "מחשב",
    "שפה",
    "למידה עמוקה",
    "בינה מלאכותית",
]


def avg_tokens(tokenizer, samples):
    vals = []

    for s in samples:
        tokens = tokenizer.encode(
            s,
            add_special_tokens=False
        )

        vals.append(
            len(tokens) / len(s.split())
        )

    return round(statistics.mean(vals), 2)


def detect_boundary(pretok):
    name = pretok.get("type", "").lower()

    # if "metaspace" in name:
    #     return "Metaspace"

    # if "split" in name:
    #     return "Split"

    # if "regex" in name:
    #     return "regex"

    return name


def detect_byte(tok_json):
    txt = json.dumps(tok_json).lower()

    if "bytefallback" in txt:
        return "byte-Fallback"

    if "bytelevel" in txt:
        return "byte-level"

    return "none"


rows = []

for model_id in models:

    print(model_id)

    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        tok_json = json.loads(
            tok.backend_tokenizer.to_str()
        )

        tokenizer_type = tok_json["model"]["type"]

        vocab_size = len(tok)

        pretok = tok_json.get(
            "pre_tokenizer",
            {}
        )

        boundary = detect_boundary(pretok)

        byte_type = detect_byte(tok_json)

        row = {
            "model_id": model_id,
            "tokenizer_type": tokenizer_type,
            "vocab_size": vocab_size,
            "special_tokens": list(tok.special_tokens_map.keys()),
            "word_boundary_strategy": boundary,
            "byte_fallback_or_byte_level": byte_type,
            "avg_tokens_per_english_word": avg_tokens(
                tok,
                english_words
            ),
            "avg_tokens_per_hebrew_words": avg_tokens(
                tok,
                hebrew_words
            ),
        }

        rows.append(row)

    except Exception as e:
        print("FAILED:", model_id)
        print(e)

with open(
    "tokenizers.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print("Saved tokenizers.csv")