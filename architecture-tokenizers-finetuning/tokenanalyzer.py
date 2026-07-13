from transformers import AutoTokenizer
from huggingface_hub import login
import os
import statistics
import json
import csv
from dotenv import load_dotenv

load_dotenv()
login(os.getenv("HF_TOKEN"))

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


def get_tokens_for_sentence(models, sentence, mode="tokenize"):
    """
    Iterates over the given list of models, loads their tokenizers,
    and tokenizes/encodes the sentence.
    Returns a dictionary mapping each model_id to its list of string tokens or integer IDs.
    """
    token_results = {}
    for model_id in models:
        print(f"Processing with {model_id}...")
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            if mode == "encode":
                tokens = tok.encode(sentence, add_special_tokens=False)
            else:
                tokens = tok.tokenize(sentence)
            token_results[model_id] = tokens
        except Exception as e:
            print(f"FAILED to process with {model_id}: {e}")
            token_results[model_id] = f"Error: {str(e)}"
    return token_results


def save_tokens_to_file(token_results, filename="tokenized_results.json"):
    """
    Writes the tokenization results dictionary to a JSON file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(token_results, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved tokenized results to '{filename}'")
    except Exception as e:
        print(f"Failed to save tokens to file: {e}")


def group_models_by_tokens(token_results):
    """
    Groups models that have the exact same list of tokens (either strings or integers).
    Returns a dictionary where the keys are tuples of model names in each group,
    and the values are the lists of tokens.
    """
    groups = {}
    for model_id, tokens in token_results.items():
        if isinstance(tokens, list):
            tokens_key = tuple(tokens)
        else:
            tokens_key = tokens

        if tokens_key not in groups:
            groups[tokens_key] = []
        groups[tokens_key].append(model_id)

    # Convert groups to a dict where keys are tuples of model names and values are the token lists
    grouped_dict = {}
    for tokens_key, model_ids in groups.items():
        group_key = tuple(model_ids)
        group_value = list(tokens_key) if isinstance(tokens_key, tuple) else tokens_key
        grouped_dict[group_key] = group_value

    return grouped_dict


def analyze_and_save_csv(models):
    """
    Analyzes the list of models and generates the tokenizers.csv summary.
    """
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
            pretok = tok_json.get("pre_tokenizer", {})
            boundary = detect_boundary(pretok)
            byte_type = detect_byte(tok_json)

            row = {
                "model_id": model_id,
                "tokenizer_type": tokenizer_type,
                "vocab_size": vocab_size,
                "special_tokens": list(tok.special_tokens_map.keys()) + tok.SPECIAL_TOKENS_ATTRIBUTES,
                "word_boundary_strategy": boundary,
                "byte_fallback_or_byte_level": byte_type,
                "avg_tokens_per_english_word": avg_tokens(tok, english_words),
                "avg_tokens_per_hebrew_words": avg_tokens(tok, hebrew_words),
            }
            rows.append(row)
        except Exception as e:
            print("FAILED:", model_id)
            print(e)

    if rows:
        with open("tokenizers.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("Saved tokenizers.csv")


def analyze_single_sentence(models, sentence, mode="tokenize", output_filename="sentence_tokens.json"):
    """
    Analyzes a single sentence for each model (either tokenize or encode), saves to file, and groups them.
    """
    results = get_tokens_for_sentence(models, sentence, mode=mode)
    save_tokens_to_file(results, output_filename)
    return group_models_by_tokens(results)


def main():
    # import sys
    # print("Select mode:")
    # print("  1: Tokenize (string fragments) [Default]")
    # print("  2: Encode (numerical token IDs)")
    # try:
    #     choice = input("Enter choice (1 or 2) [Default: 1]: ").strip()
    # except (KeyboardInterrupt, EOFError):
    #     choice = "1"
    
    # mode = "encode" if choice == "2" else "tokenize"
    mode="tokenize"
    print(f"\n--- Running Sentence Token Analysis & Grouping ({mode.capitalize()} Mode) ---")
    sentence = "The quick brown fox jumps over the lazy dog."
    groups = analyze_single_sentence(models, sentence, mode=mode)
    
    mode_title = "Tokenizations" if mode == "tokenize" else "Encodings"
    mode_label = "Tokens" if mode == "tokenize" else "Encoding"
    
    print(f"\n=== Model Groups and {mode_title} ===")
    for model_group, tokens in groups.items():
        print(f"\nGroup (Models): {model_group}")
        print(f"{mode_label}: {tokens}")
    print("==================================")


if __name__ == "__main__":
    main()