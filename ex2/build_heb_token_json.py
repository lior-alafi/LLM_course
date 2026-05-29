import json
import unicodedata
from transformers import AutoTokenizer


HEBREW_RANGES = [
    (0x0590, 0x05FF),  # Hebrew: letters, niqqud, cantillation, Hebrew punctuation
    (0xFB1D, 0xFB4F),  # Hebrew presentation forms
]


def is_hebrew_char(ch: str) -> bool:
    """
    Returns True for Hebrew Unicode characters, including:
    - Hebrew letters
    - final Hebrew letters
    - niqqud
    - cantillation marks
    - Hebrew punctuation
    - Hebrew presentation forms
    """
    cp = ord(ch)
    return any(start <= cp <= end for start, end in HEBREW_RANGES)


def token_is_hebrew_allowed(decoded: str) -> bool:
    """
    Allow tokens that contain:
    - Hebrew characters
    - numbers
    - whitespace
    - punctuation
    - emojis
    - mathematical symbols
    - general symbols
    - control/separator chars

    Reject tokens that contain:
    - any non-Hebrew letter, e.g. English, Chinese, Arabic, Greek, Cyrillic
    - any non-Hebrew combining mark, e.g. Arabic vowel marks, Latin accents

    In short:
    Hebrew chars are allowed.
    Non-letter symbols are allowed.
    Non-Hebrew letters are rejected.
    Non-Hebrew marks are rejected.
    """
    if decoded == "":
        return False

    for ch in decoded:
        # Hebrew letters, niqqud, cantillation, Hebrew punctuation, etc.
        if is_hebrew_char(ch):
            continue

        cat = unicodedata.category(ch)

        # L* = Letter
        # Reject English, Chinese, Arabic, Russian, Greek, etc.
        if cat.startswith("L"):
            return False

        # M* = Mark
        # Reject non-Hebrew combining marks such as Arabic vowel marks
        # or Latin combining accents.
        #
        # Hebrew niqqud is already allowed above because it is in Hebrew range.
        if cat.startswith("M"):
            return False

        # Everything else is allowed:
        # N* numbers
        # P* punctuation
        # S* symbols, including math symbols and emojis
        # Z* separators
        # C* control/special chars
        continue

    return True


def get_special_token_ids(tokenizer) -> set[int]:
    """
    Collect all special token IDs known to the tokenizer:
    eos, bos, pad, unk, sep, cls, mask, additional special tokens, etc.
    """
    special_ids = set()

    for token_id in tokenizer.all_special_ids:
        if token_id is not None:
            special_ids.add(int(token_id))

    return special_ids


def audit_allowed_tokens(tokenizer, allowed_ids, max_print: int = 100):
    """
    Debug function:
    Prints allowed tokens that still contain non-Hebrew letters or non-Hebrew marks.
    Ideally this should print zero bad tokens.
    """
    bad = []

    for token_id in allowed_ids:
        decoded = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

        for ch in decoded:
            if is_hebrew_char(ch):
                continue

            cat = unicodedata.category(ch)

            if cat.startswith("L") or cat.startswith("M"):
                bad.append(
                    (
                        token_id,
                        repr(decoded),
                        ch,
                        unicodedata.name(ch, "UNKNOWN"),
                        cat,
                    )
                )
                break

    print(f"Bad allowed tokens with non-Hebrew letters/marks: {len(bad)}")

    for row in bad[:max_print]:
        print(row)


def build_allowed_tokens(
    model_id: str,
    output_path: str,
    include_special_tokens: bool = False,
    json_with_metadata: bool = False,
    run_audit: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    allowed_ids = []

    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        # If this is a special token and we do not want special tokens,
        # skip it before decoding/classifying.
        if token_id in tokenizer.all_special_ids and not include_special_tokens:
            continue

        decoded = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

        if token_is_hebrew_allowed(decoded):
            allowed_ids.append(int(token_id))

    if include_special_tokens:
        allowed_ids.extend(get_special_token_ids(tokenizer))

    allowed_ids = sorted(set(allowed_ids))

    if run_audit:
        audit_allowed_tokens(tokenizer, allowed_ids)

    if json_with_metadata:
        data = {
            "model_id": model_id,
            "include_special_tokens": include_special_tokens,
            "strategy": (
                "Allowed tokens whose decoded form contains no non-Hebrew letters "
                "and no non-Hebrew combining marks. Hebrew characters are allowed. "
                "Numbers, punctuation, whitespace, separators, emojis, mathematical "
                "symbols, general symbols, and control characters are allowed. "
                "Special tokens are included only when include_special_tokens=True."
            ),
            "allowed_token_ids": allowed_ids,
            "num_allowed_tokens": len(allowed_ids),
            "special_token_ids": sorted(get_special_token_ids(tokenizer)),
        }
    else:
        # Usually safer for automatic grading: plain list of allowed token IDs.
        data = allowed_ids

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(allowed_ids)} allowed tokens to {output_path}")
    print(f"include_special_tokens = {include_special_tokens}")

    if include_special_tokens:
        print("Special token IDs added:", sorted(get_special_token_ids(tokenizer)))


if __name__ == "__main__":
    build_allowed_tokens(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        output_path="hebrew_allowed_tokens_qwen.json",
        include_special_tokens=False,
        json_with_metadata=True,
        run_audit=True,
    )

    build_allowed_tokens(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        output_path="hebrew_allowed_tokens_mistral.json",
        include_special_tokens=False,
        json_with_metadata=True,
        run_audit=True,
    )