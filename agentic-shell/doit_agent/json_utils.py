from __future__ import annotations

import json
import re
from typing import Any


def _normalize_python_literals(text: str) -> str:
    """Replace Python-style literals that are invalid JSON.

    Some local models (e.g. Ollama/gemma) output Python values like
    ``None``, ``True``, ``False`` instead of JSON ``null``, ``true``,
    ``false``.  This does a safe word-boundary substitution so that
    identifiers named e.g. ``Nonexistent`` are not touched.
    """
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    return text


def extract_json_object(raw: str | None) -> dict[str, Any]:
    """
    Tries to parse raw model output as JSON.
    If the model wrapped JSON in markdown or extra text, extract the first JSON object.
    """
    if raw is None:
        raise json.JSONDecodeError("Model returned no content", "", 0)

    raw = raw.strip()
    if not raw:
        raise json.JSONDecodeError("Model returned empty content", raw, 0)

    errors: list[str] = []

    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError as e:
        errors.append(f"raw parse failed: {e}")

    # Normalize Python literals before attempting further parsing.
    normalized = _normalize_python_literals(raw)

    try:
        return json.loads(normalized, strict=False)
    except json.JSONDecodeError as e:
        errors.append(f"normalized parse failed: {e}")

    # Remove common markdown fence.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1), strict=False)
        except json.JSONDecodeError as e:
            errors.append(f"markdown fence parse failed: {e}")

    # Fallback: first {...} block.
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(normalized[start : end + 1], strict=False)
        except json.JSONDecodeError as e:
            errors.append(f"fallback {{...}} parse failed: {e}")

    detail = "; ".join(errors) if errors else "no JSON object delimiters found"
    raise json.JSONDecodeError(f"Could not parse JSON object ({detail})", raw, 0)
