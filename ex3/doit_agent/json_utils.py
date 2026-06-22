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


def extract_json_object(raw: str) -> dict[str, Any]:
    """
    Tries to parse raw model output as JSON.
    If the model wrapped JSON in markdown or extra text, extract the first JSON object.
    """
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Normalize Python literals before attempting further parsing.
    normalized = _normalize_python_literals(raw)

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

    # Remove common markdown fence.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Fallback: first {...} block.
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(normalized[start : end + 1])

    raise json.JSONDecodeError("Could not find JSON object", raw, 0)
