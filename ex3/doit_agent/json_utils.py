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
        return json.loads(raw, strict=False)
    except json.JSONDecodeError as e:
        print("JsonDecodeError [raw parse failed]:", e)
        pass

    # Normalize Python literals before attempting further parsing.
    normalized = _normalize_python_literals(raw)

    try:
        return json.loads(normalized, strict=False)
    except json.JSONDecodeError as e:
        print("JsonDecodeError [normalized parse failed]:", e)
        pass

    # Remove common markdown fence.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1), strict=False)
        except json.JSONDecodeError as e:
            print("JsonDecodeError [markdown fence parse failed]:", e)
            pass

    # Fallback: first {...} block.
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(normalized[start : end + 1], strict=False)
        except json.JSONDecodeError as e:
            print("JsonDecodeError [fallback {...} parse failed]:", e)
            pass

    raise json.JSONDecodeError("Could not find JSON object", raw, 0)
