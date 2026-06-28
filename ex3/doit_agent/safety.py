from __future__ import annotations

import re
from pydantic import ValidationError

from doit_agent.debug import trace
from doit_agent.json_utils import extract_json_object
from doit_agent.llm_client import LLMClient
from doit_agent.prompt_log import PromptLogger
from doit_agent.prompts import build_safety_messages
from doit_agent.schemas import SafetyDecision


class RuleBasedSafetyClassifier:
    """
    Deterministic safety layer.
    This should catch obvious risky commands even if the LLM classifier fails.
    """

    DANGEROUS_PATTERNS = [
        r"\brm\s+-[^&|;]*r",               # rm -r / rm -rf
        r"\brm\s+-[^&|;]*f",               # rm -f
        r"\bsudo\b",
        r"\bmkfs\b",
        r"\bdd\s+",
        r":\(\)\s*\{",                     # fork bomb pattern
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bkillall\b",
        r"\bchmod\s+-R\b",
        r"\bchown\s+-R\b",
        r"\bcurl\b.*\|\s*(bash|sh|zsh)",
        r"\bwget\b.*\|\s*(bash|sh|zsh)",
    ]

    MODIFYING_PATTERNS = [
        r"\brm\b",
        r"\bmkdir\b",
        r"\btouch\b",
        r"\bmv\b",
        r"\bcp\b",
        r"\bchmod\b",
        r"\bchown\b",
        r"\bnpm\s+install\b",
        r"\bpip\s+install\b",
        r"\bapt\s+install\b",
        r"\bbrew\s+install\b",
        r">\s*(?!/dev/null\b|&\d)[^&|;]+",
        r">>\s*(?!/dev/null\b|&\d)[^&|;]+",
        r"\bsed\s+-i\b",
        r"\btee\b",
    ]

    READ_ONLY_PREFIXES = (
        "ls",
        "pwd",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "du",
        "df",
        "date",
        "whoami",
        "id",
        "git status",
        "git log",
        "git diff",
        "ps",
        "echo",
        "cd",       # navigation — read-only in terms of filesystem
        "pushd",
        "popd",
        "which",
        "type",
        "env",
        "printenv",
        "history",
    )

    @trace
    def classify(self, command: str) -> SafetyDecision | None:
        normalized = command.strip()

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized):
                return SafetyDecision(
                    safety_level="dangerous",
                    requires_confirmation=True,
                    allowed=True,
                    explanation="Rule-based safety detected a potentially dangerous command.",
                    source="rule_based",
                )

        for pattern in self.MODIFYING_PATTERNS:
            if re.search(pattern, normalized):
                return SafetyDecision(
                    safety_level="modifies_filesystem",
                    requires_confirmation=True,
                    allowed=True,
                    explanation="Rule-based safety detected that this command may modify the filesystem or environment.",
                    source="rule_based",
                )

        if normalized.startswith(self.READ_ONLY_PREFIXES):
            return SafetyDecision(
                safety_level="read_only",
                requires_confirmation=False,
                allowed=True,
                explanation="Rule-based safety classified this as a read-only command.",
                source="rule_based",
            )

        return None


class SafetyService:
    def __init__(
        self,
        llm: LLMClient,
        *,
        model_name: str,
        prompt_logger: PromptLogger | None = None,
    ):
        self.llm = llm
        self.model_name = model_name
        self.prompt_logger = prompt_logger
        self.rule_based = RuleBasedSafetyClassifier()

    @trace
    def classify(self, user_query: str, command: str) -> SafetyDecision:
        rule_result = self.rule_based.classify(command)

        # If deterministic rules already found modifying/dangerous/read-only,
        # return that. This keeps hard safety independent from the LLM.
        if rule_result is not None:
            return rule_result

        messages = build_safety_messages(user_query, command)
        raw = self.llm.complete_text(messages)

        try:
            data = extract_json_object(raw)
            decision = SafetyDecision.model_validate(data)
            decision.source = "llm"

            if self.prompt_logger:
                self.prompt_logger.log(
                    acdl_spec="DoitSafetyCheck",
                    model=self.model_name,
                    messages=messages,
                    raw_response=raw,
                    parsed_response=decision.model_dump(),
                )

            return decision

        except Exception as exc:
            return SafetyDecision(
                safety_level="unsupported",
                requires_confirmation=True,
                allowed=False,
                explanation=f"Could not parse safety response. Raw response: {raw}. Error: {exc}",
                source="fallback",
            )
