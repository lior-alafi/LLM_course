from __future__ import annotations

import json

from doit_agent.memory.models import MemoryRecord
from doit_agent.state.models import InteractionRecord


AGENT_SCHEMA = {
    "intent": "execute_command | conversation | answer | clarification | error",
    "command": "string or null",
    "answer": "string or null",
    "error": "string or null",
    "clarification_question": "string or null",
    "clarification_options": "list of strings",
    "explanation": "string or null",
}


SAFETY_SCHEMA = {
    "safety_level": "read_only | modifies_filesystem | dangerous | unsupported",
    "requires_confirmation": "boolean",
    "allowed": "boolean",
    "explanation": "string",
}


MEMORY_SCHEMA = {
    "memories": [
        {
            "should_store": "boolean",
            "key": "string or null",
            "value": "string or null",
            "reason": "string or null",
        }
    ]
}


def shorten(text: str | None, max_chars: int = 1200) -> str:
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]..."


def format_history(records: list[InteractionRecord]) -> str:
    if not records:
        return "No previous doit interactions."

    chunks: list[str] = []

    for r in records:
        chunk = f"""
- timestamp: {r.timestamp}
  session_id: {r.session_id}
  cwd: {r.cwd}
  user_query: {r.user_query}
  intent: {r.intent}
  command: {r.command}
  answer: {shorten(r.answer, 300)}
  returncode: {r.returncode}
  stdout: {shorten(r.stdout)}
  stderr: {shorten(r.stderr)}
""".strip()
        chunks.append(chunk)

    return "\n\n".join(chunks)


def format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "No stored user memories."

    chunks = []
    for m in memories:
        chunks.append(
            f"- {m.key}: {m.value} "
            f"(updated_at={m.updated_at})"
        )

    return "\n".join(chunks)


def build_agent_messages(
    *,
    user_query: str,
    secure: bool,
    cwd: str,
    session_id: str,
    same_session_history: list[InteractionRecord],
    global_history: list[InteractionRecord],
    memories: list[MemoryRecord],
) -> list[dict[str, str]]:
    system = f"""
You are doit, a command-line LLM agent.

Your job:
1. If the user asks for a shell action, produce exactly one bash command.
2. If the user asks a normal question, answer normally.
3. If the user refers to previous interactions, use the provided history.
4. Prefer same-session history for ambiguous references like "them", "that", "do it again".
5. Use global history only when the user explicitly refers to another terminal, another session, or earlier work.
6. Use stored memories when the user refers to remembered folders, preferences, or facts.
7. If the user request is ambiguous, ask a clarification question.
8. Return only valid JSON.

Important command rules:
- Produce only one command when intent is execute_command.
- Prefer bash commands.
- Do not wrap the command in markdown.
- Do not include explanations inside the command.
- If the desired command is cd, output a plain cd command. The caller may handle shell integration separately.

The current security mode is: env.secure={secure}.

JSON schema:
{json.dumps(AGENT_SCHEMA, indent=2)}
""".strip()

    context = f"""
Current environment:
- cwd: {cwd}
- session_id: {session_id}
- env.secure: {secure}

Stored memories:
{format_memories(memories)}

Same-session recent history:
{format_history(same_session_history)}

Global recent history:
{format_history(global_history)}
""".strip()

    user = f"""
Current user query:
{user_query}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
        {"role": "user", "content": user},
    ]


def build_safety_messages(user_query: str, command: str) -> list[dict[str, str]]:
    system = f"""
You are a shell command safety classifier.

Classify whether a proposed bash command is safe to execute directly.

Rules:
- read_only: only reads or displays information, e.g. ls, pwd, cat, grep, find without deletion.
- modifies_filesystem: creates, edits, moves, copies, renames, chmods, installs, or writes files.
- dangerous: deletes data, recursively modifies many files, uses sudo, changes permissions broadly, kills processes, formats disks, downloads and executes code, or can cause serious damage.
- unsupported: not a valid shell command or not related to the user request.

Behavior:
- read_only commands do not require confirmation.
- modifies_filesystem commands require confirmation.
- dangerous commands require confirmation and should be explained clearly.
- unsupported commands are not allowed.

Return only valid JSON.

JSON schema:
{json.dumps(SAFETY_SCHEMA, indent=2)}
""".strip()

    user = f"""
Original user query:
{user_query}

Proposed command:
{command}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_memory_extraction_messages(user_query: str, existing_memories: list[MemoryRecord]) -> list[dict[str, str]]:
    system = f"""
You extract explicit long-term memory requests from user commands.

Store a memory only when the user clearly asks to remember something, or states a stable preference/folder mapping that should be useful later.

Examples that should be stored:
- "remember that my LLM class folder is ~/school/llms/ass3"
- "this is my LLM class project folder"
- "from now on ask me before sorting by creation date"

Examples that should not be stored:
- one-time command requests
- random facts that are not useful later
- temporary command output

Return only valid JSON.

JSON schema:
{json.dumps(MEMORY_SCHEMA, indent=2)}
""".strip()

    context = f"""
Existing memories:
{format_memories(existing_memories)}
""".strip()

    user = f"""
User query:
{user_query}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
        {"role": "user", "content": user},
    ]
