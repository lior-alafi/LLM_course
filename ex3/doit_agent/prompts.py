from __future__ import annotations

import json

from doit_agent.memory.models import MemoryRecord
from doit_agent.state.models import InteractionRecord
from doit_agent.user_awareness.shell_history import ShellHistoryEntry, format_shell_history


AGENT_SCHEMA = {
    "intent": "execute_command | conversation | answer | clarification | error",
    "command": "string or null",
    "answer": "string or null",
    "error": "string or null",
    "clarification_question": "string or null",
    "clarification_options": "list of strings, use [] when there are no options, never null",
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
            "action": "store | delete | none",
            "key": "string or null",
            "value": "string or null",
            "reason": "string or null",
        }
    ]
}

CONTEXT_SUMMARY_SCHEMA = {"summary": "string"}


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
        chunks.append(
            f"""- timestamp: {r.timestamp}
  session_id: {r.session_id}
  cwd: {r.cwd}
  user_query: {r.user_query}
  intent: {r.intent}
  command: {r.command}
  answer: {shorten(r.answer, 300)}
  returncode: {r.returncode}
  stdout: {shorten(r.stdout)}
  stderr: {shorten(r.stderr)}"""
        )
    return "\n\n".join(chunks)


def format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "No stored user memories."
    return "\n".join(f"- {m.key}: {m.value} (updated_at={m.updated_at})" for m in memories)


def build_agent_messages(
    *,
    user_query: str,
    secure: bool,
    cwd: str,
    session_id: str,
    same_session_history: list[InteractionRecord],
    global_history: list[InteractionRecord],
    memories: list[MemoryRecord],
    recent_shell_commands: list[ShellHistoryEntry],
    session_summary: str | None,
) -> list[dict[str, str]]:
    system = f"""
You are doit, a command-line LLM agent.

Your job:
1. If the user asks for a shell action, produce exactly one bash command.
2. If the user asks a normal question, answer normally.
3. If the user refers to previous interactions, use the provided history.
4. Prefer same-session history for ambiguous references like "them", "that", "do it again".
5. Use global history only when the user explicitly refers to another terminal, another session, or earlier work.
6. Use recent external shell commands when the user asks about what they did manually outside doit.
7. Use stored memories when the user refers to remembered folders, preferences, or facts.
8. If the user request is ambiguous, ask a clarification question.
9. Return only valid JSON.
10. If the user refers to a named location, folder, preference, or fact (e.g. "my advanced AI course folder") that does NOT appear in stored memories and cannot be inferred from history or context:
   - If there is NO matching entry at all → use intent "answer" and tell the user you don't know which one they mean (e.g. "I couldn't find a stored folder called 'advanced AI course'. Try telling me its path or saving it with 'remember that my X folder is /path'.").
   - If there are 2 or more possible matches → use intent "clarification" and list the options so the user can pick one.
   - Never guess or fabricate a path.

Important command rules:
- Produce only one command when intent is execute_command.
- Prefer bash commands.
- Do not wrap the command in markdown.
- Do not include explanations inside the command.
- If there are no clarification options, return "clarification_options": [].
- If the desired command is cd, output a plain cd command. The caller may handle shell integration separately.
- before storing in memory , understand the entire context from the user query
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

Session summary:
{session_summary or "No compacted session summary."}

Same-session recent history:
{format_history(same_session_history)}

Global recent history:
{format_history(global_history)}

Recent external shell commands run by the user outside doit:
{format_shell_history(recent_shell_commands)}
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


def build_clarification_messages(
    *,
    original_query: str,
    clarification_question: str,
    options: list[str],
    user_answer: str,
) -> list[dict[str, str]]:
    system = f"""
You are doit, a command-line LLM agent.
The previous response asked a clarification question.
Use the user's clarification answer to produce the final decision.
Return only valid JSON.

JSON schema:
{json.dumps(AGENT_SCHEMA, indent=2)}
""".strip()

    user = f"""
Original query:
{original_query}

Clarification question:
{clarification_question}

Options:
{options}

User clarification answer:
{user_answer}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_safety_messages(user_query: str, command: str) -> list[dict[str, str]]:
    system = f"""
You are a shell command safety classifier.
Classify whether a proposed bash command is safe to execute directly.

Rules:
- read_only: only reads or displays information.
- modifies_filesystem: creates, edits, moves, copies, renames, chmods, installs, or writes files.
- dangerous: deletes data, recursively modifies files, uses sudo, changes permissions broadly, formats disks, downloads and executes code, etc.
- unsupported: not a valid shell command or unrelated.

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

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_memory_extraction_messages(
    user_query: str,
    existing_memories: list[MemoryRecord],
    cwd: str | None = None,
    command: str | None = None,
) -> list[dict[str, str]]:
    system = f"""
You extract explicit long-term memory requests from user commands.

Store a memory ONLY when the user explicitly says to remember something (e.g. "remember that...", "always use...", "my X is Y").
Store a memory for stable, named preferences that should apply across future sessions (e.g. a preferred editor, a project root path the user named).

DO NOT store:
- The current working directory unless the user explicitly says "remember this as my project root" or similar.
- Flags or modifiers from a single command (e.g. "not recursively", "sort by size") unless the user says they want this to always apply.
- Observations about what just happened or what files exist.
- Anything that is only relevant to this specific command or this session.
- Named things (folders, paths, preferences) that the user references but that do NOT appear in the existing memories and cannot be resolved from the provided cwd/command. If the reference is unresolvable, use action: none — the main agent will ask for clarification.

Examples:
- User says "list files sorted by size" → action: none (this is just a command, not a preference to remember)
- User says "list files, not recursively" → action: none (single-command modifier, not a lasting preference)
- User says "remember that my project root is ~/work/myapp" → action: store, key: project_root, value: /home/user/work/myapp
- User says "always sort by size" → action: store, key: sort_preference, value: always sort by size
- User says "cd 2 folders back , this is my LLM Folder" -> key: LLM_folder , value: absolute path of the LLM folder
- User says "move to my advanced AI course folder" but no such folder is in memories → action: none (unresolvable reference; agent will ask for clarification)

Delete a memory only when the user clearly asks to forget/remove/delete a remembered fact.
When storing any value that represents a location or path, always compute and store the final ABSOLUTE path.
Use the provided cwd and command to work out the concrete result — do not store relative paths or natural-language instructions.
Return only valid JSON.

JSON schema:
{json.dumps(MEMORY_SCHEMA, indent=2)}
""".strip()

    context_lines = []
    if cwd:
        context_lines.append(f"Current working directory: {cwd}")
    if command:
        context_lines.append(f"Command that will be executed: {command}")
    context_block = ("\n".join(context_lines) + "\n\n") if context_lines else ""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Existing memories:\n{format_memories(existing_memories)}"},
        {"role": "user", "content": f"{context_block}User query:\n{user_query}"},
    ]


def build_context_summary_messages(*, old_summary: str | None, records: list[InteractionRecord]) -> list[dict[str, str]]:
    system = f"""
You summarize command-line agent history for future context.
Preserve important folders, commands, failures, user preferences, unresolved tasks, and references that future follow-ups may depend on.
Return only valid JSON.

JSON schema:
{json.dumps(CONTEXT_SUMMARY_SCHEMA, indent=2)}
""".strip()

    user = f"""
Previous summary:
{old_summary or "No previous summary."}

Interactions to summarize:
{format_history(records)}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
