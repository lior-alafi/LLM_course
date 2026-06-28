from __future__ import annotations

import os
import sys
import json
import re


def _sanitize_command(command: str) -> str:
    # \$ inside single quotes is always wrong: single quotes pass everything
    # literally, so \$ reaches the inner shell as a literal backslash+dollar
    # which breaks $() command substitution.
    command = re.sub(r"'([^']*)'", lambda m: "'" + m.group(1).replace(r"\$", "$") + "'", command)
    # sh is POSIX dash and may lack bash features; replace with bash.
    command = re.sub(r"\bsh\s+-c\b", "bash -c", command)
    return command

from doit_agent.command_policy import SingleCommandPolicy, needs_parent_shell
from doit_agent.config import DoitConfig
from doit_agent.context_summary import ContextSummaryService
from doit_agent.debug import trace
from doit_agent.json_utils import extract_json_object
from doit_agent.llm_client import LLMClient
from doit_agent.memory.store import MemoryStore
from doit_agent.memory_service import MemoryService
from doit_agent.prompt_log import PromptLogger
from doit_agent.prompts import AGENT_SCHEMA, build_agent_messages, build_clarification_messages
from doit_agent.safety import SafetyService
from doit_agent.schemas import AgentDecision, SafetyDecision
from doit_agent.shell_exec import run_shell
from doit_agent.state.models import InteractionRecord, get_session_id
from doit_agent.state.store import StateStore
from doit_agent.user_awareness.shell_history import ShellHistoryProvider


def _normalize_confirmation_answer(answer: str) -> str:
    return "".join(ch for ch in answer.strip().lower() if "a" <= ch <= "z")


def _build_minimal_agent_messages(query: str, cwd: str) -> list[dict[str, str]]:
    system = (
        "You are doit, a command-line assistant. Return exactly one raw JSON "
        "object and no other text. If the user asks for a shell action, set "
        "intent to execute_command and provide one bash command. If the user "
        "asks a normal question, set intent to answer."
    )
    user = (
        f"cwd: {cwd}\n"
        f"user query: {query}\n\n"
        f"JSON schema:\n{json.dumps(AGENT_SCHEMA, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _deterministic_agent_decision(query: str, cwd: str) -> AgentDecision | None:
    normalized = " ".join(query.lower().split())
    target = ".." if "project" in normalized and os.path.basename(cwd) == "doit_agent" else "."

    if "todo" in normalized and any(word in normalized for word in ("find", "show", "list")):
        return AgentDecision(
            intent="execute_command",
            command=f'rg -n "TODO" {target}',
            clarification_options=[],
            explanation="Deterministic rule matched.",
        )

    if "modified today" in normalized or "changed today" in normalized:
        return AgentDecision(
            intent="execute_command",
            command=(
                f"find {target} -type f -newermt 'today' "
                "-not -path '*/.git/*' -not -path '*/.venv/*' "
                "-not -path '*/.doit/*' -not -path '*/__pycache__/*'"
            ),
            clarification_options=[],
            explanation="Deterministic rule matched.",
        )

    if "python" in normalized and "file" in normalized and (
        "recursive" in normalized or "recursively" in normalized or "all" in normalized
    ):
        return AgentDecision(
            intent="execute_command",
            command=f"find {target} -name '*.py'",
            clarification_options=[],
            explanation="Deterministic rule matched.",
        )

    mkdir_match = re.search(r"\b(?:create|make)\s+(?:a\s+)?(?:directory|folder)\s+(?:called|named)\s+([A-Za-z0-9_.-]+)\b", query, re.IGNORECASE)
    if mkdir_match:
        return AgentDecision(
            intent="execute_command",
            command=f"mkdir {mkdir_match.group(1)}",
            clarification_options=[],
            explanation="Deterministic rule matched.",
        )

    return None


class DoitAgent:
    def __init__(
        self,
        *,
        config: DoitConfig,
        llm: LLMClient,
        state_store: StateStore,
        memory_store: MemoryStore,
        prompt_logger: PromptLogger | None = None,
        verbose_level: int = 0,
    ):
        self.config = config
        self.llm = llm
        self.state_store = state_store
        self.memory_store = memory_store
        self.prompt_logger = prompt_logger
        self.verbose_level = verbose_level

        self.safety = SafetyService(
            llm,
            model_name=config.model.model,
            prompt_logger=prompt_logger,
        )

        self.memory_service = MemoryService(
            llm=llm,
            memory_store=memory_store,
            model_name=config.model.model,
            prompt_logger=prompt_logger,
        )

        self.context_summary_service = ContextSummaryService(
            llm=llm,
            model_name=config.model.model,
            prompt_logger=prompt_logger,
        )

        self.shell_history_provider = ShellHistoryProvider()
        self.command_policy = SingleCommandPolicy()
        self.session_id = get_session_id()

    @trace
    def decide(self, query: str) -> AgentDecision:
        same_session_history = self.state_store.get_recent_interactions(
            limit=self.config.context.summary_recent_keep,
            session_id=self.session_id,
        )

        # Other terminal windows as separate, labeled streams so references like
        # "the task we did in window 2" can resolve against the right session.
        other_sessions = self.state_store.get_recent_sessions(limit=3)
        other_sessions.pop(self.session_id, None)

        memories = self.memory_store.list_memories()

        recent_shell_commands = []
        if self.config.user_awareness.enabled:
            recent_shell_commands = self.shell_history_provider.get_recent_commands(
                limit=self.config.user_awareness.shell_history_limit,
            )

        session_summary = None
        if self.config.context.summaries_enabled:
            session_summary = self.context_summary_service.get_summary(self.session_id)

        messages = build_agent_messages(
            user_query=query,
            secure=self.config.agent.secure,
            cwd=os.getcwd(),
            session_id=self.session_id,
            same_session_history=same_session_history,
            other_sessions=other_sessions,
            memories=memories,
            recent_shell_commands=recent_shell_commands,
            session_summary=session_summary,
        )
        raw = self.llm.complete_text(messages)
        # print(raw)
        try:
            data = extract_json_object(raw)
            # print(data)
            decision = AgentDecision.model_validate(data)

            if self.prompt_logger:
                self.prompt_logger.log(
                    acdl_spec="DoitAgentStateful",
                    model=self.config.model.model,
                    messages=messages,
                    raw_response=raw,
                    parsed_response=decision.model_dump(),
                )

            return decision

        except Exception as first_exc:
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Return exactly one "
                        "raw JSON object matching the schema. Do not use Markdown code fences."
                    ),
                },
            ]
            retry_raw = self.llm.complete_text(retry_messages)
            try:
                data = extract_json_object(retry_raw)
                decision = AgentDecision.model_validate(data)

                if self.prompt_logger:
                    self.prompt_logger.log(
                        acdl_spec="DoitAgentStateful",
                        model=self.config.model.model,
                        messages=retry_messages,
                        raw_response=retry_raw,
                        parsed_response=decision.model_dump(),
                    )

                return decision

            except Exception as retry_exc:
                minimal_messages = _build_minimal_agent_messages(query, os.getcwd())
                minimal_raw = self.llm.complete_text(minimal_messages)
                try:
                    data = extract_json_object(minimal_raw)
                    decision = AgentDecision.model_validate(data)

                    if self.prompt_logger:
                        self.prompt_logger.log(
                            acdl_spec="DoitAgentStateful",
                            model=self.config.model.model,
                            messages=minimal_messages,
                            raw_response=minimal_raw,
                            parsed_response=decision.model_dump(),
                        )

                    return decision

                except Exception as minimal_exc:
                    deterministic_decision = _deterministic_agent_decision(query, os.getcwd())
                    if deterministic_decision is not None:
                        if self.prompt_logger:
                            self.prompt_logger.log(
                                acdl_spec="DoitAgentStateful",
                                model=self.config.model.model,
                                messages=minimal_messages,
                                raw_response=(
                                    "Deterministic fallback used after malformed model responses. "
                                    f"First raw: {raw!r}; retry raw: {retry_raw!r}; "
                                    f"minimal raw: {minimal_raw!r}"
                                ),
                                parsed_response=deterministic_decision.model_dump(),
                            )

                        return deterministic_decision

                    if self.prompt_logger:
                        self.prompt_logger.log(
                            acdl_spec="DoitAgentStateful",
                            model=self.config.model.model,
                            messages=minimal_messages,
                            raw_response=minimal_raw,
                            parsed_response=None,
                        )

                    error = (
                        "Could not parse model response as AgentDecision. "
                        f"First raw response: {raw}. First error: {first_exc}. "
                        f"Retry raw response: {retry_raw}. Retry error: {retry_exc}. "
                        f"Minimal raw response: {minimal_raw}. Minimal error: {minimal_exc}"
                    )

            return AgentDecision(
                intent="error",
                error=error,
            )

    @trace
    def clarify(self, original_query: str, decision: AgentDecision) -> AgentDecision:
        # Write directly to /dev/tty so the question is visible even when the
        # shell wrapper captures stdout with $(...).  /dev/tty is always the
        # real terminal regardless of stdout/stderr redirection.
        # Always specify encoding explicitly — WSL terminals often send UTF-8
        # but Python may default to ASCII or latin-1 on the stdin stream.
        try:
            tty = open("/dev/tty", "r+", encoding="utf-8", errors="replace")
        except OSError:
            tty = None  # fallback to sys.stdin.buffer if no tty

        def tty_print(msg: str) -> None:
            if tty:
                tty.write(msg + "\n")
                tty.flush()
            else:
                print(msg, flush=True)

        tty_print(decision.clarification_question or "I need clarification.")
        for i, option in enumerate(decision.clarification_options, start=1):
            tty_print(f"{i}. {option}")

        if tty:
            tty.write("Your answer: ")
            tty.flush()
            answer = tty.readline().strip()
            tty.close()
        else:
            # Read raw bytes and decode manually so non-UTF-8 keystrokes
            # (e.g. Hebrew keyboard input on a mis-configured locale) don't crash.
            sys.stdout.write("Your answer: ")
            sys.stdout.flush()
            raw_bytes = sys.stdin.buffer.readline()
            answer = raw_bytes.decode("utf-8", errors="replace").strip()


        messages = build_clarification_messages(
            original_query=original_query,
            clarification_question=decision.clarification_question or "",
            options=decision.clarification_options,
            user_answer=answer,
        )

        raw = self.llm.complete_text(messages)

        try:
            data = extract_json_object(raw)
            final_decision = AgentDecision.model_validate(data)

            if self.prompt_logger:
                self.prompt_logger.log(
                    acdl_spec="DoitClarification",
                    model=self.config.model.model,
                    messages=messages,
                    raw_response=raw,
                    parsed_response=final_decision.model_dump(),
                )

            return final_decision

        except Exception as exc:
            return AgentDecision(
                intent="error",
                error=f"Could not parse clarification response. Raw response: {raw}. Error: {exc}",
            )

    @trace
    def run(self, query: str) -> int:
        decision = self.decide(query)

        # If the LLM chose clarification for a query that has a known
        # deterministic answer, skip the clarification loop entirely.
        if decision.intent == "clarification":
            det = _deterministic_agent_decision(query, os.getcwd())
            if det is not None:
                decision = det

        clarification_rounds = 0
        while (
            decision.intent == "clarification"
            and clarification_rounds < self.config.agent.max_clarification_rounds
        ):
            clarification_rounds += 1
            decision = self.clarify(query, decision)

        if decision.intent == "clarification":
            msg = "Could not resolve clarification after the allowed number of rounds."
            print(msg)
            self._save_interaction(query=query, decision=decision, answer=msg)
            return 1

        if decision.intent == "error":
            msg = decision.error or "Unknown error."
            print(msg)
            self._save_interaction(query=query, decision=decision, answer=msg)
            return 1

        # Memory extraction runs AFTER decide() so the LLM has full context:
        # what the user asked, what command was chosen, and what cwd is.
        # The memory prompt instructs it to store absolute paths from that context.
        memory_actions = self.memory_service.extract_and_apply(
            query,
            cwd=os.getcwd(),
            command=decision.command,
        )

        if decision.intent in {"conversation", "answer"}:
            answer = decision.answer or decision.explanation or ""
            print(answer)

            if memory_actions:
                print(f"\nMemory actions: {', '.join(memory_actions)}")

            self._save_interaction(
                query=query,
                decision=decision,
                answer=answer,
            )
            self._maybe_update_context_summary()
            return 0

        if decision.intent != "execute_command":
            print(f"Unsupported intent: {decision.intent}")
            return 1

        if not decision.command:
            print("Model selected execute_command but did not provide a command.")
            return 1

        command = _sanitize_command(decision.command.strip())

        policy_result = self.command_policy.validate(command)
        if not policy_result.allowed:
            print("Command was not executed.")
            print(policy_result.reason)
            print(f"Proposed command: {command}")

            self._save_interaction(
                query=query,
                decision=decision,
                command=command,
                stdout="",
                stderr=policy_result.reason,
                returncode=1,
            )
            self._maybe_update_context_summary()
            return 1

        # print(command)
        print()

        safety: SafetyDecision | None = None
        user_confirmed: bool | None = None

        if self.config.agent.secure:
            safety = self.safety.classify(query, command)

            if not safety.allowed:
                print("Command was not executed.")
                print(f"Reason: {safety.explanation}")

                self._save_interaction(
                    query=query,
                    decision=decision,
                    command=command,
                    safety=safety,
                    user_confirmed=False,
                    stdout="",
                    stderr=safety.explanation,
                    returncode=1,
                )
                self._maybe_update_context_summary()
                return 1

            if safety.requires_confirmation:
                prompt_msg = (
                    f"Security check: {safety.safety_level} [{safety.source}]\n"
                    f"{safety.explanation}\n"
                    f"Command: {command}\n"
                    f"Proceed? [y/n]\n"
                )
                try:
                    with open("/dev/tty", "r+", encoding="utf-8", errors="replace") as tty:
                        tty.write(prompt_msg)
                        tty.flush()
                        answer = _normalize_confirmation_answer(tty.readline())
                        print(f"\nYour answer: {answer}")
                except OSError:
                    sys.stderr.write(prompt_msg)
                    sys.stderr.flush()
                    raw_bytes = sys.stdin.buffer.readline()
                    answer = _normalize_confirmation_answer(
                        raw_bytes.decode("utf-8", errors="replace")
                    )
                    print(f"\nYour answer: {answer}")
                
                user_confirmed = answer in {"y", "yes"}

                if not user_confirmed:
                    print("Cancelled.")
                    self._save_interaction(
                        query=query,
                        decision=decision,
                        command=command,
                        safety=safety,
                        user_confirmed=False,
                        stdout="",
                        stderr="User cancelled command.",
                        returncode=1,
                    )
                    self._maybe_update_context_summary()
                    return 1

        if memory_actions:
            print(f"\nMemory actions: {', '.join(memory_actions)}")

        if needs_parent_shell(command):
            # Shell-mutating commands (cd, export, source, alias, ...) must run
            # in the real shell. Emit a marker the wrapper will eval there.
            # We cannot capture their output, and they produce little anyway.
            print(f"DOIT_EXEC:{command}")

            self._save_interaction(
                query=query,
                decision=decision,
                command=command,
                safety=safety,
                user_confirmed=user_confirmed,
                stdout="",   # ran in the real shell, not captured here
                stderr="",
                returncode=0,
            )
            self._maybe_update_context_summary()
            return 0

        # Everything else runs in-process so we can capture stdout/stderr/rc.
        # This is what makes output-awareness ("why did that fail?", "which is
        # safe to delete?") work: the captured output is saved to history.
        # The subprocess inherits Python's cwd, which already tracks the user's
        # terminal (prior `doit cd` went through the DOIT_EXEC path above).
        print(f"\033[0;36m▶ {command}\033[0m")
        result = run_shell(
            command,
            shell=self.config.agent.shell,
            timeout=self.config.agent.command_timeout,
        )
        stdout = str(result["stdout"])
        stderr = str(result["stderr"])
        returncode = int(result["returncode"])

        # Always emit newline-terminated lines: the shell wrapper reads our
        # output with `while read`, which silently drops a trailing line that
        # has no newline. Never leave the user with a blank, silent result.
        def _emit(text: str, stream) -> None:
            if not text:
                return
            stream.write(text if text.endswith("\n") else text + "\n")
            stream.flush()

        _emit(stdout, sys.stdout)
        _emit(stderr, sys.stderr)
        if not stdout and not stderr:
            print(f"(command exited with code {returncode}, no output)")

        self._save_interaction(
            query=query,
            decision=decision,
            command=command,
            safety=safety,
            user_confirmed=user_confirmed,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )

        self._maybe_update_context_summary()
        return returncode

    @trace
    def _maybe_update_context_summary(self) -> None:
        if not self.config.context.summaries_enabled:
            return

        records = self.state_store.get_recent_interactions(
            limit=1000,
            session_id=self.session_id,
        )
        self.context_summary_service.maybe_update_summary(
            session_id=self.session_id,
            records=records,
            threshold=self.config.context.summary_threshold,
            recent_keep=self.config.context.summary_recent_keep,
        )

    @trace
    def _save_interaction(
        self,
        *,
        query: str,
        decision: AgentDecision,
        command: str | None = None,
        answer: str | None = None,
        safety: SafetyDecision | None = None,
        user_confirmed: bool | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        returncode: int | None = None,
    ) -> None:
        record = InteractionRecord.create(
            session_id=self.session_id,
            cwd=os.getcwd(),
            user_query=query,
            intent=decision.intent,
            command=command or decision.command,
            answer=answer or decision.answer,
            explanation=decision.explanation,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            safety_level=safety.safety_level if safety else None,
            safety_requires_confirmation=safety.requires_confirmation if safety else None,
            user_confirmed=user_confirmed,
            model=self.config.model.model,
        )

        self.state_store.append_interaction(record)
