from __future__ import annotations

import os
from pydantic import ValidationError

from doit_agent.config import DoitConfig
from doit_agent.json_utils import extract_json_object
from doit_agent.llm_client import LLMClient
from doit_agent.memory.store import MemoryStore
from doit_agent.memory_service import MemoryService
from doit_agent.prompt_log import PromptLogger
from doit_agent.prompts import build_agent_messages
from doit_agent.safety import SafetyService
from doit_agent.schemas import AgentDecision, SafetyDecision
from doit_agent.shell_exec import run_shell
from doit_agent.state.models import InteractionRecord, get_session_id
from doit_agent.state.store import StateStore


class DoitAgent:
    def __init__(
        self,
        *,
        config: DoitConfig,
        llm: LLMClient,
        state_store: StateStore,
        memory_store: MemoryStore,
        prompt_logger: PromptLogger | None = None,
    ):
        self.config = config
        self.llm = llm
        self.state_store = state_store
        self.memory_store = memory_store
        self.prompt_logger = prompt_logger

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

        self.session_id = get_session_id()

    def decide(self, query: str) -> AgentDecision:
        same_session_history = self.state_store.get_recent_interactions(
            limit=8,
            session_id=self.session_id,
        )

        # Keep global history small to avoid confusing the model.
        global_history = self.state_store.get_recent_interactions(
            limit=3,
            session_id=None,
        )

        memories = self.memory_store.list_memories()

        messages = build_agent_messages(
            user_query=query,
            secure=self.config.agent.secure,
            cwd=os.getcwd(),
            session_id=self.session_id,
            same_session_history=same_session_history,
            global_history=global_history,
            memories=memories,
        )

        raw = self.llm.complete_text(messages)

        try:
            data = extract_json_object(raw)
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

        except Exception as exc:
            return AgentDecision(
                intent="error",
                error=f"Could not parse model response as AgentDecision. Raw response: {raw}. Error: {exc}",
            )

    def run(self, query: str) -> int:
        # Memory extraction is separate from command planning.
        # This allows commands such as:
        # "move to ~/school/llms/ass3. this is my LLM class project folder"
        # to both execute and store a memory.
        stored_memory_keys = self.memory_service.extract_and_store(query)

        decision = self.decide(query)

        if decision.intent == "error":
            msg = decision.error or "Unknown error."
            print(msg)

            self._save_interaction(
                query=query,
                decision=decision,
                answer=msg,
            )

            return 1

        if decision.intent in {"conversation", "answer"}:
            answer = decision.answer or decision.explanation or ""
            print(answer)

            if stored_memory_keys:
                print(f"\nStored memory: {', '.join(stored_memory_keys)}")

            self._save_interaction(
                query=query,
                decision=decision,
                answer=answer,
            )

            return 0

        if decision.intent == "clarification":
            print(decision.clarification_question or "I need clarification.")

            for i, option in enumerate(decision.clarification_options, start=1):
                print(f"{i}. {option}")

            self._save_interaction(
                query=query,
                decision=decision,
                answer=decision.clarification_question,
            )

            return 0

        if decision.intent != "execute_command":
            print(f"Unsupported intent: {decision.intent}")
            return 1

        if not decision.command:
            print("Model selected execute_command but did not provide a command.")
            return 1

        command = decision.command
        print(command)
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

                return 1

            if safety.requires_confirmation:
                print(f"Security check: {safety.safety_level} [{safety.source}]")
                print(safety.explanation)
                answer = input("Proceed? [y/N] ").strip().lower()
                user_confirmed = answer == "y"

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

                    return 1

        result = run_shell(command, shell=self.config.agent.shell)

        if result["stdout"]:
            print(result["stdout"], end="")

        if result["stderr"]:
            print("Error Output:", result["stderr"], end="")

        if stored_memory_keys:
            print(f"\nStored memory: {', '.join(stored_memory_keys)}")

        self._save_interaction(
            query=query,
            decision=decision,
            command=command,
            safety=safety,
            user_confirmed=user_confirmed,
            stdout=str(result["stdout"]),
            stderr=str(result["stderr"]),
            returncode=int(result["returncode"]),
        )

        return int(result["returncode"])

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
