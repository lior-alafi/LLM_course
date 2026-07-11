# Full ACDL package: prompt-level + scenario-level

This package contains two kinds of ACDL files.

## 1. Prompt-level ACDL

Directory:

```text
acdl/prompt/
```

These files correspond to actual LLM calls in the code and should be used by
`PromptLogger`:

- `new_doit_agent_stateful.acdl`
- `new_doit_safety_check.acdl`
- `new_doit_memory_extraction.acdl`
- `new_doit_clarification.acdl`
- `new_doit_context_summary.acdl`

PromptLogger should map only these files, because a single prompt log represents
one LLM call.

## 2. Scenario-level ACDL

Directory:

```text
acdl/scenarios/
```

These files document complete end-to-end invocations of `doit`, including what
the runtime/tool layer did:

- config/state/memory/shell-history loading
- main LLM decision
- JSON parsing
- single-command policy validation
- rule-based safety
- optional LLM safety fallback
- optional user confirmation
- shell execution
- stdout/stderr/returncode capture
- memory extraction
- state/log/summary updates

These files are for report documentation and generated visual representations.
They are not used as the `acdl_spec` for a single LLM call.

## Why both are needed

The assignment explicitly asks to document the context sent to the LLM using ACDL,
but it also says that ACDL should make it possible to compare the report, logs,
prompts, memory/session state, and code behavior. Therefore, the cleanest design is:

```text
prompt-level ACDL   = what each LLM call sees and returns
scenario-level ACDL = what happened during a full doit invocation
```

This avoids pretending that runtime events such as shell execution were part of
the main LLM prompt, while still documenting the complete tool behavior.
