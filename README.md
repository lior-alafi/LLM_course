# LLM Course

Three projects from the LLM course (Bar-Ilan, 2025/2026), each built from a public assignment spec by [Yoav Goldberg](https://yoavg.github.io/llm-class-2025-2026/).

- **[`transformer-language-model/`](transformer-language-model)** — a decoder-only transformer language model built from scratch (attention, multi-head attention, causal masking, residual/norm blocks) at the character level, trained on English (Shakespeare) and Hebrew (Bialik/Rachel poetry) corpora, with attention-pattern analysis. Report: [`report.pdf`](transformer-language-model/report.pdf). [Assignment spec](https://yoavg.github.io/llm-class-2025-2026/ass1-transformers/).
- **[`architecture-tokenizers-finetuning/`](architecture-tokenizers-finetuning)** — comparative analysis of 10 open LLMs' architectures and tokenizers, constrained decoding to force Hebrew-only output from an English prompt, and LoRA fine-tuning of Qwen2.5-1.5B to answer English questions in Hebrew. Report: [`Report.pdf`](architecture-tokenizers-finetuning/Report.pdf). [Assignment spec](https://yoavg.github.io/llm-class-2025-2026/ass2-architectural-choices-tokenizers-finetuning/).
- **[`agentic-shell/`](agentic-shell)** — `doit`, an LLM-powered shell agent that turns natural language into shell commands: destructive-action confirmation, multi-model support (API + local, tool-calling + non-tool-calling) via LiteLLM, persistent cross-session memory, shell-history awareness, and multi-terminal session handling. Report: [`report.pdf`](agentic-shell/report.pdf). [Assignment spec](https://yoavg.github.io/llm-class-2025-2026/ass3-agentic-shell/).

Each project is a standalone `uv` project — see its own README for setup and run instructions.
