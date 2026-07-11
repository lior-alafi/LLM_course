# Transformer Language Model from Scratch

A decoder-only transformer language model implemented from first principles (no `nn.TransformerDecoder` / attention library) — built to understand the architecture, not just call it.

[Assignment spec](https://yoavg.github.io/llm-class-2025-2026/ass1-transformers/) · code in [`code-and-data/code/`](code-and-data/code)

## What's implemented

- **Attention** (`attention.py`) — Q/K/V projections, scaled dot-product attention scores, causal masking, multi-head attention with head concatenation and output projection.
- **Transformer block** (`transformer.py`) — residual connections around attention/MLP sublayers, configurable pre-norm/post-norm layer normalization.
- **Language model** (`lm.py`) — token + positional embeddings, weight initialization, top-k / temperature sampling for generation (`better_sample_continuation`).
- **Data pipeline** (`data.py`) — character-level tokenizer, batching into input/output pairs for next-token prediction.
- **Training** (`main.py`, `training.ipynb`) — cross-entropy training loop over both corpora.
- **Hyperparameter search** (`params_search.py`) — sweeps over depth/width/heads/embedding-dim/context-length/learning-rate.
- **MLP** (`mlp.py`), **tests** (`tests.py`), **visualization** (`visualize.py`, `data_analysis.py`).

## Data

- English: the Shakespeare corpus (character-level).
- Hebrew: ~1.5M characters of Bialik and Rachel poetry.

## Run

```bash
cd code-and-data/code
uv run main.py
```
