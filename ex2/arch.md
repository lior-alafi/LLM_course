# meta-llama/Llama-3.1-8B-Instruct

## Parameter shapes

```text
model.embed_tokens (128256, 4096)
model.layers.0.self_attn.q_proj (4096, 4096)
model.layers.0.self_attn.k_proj (1024, 4096)
model.layers.0.self_attn.v_proj (1024, 4096)
model.layers.0.self_attn.o_proj (4096, 4096)
model.layers.0.mlp.gate_proj (14336, 4096)
model.layers.0.mlp.up_proj (14336, 4096)
model.layers.0.mlp.down_proj (4096, 14336)
model.layers.0.input_layernorm (4096,)
model.layers.0.post_attention_layernorm (4096,)
model.layers.1.self_attn.q_proj (4096, 4096)
model.layers.1.self_attn.k_proj (1024, 4096)
model.layers.1.self_attn.v_proj (1024, 4096)
model.layers.1.self_attn.o_proj (4096, 4096)
model.layers.1.mlp.gate_proj (14336, 4096)
model.layers.1.mlp.up_proj (14336, 4096)
model.layers.1.mlp.down_proj (4096, 14336)
model.layers.1.input_layernorm (4096,)
model.layers.1.post_attention_layernorm (4096,)
model.layers.2.self_attn.q_proj (4096, 4096)
model.layers.2.self_attn.k_proj (1024, 4096)
model.layers.2.self_attn.v_proj (1024, 4096)
model.layers.2.self_attn.o_proj (4096, 4096)
model.layers.2.mlp.gate_proj (14336, 4096)
model.layers.2.mlp.up_proj (14336, 4096)
model.layers.2.mlp.down_proj (4096, 14336)
model.layers.2.input_layernorm (4096,)
model.layers.2.post_attention_layernorm (4096,)
model.layers.3.self_attn.q_proj (4096, 4096)
model.layers.3.self_attn.k_proj (1024, 4096)
model.layers.3.self_attn.v_proj (1024, 4096)
model.layers.3.self_attn.o_proj (4096, 4096)
model.layers.3.mlp.gate_proj (14336, 4096)
model.layers.3.mlp.up_proj (14336, 4096)
model.layers.3.mlp.down_proj (4096, 14336)
model.layers.3.input_layernorm (4096,)
model.layers.3.post_attention_layernorm (4096,)
model.layers.4.self_attn.q_proj (4096, 4096)
model.layers.4.self_attn.k_proj (1024, 4096)
model.layers.4.self_attn.v_proj (1024, 4096)
model.layers.4.self_attn.o_proj (4096, 4096)
model.layers.4.mlp.gate_proj (14336, 4096)
model.layers.4.mlp.up_proj (14336, 4096)
model.layers.4.mlp.down_proj (4096, 14336)
model.layers.4.input_layernorm (4096,)
model.layers.4.post_attention_layernorm (4096,)
model.layers.5.self_attn.q_proj (4096, 4096)
model.layers.5.self_attn.k_proj (1024, 4096)
model.layers.5.self_attn.v_proj (1024, 4096)
model.layers.5.self_attn.o_proj (4096, 4096)
model.layers.5.mlp.gate_proj (14336, 4096)
model.layers.5.mlp.up_proj (14336, 4096)
model.layers.5.mlp.down_proj (4096, 14336)
model.layers.5.input_layernorm (4096,)
model.layers.5.post_attention_layernorm (4096,)
model.layers.6.self_attn.q_proj (4096, 4096)
model.layers.6.self_attn.k_proj (1024, 4096)
model.layers.6.self_attn.v_proj (1024, 4096)
model.layers.6.self_attn.o_proj (4096, 4096)
model.layers.6.mlp.gate_proj (14336, 4096)
model.layers.6.mlp.up_proj (14336, 4096)
model.layers.6.mlp.down_proj (4096, 14336)
model.layers.6.input_layernorm (4096,)
model.layers.6.post_attention_layernorm (4096,)
model.layers.7.self_attn.q_proj (4096, 4096)
model.layers.7.self_attn.k_proj (1024, 4096)
model.layers.7.self_attn.v_proj (1024, 4096)
model.layers.7.self_attn.o_proj (4096, 4096)
model.layers.7.mlp.gate_proj (14336, 4096)
model.layers.7.mlp.up_proj (14336, 4096)
model.layers.7.mlp.down_proj (4096, 14336)
model.layers.7.input_layernorm (4096,)
model.layers.7.post_attention_layernorm (4096,)
model.layers.8.self_attn.q_proj (4096, 4096)
model.layers.8.self_attn.k_proj (1024, 4096)
model.layers.8.self_attn.v_proj (1024, 4096)
model.layers.8.self_attn.o_proj (4096, 4096)
model.layers.8.mlp.gate_proj (14336, 4096)
model.layers.8.mlp.up_proj (14336, 4096)
model.layers.8.mlp.down_proj (4096, 14336)
model.layers.8.input_layernorm (4096,)
model.layers.8.post_attention_layernorm (4096,)
model.layers.9.self_attn.q_proj (4096, 4096)
model.layers.9.self_attn.k_proj (1024, 4096)
model.layers.9.self_attn.v_proj (1024, 4096)
model.layers.9.self_attn.o_proj (4096, 4096)
model.layers.9.mlp.gate_proj (14336, 4096)
model.layers.9.mlp.up_proj (14336, 4096)
model.layers.9.mlp.down_proj (4096, 14336)
model.layers.9.input_layernorm (4096,)
model.layers.9.post_attention_layernorm (4096,)
model.layers.10.self_attn.q_proj (4096, 4096)
model.layers.10.self_attn.k_proj (1024, 4096)
model.layers.10.self_attn.v_proj (1024, 4096)
model.layers.10.self_attn.o_proj (4096, 4096)
model.layers.10.mlp.gate_proj (14336, 4096)
model.layers.10.mlp.up_proj (14336, 4096)
model.layers.10.mlp.down_proj (4096, 14336)
model.layers.10.input_layernorm (4096,)
model.layers.10.post_attention_layernorm (4096,)
model.layers.11.self_attn.q_proj (4096, 4096)
model.layers.11.self_attn.k_proj (1024, 4096)
model.layers.11.self_attn.v_proj (1024, 4096)
model.layers.11.self_attn.o_proj (4096, 4096)
model.layers.11.mlp.gate_proj (14336, 4096)
model.layers.11.mlp.up_proj (14336, 4096)
model.layers.11.mlp.down_proj (4096, 14336)
model.layers.11.input_layernorm (4096,)
model.layers.11.post_attention_layernorm (4096,)
model.layers.12.self_attn.q_proj (4096, 4096)
model.layers.12.self_attn.k_proj (1024, 4096)
model.layers.12.self_attn.v_proj (1024, 4096)
model.layers.12.self_attn.o_proj (4096, 4096)
model.layers.12.mlp.gate_proj (14336, 4096)
model.layers.12.mlp.up_proj (14336, 4096)
model.layers.12.mlp.down_proj (4096, 14336)
model.layers.12.input_layernorm (4096,)
model.layers.12.post_attention_layernorm (4096,)
model.layers.13.self_attn.q_proj (4096, 4096)
model.layers.13.self_attn.k_proj (1024, 4096)
model.layers.13.self_attn.v_proj (1024, 4096)
model.layers.13.self_attn.o_proj (4096, 4096)
model.layers.13.mlp.gate_proj (14336, 4096)
model.layers.13.mlp.up_proj (14336, 4096)
model.layers.13.mlp.down_proj (4096, 14336)
model.layers.13.input_layernorm (4096,)
model.layers.13.post_attention_layernorm (4096,)
model.layers.14.self_attn.q_proj (4096, 4096)
model.layers.14.self_attn.k_proj (1024, 4096)
model.layers.14.self_attn.v_proj (1024, 4096)
model.layers.14.self_attn.o_proj (4096, 4096)
model.layers.14.mlp.gate_proj (14336, 4096)
model.layers.14.mlp.up_proj (14336, 4096)
model.layers.14.mlp.down_proj (4096, 14336)
model.layers.14.input_layernorm (4096,)
model.layers.14.post_attention_layernorm (4096,)
model.layers.15.self_attn.q_proj (4096, 4096)
model.layers.15.self_attn.k_proj (1024, 4096)
model.layers.15.self_attn.v_proj (1024, 4096)
model.layers.15.self_attn.o_proj (4096, 4096)
model.layers.15.mlp.gate_proj (14336, 4096)
model.layers.15.mlp.up_proj (14336, 4096)
model.layers.15.mlp.down_proj (4096, 14336)
model.layers.15.input_layernorm (4096,)
model.layers.15.post_attention_layernorm (4096,)
model.layers.16.self_attn.q_proj (4096, 4096)
model.layers.16.self_attn.k_proj (1024, 4096)
model.layers.16.self_attn.v_proj (1024, 4096)
model.layers.16.self_attn.o_proj (4096, 4096)
model.layers.16.mlp.gate_proj (14336, 4096)
model.layers.16.mlp.up_proj (14336, 4096)
model.layers.16.mlp.down_proj (4096, 14336)
model.layers.16.input_layernorm (4096,)
model.layers.16.post_attention_layernorm (4096,)
model.layers.17.self_attn.q_proj (4096, 4096)
model.layers.17.self_attn.k_proj (1024, 4096)
model.layers.17.self_attn.v_proj (1024, 4096)
model.layers.17.self_attn.o_proj (4096, 4096)
model.layers.17.mlp.gate_proj (14336, 4096)
model.layers.17.mlp.up_proj (14336, 4096)
model.layers.17.mlp.down_proj (4096, 14336)
model.layers.17.input_layernorm (4096,)
model.layers.17.post_attention_layernorm (4096,)
model.layers.18.self_attn.q_proj (4096, 4096)
model.layers.18.self_attn.k_proj (1024, 4096)
model.layers.18.self_attn.v_proj (1024, 4096)
model.layers.18.self_attn.o_proj (4096, 4096)
model.layers.18.mlp.gate_proj (14336, 4096)
model.layers.18.mlp.up_proj (14336, 4096)
model.layers.18.mlp.down_proj (4096, 14336)
model.layers.18.input_layernorm (4096,)
model.layers.18.post_attention_layernorm (4096,)
model.layers.19.self_attn.q_proj (4096, 4096)
model.layers.19.self_attn.k_proj (1024, 4096)
model.layers.19.self_attn.v_proj (1024, 4096)
model.layers.19.self_attn.o_proj (4096, 4096)
model.layers.19.mlp.gate_proj (14336, 4096)
model.layers.19.mlp.up_proj (14336, 4096)
model.layers.19.mlp.down_proj (4096, 14336)
model.layers.19.input_layernorm (4096,)
model.layers.19.post_attention_layernorm (4096,)
model.layers.20.self_attn.q_proj (4096, 4096)
model.layers.20.self_attn.k_proj (1024, 4096)
model.layers.20.self_attn.v_proj (1024, 4096)
model.layers.20.self_attn.o_proj (4096, 4096)
model.layers.20.mlp.gate_proj (14336, 4096)
model.layers.20.mlp.up_proj (14336, 4096)
model.layers.20.mlp.down_proj (4096, 14336)
model.layers.20.input_layernorm (4096,)
model.layers.20.post_attention_layernorm (4096,)
model.layers.21.self_attn.q_proj (4096, 4096)
model.layers.21.self_attn.k_proj (1024, 4096)
model.layers.21.self_attn.v_proj (1024, 4096)
model.layers.21.self_attn.o_proj (4096, 4096)
model.layers.21.mlp.gate_proj (14336, 4096)
model.layers.21.mlp.up_proj (14336, 4096)
model.layers.21.mlp.down_proj (4096, 14336)
model.layers.21.input_layernorm (4096,)
model.layers.21.post_attention_layernorm (4096,)
model.layers.22.self_attn.q_proj (4096, 4096)
model.layers.22.self_attn.k_proj (1024, 4096)
model.layers.22.self_attn.v_proj (1024, 4096)
model.layers.22.self_attn.o_proj (4096, 4096)
model.layers.22.mlp.gate_proj (14336, 4096)
model.layers.22.mlp.up_proj (14336, 4096)
model.layers.22.mlp.down_proj (4096, 14336)
model.layers.22.input_layernorm (4096,)
model.layers.22.post_attention_layernorm (4096,)
model.layers.23.self_attn.q_proj (4096, 4096)
model.layers.23.self_attn.k_proj (1024, 4096)
model.layers.23.self_attn.v_proj (1024, 4096)
model.layers.23.self_attn.o_proj (4096, 4096)
model.layers.23.mlp.gate_proj (14336, 4096)
model.layers.23.mlp.up_proj (14336, 4096)
model.layers.23.mlp.down_proj (4096, 14336)
model.layers.23.input_layernorm (4096,)
model.layers.23.post_attention_layernorm (4096,)
model.layers.24.self_attn.q_proj (4096, 4096)
model.layers.24.self_attn.k_proj (1024, 4096)
model.layers.24.self_attn.v_proj (1024, 4096)
model.layers.24.self_attn.o_proj (4096, 4096)
model.layers.24.mlp.gate_proj (14336, 4096)
model.layers.24.mlp.up_proj (14336, 4096)
model.layers.24.mlp.down_proj (4096, 14336)
model.layers.24.input_layernorm (4096,)
model.layers.24.post_attention_layernorm (4096,)
model.layers.25.self_attn.q_proj (4096, 4096)
model.layers.25.self_attn.k_proj (1024, 4096)
model.layers.25.self_attn.v_proj (1024, 4096)
model.layers.25.self_attn.o_proj (4096, 4096)
model.layers.25.mlp.gate_proj (14336, 4096)
model.layers.25.mlp.up_proj (14336, 4096)
model.layers.25.mlp.down_proj (4096, 14336)
model.layers.25.input_layernorm (4096,)
model.layers.25.post_attention_layernorm (4096,)
model.layers.26.self_attn.q_proj (4096, 4096)
model.layers.26.self_attn.k_proj (1024, 4096)
model.layers.26.self_attn.v_proj (1024, 4096)
model.layers.26.self_attn.o_proj (4096, 4096)
model.layers.26.mlp.gate_proj (14336, 4096)
model.layers.26.mlp.up_proj (14336, 4096)
model.layers.26.mlp.down_proj (4096, 14336)
model.layers.26.input_layernorm (4096,)
model.layers.26.post_attention_layernorm (4096,)
model.layers.27.self_attn.q_proj (4096, 4096)
model.layers.27.self_attn.k_proj (1024, 4096)
model.layers.27.self_attn.v_proj (1024, 4096)
model.layers.27.self_attn.o_proj (4096, 4096)
model.layers.27.mlp.gate_proj (14336, 4096)
model.layers.27.mlp.up_proj (14336, 4096)
model.layers.27.mlp.down_proj (4096, 14336)
model.layers.27.input_layernorm (4096,)
model.layers.27.post_attention_layernorm (4096,)
model.layers.28.self_attn.q_proj (4096, 4096)
model.layers.28.self_attn.k_proj (1024, 4096)
model.layers.28.self_attn.v_proj (1024, 4096)
model.layers.28.self_attn.o_proj (4096, 4096)
model.layers.28.mlp.gate_proj (14336, 4096)
model.layers.28.mlp.up_proj (14336, 4096)
model.layers.28.mlp.down_proj (4096, 14336)
model.layers.28.input_layernorm (4096,)
model.layers.28.post_attention_layernorm (4096,)
model.layers.29.self_attn.q_proj (4096, 4096)
model.layers.29.self_attn.k_proj (1024, 4096)
model.layers.29.self_attn.v_proj (1024, 4096)
model.layers.29.self_attn.o_proj (4096, 4096)
model.layers.29.mlp.gate_proj (14336, 4096)
model.layers.29.mlp.up_proj (14336, 4096)
model.layers.29.mlp.down_proj (4096, 14336)
model.layers.29.input_layernorm (4096,)
model.layers.29.post_attention_layernorm (4096,)
model.layers.30.self_attn.q_proj (4096, 4096)
model.layers.30.self_attn.k_proj (1024, 4096)
model.layers.30.self_attn.v_proj (1024, 4096)
model.layers.30.self_attn.o_proj (4096, 4096)
model.layers.30.mlp.gate_proj (14336, 4096)
model.layers.30.mlp.up_proj (14336, 4096)
model.layers.30.mlp.down_proj (4096, 14336)
model.layers.30.input_layernorm (4096,)
model.layers.30.post_attention_layernorm (4096,)
model.layers.31.self_attn.q_proj (4096, 4096)
model.layers.31.self_attn.k_proj (1024, 4096)
model.layers.31.self_attn.v_proj (1024, 4096)
model.layers.31.self_attn.o_proj (4096, 4096)
model.layers.31.mlp.gate_proj (14336, 4096)
model.layers.31.mlp.up_proj (14336, 4096)
model.layers.31.mlp.down_proj (4096, 14336)
model.layers.31.input_layernorm (4096,)
model.layers.31.post_attention_layernorm (4096,)
model.norm (4096,)
lm_head (128256, 4096)
```

## config.json

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "use_cache": true,
  "vocab_size": 128256
}
```

---

# mistralai/Mistral-7B-Instruct-v0.3

## Parameter shapes

```text
model.embed_tokens (32768, 4096)
model.layers.0.self_attn.q_proj (4096, 4096)
model.layers.0.self_attn.k_proj (1024, 4096)
model.layers.0.self_attn.v_proj (1024, 4096)
model.layers.0.self_attn.o_proj (4096, 4096)
model.layers.0.mlp.gate_proj (14336, 4096)
model.layers.0.mlp.up_proj (14336, 4096)
model.layers.0.mlp.down_proj (4096, 14336)
model.layers.0.input_layernorm (4096,)
model.layers.0.post_attention_layernorm (4096,)
model.layers.1.self_attn.q_proj (4096, 4096)
model.layers.1.self_attn.k_proj (1024, 4096)
model.layers.1.self_attn.v_proj (1024, 4096)
model.layers.1.self_attn.o_proj (4096, 4096)
model.layers.1.mlp.gate_proj (14336, 4096)
model.layers.1.mlp.up_proj (14336, 4096)
model.layers.1.mlp.down_proj (4096, 14336)
model.layers.1.input_layernorm (4096,)
model.layers.1.post_attention_layernorm (4096,)
model.layers.2.self_attn.q_proj (4096, 4096)
model.layers.2.self_attn.k_proj (1024, 4096)
model.layers.2.self_attn.v_proj (1024, 4096)
model.layers.2.self_attn.o_proj (4096, 4096)
model.layers.2.mlp.gate_proj (14336, 4096)
model.layers.2.mlp.up_proj (14336, 4096)
model.layers.2.mlp.down_proj (4096, 14336)
model.layers.2.input_layernorm (4096,)
model.layers.2.post_attention_layernorm (4096,)
model.layers.3.self_attn.q_proj (4096, 4096)
model.layers.3.self_attn.k_proj (1024, 4096)
model.layers.3.self_attn.v_proj (1024, 4096)
model.layers.3.self_attn.o_proj (4096, 4096)
model.layers.3.mlp.gate_proj (14336, 4096)
model.layers.3.mlp.up_proj (14336, 4096)
model.layers.3.mlp.down_proj (4096, 14336)
model.layers.3.input_layernorm (4096,)
model.layers.3.post_attention_layernorm (4096,)
model.layers.4.self_attn.q_proj (4096, 4096)
model.layers.4.self_attn.k_proj (1024, 4096)
model.layers.4.self_attn.v_proj (1024, 4096)
model.layers.4.self_attn.o_proj (4096, 4096)
model.layers.4.mlp.gate_proj (14336, 4096)
model.layers.4.mlp.up_proj (14336, 4096)
model.layers.4.mlp.down_proj (4096, 14336)
model.layers.4.input_layernorm (4096,)
model.layers.4.post_attention_layernorm (4096,)
model.layers.5.self_attn.q_proj (4096, 4096)
model.layers.5.self_attn.k_proj (1024, 4096)
model.layers.5.self_attn.v_proj (1024, 4096)
model.layers.5.self_attn.o_proj (4096, 4096)
model.layers.5.mlp.gate_proj (14336, 4096)
model.layers.5.mlp.up_proj (14336, 4096)
model.layers.5.mlp.down_proj (4096, 14336)
model.layers.5.input_layernorm (4096,)
model.layers.5.post_attention_layernorm (4096,)
model.layers.6.self_attn.q_proj (4096, 4096)
model.layers.6.self_attn.k_proj (1024, 4096)
model.layers.6.self_attn.v_proj (1024, 4096)
model.layers.6.self_attn.o_proj (4096, 4096)
model.layers.6.mlp.gate_proj (14336, 4096)
model.layers.6.mlp.up_proj (14336, 4096)
model.layers.6.mlp.down_proj (4096, 14336)
model.layers.6.input_layernorm (4096,)
model.layers.6.post_attention_layernorm (4096,)
model.layers.7.self_attn.q_proj (4096, 4096)
model.layers.7.self_attn.k_proj (1024, 4096)
model.layers.7.self_attn.v_proj (1024, 4096)
model.layers.7.self_attn.o_proj (4096, 4096)
model.layers.7.mlp.gate_proj (14336, 4096)
model.layers.7.mlp.up_proj (14336, 4096)
model.layers.7.mlp.down_proj (4096, 14336)
model.layers.7.input_layernorm (4096,)
model.layers.7.post_attention_layernorm (4096,)
model.layers.8.self_attn.q_proj (4096, 4096)
model.layers.8.self_attn.k_proj (1024, 4096)
model.layers.8.self_attn.v_proj (1024, 4096)
model.layers.8.self_attn.o_proj (4096, 4096)
model.layers.8.mlp.gate_proj (14336, 4096)
model.layers.8.mlp.up_proj (14336, 4096)
model.layers.8.mlp.down_proj (4096, 14336)
model.layers.8.input_layernorm (4096,)
model.layers.8.post_attention_layernorm (4096,)
model.layers.9.self_attn.q_proj (4096, 4096)
model.layers.9.self_attn.k_proj (1024, 4096)
model.layers.9.self_attn.v_proj (1024, 4096)
model.layers.9.self_attn.o_proj (4096, 4096)
model.layers.9.mlp.gate_proj (14336, 4096)
model.layers.9.mlp.up_proj (14336, 4096)
model.layers.9.mlp.down_proj (4096, 14336)
model.layers.9.input_layernorm (4096,)
model.layers.9.post_attention_layernorm (4096,)
model.layers.10.self_attn.q_proj (4096, 4096)
model.layers.10.self_attn.k_proj (1024, 4096)
model.layers.10.self_attn.v_proj (1024, 4096)
model.layers.10.self_attn.o_proj (4096, 4096)
model.layers.10.mlp.gate_proj (14336, 4096)
model.layers.10.mlp.up_proj (14336, 4096)
model.layers.10.mlp.down_proj (4096, 14336)
model.layers.10.input_layernorm (4096,)
model.layers.10.post_attention_layernorm (4096,)
model.layers.11.self_attn.q_proj (4096, 4096)
model.layers.11.self_attn.k_proj (1024, 4096)
model.layers.11.self_attn.v_proj (1024, 4096)
model.layers.11.self_attn.o_proj (4096, 4096)
model.layers.11.mlp.gate_proj (14336, 4096)
model.layers.11.mlp.up_proj (14336, 4096)
model.layers.11.mlp.down_proj (4096, 14336)
model.layers.11.input_layernorm (4096,)
model.layers.11.post_attention_layernorm (4096,)
model.layers.12.self_attn.q_proj (4096, 4096)
model.layers.12.self_attn.k_proj (1024, 4096)
model.layers.12.self_attn.v_proj (1024, 4096)
model.layers.12.self_attn.o_proj (4096, 4096)
model.layers.12.mlp.gate_proj (14336, 4096)
model.layers.12.mlp.up_proj (14336, 4096)
model.layers.12.mlp.down_proj (4096, 14336)
model.layers.12.input_layernorm (4096,)
model.layers.12.post_attention_layernorm (4096,)
model.layers.13.self_attn.q_proj (4096, 4096)
model.layers.13.self_attn.k_proj (1024, 4096)
model.layers.13.self_attn.v_proj (1024, 4096)
model.layers.13.self_attn.o_proj (4096, 4096)
model.layers.13.mlp.gate_proj (14336, 4096)
model.layers.13.mlp.up_proj (14336, 4096)
model.layers.13.mlp.down_proj (4096, 14336)
model.layers.13.input_layernorm (4096,)
model.layers.13.post_attention_layernorm (4096,)
model.layers.14.self_attn.q_proj (4096, 4096)
model.layers.14.self_attn.k_proj (1024, 4096)
model.layers.14.self_attn.v_proj (1024, 4096)
model.layers.14.self_attn.o_proj (4096, 4096)
model.layers.14.mlp.gate_proj (14336, 4096)
model.layers.14.mlp.up_proj (14336, 4096)
model.layers.14.mlp.down_proj (4096, 14336)
model.layers.14.input_layernorm (4096,)
model.layers.14.post_attention_layernorm (4096,)
model.layers.15.self_attn.q_proj (4096, 4096)
model.layers.15.self_attn.k_proj (1024, 4096)
model.layers.15.self_attn.v_proj (1024, 4096)
model.layers.15.self_attn.o_proj (4096, 4096)
model.layers.15.mlp.gate_proj (14336, 4096)
model.layers.15.mlp.up_proj (14336, 4096)
model.layers.15.mlp.down_proj (4096, 14336)
model.layers.15.input_layernorm (4096,)
model.layers.15.post_attention_layernorm (4096,)
model.layers.16.self_attn.q_proj (4096, 4096)
model.layers.16.self_attn.k_proj (1024, 4096)
model.layers.16.self_attn.v_proj (1024, 4096)
model.layers.16.self_attn.o_proj (4096, 4096)
model.layers.16.mlp.gate_proj (14336, 4096)
model.layers.16.mlp.up_proj (14336, 4096)
model.layers.16.mlp.down_proj (4096, 14336)
model.layers.16.input_layernorm (4096,)
model.layers.16.post_attention_layernorm (4096,)
model.layers.17.self_attn.q_proj (4096, 4096)
model.layers.17.self_attn.k_proj (1024, 4096)
model.layers.17.self_attn.v_proj (1024, 4096)
model.layers.17.self_attn.o_proj (4096, 4096)
model.layers.17.mlp.gate_proj (14336, 4096)
model.layers.17.mlp.up_proj (14336, 4096)
model.layers.17.mlp.down_proj (4096, 14336)
model.layers.17.input_layernorm (4096,)
model.layers.17.post_attention_layernorm (4096,)
model.layers.18.self_attn.q_proj (4096, 4096)
model.layers.18.self_attn.k_proj (1024, 4096)
model.layers.18.self_attn.v_proj (1024, 4096)
model.layers.18.self_attn.o_proj (4096, 4096)
model.layers.18.mlp.gate_proj (14336, 4096)
model.layers.18.mlp.up_proj (14336, 4096)
model.layers.18.mlp.down_proj (4096, 14336)
model.layers.18.input_layernorm (4096,)
model.layers.18.post_attention_layernorm (4096,)
model.layers.19.self_attn.q_proj (4096, 4096)
model.layers.19.self_attn.k_proj (1024, 4096)
model.layers.19.self_attn.v_proj (1024, 4096)
model.layers.19.self_attn.o_proj (4096, 4096)
model.layers.19.mlp.gate_proj (14336, 4096)
model.layers.19.mlp.up_proj (14336, 4096)
model.layers.19.mlp.down_proj (4096, 14336)
model.layers.19.input_layernorm (4096,)
model.layers.19.post_attention_layernorm (4096,)
model.layers.20.self_attn.q_proj (4096, 4096)
model.layers.20.self_attn.k_proj (1024, 4096)
model.layers.20.self_attn.v_proj (1024, 4096)
model.layers.20.self_attn.o_proj (4096, 4096)
model.layers.20.mlp.gate_proj (14336, 4096)
model.layers.20.mlp.up_proj (14336, 4096)
model.layers.20.mlp.down_proj (4096, 14336)
model.layers.20.input_layernorm (4096,)
model.layers.20.post_attention_layernorm (4096,)
model.layers.21.self_attn.q_proj (4096, 4096)
model.layers.21.self_attn.k_proj (1024, 4096)
model.layers.21.self_attn.v_proj (1024, 4096)
model.layers.21.self_attn.o_proj (4096, 4096)
model.layers.21.mlp.gate_proj (14336, 4096)
model.layers.21.mlp.up_proj (14336, 4096)
model.layers.21.mlp.down_proj (4096, 14336)
model.layers.21.input_layernorm (4096,)
model.layers.21.post_attention_layernorm (4096,)
model.layers.22.self_attn.q_proj (4096, 4096)
model.layers.22.self_attn.k_proj (1024, 4096)
model.layers.22.self_attn.v_proj (1024, 4096)
model.layers.22.self_attn.o_proj (4096, 4096)
model.layers.22.mlp.gate_proj (14336, 4096)
model.layers.22.mlp.up_proj (14336, 4096)
model.layers.22.mlp.down_proj (4096, 14336)
model.layers.22.input_layernorm (4096,)
model.layers.22.post_attention_layernorm (4096,)
model.layers.23.self_attn.q_proj (4096, 4096)
model.layers.23.self_attn.k_proj (1024, 4096)
model.layers.23.self_attn.v_proj (1024, 4096)
model.layers.23.self_attn.o_proj (4096, 4096)
model.layers.23.mlp.gate_proj (14336, 4096)
model.layers.23.mlp.up_proj (14336, 4096)
model.layers.23.mlp.down_proj (4096, 14336)
model.layers.23.input_layernorm (4096,)
model.layers.23.post_attention_layernorm (4096,)
model.layers.24.self_attn.q_proj (4096, 4096)
model.layers.24.self_attn.k_proj (1024, 4096)
model.layers.24.self_attn.v_proj (1024, 4096)
model.layers.24.self_attn.o_proj (4096, 4096)
model.layers.24.mlp.gate_proj (14336, 4096)
model.layers.24.mlp.up_proj (14336, 4096)
model.layers.24.mlp.down_proj (4096, 14336)
model.layers.24.input_layernorm (4096,)
model.layers.24.post_attention_layernorm (4096,)
model.layers.25.self_attn.q_proj (4096, 4096)
model.layers.25.self_attn.k_proj (1024, 4096)
model.layers.25.self_attn.v_proj (1024, 4096)
model.layers.25.self_attn.o_proj (4096, 4096)
model.layers.25.mlp.gate_proj (14336, 4096)
model.layers.25.mlp.up_proj (14336, 4096)
model.layers.25.mlp.down_proj (4096, 14336)
model.layers.25.input_layernorm (4096,)
model.layers.25.post_attention_layernorm (4096,)
model.layers.26.self_attn.q_proj (4096, 4096)
model.layers.26.self_attn.k_proj (1024, 4096)
model.layers.26.self_attn.v_proj (1024, 4096)
model.layers.26.self_attn.o_proj (4096, 4096)
model.layers.26.mlp.gate_proj (14336, 4096)
model.layers.26.mlp.up_proj (14336, 4096)
model.layers.26.mlp.down_proj (4096, 14336)
model.layers.26.input_layernorm (4096,)
model.layers.26.post_attention_layernorm (4096,)
model.layers.27.self_attn.q_proj (4096, 4096)
model.layers.27.self_attn.k_proj (1024, 4096)
model.layers.27.self_attn.v_proj (1024, 4096)
model.layers.27.self_attn.o_proj (4096, 4096)
model.layers.27.mlp.gate_proj (14336, 4096)
model.layers.27.mlp.up_proj (14336, 4096)
model.layers.27.mlp.down_proj (4096, 14336)
model.layers.27.input_layernorm (4096,)
model.layers.27.post_attention_layernorm (4096,)
model.layers.28.self_attn.q_proj (4096, 4096)
model.layers.28.self_attn.k_proj (1024, 4096)
model.layers.28.self_attn.v_proj (1024, 4096)
model.layers.28.self_attn.o_proj (4096, 4096)
model.layers.28.mlp.gate_proj (14336, 4096)
model.layers.28.mlp.up_proj (14336, 4096)
model.layers.28.mlp.down_proj (4096, 14336)
model.layers.28.input_layernorm (4096,)
model.layers.28.post_attention_layernorm (4096,)
model.layers.29.self_attn.q_proj (4096, 4096)
model.layers.29.self_attn.k_proj (1024, 4096)
model.layers.29.self_attn.v_proj (1024, 4096)
model.layers.29.self_attn.o_proj (4096, 4096)
model.layers.29.mlp.gate_proj (14336, 4096)
model.layers.29.mlp.up_proj (14336, 4096)
model.layers.29.mlp.down_proj (4096, 14336)
model.layers.29.input_layernorm (4096,)
model.layers.29.post_attention_layernorm (4096,)
model.layers.30.self_attn.q_proj (4096, 4096)
model.layers.30.self_attn.k_proj (1024, 4096)
model.layers.30.self_attn.v_proj (1024, 4096)
model.layers.30.self_attn.o_proj (4096, 4096)
model.layers.30.mlp.gate_proj (14336, 4096)
model.layers.30.mlp.up_proj (14336, 4096)
model.layers.30.mlp.down_proj (4096, 14336)
model.layers.30.input_layernorm (4096,)
model.layers.30.post_attention_layernorm (4096,)
model.layers.31.self_attn.q_proj (4096, 4096)
model.layers.31.self_attn.k_proj (1024, 4096)
model.layers.31.self_attn.v_proj (1024, 4096)
model.layers.31.self_attn.o_proj (4096, 4096)
model.layers.31.mlp.gate_proj (14336, 4096)
model.layers.31.mlp.up_proj (14336, 4096)
model.layers.31.mlp.down_proj (4096, 14336)
model.layers.31.input_layernorm (4096,)
model.layers.31.post_attention_layernorm (4096,)
model.norm (4096,)
lm_head (32768, 4096)
```

## config.json

```json
{
  "architectures": [
    "MistralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.0.dev0",
  "use_cache": true,
  "vocab_size": 32768
}
```

---

# Qwen/Qwen2.5-7B-Instruct

## Parameter shapes

```text
model.embed_tokens (152064, 3584)
model.layers.0.self_attn.q_proj (3584, 3584)
model.layers.0.self_attn.k_proj (512, 3584)
model.layers.0.self_attn.v_proj (512, 3584)
model.layers.0.self_attn.o_proj (3584, 3584)
model.layers.0.mlp.gate_proj (18944, 3584)
model.layers.0.mlp.up_proj (18944, 3584)
model.layers.0.mlp.down_proj (3584, 18944)
model.layers.0.input_layernorm (3584,)
model.layers.0.post_attention_layernorm (3584,)
model.layers.1.self_attn.q_proj (3584, 3584)
model.layers.1.self_attn.k_proj (512, 3584)
model.layers.1.self_attn.v_proj (512, 3584)
model.layers.1.self_attn.o_proj (3584, 3584)
model.layers.1.mlp.gate_proj (18944, 3584)
model.layers.1.mlp.up_proj (18944, 3584)
model.layers.1.mlp.down_proj (3584, 18944)
model.layers.1.input_layernorm (3584,)
model.layers.1.post_attention_layernorm (3584,)
model.layers.2.self_attn.q_proj (3584, 3584)
model.layers.2.self_attn.k_proj (512, 3584)
model.layers.2.self_attn.v_proj (512, 3584)
model.layers.2.self_attn.o_proj (3584, 3584)
model.layers.2.mlp.gate_proj (18944, 3584)
model.layers.2.mlp.up_proj (18944, 3584)
model.layers.2.mlp.down_proj (3584, 18944)
model.layers.2.input_layernorm (3584,)
model.layers.2.post_attention_layernorm (3584,)
model.layers.3.self_attn.q_proj (3584, 3584)
model.layers.3.self_attn.k_proj (512, 3584)
model.layers.3.self_attn.v_proj (512, 3584)
model.layers.3.self_attn.o_proj (3584, 3584)
model.layers.3.mlp.gate_proj (18944, 3584)
model.layers.3.mlp.up_proj (18944, 3584)
model.layers.3.mlp.down_proj (3584, 18944)
model.layers.3.input_layernorm (3584,)
model.layers.3.post_attention_layernorm (3584,)
model.layers.4.self_attn.q_proj (3584, 3584)
model.layers.4.self_attn.k_proj (512, 3584)
model.layers.4.self_attn.v_proj (512, 3584)
model.layers.4.self_attn.o_proj (3584, 3584)
model.layers.4.mlp.gate_proj (18944, 3584)
model.layers.4.mlp.up_proj (18944, 3584)
model.layers.4.mlp.down_proj (3584, 18944)
model.layers.4.input_layernorm (3584,)
model.layers.4.post_attention_layernorm (3584,)
model.layers.5.self_attn.q_proj (3584, 3584)
model.layers.5.self_attn.k_proj (512, 3584)
model.layers.5.self_attn.v_proj (512, 3584)
model.layers.5.self_attn.o_proj (3584, 3584)
model.layers.5.mlp.gate_proj (18944, 3584)
model.layers.5.mlp.up_proj (18944, 3584)
model.layers.5.mlp.down_proj (3584, 18944)
model.layers.5.input_layernorm (3584,)
model.layers.5.post_attention_layernorm (3584,)
model.layers.6.self_attn.q_proj (3584, 3584)
model.layers.6.self_attn.k_proj (512, 3584)
model.layers.6.self_attn.v_proj (512, 3584)
model.layers.6.self_attn.o_proj (3584, 3584)
model.layers.6.mlp.gate_proj (18944, 3584)
model.layers.6.mlp.up_proj (18944, 3584)
model.layers.6.mlp.down_proj (3584, 18944)
model.layers.6.input_layernorm (3584,)
model.layers.6.post_attention_layernorm (3584,)
model.layers.7.self_attn.q_proj (3584, 3584)
model.layers.7.self_attn.k_proj (512, 3584)
model.layers.7.self_attn.v_proj (512, 3584)
model.layers.7.self_attn.o_proj (3584, 3584)
model.layers.7.mlp.gate_proj (18944, 3584)
model.layers.7.mlp.up_proj (18944, 3584)
model.layers.7.mlp.down_proj (3584, 18944)
model.layers.7.input_layernorm (3584,)
model.layers.7.post_attention_layernorm (3584,)
model.layers.8.self_attn.q_proj (3584, 3584)
model.layers.8.self_attn.k_proj (512, 3584)
model.layers.8.self_attn.v_proj (512, 3584)
model.layers.8.self_attn.o_proj (3584, 3584)
model.layers.8.mlp.gate_proj (18944, 3584)
model.layers.8.mlp.up_proj (18944, 3584)
model.layers.8.mlp.down_proj (3584, 18944)
model.layers.8.input_layernorm (3584,)
model.layers.8.post_attention_layernorm (3584,)
model.layers.9.self_attn.q_proj (3584, 3584)
model.layers.9.self_attn.k_proj (512, 3584)
model.layers.9.self_attn.v_proj (512, 3584)
model.layers.9.self_attn.o_proj (3584, 3584)
model.layers.9.mlp.gate_proj (18944, 3584)
model.layers.9.mlp.up_proj (18944, 3584)
model.layers.9.mlp.down_proj (3584, 18944)
model.layers.9.input_layernorm (3584,)
model.layers.9.post_attention_layernorm (3584,)
model.layers.10.self_attn.q_proj (3584, 3584)
model.layers.10.self_attn.k_proj (512, 3584)
model.layers.10.self_attn.v_proj (512, 3584)
model.layers.10.self_attn.o_proj (3584, 3584)
model.layers.10.mlp.gate_proj (18944, 3584)
model.layers.10.mlp.up_proj (18944, 3584)
model.layers.10.mlp.down_proj (3584, 18944)
model.layers.10.input_layernorm (3584,)
model.layers.10.post_attention_layernorm (3584,)
model.layers.11.self_attn.q_proj (3584, 3584)
model.layers.11.self_attn.k_proj (512, 3584)
model.layers.11.self_attn.v_proj (512, 3584)
model.layers.11.self_attn.o_proj (3584, 3584)
model.layers.11.mlp.gate_proj (18944, 3584)
model.layers.11.mlp.up_proj (18944, 3584)
model.layers.11.mlp.down_proj (3584, 18944)
model.layers.11.input_layernorm (3584,)
model.layers.11.post_attention_layernorm (3584,)
model.layers.12.self_attn.q_proj (3584, 3584)
model.layers.12.self_attn.k_proj (512, 3584)
model.layers.12.self_attn.v_proj (512, 3584)
model.layers.12.self_attn.o_proj (3584, 3584)
model.layers.12.mlp.gate_proj (18944, 3584)
model.layers.12.mlp.up_proj (18944, 3584)
model.layers.12.mlp.down_proj (3584, 18944)
model.layers.12.input_layernorm (3584,)
model.layers.12.post_attention_layernorm (3584,)
model.layers.13.self_attn.q_proj (3584, 3584)
model.layers.13.self_attn.k_proj (512, 3584)
model.layers.13.self_attn.v_proj (512, 3584)
model.layers.13.self_attn.o_proj (3584, 3584)
model.layers.13.mlp.gate_proj (18944, 3584)
model.layers.13.mlp.up_proj (18944, 3584)
model.layers.13.mlp.down_proj (3584, 18944)
model.layers.13.input_layernorm (3584,)
model.layers.13.post_attention_layernorm (3584,)
model.layers.14.self_attn.q_proj (3584, 3584)
model.layers.14.self_attn.k_proj (512, 3584)
model.layers.14.self_attn.v_proj (512, 3584)
model.layers.14.self_attn.o_proj (3584, 3584)
model.layers.14.mlp.gate_proj (18944, 3584)
model.layers.14.mlp.up_proj (18944, 3584)
model.layers.14.mlp.down_proj (3584, 18944)
model.layers.14.input_layernorm (3584,)
model.layers.14.post_attention_layernorm (3584,)
model.layers.15.self_attn.q_proj (3584, 3584)
model.layers.15.self_attn.k_proj (512, 3584)
model.layers.15.self_attn.v_proj (512, 3584)
model.layers.15.self_attn.o_proj (3584, 3584)
model.layers.15.mlp.gate_proj (18944, 3584)
model.layers.15.mlp.up_proj (18944, 3584)
model.layers.15.mlp.down_proj (3584, 18944)
model.layers.15.input_layernorm (3584,)
model.layers.15.post_attention_layernorm (3584,)
model.layers.16.self_attn.q_proj (3584, 3584)
model.layers.16.self_attn.k_proj (512, 3584)
model.layers.16.self_attn.v_proj (512, 3584)
model.layers.16.self_attn.o_proj (3584, 3584)
model.layers.16.mlp.gate_proj (18944, 3584)
model.layers.16.mlp.up_proj (18944, 3584)
model.layers.16.mlp.down_proj (3584, 18944)
model.layers.16.input_layernorm (3584,)
model.layers.16.post_attention_layernorm (3584,)
model.layers.17.self_attn.q_proj (3584, 3584)
model.layers.17.self_attn.k_proj (512, 3584)
model.layers.17.self_attn.v_proj (512, 3584)
model.layers.17.self_attn.o_proj (3584, 3584)
model.layers.17.mlp.gate_proj (18944, 3584)
model.layers.17.mlp.up_proj (18944, 3584)
model.layers.17.mlp.down_proj (3584, 18944)
model.layers.17.input_layernorm (3584,)
model.layers.17.post_attention_layernorm (3584,)
model.layers.18.self_attn.q_proj (3584, 3584)
model.layers.18.self_attn.k_proj (512, 3584)
model.layers.18.self_attn.v_proj (512, 3584)
model.layers.18.self_attn.o_proj (3584, 3584)
model.layers.18.mlp.gate_proj (18944, 3584)
model.layers.18.mlp.up_proj (18944, 3584)
model.layers.18.mlp.down_proj (3584, 18944)
model.layers.18.input_layernorm (3584,)
model.layers.18.post_attention_layernorm (3584,)
model.layers.19.self_attn.q_proj (3584, 3584)
model.layers.19.self_attn.k_proj (512, 3584)
model.layers.19.self_attn.v_proj (512, 3584)
model.layers.19.self_attn.o_proj (3584, 3584)
model.layers.19.mlp.gate_proj (18944, 3584)
model.layers.19.mlp.up_proj (18944, 3584)
model.layers.19.mlp.down_proj (3584, 18944)
model.layers.19.input_layernorm (3584,)
model.layers.19.post_attention_layernorm (3584,)
model.layers.20.self_attn.q_proj (3584, 3584)
model.layers.20.self_attn.k_proj (512, 3584)
model.layers.20.self_attn.v_proj (512, 3584)
model.layers.20.self_attn.o_proj (3584, 3584)
model.layers.20.mlp.gate_proj (18944, 3584)
model.layers.20.mlp.up_proj (18944, 3584)
model.layers.20.mlp.down_proj (3584, 18944)
model.layers.20.input_layernorm (3584,)
model.layers.20.post_attention_layernorm (3584,)
model.layers.21.self_attn.q_proj (3584, 3584)
model.layers.21.self_attn.k_proj (512, 3584)
model.layers.21.self_attn.v_proj (512, 3584)
model.layers.21.self_attn.o_proj (3584, 3584)
model.layers.21.mlp.gate_proj (18944, 3584)
model.layers.21.mlp.up_proj (18944, 3584)
model.layers.21.mlp.down_proj (3584, 18944)
model.layers.21.input_layernorm (3584,)
model.layers.21.post_attention_layernorm (3584,)
model.layers.22.self_attn.q_proj (3584, 3584)
model.layers.22.self_attn.k_proj (512, 3584)
model.layers.22.self_attn.v_proj (512, 3584)
model.layers.22.self_attn.o_proj (3584, 3584)
model.layers.22.mlp.gate_proj (18944, 3584)
model.layers.22.mlp.up_proj (18944, 3584)
model.layers.22.mlp.down_proj (3584, 18944)
model.layers.22.input_layernorm (3584,)
model.layers.22.post_attention_layernorm (3584,)
model.layers.23.self_attn.q_proj (3584, 3584)
model.layers.23.self_attn.k_proj (512, 3584)
model.layers.23.self_attn.v_proj (512, 3584)
model.layers.23.self_attn.o_proj (3584, 3584)
model.layers.23.mlp.gate_proj (18944, 3584)
model.layers.23.mlp.up_proj (18944, 3584)
model.layers.23.mlp.down_proj (3584, 18944)
model.layers.23.input_layernorm (3584,)
model.layers.23.post_attention_layernorm (3584,)
model.layers.24.self_attn.q_proj (3584, 3584)
model.layers.24.self_attn.k_proj (512, 3584)
model.layers.24.self_attn.v_proj (512, 3584)
model.layers.24.self_attn.o_proj (3584, 3584)
model.layers.24.mlp.gate_proj (18944, 3584)
model.layers.24.mlp.up_proj (18944, 3584)
model.layers.24.mlp.down_proj (3584, 18944)
model.layers.24.input_layernorm (3584,)
model.layers.24.post_attention_layernorm (3584,)
model.layers.25.self_attn.q_proj (3584, 3584)
model.layers.25.self_attn.k_proj (512, 3584)
model.layers.25.self_attn.v_proj (512, 3584)
model.layers.25.self_attn.o_proj (3584, 3584)
model.layers.25.mlp.gate_proj (18944, 3584)
model.layers.25.mlp.up_proj (18944, 3584)
model.layers.25.mlp.down_proj (3584, 18944)
model.layers.25.input_layernorm (3584,)
model.layers.25.post_attention_layernorm (3584,)
model.layers.26.self_attn.q_proj (3584, 3584)
model.layers.26.self_attn.k_proj (512, 3584)
model.layers.26.self_attn.v_proj (512, 3584)
model.layers.26.self_attn.o_proj (3584, 3584)
model.layers.26.mlp.gate_proj (18944, 3584)
model.layers.26.mlp.up_proj (18944, 3584)
model.layers.26.mlp.down_proj (3584, 18944)
model.layers.26.input_layernorm (3584,)
model.layers.26.post_attention_layernorm (3584,)
model.layers.27.self_attn.q_proj (3584, 3584)
model.layers.27.self_attn.k_proj (512, 3584)
model.layers.27.self_attn.v_proj (512, 3584)
model.layers.27.self_attn.o_proj (3584, 3584)
model.layers.27.mlp.gate_proj (18944, 3584)
model.layers.27.mlp.up_proj (18944, 3584)
model.layers.27.mlp.down_proj (3584, 18944)
model.layers.27.input_layernorm (3584,)
model.layers.27.post_attention_layernorm (3584,)
model.norm (3584,)
lm_head (152064, 3584)
```

## config.json

```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}
```

---

# allenai/OLMo-2-1124-7B-Instruct

## Parameter shapes

```text
model.embed_tokens (100352, 4096)
model.layers.0.self_attn.q_proj (4096, 4096)
model.layers.0.self_attn.k_proj (4096, 4096)
model.layers.0.self_attn.v_proj (4096, 4096)
model.layers.0.self_attn.o_proj (4096, 4096)
model.layers.0.self_attn.q_norm (4096,)
model.layers.0.self_attn.k_norm (4096,)
model.layers.0.mlp.gate_proj (11008, 4096)
model.layers.0.mlp.up_proj (11008, 4096)
model.layers.0.mlp.down_proj (4096, 11008)
model.layers.0.post_attention_layernorm (4096,)
model.layers.0.post_feedforward_layernorm (4096,)
model.layers.1.self_attn.q_proj (4096, 4096)
model.layers.1.self_attn.k_proj (4096, 4096)
model.layers.1.self_attn.v_proj (4096, 4096)
model.layers.1.self_attn.o_proj (4096, 4096)
model.layers.1.self_attn.q_norm (4096,)
model.layers.1.self_attn.k_norm (4096,)
model.layers.1.mlp.gate_proj (11008, 4096)
model.layers.1.mlp.up_proj (11008, 4096)
model.layers.1.mlp.down_proj (4096, 11008)
model.layers.1.post_attention_layernorm (4096,)
model.layers.1.post_feedforward_layernorm (4096,)
model.layers.2.self_attn.q_proj (4096, 4096)
model.layers.2.self_attn.k_proj (4096, 4096)
model.layers.2.self_attn.v_proj (4096, 4096)
model.layers.2.self_attn.o_proj (4096, 4096)
model.layers.2.self_attn.q_norm (4096,)
model.layers.2.self_attn.k_norm (4096,)
model.layers.2.mlp.gate_proj (11008, 4096)
model.layers.2.mlp.up_proj (11008, 4096)
model.layers.2.mlp.down_proj (4096, 11008)
model.layers.2.post_attention_layernorm (4096,)
model.layers.2.post_feedforward_layernorm (4096,)
model.layers.3.self_attn.q_proj (4096, 4096)
model.layers.3.self_attn.k_proj (4096, 4096)
model.layers.3.self_attn.v_proj (4096, 4096)
model.layers.3.self_attn.o_proj (4096, 4096)
model.layers.3.self_attn.q_norm (4096,)
model.layers.3.self_attn.k_norm (4096,)
model.layers.3.mlp.gate_proj (11008, 4096)
model.layers.3.mlp.up_proj (11008, 4096)
model.layers.3.mlp.down_proj (4096, 11008)
model.layers.3.post_attention_layernorm (4096,)
model.layers.3.post_feedforward_layernorm (4096,)
model.layers.4.self_attn.q_proj (4096, 4096)
model.layers.4.self_attn.k_proj (4096, 4096)
model.layers.4.self_attn.v_proj (4096, 4096)
model.layers.4.self_attn.o_proj (4096, 4096)
model.layers.4.self_attn.q_norm (4096,)
model.layers.4.self_attn.k_norm (4096,)
model.layers.4.mlp.gate_proj (11008, 4096)
model.layers.4.mlp.up_proj (11008, 4096)
model.layers.4.mlp.down_proj (4096, 11008)
model.layers.4.post_attention_layernorm (4096,)
model.layers.4.post_feedforward_layernorm (4096,)
model.layers.5.self_attn.q_proj (4096, 4096)
model.layers.5.self_attn.k_proj (4096, 4096)
model.layers.5.self_attn.v_proj (4096, 4096)
model.layers.5.self_attn.o_proj (4096, 4096)
model.layers.5.self_attn.q_norm (4096,)
model.layers.5.self_attn.k_norm (4096,)
model.layers.5.mlp.gate_proj (11008, 4096)
model.layers.5.mlp.up_proj (11008, 4096)
model.layers.5.mlp.down_proj (4096, 11008)
model.layers.5.post_attention_layernorm (4096,)
model.layers.5.post_feedforward_layernorm (4096,)
model.layers.6.self_attn.q_proj (4096, 4096)
model.layers.6.self_attn.k_proj (4096, 4096)
model.layers.6.self_attn.v_proj (4096, 4096)
model.layers.6.self_attn.o_proj (4096, 4096)
model.layers.6.self_attn.q_norm (4096,)
model.layers.6.self_attn.k_norm (4096,)
model.layers.6.mlp.gate_proj (11008, 4096)
model.layers.6.mlp.up_proj (11008, 4096)
model.layers.6.mlp.down_proj (4096, 11008)
model.layers.6.post_attention_layernorm (4096,)
model.layers.6.post_feedforward_layernorm (4096,)
model.layers.7.self_attn.q_proj (4096, 4096)
model.layers.7.self_attn.k_proj (4096, 4096)
model.layers.7.self_attn.v_proj (4096, 4096)
model.layers.7.self_attn.o_proj (4096, 4096)
model.layers.7.self_attn.q_norm (4096,)
model.layers.7.self_attn.k_norm (4096,)
model.layers.7.mlp.gate_proj (11008, 4096)
model.layers.7.mlp.up_proj (11008, 4096)
model.layers.7.mlp.down_proj (4096, 11008)
model.layers.7.post_attention_layernorm (4096,)
model.layers.7.post_feedforward_layernorm (4096,)
model.layers.8.self_attn.q_proj (4096, 4096)
model.layers.8.self_attn.k_proj (4096, 4096)
model.layers.8.self_attn.v_proj (4096, 4096)
model.layers.8.self_attn.o_proj (4096, 4096)
model.layers.8.self_attn.q_norm (4096,)
model.layers.8.self_attn.k_norm (4096,)
model.layers.8.mlp.gate_proj (11008, 4096)
model.layers.8.mlp.up_proj (11008, 4096)
model.layers.8.mlp.down_proj (4096, 11008)
model.layers.8.post_attention_layernorm (4096,)
model.layers.8.post_feedforward_layernorm (4096,)
model.layers.9.self_attn.q_proj (4096, 4096)
model.layers.9.self_attn.k_proj (4096, 4096)
model.layers.9.self_attn.v_proj (4096, 4096)
model.layers.9.self_attn.o_proj (4096, 4096)
model.layers.9.self_attn.q_norm (4096,)
model.layers.9.self_attn.k_norm (4096,)
model.layers.9.mlp.gate_proj (11008, 4096)
model.layers.9.mlp.up_proj (11008, 4096)
model.layers.9.mlp.down_proj (4096, 11008)
model.layers.9.post_attention_layernorm (4096,)
model.layers.9.post_feedforward_layernorm (4096,)
model.layers.10.self_attn.q_proj (4096, 4096)
model.layers.10.self_attn.k_proj (4096, 4096)
model.layers.10.self_attn.v_proj (4096, 4096)
model.layers.10.self_attn.o_proj (4096, 4096)
model.layers.10.self_attn.q_norm (4096,)
model.layers.10.self_attn.k_norm (4096,)
model.layers.10.mlp.gate_proj (11008, 4096)
model.layers.10.mlp.up_proj (11008, 4096)
model.layers.10.mlp.down_proj (4096, 11008)
model.layers.10.post_attention_layernorm (4096,)
model.layers.10.post_feedforward_layernorm (4096,)
model.layers.11.self_attn.q_proj (4096, 4096)
model.layers.11.self_attn.k_proj (4096, 4096)
model.layers.11.self_attn.v_proj (4096, 4096)
model.layers.11.self_attn.o_proj (4096, 4096)
model.layers.11.self_attn.q_norm (4096,)
model.layers.11.self_attn.k_norm (4096,)
model.layers.11.mlp.gate_proj (11008, 4096)
model.layers.11.mlp.up_proj (11008, 4096)
model.layers.11.mlp.down_proj (4096, 11008)
model.layers.11.post_attention_layernorm (4096,)
model.layers.11.post_feedforward_layernorm (4096,)
model.layers.12.self_attn.q_proj (4096, 4096)
model.layers.12.self_attn.k_proj (4096, 4096)
model.layers.12.self_attn.v_proj (4096, 4096)
model.layers.12.self_attn.o_proj (4096, 4096)
model.layers.12.self_attn.q_norm (4096,)
model.layers.12.self_attn.k_norm (4096,)
model.layers.12.mlp.gate_proj (11008, 4096)
model.layers.12.mlp.up_proj (11008, 4096)
model.layers.12.mlp.down_proj (4096, 11008)
model.layers.12.post_attention_layernorm (4096,)
model.layers.12.post_feedforward_layernorm (4096,)
model.layers.13.self_attn.q_proj (4096, 4096)
model.layers.13.self_attn.k_proj (4096, 4096)
model.layers.13.self_attn.v_proj (4096, 4096)
model.layers.13.self_attn.o_proj (4096, 4096)
model.layers.13.self_attn.q_norm (4096,)
model.layers.13.self_attn.k_norm (4096,)
model.layers.13.mlp.gate_proj (11008, 4096)
model.layers.13.mlp.up_proj (11008, 4096)
model.layers.13.mlp.down_proj (4096, 11008)
model.layers.13.post_attention_layernorm (4096,)
model.layers.13.post_feedforward_layernorm (4096,)
model.layers.14.self_attn.q_proj (4096, 4096)
model.layers.14.self_attn.k_proj (4096, 4096)
model.layers.14.self_attn.v_proj (4096, 4096)
model.layers.14.self_attn.o_proj (4096, 4096)
model.layers.14.self_attn.q_norm (4096,)
model.layers.14.self_attn.k_norm (4096,)
model.layers.14.mlp.gate_proj (11008, 4096)
model.layers.14.mlp.up_proj (11008, 4096)
model.layers.14.mlp.down_proj (4096, 11008)
model.layers.14.post_attention_layernorm (4096,)
model.layers.14.post_feedforward_layernorm (4096,)
model.layers.15.self_attn.q_proj (4096, 4096)
model.layers.15.self_attn.k_proj (4096, 4096)
model.layers.15.self_attn.v_proj (4096, 4096)
model.layers.15.self_attn.o_proj (4096, 4096)
model.layers.15.self_attn.q_norm (4096,)
model.layers.15.self_attn.k_norm (4096,)
model.layers.15.mlp.gate_proj (11008, 4096)
model.layers.15.mlp.up_proj (11008, 4096)
model.layers.15.mlp.down_proj (4096, 11008)
model.layers.15.post_attention_layernorm (4096,)
model.layers.15.post_feedforward_layernorm (4096,)
model.layers.16.self_attn.q_proj (4096, 4096)
model.layers.16.self_attn.k_proj (4096, 4096)
model.layers.16.self_attn.v_proj (4096, 4096)
model.layers.16.self_attn.o_proj (4096, 4096)
model.layers.16.self_attn.q_norm (4096,)
model.layers.16.self_attn.k_norm (4096,)
model.layers.16.mlp.gate_proj (11008, 4096)
model.layers.16.mlp.up_proj (11008, 4096)
model.layers.16.mlp.down_proj (4096, 11008)
model.layers.16.post_attention_layernorm (4096,)
model.layers.16.post_feedforward_layernorm (4096,)
model.layers.17.self_attn.q_proj (4096, 4096)
model.layers.17.self_attn.k_proj (4096, 4096)
model.layers.17.self_attn.v_proj (4096, 4096)
model.layers.17.self_attn.o_proj (4096, 4096)
model.layers.17.self_attn.q_norm (4096,)
model.layers.17.self_attn.k_norm (4096,)
model.layers.17.mlp.gate_proj (11008, 4096)
model.layers.17.mlp.up_proj (11008, 4096)
model.layers.17.mlp.down_proj (4096, 11008)
model.layers.17.post_attention_layernorm (4096,)
model.layers.17.post_feedforward_layernorm (4096,)
model.layers.18.self_attn.q_proj (4096, 4096)
model.layers.18.self_attn.k_proj (4096, 4096)
model.layers.18.self_attn.v_proj (4096, 4096)
model.layers.18.self_attn.o_proj (4096, 4096)
model.layers.18.self_attn.q_norm (4096,)
model.layers.18.self_attn.k_norm (4096,)
model.layers.18.mlp.gate_proj (11008, 4096)
model.layers.18.mlp.up_proj (11008, 4096)
model.layers.18.mlp.down_proj (4096, 11008)
model.layers.18.post_attention_layernorm (4096,)
model.layers.18.post_feedforward_layernorm (4096,)
model.layers.19.self_attn.q_proj (4096, 4096)
model.layers.19.self_attn.k_proj (4096, 4096)
model.layers.19.self_attn.v_proj (4096, 4096)
model.layers.19.self_attn.o_proj (4096, 4096)
model.layers.19.self_attn.q_norm (4096,)
model.layers.19.self_attn.k_norm (4096,)
model.layers.19.mlp.gate_proj (11008, 4096)
model.layers.19.mlp.up_proj (11008, 4096)
model.layers.19.mlp.down_proj (4096, 11008)
model.layers.19.post_attention_layernorm (4096,)
model.layers.19.post_feedforward_layernorm (4096,)
model.layers.20.self_attn.q_proj (4096, 4096)
model.layers.20.self_attn.k_proj (4096, 4096)
model.layers.20.self_attn.v_proj (4096, 4096)
model.layers.20.self_attn.o_proj (4096, 4096)
model.layers.20.self_attn.q_norm (4096,)
model.layers.20.self_attn.k_norm (4096,)
model.layers.20.mlp.gate_proj (11008, 4096)
model.layers.20.mlp.up_proj (11008, 4096)
model.layers.20.mlp.down_proj (4096, 11008)
model.layers.20.post_attention_layernorm (4096,)
model.layers.20.post_feedforward_layernorm (4096,)
model.layers.21.self_attn.q_proj (4096, 4096)
model.layers.21.self_attn.k_proj (4096, 4096)
model.layers.21.self_attn.v_proj (4096, 4096)
model.layers.21.self_attn.o_proj (4096, 4096)
model.layers.21.self_attn.q_norm (4096,)
model.layers.21.self_attn.k_norm (4096,)
model.layers.21.mlp.gate_proj (11008, 4096)
model.layers.21.mlp.up_proj (11008, 4096)
model.layers.21.mlp.down_proj (4096, 11008)
model.layers.21.post_attention_layernorm (4096,)
model.layers.21.post_feedforward_layernorm (4096,)
model.layers.22.self_attn.q_proj (4096, 4096)
model.layers.22.self_attn.k_proj (4096, 4096)
model.layers.22.self_attn.v_proj (4096, 4096)
model.layers.22.self_attn.o_proj (4096, 4096)
model.layers.22.self_attn.q_norm (4096,)
model.layers.22.self_attn.k_norm (4096,)
model.layers.22.mlp.gate_proj (11008, 4096)
model.layers.22.mlp.up_proj (11008, 4096)
model.layers.22.mlp.down_proj (4096, 11008)
model.layers.22.post_attention_layernorm (4096,)
model.layers.22.post_feedforward_layernorm (4096,)
model.layers.23.self_attn.q_proj (4096, 4096)
model.layers.23.self_attn.k_proj (4096, 4096)
model.layers.23.self_attn.v_proj (4096, 4096)
model.layers.23.self_attn.o_proj (4096, 4096)
model.layers.23.self_attn.q_norm (4096,)
model.layers.23.self_attn.k_norm (4096,)
model.layers.23.mlp.gate_proj (11008, 4096)
model.layers.23.mlp.up_proj (11008, 4096)
model.layers.23.mlp.down_proj (4096, 11008)
model.layers.23.post_attention_layernorm (4096,)
model.layers.23.post_feedforward_layernorm (4096,)
model.layers.24.self_attn.q_proj (4096, 4096)
model.layers.24.self_attn.k_proj (4096, 4096)
model.layers.24.self_attn.v_proj (4096, 4096)
model.layers.24.self_attn.o_proj (4096, 4096)
model.layers.24.self_attn.q_norm (4096,)
model.layers.24.self_attn.k_norm (4096,)
model.layers.24.mlp.gate_proj (11008, 4096)
model.layers.24.mlp.up_proj (11008, 4096)
model.layers.24.mlp.down_proj (4096, 11008)
model.layers.24.post_attention_layernorm (4096,)
model.layers.24.post_feedforward_layernorm (4096,)
model.layers.25.self_attn.q_proj (4096, 4096)
model.layers.25.self_attn.k_proj (4096, 4096)
model.layers.25.self_attn.v_proj (4096, 4096)
model.layers.25.self_attn.o_proj (4096, 4096)
model.layers.25.self_attn.q_norm (4096,)
model.layers.25.self_attn.k_norm (4096,)
model.layers.25.mlp.gate_proj (11008, 4096)
model.layers.25.mlp.up_proj (11008, 4096)
model.layers.25.mlp.down_proj (4096, 11008)
model.layers.25.post_attention_layernorm (4096,)
model.layers.25.post_feedforward_layernorm (4096,)
model.layers.26.self_attn.q_proj (4096, 4096)
model.layers.26.self_attn.k_proj (4096, 4096)
model.layers.26.self_attn.v_proj (4096, 4096)
model.layers.26.self_attn.o_proj (4096, 4096)
model.layers.26.self_attn.q_norm (4096,)
model.layers.26.self_attn.k_norm (4096,)
model.layers.26.mlp.gate_proj (11008, 4096)
model.layers.26.mlp.up_proj (11008, 4096)
model.layers.26.mlp.down_proj (4096, 11008)
model.layers.26.post_attention_layernorm (4096,)
model.layers.26.post_feedforward_layernorm (4096,)
model.layers.27.self_attn.q_proj (4096, 4096)
model.layers.27.self_attn.k_proj (4096, 4096)
model.layers.27.self_attn.v_proj (4096, 4096)
model.layers.27.self_attn.o_proj (4096, 4096)
model.layers.27.self_attn.q_norm (4096,)
model.layers.27.self_attn.k_norm (4096,)
model.layers.27.mlp.gate_proj (11008, 4096)
model.layers.27.mlp.up_proj (11008, 4096)
model.layers.27.mlp.down_proj (4096, 11008)
model.layers.27.post_attention_layernorm (4096,)
model.layers.27.post_feedforward_layernorm (4096,)
model.layers.28.self_attn.q_proj (4096, 4096)
model.layers.28.self_attn.k_proj (4096, 4096)
model.layers.28.self_attn.v_proj (4096, 4096)
model.layers.28.self_attn.o_proj (4096, 4096)
model.layers.28.self_attn.q_norm (4096,)
model.layers.28.self_attn.k_norm (4096,)
model.layers.28.mlp.gate_proj (11008, 4096)
model.layers.28.mlp.up_proj (11008, 4096)
model.layers.28.mlp.down_proj (4096, 11008)
model.layers.28.post_attention_layernorm (4096,)
model.layers.28.post_feedforward_layernorm (4096,)
model.layers.29.self_attn.q_proj (4096, 4096)
model.layers.29.self_attn.k_proj (4096, 4096)
model.layers.29.self_attn.v_proj (4096, 4096)
model.layers.29.self_attn.o_proj (4096, 4096)
model.layers.29.self_attn.q_norm (4096,)
model.layers.29.self_attn.k_norm (4096,)
model.layers.29.mlp.gate_proj (11008, 4096)
model.layers.29.mlp.up_proj (11008, 4096)
model.layers.29.mlp.down_proj (4096, 11008)
model.layers.29.post_attention_layernorm (4096,)
model.layers.29.post_feedforward_layernorm (4096,)
model.layers.30.self_attn.q_proj (4096, 4096)
model.layers.30.self_attn.k_proj (4096, 4096)
model.layers.30.self_attn.v_proj (4096, 4096)
model.layers.30.self_attn.o_proj (4096, 4096)
model.layers.30.self_attn.q_norm (4096,)
model.layers.30.self_attn.k_norm (4096,)
model.layers.30.mlp.gate_proj (11008, 4096)
model.layers.30.mlp.up_proj (11008, 4096)
model.layers.30.mlp.down_proj (4096, 11008)
model.layers.30.post_attention_layernorm (4096,)
model.layers.30.post_feedforward_layernorm (4096,)
model.layers.31.self_attn.q_proj (4096, 4096)
model.layers.31.self_attn.k_proj (4096, 4096)
model.layers.31.self_attn.v_proj (4096, 4096)
model.layers.31.self_attn.o_proj (4096, 4096)
model.layers.31.self_attn.q_norm (4096,)
model.layers.31.self_attn.k_norm (4096,)
model.layers.31.mlp.gate_proj (11008, 4096)
model.layers.31.mlp.up_proj (11008, 4096)
model.layers.31.mlp.down_proj (4096, 11008)
model.layers.31.post_attention_layernorm (4096,)
model.layers.31.post_feedforward_layernorm (4096,)
model.norm (4096,)
lm_head (100352, 4096)
```

## config.json

```json
{
  "_name_or_path": "allenai/open_instruct_dev",
  "architectures": [
    "Olmo2ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "eos_token_id": 100257,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "olmo2",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 100277,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 500000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.47.0.dev0",
  "use_cache": false,
  "vocab_size": 100352
}
```

---

# ibm-granite/granite-3.3-8b-instruct

## Parameter shapes

```text
model.embed_tokens (49159, 4096)
model.layers.0.self_attn.q_proj (4096, 4096)
model.layers.0.self_attn.k_proj (1024, 4096)
model.layers.0.self_attn.v_proj (1024, 4096)
model.layers.0.self_attn.o_proj (4096, 4096)
model.layers.0.mlp.gate_proj (12800, 4096)
model.layers.0.mlp.up_proj (12800, 4096)
model.layers.0.mlp.down_proj (4096, 12800)
model.layers.0.input_layernorm (4096,)
model.layers.0.post_attention_layernorm (4096,)
model.layers.1.self_attn.q_proj (4096, 4096)
model.layers.1.self_attn.k_proj (1024, 4096)
model.layers.1.self_attn.v_proj (1024, 4096)
model.layers.1.self_attn.o_proj (4096, 4096)
model.layers.1.mlp.gate_proj (12800, 4096)
model.layers.1.mlp.up_proj (12800, 4096)
model.layers.1.mlp.down_proj (4096, 12800)
model.layers.1.input_layernorm (4096,)
model.layers.1.post_attention_layernorm (4096,)
model.layers.2.self_attn.q_proj (4096, 4096)
model.layers.2.self_attn.k_proj (1024, 4096)
model.layers.2.self_attn.v_proj (1024, 4096)
model.layers.2.self_attn.o_proj (4096, 4096)
model.layers.2.mlp.gate_proj (12800, 4096)
model.layers.2.mlp.up_proj (12800, 4096)
model.layers.2.mlp.down_proj (4096, 12800)
model.layers.2.input_layernorm (4096,)
model.layers.2.post_attention_layernorm (4096,)
model.layers.3.self_attn.q_proj (4096, 4096)
model.layers.3.self_attn.k_proj (1024, 4096)
model.layers.3.self_attn.v_proj (1024, 4096)
model.layers.3.self_attn.o_proj (4096, 4096)
model.layers.3.mlp.gate_proj (12800, 4096)
model.layers.3.mlp.up_proj (12800, 4096)
model.layers.3.mlp.down_proj (4096, 12800)
model.layers.3.input_layernorm (4096,)
model.layers.3.post_attention_layernorm (4096,)
model.layers.4.self_attn.q_proj (4096, 4096)
model.layers.4.self_attn.k_proj (1024, 4096)
model.layers.4.self_attn.v_proj (1024, 4096)
model.layers.4.self_attn.o_proj (4096, 4096)
model.layers.4.mlp.gate_proj (12800, 4096)
model.layers.4.mlp.up_proj (12800, 4096)
model.layers.4.mlp.down_proj (4096, 12800)
model.layers.4.input_layernorm (4096,)
model.layers.4.post_attention_layernorm (4096,)
model.layers.5.self_attn.q_proj (4096, 4096)
model.layers.5.self_attn.k_proj (1024, 4096)
model.layers.5.self_attn.v_proj (1024, 4096)
model.layers.5.self_attn.o_proj (4096, 4096)
model.layers.5.mlp.gate_proj (12800, 4096)
model.layers.5.mlp.up_proj (12800, 4096)
model.layers.5.mlp.down_proj (4096, 12800)
model.layers.5.input_layernorm (4096,)
model.layers.5.post_attention_layernorm (4096,)
model.layers.6.self_attn.q_proj (4096, 4096)
model.layers.6.self_attn.k_proj (1024, 4096)
model.layers.6.self_attn.v_proj (1024, 4096)
model.layers.6.self_attn.o_proj (4096, 4096)
model.layers.6.mlp.gate_proj (12800, 4096)
model.layers.6.mlp.up_proj (12800, 4096)
model.layers.6.mlp.down_proj (4096, 12800)
model.layers.6.input_layernorm (4096,)
model.layers.6.post_attention_layernorm (4096,)
model.layers.7.self_attn.q_proj (4096, 4096)
model.layers.7.self_attn.k_proj (1024, 4096)
model.layers.7.self_attn.v_proj (1024, 4096)
model.layers.7.self_attn.o_proj (4096, 4096)
model.layers.7.mlp.gate_proj (12800, 4096)
model.layers.7.mlp.up_proj (12800, 4096)
model.layers.7.mlp.down_proj (4096, 12800)
model.layers.7.input_layernorm (4096,)
model.layers.7.post_attention_layernorm (4096,)
model.layers.8.self_attn.q_proj (4096, 4096)
model.layers.8.self_attn.k_proj (1024, 4096)
model.layers.8.self_attn.v_proj (1024, 4096)
model.layers.8.self_attn.o_proj (4096, 4096)
model.layers.8.mlp.gate_proj (12800, 4096)
model.layers.8.mlp.up_proj (12800, 4096)
model.layers.8.mlp.down_proj (4096, 12800)
model.layers.8.input_layernorm (4096,)
model.layers.8.post_attention_layernorm (4096,)
model.layers.9.self_attn.q_proj (4096, 4096)
model.layers.9.self_attn.k_proj (1024, 4096)
model.layers.9.self_attn.v_proj (1024, 4096)
model.layers.9.self_attn.o_proj (4096, 4096)
model.layers.9.mlp.gate_proj (12800, 4096)
model.layers.9.mlp.up_proj (12800, 4096)
model.layers.9.mlp.down_proj (4096, 12800)
model.layers.9.input_layernorm (4096,)
model.layers.9.post_attention_layernorm (4096,)
model.layers.10.self_attn.q_proj (4096, 4096)
model.layers.10.self_attn.k_proj (1024, 4096)
model.layers.10.self_attn.v_proj (1024, 4096)
model.layers.10.self_attn.o_proj (4096, 4096)
model.layers.10.mlp.gate_proj (12800, 4096)
model.layers.10.mlp.up_proj (12800, 4096)
model.layers.10.mlp.down_proj (4096, 12800)
model.layers.10.input_layernorm (4096,)
model.layers.10.post_attention_layernorm (4096,)
model.layers.11.self_attn.q_proj (4096, 4096)
model.layers.11.self_attn.k_proj (1024, 4096)
model.layers.11.self_attn.v_proj (1024, 4096)
model.layers.11.self_attn.o_proj (4096, 4096)
model.layers.11.mlp.gate_proj (12800, 4096)
model.layers.11.mlp.up_proj (12800, 4096)
model.layers.11.mlp.down_proj (4096, 12800)
model.layers.11.input_layernorm (4096,)
model.layers.11.post_attention_layernorm (4096,)
model.layers.12.self_attn.q_proj (4096, 4096)
model.layers.12.self_attn.k_proj (1024, 4096)
model.layers.12.self_attn.v_proj (1024, 4096)
model.layers.12.self_attn.o_proj (4096, 4096)
model.layers.12.mlp.gate_proj (12800, 4096)
model.layers.12.mlp.up_proj (12800, 4096)
model.layers.12.mlp.down_proj (4096, 12800)
model.layers.12.input_layernorm (4096,)
model.layers.12.post_attention_layernorm (4096,)
model.layers.13.self_attn.q_proj (4096, 4096)
model.layers.13.self_attn.k_proj (1024, 4096)
model.layers.13.self_attn.v_proj (1024, 4096)
model.layers.13.self_attn.o_proj (4096, 4096)
model.layers.13.mlp.gate_proj (12800, 4096)
model.layers.13.mlp.up_proj (12800, 4096)
model.layers.13.mlp.down_proj (4096, 12800)
model.layers.13.input_layernorm (4096,)
model.layers.13.post_attention_layernorm (4096,)
model.layers.14.self_attn.q_proj (4096, 4096)
model.layers.14.self_attn.k_proj (1024, 4096)
model.layers.14.self_attn.v_proj (1024, 4096)
model.layers.14.self_attn.o_proj (4096, 4096)
model.layers.14.mlp.gate_proj (12800, 4096)
model.layers.14.mlp.up_proj (12800, 4096)
model.layers.14.mlp.down_proj (4096, 12800)
model.layers.14.input_layernorm (4096,)
model.layers.14.post_attention_layernorm (4096,)
model.layers.15.self_attn.q_proj (4096, 4096)
model.layers.15.self_attn.k_proj (1024, 4096)
model.layers.15.self_attn.v_proj (1024, 4096)
model.layers.15.self_attn.o_proj (4096, 4096)
model.layers.15.mlp.gate_proj (12800, 4096)
model.layers.15.mlp.up_proj (12800, 4096)
model.layers.15.mlp.down_proj (4096, 12800)
model.layers.15.input_layernorm (4096,)
model.layers.15.post_attention_layernorm (4096,)
model.layers.16.self_attn.q_proj (4096, 4096)
model.layers.16.self_attn.k_proj (1024, 4096)
model.layers.16.self_attn.v_proj (1024, 4096)
model.layers.16.self_attn.o_proj (4096, 4096)
model.layers.16.mlp.gate_proj (12800, 4096)
model.layers.16.mlp.up_proj (12800, 4096)
model.layers.16.mlp.down_proj (4096, 12800)
model.layers.16.input_layernorm (4096,)
model.layers.16.post_attention_layernorm (4096,)
model.layers.17.self_attn.q_proj (4096, 4096)
model.layers.17.self_attn.k_proj (1024, 4096)
model.layers.17.self_attn.v_proj (1024, 4096)
model.layers.17.self_attn.o_proj (4096, 4096)
model.layers.17.mlp.gate_proj (12800, 4096)
model.layers.17.mlp.up_proj (12800, 4096)
model.layers.17.mlp.down_proj (4096, 12800)
model.layers.17.input_layernorm (4096,)
model.layers.17.post_attention_layernorm (4096,)
model.layers.18.self_attn.q_proj (4096, 4096)
model.layers.18.self_attn.k_proj (1024, 4096)
model.layers.18.self_attn.v_proj (1024, 4096)
model.layers.18.self_attn.o_proj (4096, 4096)
model.layers.18.mlp.gate_proj (12800, 4096)
model.layers.18.mlp.up_proj (12800, 4096)
model.layers.18.mlp.down_proj (4096, 12800)
model.layers.18.input_layernorm (4096,)
model.layers.18.post_attention_layernorm (4096,)
model.layers.19.self_attn.q_proj (4096, 4096)
model.layers.19.self_attn.k_proj (1024, 4096)
model.layers.19.self_attn.v_proj (1024, 4096)
model.layers.19.self_attn.o_proj (4096, 4096)
model.layers.19.mlp.gate_proj (12800, 4096)
model.layers.19.mlp.up_proj (12800, 4096)
model.layers.19.mlp.down_proj (4096, 12800)
model.layers.19.input_layernorm (4096,)
model.layers.19.post_attention_layernorm (4096,)
model.layers.20.self_attn.q_proj (4096, 4096)
model.layers.20.self_attn.k_proj (1024, 4096)
model.layers.20.self_attn.v_proj (1024, 4096)
model.layers.20.self_attn.o_proj (4096, 4096)
model.layers.20.mlp.gate_proj (12800, 4096)
model.layers.20.mlp.up_proj (12800, 4096)
model.layers.20.mlp.down_proj (4096, 12800)
model.layers.20.input_layernorm (4096,)
model.layers.20.post_attention_layernorm (4096,)
model.layers.21.self_attn.q_proj (4096, 4096)
model.layers.21.self_attn.k_proj (1024, 4096)
model.layers.21.self_attn.v_proj (1024, 4096)
model.layers.21.self_attn.o_proj (4096, 4096)
model.layers.21.mlp.gate_proj (12800, 4096)
model.layers.21.mlp.up_proj (12800, 4096)
model.layers.21.mlp.down_proj (4096, 12800)
model.layers.21.input_layernorm (4096,)
model.layers.21.post_attention_layernorm (4096,)
model.layers.22.self_attn.q_proj (4096, 4096)
model.layers.22.self_attn.k_proj (1024, 4096)
model.layers.22.self_attn.v_proj (1024, 4096)
model.layers.22.self_attn.o_proj (4096, 4096)
model.layers.22.mlp.gate_proj (12800, 4096)
model.layers.22.mlp.up_proj (12800, 4096)
model.layers.22.mlp.down_proj (4096, 12800)
model.layers.22.input_layernorm (4096,)
model.layers.22.post_attention_layernorm (4096,)
model.layers.23.self_attn.q_proj (4096, 4096)
model.layers.23.self_attn.k_proj (1024, 4096)
model.layers.23.self_attn.v_proj (1024, 4096)
model.layers.23.self_attn.o_proj (4096, 4096)
model.layers.23.mlp.gate_proj (12800, 4096)
model.layers.23.mlp.up_proj (12800, 4096)
model.layers.23.mlp.down_proj (4096, 12800)
model.layers.23.input_layernorm (4096,)
model.layers.23.post_attention_layernorm (4096,)
model.layers.24.self_attn.q_proj (4096, 4096)
model.layers.24.self_attn.k_proj (1024, 4096)
model.layers.24.self_attn.v_proj (1024, 4096)
model.layers.24.self_attn.o_proj (4096, 4096)
model.layers.24.mlp.gate_proj (12800, 4096)
model.layers.24.mlp.up_proj (12800, 4096)
model.layers.24.mlp.down_proj (4096, 12800)
model.layers.24.input_layernorm (4096,)
model.layers.24.post_attention_layernorm (4096,)
model.layers.25.self_attn.q_proj (4096, 4096)
model.layers.25.self_attn.k_proj (1024, 4096)
model.layers.25.self_attn.v_proj (1024, 4096)
model.layers.25.self_attn.o_proj (4096, 4096)
model.layers.25.mlp.gate_proj (12800, 4096)
model.layers.25.mlp.up_proj (12800, 4096)
model.layers.25.mlp.down_proj (4096, 12800)
model.layers.25.input_layernorm (4096,)
model.layers.25.post_attention_layernorm (4096,)
model.layers.26.self_attn.q_proj (4096, 4096)
model.layers.26.self_attn.k_proj (1024, 4096)
model.layers.26.self_attn.v_proj (1024, 4096)
model.layers.26.self_attn.o_proj (4096, 4096)
model.layers.26.mlp.gate_proj (12800, 4096)
model.layers.26.mlp.up_proj (12800, 4096)
model.layers.26.mlp.down_proj (4096, 12800)
model.layers.26.input_layernorm (4096,)
model.layers.26.post_attention_layernorm (4096,)
model.layers.27.self_attn.q_proj (4096, 4096)
model.layers.27.self_attn.k_proj (1024, 4096)
model.layers.27.self_attn.v_proj (1024, 4096)
model.layers.27.self_attn.o_proj (4096, 4096)
model.layers.27.mlp.gate_proj (12800, 4096)
model.layers.27.mlp.up_proj (12800, 4096)
model.layers.27.mlp.down_proj (4096, 12800)
model.layers.27.input_layernorm (4096,)
model.layers.27.post_attention_layernorm (4096,)
model.layers.28.self_attn.q_proj (4096, 4096)
model.layers.28.self_attn.k_proj (1024, 4096)
model.layers.28.self_attn.v_proj (1024, 4096)
model.layers.28.self_attn.o_proj (4096, 4096)
model.layers.28.mlp.gate_proj (12800, 4096)
model.layers.28.mlp.up_proj (12800, 4096)
model.layers.28.mlp.down_proj (4096, 12800)
model.layers.28.input_layernorm (4096,)
model.layers.28.post_attention_layernorm (4096,)
model.layers.29.self_attn.q_proj (4096, 4096)
model.layers.29.self_attn.k_proj (1024, 4096)
model.layers.29.self_attn.v_proj (1024, 4096)
model.layers.29.self_attn.o_proj (4096, 4096)
model.layers.29.mlp.gate_proj (12800, 4096)
model.layers.29.mlp.up_proj (12800, 4096)
model.layers.29.mlp.down_proj (4096, 12800)
model.layers.29.input_layernorm (4096,)
model.layers.29.post_attention_layernorm (4096,)
model.layers.30.self_attn.q_proj (4096, 4096)
model.layers.30.self_attn.k_proj (1024, 4096)
model.layers.30.self_attn.v_proj (1024, 4096)
model.layers.30.self_attn.o_proj (4096, 4096)
model.layers.30.mlp.gate_proj (12800, 4096)
model.layers.30.mlp.up_proj (12800, 4096)
model.layers.30.mlp.down_proj (4096, 12800)
model.layers.30.input_layernorm (4096,)
model.layers.30.post_attention_layernorm (4096,)
model.layers.31.self_attn.q_proj (4096, 4096)
model.layers.31.self_attn.k_proj (1024, 4096)
model.layers.31.self_attn.v_proj (1024, 4096)
model.layers.31.self_attn.o_proj (4096, 4096)
model.layers.31.mlp.gate_proj (12800, 4096)
model.layers.31.mlp.up_proj (12800, 4096)
model.layers.31.mlp.down_proj (4096, 12800)
model.layers.31.input_layernorm (4096,)
model.layers.31.post_attention_layernorm (4096,)
model.layers.32.self_attn.q_proj (4096, 4096)
model.layers.32.self_attn.k_proj (1024, 4096)
model.layers.32.self_attn.v_proj (1024, 4096)
model.layers.32.self_attn.o_proj (4096, 4096)
model.layers.32.mlp.gate_proj (12800, 4096)
model.layers.32.mlp.up_proj (12800, 4096)
model.layers.32.mlp.down_proj (4096, 12800)
model.layers.32.input_layernorm (4096,)
model.layers.32.post_attention_layernorm (4096,)
model.layers.33.self_attn.q_proj (4096, 4096)
model.layers.33.self_attn.k_proj (1024, 4096)
model.layers.33.self_attn.v_proj (1024, 4096)
model.layers.33.self_attn.o_proj (4096, 4096)
model.layers.33.mlp.gate_proj (12800, 4096)
model.layers.33.mlp.up_proj (12800, 4096)
model.layers.33.mlp.down_proj (4096, 12800)
model.layers.33.input_layernorm (4096,)
model.layers.33.post_attention_layernorm (4096,)
model.layers.34.self_attn.q_proj (4096, 4096)
model.layers.34.self_attn.k_proj (1024, 4096)
model.layers.34.self_attn.v_proj (1024, 4096)
model.layers.34.self_attn.o_proj (4096, 4096)
model.layers.34.mlp.gate_proj (12800, 4096)
model.layers.34.mlp.up_proj (12800, 4096)
model.layers.34.mlp.down_proj (4096, 12800)
model.layers.34.input_layernorm (4096,)
model.layers.34.post_attention_layernorm (4096,)
model.layers.35.self_attn.q_proj (4096, 4096)
model.layers.35.self_attn.k_proj (1024, 4096)
model.layers.35.self_attn.v_proj (1024, 4096)
model.layers.35.self_attn.o_proj (4096, 4096)
model.layers.35.mlp.gate_proj (12800, 4096)
model.layers.35.mlp.up_proj (12800, 4096)
model.layers.35.mlp.down_proj (4096, 12800)
model.layers.35.input_layernorm (4096,)
model.layers.35.post_attention_layernorm (4096,)
model.layers.36.self_attn.q_proj (4096, 4096)
model.layers.36.self_attn.k_proj (1024, 4096)
model.layers.36.self_attn.v_proj (1024, 4096)
model.layers.36.self_attn.o_proj (4096, 4096)
model.layers.36.mlp.gate_proj (12800, 4096)
model.layers.36.mlp.up_proj (12800, 4096)
model.layers.36.mlp.down_proj (4096, 12800)
model.layers.36.input_layernorm (4096,)
model.layers.36.post_attention_layernorm (4096,)
model.layers.37.self_attn.q_proj (4096, 4096)
model.layers.37.self_attn.k_proj (1024, 4096)
model.layers.37.self_attn.v_proj (1024, 4096)
model.layers.37.self_attn.o_proj (4096, 4096)
model.layers.37.mlp.gate_proj (12800, 4096)
model.layers.37.mlp.up_proj (12800, 4096)
model.layers.37.mlp.down_proj (4096, 12800)
model.layers.37.input_layernorm (4096,)
model.layers.37.post_attention_layernorm (4096,)
model.layers.38.self_attn.q_proj (4096, 4096)
model.layers.38.self_attn.k_proj (1024, 4096)
model.layers.38.self_attn.v_proj (1024, 4096)
model.layers.38.self_attn.o_proj (4096, 4096)
model.layers.38.mlp.gate_proj (12800, 4096)
model.layers.38.mlp.up_proj (12800, 4096)
model.layers.38.mlp.down_proj (4096, 12800)
model.layers.38.input_layernorm (4096,)
model.layers.38.post_attention_layernorm (4096,)
model.layers.39.self_attn.q_proj (4096, 4096)
model.layers.39.self_attn.k_proj (1024, 4096)
model.layers.39.self_attn.v_proj (1024, 4096)
model.layers.39.self_attn.o_proj (4096, 4096)
model.layers.39.mlp.gate_proj (12800, 4096)
model.layers.39.mlp.up_proj (12800, 4096)
model.layers.39.mlp.down_proj (4096, 12800)
model.layers.39.input_layernorm (4096,)
model.layers.39.post_attention_layernorm (4096,)
model.norm (4096,)
lm_head (49159, 4096)
```

## config.json

```json
{
  "architectures": [
    "GraniteForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attention_multiplier": 0.0078125,
  "bos_token_id": 0,
  "embedding_multiplier": 12.0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12800,
  "logits_scaling": 16.0,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "granite",
  "num_attention_heads": 32,
  "num_hidden_layers": 40,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "residual_multiplier": 0.22,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.49.0",
  "use_cache": true,
  "vocab_size": 49159
}
```

---

# deepseek-ai/DeepSeek-V3

## Parameter shapes

```text
ERROR while building empty model / extracting architecture:
cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils' (C:\Users\liora\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\utils\import_utils.py)
```

## config.json

```json
{
  "architectures": [
    "DeepseekV3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV3Config",
    "AutoModel": "modeling_deepseek.DeepseekV3Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
  },
  "bos_token_id": 0,
  "eos_token_id": 1,
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "initializer_range": 0.02,
  "intermediate_size": 18432,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v3",
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 128,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 61,
  "num_key_value_heads": 128,
  "num_nextn_predict_layers": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 4,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.33.1",
  "use_cache": true,
  "v_head_dim": 128,
  "vocab_size": 129280
}
```

---

# HuggingFaceTB/SmolLM2-1.7B-Instruct

## Parameter shapes

```text
model.embed_tokens (49152, 2048)
model.layers.0.self_attn.q_proj (2048, 2048)
model.layers.0.self_attn.k_proj (2048, 2048)
model.layers.0.self_attn.v_proj (2048, 2048)
model.layers.0.self_attn.o_proj (2048, 2048)
model.layers.0.mlp.gate_proj (8192, 2048)
model.layers.0.mlp.up_proj (8192, 2048)
model.layers.0.mlp.down_proj (2048, 8192)
model.layers.0.input_layernorm (2048,)
model.layers.0.post_attention_layernorm (2048,)
model.layers.1.self_attn.q_proj (2048, 2048)
model.layers.1.self_attn.k_proj (2048, 2048)
model.layers.1.self_attn.v_proj (2048, 2048)
model.layers.1.self_attn.o_proj (2048, 2048)
model.layers.1.mlp.gate_proj (8192, 2048)
model.layers.1.mlp.up_proj (8192, 2048)
model.layers.1.mlp.down_proj (2048, 8192)
model.layers.1.input_layernorm (2048,)
model.layers.1.post_attention_layernorm (2048,)
model.layers.2.self_attn.q_proj (2048, 2048)
model.layers.2.self_attn.k_proj (2048, 2048)
model.layers.2.self_attn.v_proj (2048, 2048)
model.layers.2.self_attn.o_proj (2048, 2048)
model.layers.2.mlp.gate_proj (8192, 2048)
model.layers.2.mlp.up_proj (8192, 2048)
model.layers.2.mlp.down_proj (2048, 8192)
model.layers.2.input_layernorm (2048,)
model.layers.2.post_attention_layernorm (2048,)
model.layers.3.self_attn.q_proj (2048, 2048)
model.layers.3.self_attn.k_proj (2048, 2048)
model.layers.3.self_attn.v_proj (2048, 2048)
model.layers.3.self_attn.o_proj (2048, 2048)
model.layers.3.mlp.gate_proj (8192, 2048)
model.layers.3.mlp.up_proj (8192, 2048)
model.layers.3.mlp.down_proj (2048, 8192)
model.layers.3.input_layernorm (2048,)
model.layers.3.post_attention_layernorm (2048,)
model.layers.4.self_attn.q_proj (2048, 2048)
model.layers.4.self_attn.k_proj (2048, 2048)
model.layers.4.self_attn.v_proj (2048, 2048)
model.layers.4.self_attn.o_proj (2048, 2048)
model.layers.4.mlp.gate_proj (8192, 2048)
model.layers.4.mlp.up_proj (8192, 2048)
model.layers.4.mlp.down_proj (2048, 8192)
model.layers.4.input_layernorm (2048,)
model.layers.4.post_attention_layernorm (2048,)
model.layers.5.self_attn.q_proj (2048, 2048)
model.layers.5.self_attn.k_proj (2048, 2048)
model.layers.5.self_attn.v_proj (2048, 2048)
model.layers.5.self_attn.o_proj (2048, 2048)
model.layers.5.mlp.gate_proj (8192, 2048)
model.layers.5.mlp.up_proj (8192, 2048)
model.layers.5.mlp.down_proj (2048, 8192)
model.layers.5.input_layernorm (2048,)
model.layers.5.post_attention_layernorm (2048,)
model.layers.6.self_attn.q_proj (2048, 2048)
model.layers.6.self_attn.k_proj (2048, 2048)
model.layers.6.self_attn.v_proj (2048, 2048)
model.layers.6.self_attn.o_proj (2048, 2048)
model.layers.6.mlp.gate_proj (8192, 2048)
model.layers.6.mlp.up_proj (8192, 2048)
model.layers.6.mlp.down_proj (2048, 8192)
model.layers.6.input_layernorm (2048,)
model.layers.6.post_attention_layernorm (2048,)
model.layers.7.self_attn.q_proj (2048, 2048)
model.layers.7.self_attn.k_proj (2048, 2048)
model.layers.7.self_attn.v_proj (2048, 2048)
model.layers.7.self_attn.o_proj (2048, 2048)
model.layers.7.mlp.gate_proj (8192, 2048)
model.layers.7.mlp.up_proj (8192, 2048)
model.layers.7.mlp.down_proj (2048, 8192)
model.layers.7.input_layernorm (2048,)
model.layers.7.post_attention_layernorm (2048,)
model.layers.8.self_attn.q_proj (2048, 2048)
model.layers.8.self_attn.k_proj (2048, 2048)
model.layers.8.self_attn.v_proj (2048, 2048)
model.layers.8.self_attn.o_proj (2048, 2048)
model.layers.8.mlp.gate_proj (8192, 2048)
model.layers.8.mlp.up_proj (8192, 2048)
model.layers.8.mlp.down_proj (2048, 8192)
model.layers.8.input_layernorm (2048,)
model.layers.8.post_attention_layernorm (2048,)
model.layers.9.self_attn.q_proj (2048, 2048)
model.layers.9.self_attn.k_proj (2048, 2048)
model.layers.9.self_attn.v_proj (2048, 2048)
model.layers.9.self_attn.o_proj (2048, 2048)
model.layers.9.mlp.gate_proj (8192, 2048)
model.layers.9.mlp.up_proj (8192, 2048)
model.layers.9.mlp.down_proj (2048, 8192)
model.layers.9.input_layernorm (2048,)
model.layers.9.post_attention_layernorm (2048,)
model.layers.10.self_attn.q_proj (2048, 2048)
model.layers.10.self_attn.k_proj (2048, 2048)
model.layers.10.self_attn.v_proj (2048, 2048)
model.layers.10.self_attn.o_proj (2048, 2048)
model.layers.10.mlp.gate_proj (8192, 2048)
model.layers.10.mlp.up_proj (8192, 2048)
model.layers.10.mlp.down_proj (2048, 8192)
model.layers.10.input_layernorm (2048,)
model.layers.10.post_attention_layernorm (2048,)
model.layers.11.self_attn.q_proj (2048, 2048)
model.layers.11.self_attn.k_proj (2048, 2048)
model.layers.11.self_attn.v_proj (2048, 2048)
model.layers.11.self_attn.o_proj (2048, 2048)
model.layers.11.mlp.gate_proj (8192, 2048)
model.layers.11.mlp.up_proj (8192, 2048)
model.layers.11.mlp.down_proj (2048, 8192)
model.layers.11.input_layernorm (2048,)
model.layers.11.post_attention_layernorm (2048,)
model.layers.12.self_attn.q_proj (2048, 2048)
model.layers.12.self_attn.k_proj (2048, 2048)
model.layers.12.self_attn.v_proj (2048, 2048)
model.layers.12.self_attn.o_proj (2048, 2048)
model.layers.12.mlp.gate_proj (8192, 2048)
model.layers.12.mlp.up_proj (8192, 2048)
model.layers.12.mlp.down_proj (2048, 8192)
model.layers.12.input_layernorm (2048,)
model.layers.12.post_attention_layernorm (2048,)
model.layers.13.self_attn.q_proj (2048, 2048)
model.layers.13.self_attn.k_proj (2048, 2048)
model.layers.13.self_attn.v_proj (2048, 2048)
model.layers.13.self_attn.o_proj (2048, 2048)
model.layers.13.mlp.gate_proj (8192, 2048)
model.layers.13.mlp.up_proj (8192, 2048)
model.layers.13.mlp.down_proj (2048, 8192)
model.layers.13.input_layernorm (2048,)
model.layers.13.post_attention_layernorm (2048,)
model.layers.14.self_attn.q_proj (2048, 2048)
model.layers.14.self_attn.k_proj (2048, 2048)
model.layers.14.self_attn.v_proj (2048, 2048)
model.layers.14.self_attn.o_proj (2048, 2048)
model.layers.14.mlp.gate_proj (8192, 2048)
model.layers.14.mlp.up_proj (8192, 2048)
model.layers.14.mlp.down_proj (2048, 8192)
model.layers.14.input_layernorm (2048,)
model.layers.14.post_attention_layernorm (2048,)
model.layers.15.self_attn.q_proj (2048, 2048)
model.layers.15.self_attn.k_proj (2048, 2048)
model.layers.15.self_attn.v_proj (2048, 2048)
model.layers.15.self_attn.o_proj (2048, 2048)
model.layers.15.mlp.gate_proj (8192, 2048)
model.layers.15.mlp.up_proj (8192, 2048)
model.layers.15.mlp.down_proj (2048, 8192)
model.layers.15.input_layernorm (2048,)
model.layers.15.post_attention_layernorm (2048,)
model.layers.16.self_attn.q_proj (2048, 2048)
model.layers.16.self_attn.k_proj (2048, 2048)
model.layers.16.self_attn.v_proj (2048, 2048)
model.layers.16.self_attn.o_proj (2048, 2048)
model.layers.16.mlp.gate_proj (8192, 2048)
model.layers.16.mlp.up_proj (8192, 2048)
model.layers.16.mlp.down_proj (2048, 8192)
model.layers.16.input_layernorm (2048,)
model.layers.16.post_attention_layernorm (2048,)
model.layers.17.self_attn.q_proj (2048, 2048)
model.layers.17.self_attn.k_proj (2048, 2048)
model.layers.17.self_attn.v_proj (2048, 2048)
model.layers.17.self_attn.o_proj (2048, 2048)
model.layers.17.mlp.gate_proj (8192, 2048)
model.layers.17.mlp.up_proj (8192, 2048)
model.layers.17.mlp.down_proj (2048, 8192)
model.layers.17.input_layernorm (2048,)
model.layers.17.post_attention_layernorm (2048,)
model.layers.18.self_attn.q_proj (2048, 2048)
model.layers.18.self_attn.k_proj (2048, 2048)
model.layers.18.self_attn.v_proj (2048, 2048)
model.layers.18.self_attn.o_proj (2048, 2048)
model.layers.18.mlp.gate_proj (8192, 2048)
model.layers.18.mlp.up_proj (8192, 2048)
model.layers.18.mlp.down_proj (2048, 8192)
model.layers.18.input_layernorm (2048,)
model.layers.18.post_attention_layernorm (2048,)
model.layers.19.self_attn.q_proj (2048, 2048)
model.layers.19.self_attn.k_proj (2048, 2048)
model.layers.19.self_attn.v_proj (2048, 2048)
model.layers.19.self_attn.o_proj (2048, 2048)
model.layers.19.mlp.gate_proj (8192, 2048)
model.layers.19.mlp.up_proj (8192, 2048)
model.layers.19.mlp.down_proj (2048, 8192)
model.layers.19.input_layernorm (2048,)
model.layers.19.post_attention_layernorm (2048,)
model.layers.20.self_attn.q_proj (2048, 2048)
model.layers.20.self_attn.k_proj (2048, 2048)
model.layers.20.self_attn.v_proj (2048, 2048)
model.layers.20.self_attn.o_proj (2048, 2048)
model.layers.20.mlp.gate_proj (8192, 2048)
model.layers.20.mlp.up_proj (8192, 2048)
model.layers.20.mlp.down_proj (2048, 8192)
model.layers.20.input_layernorm (2048,)
model.layers.20.post_attention_layernorm (2048,)
model.layers.21.self_attn.q_proj (2048, 2048)
model.layers.21.self_attn.k_proj (2048, 2048)
model.layers.21.self_attn.v_proj (2048, 2048)
model.layers.21.self_attn.o_proj (2048, 2048)
model.layers.21.mlp.gate_proj (8192, 2048)
model.layers.21.mlp.up_proj (8192, 2048)
model.layers.21.mlp.down_proj (2048, 8192)
model.layers.21.input_layernorm (2048,)
model.layers.21.post_attention_layernorm (2048,)
model.layers.22.self_attn.q_proj (2048, 2048)
model.layers.22.self_attn.k_proj (2048, 2048)
model.layers.22.self_attn.v_proj (2048, 2048)
model.layers.22.self_attn.o_proj (2048, 2048)
model.layers.22.mlp.gate_proj (8192, 2048)
model.layers.22.mlp.up_proj (8192, 2048)
model.layers.22.mlp.down_proj (2048, 8192)
model.layers.22.input_layernorm (2048,)
model.layers.22.post_attention_layernorm (2048,)
model.layers.23.self_attn.q_proj (2048, 2048)
model.layers.23.self_attn.k_proj (2048, 2048)
model.layers.23.self_attn.v_proj (2048, 2048)
model.layers.23.self_attn.o_proj (2048, 2048)
model.layers.23.mlp.gate_proj (8192, 2048)
model.layers.23.mlp.up_proj (8192, 2048)
model.layers.23.mlp.down_proj (2048, 8192)
model.layers.23.input_layernorm (2048,)
model.layers.23.post_attention_layernorm (2048,)
model.norm (2048,)
lm_head (49152, 2048)
```

## config.json

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "pad_token_id": 2,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 130000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "transformers.js_config": {
    "dtype": "q4",
    "kv_cache_dtype": {
      "q4f16": "float16",
      "fp16": "float16"
    },
    "use_external_data_format": {
      "model.onnx": true,
      "model_fp16.onnx": true
    }
  },
  "use_cache": true,
  "vocab_size": 49152
}
```

---

# microsoft/Phi-4-mini-instruct

## Parameter shapes

```text
ERROR while building empty model / extracting architecture:
cannot import name 'LossKwargs' from 'transformers.utils' (C:\Users\liora\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\utils\__init__.py)
```

## config.json

```json
{
  "_name_or_path": "Phi-4-mini-instruct",
  "architectures": [
    "Phi3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_phi3.Phi3Config",
    "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
    "AutoTokenizer": "Xenova/gpt-4o"
  },
  "bos_token_id": 199999,
  "embd_pdrop": 0.0,
  "eos_token_id": 199999,
  "full_attn_mod": 1,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "interpolate_factor": 1,
  "lm_head_bias": false,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "phi3",
  "num_attention_heads": 24,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "original_max_position_embeddings": 4096,
  "pad_token_id": 199999,
  "partial_rotary_factor": 0.75,
  "resid_pdrop": 0.0,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "long_factor": [
      1,
      1.118320672,
      1.250641126,
      1.398617824,
      1.564103225,
      1.74916897,
      1.956131817,
      2.187582649,
      2.446418898,
      2.735880826,
      3.059592084,
      3.421605075,
      3.826451687,
      4.279200023,
      4.785517845,
      5.351743533,
      5.984965424,
      6.693110555,
      7.485043894,
      8.370679318,
      9.36110372,
      10.4687158,
      11.70738129,
      13.09260651,
      14.64173252,
      16.37415215,
      18.31155283,
      20.47818807,
      22.90118105,
      25.61086418,
      28.64115884,
      32.03,
      32.1,
      32.13,
      32.23,
      32.6,
      32.61,
      32.64,
      32.66,
      32.7,
      32.71,
      32.93,
      32.97,
      33.28,
      33.49,
      33.5,
      44.16,
      47.77
    ],
    "short_factor": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "type": "longrope"
  },
  "rope_theta": 10000.0,
  "sliding_window": 262144,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 200064
}
```

---

# tiiuae/Falcon3-7B-Instruct

## Parameter shapes

```text
model.embed_tokens (131072, 3072)
model.layers.0.self_attn.q_proj (3072, 3072)
model.layers.0.self_attn.k_proj (1024, 3072)
model.layers.0.self_attn.v_proj (1024, 3072)
model.layers.0.self_attn.o_proj (3072, 3072)
model.layers.0.mlp.gate_proj (23040, 3072)
model.layers.0.mlp.up_proj (23040, 3072)
model.layers.0.mlp.down_proj (3072, 23040)
model.layers.0.input_layernorm (3072,)
model.layers.0.post_attention_layernorm (3072,)
model.layers.1.self_attn.q_proj (3072, 3072)
model.layers.1.self_attn.k_proj (1024, 3072)
model.layers.1.self_attn.v_proj (1024, 3072)
model.layers.1.self_attn.o_proj (3072, 3072)
model.layers.1.mlp.gate_proj (23040, 3072)
model.layers.1.mlp.up_proj (23040, 3072)
model.layers.1.mlp.down_proj (3072, 23040)
model.layers.1.input_layernorm (3072,)
model.layers.1.post_attention_layernorm (3072,)
model.layers.2.self_attn.q_proj (3072, 3072)
model.layers.2.self_attn.k_proj (1024, 3072)
model.layers.2.self_attn.v_proj (1024, 3072)
model.layers.2.self_attn.o_proj (3072, 3072)
model.layers.2.mlp.gate_proj (23040, 3072)
model.layers.2.mlp.up_proj (23040, 3072)
model.layers.2.mlp.down_proj (3072, 23040)
model.layers.2.input_layernorm (3072,)
model.layers.2.post_attention_layernorm (3072,)
model.layers.3.self_attn.q_proj (3072, 3072)
model.layers.3.self_attn.k_proj (1024, 3072)
model.layers.3.self_attn.v_proj (1024, 3072)
model.layers.3.self_attn.o_proj (3072, 3072)
model.layers.3.mlp.gate_proj (23040, 3072)
model.layers.3.mlp.up_proj (23040, 3072)
model.layers.3.mlp.down_proj (3072, 23040)
model.layers.3.input_layernorm (3072,)
model.layers.3.post_attention_layernorm (3072,)
model.layers.4.self_attn.q_proj (3072, 3072)
model.layers.4.self_attn.k_proj (1024, 3072)
model.layers.4.self_attn.v_proj (1024, 3072)
model.layers.4.self_attn.o_proj (3072, 3072)
model.layers.4.mlp.gate_proj (23040, 3072)
model.layers.4.mlp.up_proj (23040, 3072)
model.layers.4.mlp.down_proj (3072, 23040)
model.layers.4.input_layernorm (3072,)
model.layers.4.post_attention_layernorm (3072,)
model.layers.5.self_attn.q_proj (3072, 3072)
model.layers.5.self_attn.k_proj (1024, 3072)
model.layers.5.self_attn.v_proj (1024, 3072)
model.layers.5.self_attn.o_proj (3072, 3072)
model.layers.5.mlp.gate_proj (23040, 3072)
model.layers.5.mlp.up_proj (23040, 3072)
model.layers.5.mlp.down_proj (3072, 23040)
model.layers.5.input_layernorm (3072,)
model.layers.5.post_attention_layernorm (3072,)
model.layers.6.self_attn.q_proj (3072, 3072)
model.layers.6.self_attn.k_proj (1024, 3072)
model.layers.6.self_attn.v_proj (1024, 3072)
model.layers.6.self_attn.o_proj (3072, 3072)
model.layers.6.mlp.gate_proj (23040, 3072)
model.layers.6.mlp.up_proj (23040, 3072)
model.layers.6.mlp.down_proj (3072, 23040)
model.layers.6.input_layernorm (3072,)
model.layers.6.post_attention_layernorm (3072,)
model.layers.7.self_attn.q_proj (3072, 3072)
model.layers.7.self_attn.k_proj (1024, 3072)
model.layers.7.self_attn.v_proj (1024, 3072)
model.layers.7.self_attn.o_proj (3072, 3072)
model.layers.7.mlp.gate_proj (23040, 3072)
model.layers.7.mlp.up_proj (23040, 3072)
model.layers.7.mlp.down_proj (3072, 23040)
model.layers.7.input_layernorm (3072,)
model.layers.7.post_attention_layernorm (3072,)
model.layers.8.self_attn.q_proj (3072, 3072)
model.layers.8.self_attn.k_proj (1024, 3072)
model.layers.8.self_attn.v_proj (1024, 3072)
model.layers.8.self_attn.o_proj (3072, 3072)
model.layers.8.mlp.gate_proj (23040, 3072)
model.layers.8.mlp.up_proj (23040, 3072)
model.layers.8.mlp.down_proj (3072, 23040)
model.layers.8.input_layernorm (3072,)
model.layers.8.post_attention_layernorm (3072,)
model.layers.9.self_attn.q_proj (3072, 3072)
model.layers.9.self_attn.k_proj (1024, 3072)
model.layers.9.self_attn.v_proj (1024, 3072)
model.layers.9.self_attn.o_proj (3072, 3072)
model.layers.9.mlp.gate_proj (23040, 3072)
model.layers.9.mlp.up_proj (23040, 3072)
model.layers.9.mlp.down_proj (3072, 23040)
model.layers.9.input_layernorm (3072,)
model.layers.9.post_attention_layernorm (3072,)
model.layers.10.self_attn.q_proj (3072, 3072)
model.layers.10.self_attn.k_proj (1024, 3072)
model.layers.10.self_attn.v_proj (1024, 3072)
model.layers.10.self_attn.o_proj (3072, 3072)
model.layers.10.mlp.gate_proj (23040, 3072)
model.layers.10.mlp.up_proj (23040, 3072)
model.layers.10.mlp.down_proj (3072, 23040)
model.layers.10.input_layernorm (3072,)
model.layers.10.post_attention_layernorm (3072,)
model.layers.11.self_attn.q_proj (3072, 3072)
model.layers.11.self_attn.k_proj (1024, 3072)
model.layers.11.self_attn.v_proj (1024, 3072)
model.layers.11.self_attn.o_proj (3072, 3072)
model.layers.11.mlp.gate_proj (23040, 3072)
model.layers.11.mlp.up_proj (23040, 3072)
model.layers.11.mlp.down_proj (3072, 23040)
model.layers.11.input_layernorm (3072,)
model.layers.11.post_attention_layernorm (3072,)
model.layers.12.self_attn.q_proj (3072, 3072)
model.layers.12.self_attn.k_proj (1024, 3072)
model.layers.12.self_attn.v_proj (1024, 3072)
model.layers.12.self_attn.o_proj (3072, 3072)
model.layers.12.mlp.gate_proj (23040, 3072)
model.layers.12.mlp.up_proj (23040, 3072)
model.layers.12.mlp.down_proj (3072, 23040)
model.layers.12.input_layernorm (3072,)
model.layers.12.post_attention_layernorm (3072,)
model.layers.13.self_attn.q_proj (3072, 3072)
model.layers.13.self_attn.k_proj (1024, 3072)
model.layers.13.self_attn.v_proj (1024, 3072)
model.layers.13.self_attn.o_proj (3072, 3072)
model.layers.13.mlp.gate_proj (23040, 3072)
model.layers.13.mlp.up_proj (23040, 3072)
model.layers.13.mlp.down_proj (3072, 23040)
model.layers.13.input_layernorm (3072,)
model.layers.13.post_attention_layernorm (3072,)
model.layers.14.self_attn.q_proj (3072, 3072)
model.layers.14.self_attn.k_proj (1024, 3072)
model.layers.14.self_attn.v_proj (1024, 3072)
model.layers.14.self_attn.o_proj (3072, 3072)
model.layers.14.mlp.gate_proj (23040, 3072)
model.layers.14.mlp.up_proj (23040, 3072)
model.layers.14.mlp.down_proj (3072, 23040)
model.layers.14.input_layernorm (3072,)
model.layers.14.post_attention_layernorm (3072,)
model.layers.15.self_attn.q_proj (3072, 3072)
model.layers.15.self_attn.k_proj (1024, 3072)
model.layers.15.self_attn.v_proj (1024, 3072)
model.layers.15.self_attn.o_proj (3072, 3072)
model.layers.15.mlp.gate_proj (23040, 3072)
model.layers.15.mlp.up_proj (23040, 3072)
model.layers.15.mlp.down_proj (3072, 23040)
model.layers.15.input_layernorm (3072,)
model.layers.15.post_attention_layernorm (3072,)
model.layers.16.self_attn.q_proj (3072, 3072)
model.layers.16.self_attn.k_proj (1024, 3072)
model.layers.16.self_attn.v_proj (1024, 3072)
model.layers.16.self_attn.o_proj (3072, 3072)
model.layers.16.mlp.gate_proj (23040, 3072)
model.layers.16.mlp.up_proj (23040, 3072)
model.layers.16.mlp.down_proj (3072, 23040)
model.layers.16.input_layernorm (3072,)
model.layers.16.post_attention_layernorm (3072,)
model.layers.17.self_attn.q_proj (3072, 3072)
model.layers.17.self_attn.k_proj (1024, 3072)
model.layers.17.self_attn.v_proj (1024, 3072)
model.layers.17.self_attn.o_proj (3072, 3072)
model.layers.17.mlp.gate_proj (23040, 3072)
model.layers.17.mlp.up_proj (23040, 3072)
model.layers.17.mlp.down_proj (3072, 23040)
model.layers.17.input_layernorm (3072,)
model.layers.17.post_attention_layernorm (3072,)
model.layers.18.self_attn.q_proj (3072, 3072)
model.layers.18.self_attn.k_proj (1024, 3072)
model.layers.18.self_attn.v_proj (1024, 3072)
model.layers.18.self_attn.o_proj (3072, 3072)
model.layers.18.mlp.gate_proj (23040, 3072)
model.layers.18.mlp.up_proj (23040, 3072)
model.layers.18.mlp.down_proj (3072, 23040)
model.layers.18.input_layernorm (3072,)
model.layers.18.post_attention_layernorm (3072,)
model.layers.19.self_attn.q_proj (3072, 3072)
model.layers.19.self_attn.k_proj (1024, 3072)
model.layers.19.self_attn.v_proj (1024, 3072)
model.layers.19.self_attn.o_proj (3072, 3072)
model.layers.19.mlp.gate_proj (23040, 3072)
model.layers.19.mlp.up_proj (23040, 3072)
model.layers.19.mlp.down_proj (3072, 23040)
model.layers.19.input_layernorm (3072,)
model.layers.19.post_attention_layernorm (3072,)
model.layers.20.self_attn.q_proj (3072, 3072)
model.layers.20.self_attn.k_proj (1024, 3072)
model.layers.20.self_attn.v_proj (1024, 3072)
model.layers.20.self_attn.o_proj (3072, 3072)
model.layers.20.mlp.gate_proj (23040, 3072)
model.layers.20.mlp.up_proj (23040, 3072)
model.layers.20.mlp.down_proj (3072, 23040)
model.layers.20.input_layernorm (3072,)
model.layers.20.post_attention_layernorm (3072,)
model.layers.21.self_attn.q_proj (3072, 3072)
model.layers.21.self_attn.k_proj (1024, 3072)
model.layers.21.self_attn.v_proj (1024, 3072)
model.layers.21.self_attn.o_proj (3072, 3072)
model.layers.21.mlp.gate_proj (23040, 3072)
model.layers.21.mlp.up_proj (23040, 3072)
model.layers.21.mlp.down_proj (3072, 23040)
model.layers.21.input_layernorm (3072,)
model.layers.21.post_attention_layernorm (3072,)
model.layers.22.self_attn.q_proj (3072, 3072)
model.layers.22.self_attn.k_proj (1024, 3072)
model.layers.22.self_attn.v_proj (1024, 3072)
model.layers.22.self_attn.o_proj (3072, 3072)
model.layers.22.mlp.gate_proj (23040, 3072)
model.layers.22.mlp.up_proj (23040, 3072)
model.layers.22.mlp.down_proj (3072, 23040)
model.layers.22.input_layernorm (3072,)
model.layers.22.post_attention_layernorm (3072,)
model.layers.23.self_attn.q_proj (3072, 3072)
model.layers.23.self_attn.k_proj (1024, 3072)
model.layers.23.self_attn.v_proj (1024, 3072)
model.layers.23.self_attn.o_proj (3072, 3072)
model.layers.23.mlp.gate_proj (23040, 3072)
model.layers.23.mlp.up_proj (23040, 3072)
model.layers.23.mlp.down_proj (3072, 23040)
model.layers.23.input_layernorm (3072,)
model.layers.23.post_attention_layernorm (3072,)
model.layers.24.self_attn.q_proj (3072, 3072)
model.layers.24.self_attn.k_proj (1024, 3072)
model.layers.24.self_attn.v_proj (1024, 3072)
model.layers.24.self_attn.o_proj (3072, 3072)
model.layers.24.mlp.gate_proj (23040, 3072)
model.layers.24.mlp.up_proj (23040, 3072)
model.layers.24.mlp.down_proj (3072, 23040)
model.layers.24.input_layernorm (3072,)
model.layers.24.post_attention_layernorm (3072,)
model.layers.25.self_attn.q_proj (3072, 3072)
model.layers.25.self_attn.k_proj (1024, 3072)
model.layers.25.self_attn.v_proj (1024, 3072)
model.layers.25.self_attn.o_proj (3072, 3072)
model.layers.25.mlp.gate_proj (23040, 3072)
model.layers.25.mlp.up_proj (23040, 3072)
model.layers.25.mlp.down_proj (3072, 23040)
model.layers.25.input_layernorm (3072,)
model.layers.25.post_attention_layernorm (3072,)
model.layers.26.self_attn.q_proj (3072, 3072)
model.layers.26.self_attn.k_proj (1024, 3072)
model.layers.26.self_attn.v_proj (1024, 3072)
model.layers.26.self_attn.o_proj (3072, 3072)
model.layers.26.mlp.gate_proj (23040, 3072)
model.layers.26.mlp.up_proj (23040, 3072)
model.layers.26.mlp.down_proj (3072, 23040)
model.layers.26.input_layernorm (3072,)
model.layers.26.post_attention_layernorm (3072,)
model.layers.27.self_attn.q_proj (3072, 3072)
model.layers.27.self_attn.k_proj (1024, 3072)
model.layers.27.self_attn.v_proj (1024, 3072)
model.layers.27.self_attn.o_proj (3072, 3072)
model.layers.27.mlp.gate_proj (23040, 3072)
model.layers.27.mlp.up_proj (23040, 3072)
model.layers.27.mlp.down_proj (3072, 23040)
model.layers.27.input_layernorm (3072,)
model.layers.27.post_attention_layernorm (3072,)
model.norm (3072,)
lm_head (131072, 3072)
```

## config.json

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 11,
  "eos_token_id": 11,
  "head_dim": 256,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "intermediate_size": 23040,
  "max_position_embeddings": 32768,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000042,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 131072
}
```

---

# dicta-il/dictalm2.0-instruct

## Parameter shapes

```text
model.embed_tokens (33152, 4096)
model.layers.0.self_attn.q_proj (4096, 4096)
model.layers.0.self_attn.k_proj (1024, 4096)
model.layers.0.self_attn.v_proj (1024, 4096)
model.layers.0.self_attn.o_proj (4096, 4096)
model.layers.0.mlp.gate_proj (14336, 4096)
model.layers.0.mlp.up_proj (14336, 4096)
model.layers.0.mlp.down_proj (4096, 14336)
model.layers.0.input_layernorm (4096,)
model.layers.0.post_attention_layernorm (4096,)
model.layers.1.self_attn.q_proj (4096, 4096)
model.layers.1.self_attn.k_proj (1024, 4096)
model.layers.1.self_attn.v_proj (1024, 4096)
model.layers.1.self_attn.o_proj (4096, 4096)
model.layers.1.mlp.gate_proj (14336, 4096)
model.layers.1.mlp.up_proj (14336, 4096)
model.layers.1.mlp.down_proj (4096, 14336)
model.layers.1.input_layernorm (4096,)
model.layers.1.post_attention_layernorm (4096,)
model.layers.2.self_attn.q_proj (4096, 4096)
model.layers.2.self_attn.k_proj (1024, 4096)
model.layers.2.self_attn.v_proj (1024, 4096)
model.layers.2.self_attn.o_proj (4096, 4096)
model.layers.2.mlp.gate_proj (14336, 4096)
model.layers.2.mlp.up_proj (14336, 4096)
model.layers.2.mlp.down_proj (4096, 14336)
model.layers.2.input_layernorm (4096,)
model.layers.2.post_attention_layernorm (4096,)
model.layers.3.self_attn.q_proj (4096, 4096)
model.layers.3.self_attn.k_proj (1024, 4096)
model.layers.3.self_attn.v_proj (1024, 4096)
model.layers.3.self_attn.o_proj (4096, 4096)
model.layers.3.mlp.gate_proj (14336, 4096)
model.layers.3.mlp.up_proj (14336, 4096)
model.layers.3.mlp.down_proj (4096, 14336)
model.layers.3.input_layernorm (4096,)
model.layers.3.post_attention_layernorm (4096,)
model.layers.4.self_attn.q_proj (4096, 4096)
model.layers.4.self_attn.k_proj (1024, 4096)
model.layers.4.self_attn.v_proj (1024, 4096)
model.layers.4.self_attn.o_proj (4096, 4096)
model.layers.4.mlp.gate_proj (14336, 4096)
model.layers.4.mlp.up_proj (14336, 4096)
model.layers.4.mlp.down_proj (4096, 14336)
model.layers.4.input_layernorm (4096,)
model.layers.4.post_attention_layernorm (4096,)
model.layers.5.self_attn.q_proj (4096, 4096)
model.layers.5.self_attn.k_proj (1024, 4096)
model.layers.5.self_attn.v_proj (1024, 4096)
model.layers.5.self_attn.o_proj (4096, 4096)
model.layers.5.mlp.gate_proj (14336, 4096)
model.layers.5.mlp.up_proj (14336, 4096)
model.layers.5.mlp.down_proj (4096, 14336)
model.layers.5.input_layernorm (4096,)
model.layers.5.post_attention_layernorm (4096,)
model.layers.6.self_attn.q_proj (4096, 4096)
model.layers.6.self_attn.k_proj (1024, 4096)
model.layers.6.self_attn.v_proj (1024, 4096)
model.layers.6.self_attn.o_proj (4096, 4096)
model.layers.6.mlp.gate_proj (14336, 4096)
model.layers.6.mlp.up_proj (14336, 4096)
model.layers.6.mlp.down_proj (4096, 14336)
model.layers.6.input_layernorm (4096,)
model.layers.6.post_attention_layernorm (4096,)
model.layers.7.self_attn.q_proj (4096, 4096)
model.layers.7.self_attn.k_proj (1024, 4096)
model.layers.7.self_attn.v_proj (1024, 4096)
model.layers.7.self_attn.o_proj (4096, 4096)
model.layers.7.mlp.gate_proj (14336, 4096)
model.layers.7.mlp.up_proj (14336, 4096)
model.layers.7.mlp.down_proj (4096, 14336)
model.layers.7.input_layernorm (4096,)
model.layers.7.post_attention_layernorm (4096,)
model.layers.8.self_attn.q_proj (4096, 4096)
model.layers.8.self_attn.k_proj (1024, 4096)
model.layers.8.self_attn.v_proj (1024, 4096)
model.layers.8.self_attn.o_proj (4096, 4096)
model.layers.8.mlp.gate_proj (14336, 4096)
model.layers.8.mlp.up_proj (14336, 4096)
model.layers.8.mlp.down_proj (4096, 14336)
model.layers.8.input_layernorm (4096,)
model.layers.8.post_attention_layernorm (4096,)
model.layers.9.self_attn.q_proj (4096, 4096)
model.layers.9.self_attn.k_proj (1024, 4096)
model.layers.9.self_attn.v_proj (1024, 4096)
model.layers.9.self_attn.o_proj (4096, 4096)
model.layers.9.mlp.gate_proj (14336, 4096)
model.layers.9.mlp.up_proj (14336, 4096)
model.layers.9.mlp.down_proj (4096, 14336)
model.layers.9.input_layernorm (4096,)
model.layers.9.post_attention_layernorm (4096,)
model.layers.10.self_attn.q_proj (4096, 4096)
model.layers.10.self_attn.k_proj (1024, 4096)
model.layers.10.self_attn.v_proj (1024, 4096)
model.layers.10.self_attn.o_proj (4096, 4096)
model.layers.10.mlp.gate_proj (14336, 4096)
model.layers.10.mlp.up_proj (14336, 4096)
model.layers.10.mlp.down_proj (4096, 14336)
model.layers.10.input_layernorm (4096,)
model.layers.10.post_attention_layernorm (4096,)
model.layers.11.self_attn.q_proj (4096, 4096)
model.layers.11.self_attn.k_proj (1024, 4096)
model.layers.11.self_attn.v_proj (1024, 4096)
model.layers.11.self_attn.o_proj (4096, 4096)
model.layers.11.mlp.gate_proj (14336, 4096)
model.layers.11.mlp.up_proj (14336, 4096)
model.layers.11.mlp.down_proj (4096, 14336)
model.layers.11.input_layernorm (4096,)
model.layers.11.post_attention_layernorm (4096,)
model.layers.12.self_attn.q_proj (4096, 4096)
model.layers.12.self_attn.k_proj (1024, 4096)
model.layers.12.self_attn.v_proj (1024, 4096)
model.layers.12.self_attn.o_proj (4096, 4096)
model.layers.12.mlp.gate_proj (14336, 4096)
model.layers.12.mlp.up_proj (14336, 4096)
model.layers.12.mlp.down_proj (4096, 14336)
model.layers.12.input_layernorm (4096,)
model.layers.12.post_attention_layernorm (4096,)
model.layers.13.self_attn.q_proj (4096, 4096)
model.layers.13.self_attn.k_proj (1024, 4096)
model.layers.13.self_attn.v_proj (1024, 4096)
model.layers.13.self_attn.o_proj (4096, 4096)
model.layers.13.mlp.gate_proj (14336, 4096)
model.layers.13.mlp.up_proj (14336, 4096)
model.layers.13.mlp.down_proj (4096, 14336)
model.layers.13.input_layernorm (4096,)
model.layers.13.post_attention_layernorm (4096,)
model.layers.14.self_attn.q_proj (4096, 4096)
model.layers.14.self_attn.k_proj (1024, 4096)
model.layers.14.self_attn.v_proj (1024, 4096)
model.layers.14.self_attn.o_proj (4096, 4096)
model.layers.14.mlp.gate_proj (14336, 4096)
model.layers.14.mlp.up_proj (14336, 4096)
model.layers.14.mlp.down_proj (4096, 14336)
model.layers.14.input_layernorm (4096,)
model.layers.14.post_attention_layernorm (4096,)
model.layers.15.self_attn.q_proj (4096, 4096)
model.layers.15.self_attn.k_proj (1024, 4096)
model.layers.15.self_attn.v_proj (1024, 4096)
model.layers.15.self_attn.o_proj (4096, 4096)
model.layers.15.mlp.gate_proj (14336, 4096)
model.layers.15.mlp.up_proj (14336, 4096)
model.layers.15.mlp.down_proj (4096, 14336)
model.layers.15.input_layernorm (4096,)
model.layers.15.post_attention_layernorm (4096,)
model.layers.16.self_attn.q_proj (4096, 4096)
model.layers.16.self_attn.k_proj (1024, 4096)
model.layers.16.self_attn.v_proj (1024, 4096)
model.layers.16.self_attn.o_proj (4096, 4096)
model.layers.16.mlp.gate_proj (14336, 4096)
model.layers.16.mlp.up_proj (14336, 4096)
model.layers.16.mlp.down_proj (4096, 14336)
model.layers.16.input_layernorm (4096,)
model.layers.16.post_attention_layernorm (4096,)
model.layers.17.self_attn.q_proj (4096, 4096)
model.layers.17.self_attn.k_proj (1024, 4096)
model.layers.17.self_attn.v_proj (1024, 4096)
model.layers.17.self_attn.o_proj (4096, 4096)
model.layers.17.mlp.gate_proj (14336, 4096)
model.layers.17.mlp.up_proj (14336, 4096)
model.layers.17.mlp.down_proj (4096, 14336)
model.layers.17.input_layernorm (4096,)
model.layers.17.post_attention_layernorm (4096,)
model.layers.18.self_attn.q_proj (4096, 4096)
model.layers.18.self_attn.k_proj (1024, 4096)
model.layers.18.self_attn.v_proj (1024, 4096)
model.layers.18.self_attn.o_proj (4096, 4096)
model.layers.18.mlp.gate_proj (14336, 4096)
model.layers.18.mlp.up_proj (14336, 4096)
model.layers.18.mlp.down_proj (4096, 14336)
model.layers.18.input_layernorm (4096,)
model.layers.18.post_attention_layernorm (4096,)
model.layers.19.self_attn.q_proj (4096, 4096)
model.layers.19.self_attn.k_proj (1024, 4096)
model.layers.19.self_attn.v_proj (1024, 4096)
model.layers.19.self_attn.o_proj (4096, 4096)
model.layers.19.mlp.gate_proj (14336, 4096)
model.layers.19.mlp.up_proj (14336, 4096)
model.layers.19.mlp.down_proj (4096, 14336)
model.layers.19.input_layernorm (4096,)
model.layers.19.post_attention_layernorm (4096,)
model.layers.20.self_attn.q_proj (4096, 4096)
model.layers.20.self_attn.k_proj (1024, 4096)
model.layers.20.self_attn.v_proj (1024, 4096)
model.layers.20.self_attn.o_proj (4096, 4096)
model.layers.20.mlp.gate_proj (14336, 4096)
model.layers.20.mlp.up_proj (14336, 4096)
model.layers.20.mlp.down_proj (4096, 14336)
model.layers.20.input_layernorm (4096,)
model.layers.20.post_attention_layernorm (4096,)
model.layers.21.self_attn.q_proj (4096, 4096)
model.layers.21.self_attn.k_proj (1024, 4096)
model.layers.21.self_attn.v_proj (1024, 4096)
model.layers.21.self_attn.o_proj (4096, 4096)
model.layers.21.mlp.gate_proj (14336, 4096)
model.layers.21.mlp.up_proj (14336, 4096)
model.layers.21.mlp.down_proj (4096, 14336)
model.layers.21.input_layernorm (4096,)
model.layers.21.post_attention_layernorm (4096,)
model.layers.22.self_attn.q_proj (4096, 4096)
model.layers.22.self_attn.k_proj (1024, 4096)
model.layers.22.self_attn.v_proj (1024, 4096)
model.layers.22.self_attn.o_proj (4096, 4096)
model.layers.22.mlp.gate_proj (14336, 4096)
model.layers.22.mlp.up_proj (14336, 4096)
model.layers.22.mlp.down_proj (4096, 14336)
model.layers.22.input_layernorm (4096,)
model.layers.22.post_attention_layernorm (4096,)
model.layers.23.self_attn.q_proj (4096, 4096)
model.layers.23.self_attn.k_proj (1024, 4096)
model.layers.23.self_attn.v_proj (1024, 4096)
model.layers.23.self_attn.o_proj (4096, 4096)
model.layers.23.mlp.gate_proj (14336, 4096)
model.layers.23.mlp.up_proj (14336, 4096)
model.layers.23.mlp.down_proj (4096, 14336)
model.layers.23.input_layernorm (4096,)
model.layers.23.post_attention_layernorm (4096,)
model.layers.24.self_attn.q_proj (4096, 4096)
model.layers.24.self_attn.k_proj (1024, 4096)
model.layers.24.self_attn.v_proj (1024, 4096)
model.layers.24.self_attn.o_proj (4096, 4096)
model.layers.24.mlp.gate_proj (14336, 4096)
model.layers.24.mlp.up_proj (14336, 4096)
model.layers.24.mlp.down_proj (4096, 14336)
model.layers.24.input_layernorm (4096,)
model.layers.24.post_attention_layernorm (4096,)
model.layers.25.self_attn.q_proj (4096, 4096)
model.layers.25.self_attn.k_proj (1024, 4096)
model.layers.25.self_attn.v_proj (1024, 4096)
model.layers.25.self_attn.o_proj (4096, 4096)
model.layers.25.mlp.gate_proj (14336, 4096)
model.layers.25.mlp.up_proj (14336, 4096)
model.layers.25.mlp.down_proj (4096, 14336)
model.layers.25.input_layernorm (4096,)
model.layers.25.post_attention_layernorm (4096,)
model.layers.26.self_attn.q_proj (4096, 4096)
model.layers.26.self_attn.k_proj (1024, 4096)
model.layers.26.self_attn.v_proj (1024, 4096)
model.layers.26.self_attn.o_proj (4096, 4096)
model.layers.26.mlp.gate_proj (14336, 4096)
model.layers.26.mlp.up_proj (14336, 4096)
model.layers.26.mlp.down_proj (4096, 14336)
model.layers.26.input_layernorm (4096,)
model.layers.26.post_attention_layernorm (4096,)
model.layers.27.self_attn.q_proj (4096, 4096)
model.layers.27.self_attn.k_proj (1024, 4096)
model.layers.27.self_attn.v_proj (1024, 4096)
model.layers.27.self_attn.o_proj (4096, 4096)
model.layers.27.mlp.gate_proj (14336, 4096)
model.layers.27.mlp.up_proj (14336, 4096)
model.layers.27.mlp.down_proj (4096, 14336)
model.layers.27.input_layernorm (4096,)
model.layers.27.post_attention_layernorm (4096,)
model.layers.28.self_attn.q_proj (4096, 4096)
model.layers.28.self_attn.k_proj (1024, 4096)
model.layers.28.self_attn.v_proj (1024, 4096)
model.layers.28.self_attn.o_proj (4096, 4096)
model.layers.28.mlp.gate_proj (14336, 4096)
model.layers.28.mlp.up_proj (14336, 4096)
model.layers.28.mlp.down_proj (4096, 14336)
model.layers.28.input_layernorm (4096,)
model.layers.28.post_attention_layernorm (4096,)
model.layers.29.self_attn.q_proj (4096, 4096)
model.layers.29.self_attn.k_proj (1024, 4096)
model.layers.29.self_attn.v_proj (1024, 4096)
model.layers.29.self_attn.o_proj (4096, 4096)
model.layers.29.mlp.gate_proj (14336, 4096)
model.layers.29.mlp.up_proj (14336, 4096)
model.layers.29.mlp.down_proj (4096, 14336)
model.layers.29.input_layernorm (4096,)
model.layers.29.post_attention_layernorm (4096,)
model.layers.30.self_attn.q_proj (4096, 4096)
model.layers.30.self_attn.k_proj (1024, 4096)
model.layers.30.self_attn.v_proj (1024, 4096)
model.layers.30.self_attn.o_proj (4096, 4096)
model.layers.30.mlp.gate_proj (14336, 4096)
model.layers.30.mlp.up_proj (14336, 4096)
model.layers.30.mlp.down_proj (4096, 14336)
model.layers.30.input_layernorm (4096,)
model.layers.30.post_attention_layernorm (4096,)
model.layers.31.self_attn.q_proj (4096, 4096)
model.layers.31.self_attn.k_proj (1024, 4096)
model.layers.31.self_attn.v_proj (1024, 4096)
model.layers.31.self_attn.o_proj (4096, 4096)
model.layers.31.mlp.gate_proj (14336, 4096)
model.layers.31.mlp.up_proj (14336, 4096)
model.layers.31.mlp.down_proj (4096, 14336)
model.layers.31.input_layernorm (4096,)
model.layers.31.post_attention_layernorm (4096,)
model.norm (4096,)
lm_head (33152, 4096)
```

## config.json

```json
{
  "architectures": [
    "MistralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "document_attention": true,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.38.2",
  "use_cache": false,
  "vocab_size": 33152
}
```

---

