from huggingface_hub import hf_hub_download, login
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

import json
import csv
import data.at as at


login(at.AT)


models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.3-8b-instruct",
    "deepseek-ai/DeepSeek-V3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "dicta-il/dictalm2.0-instruct",
]


CSV_COLUMNS = [
    "model_id",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "num_kv_heads",
    "mlp_size",
    "activation",
    "norm_type",
    "position_encoding",
    "context_length",
    "vocab_size",
    "moe_details",
]


def get_model_cfg(model_id):
    path = hf_hub_download(
        repo_id=model_id,
        filename="config.json"
    )

    print(path)

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
        return config


def get_model(model_id):
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True
        )
        return model


def first_existing(cfg, keys, default=""):
    """
    Return the first existing key from cfg.
    Useful because different model families use slightly different names.
    """
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def detect_norm_type(cfg):
    """
    The config usually does not literally say 'RMSNorm',
    but rms_norm_eps is a strong indication.
    """
    if "rms_norm_eps" in cfg:
        return "rmsnorm"

    if "layer_norm_eps" in cfg:
        return "layernorm"

    if "norm_eps" in cfg:
        return "norm"

    return ""


def detect_position_encoding(cfg):
    """
    Extract a readable positional encoding description from the config.
    Most of these models use RoPE, sometimes with scaling variants.
    """
    rope_scaling = cfg.get("rope_scaling", None)

    if isinstance(rope_scaling, dict):
        rope_type = (
            rope_scaling.get("type")
            or rope_scaling.get("rope_type")
            or rope_scaling.get("name")
        )

        if rope_type is not None:
            rope_type = str(rope_type).lower()

            if rope_type == "longrope":
                return "LongRoPE"

            if rope_type == "yarn":
                return "RoPE(YaRN)"

            return f"RoPE({rope_type})"

        return "RoPE(scaled)"

    if "rope_theta" in cfg or "rotary_emb_base" in cfg:
        return "RoPE"

    if "alibi" in str(cfg).lower():
        return "ALiBi"

    return ""


def extract_context_length(cfg):
    """
    Use config fields only.
    Main field is max_position_embeddings.
    If absent, fall back to other common names.
    """
    return first_existing(
        cfg,
        [
            "max_position_embeddings",
            "seq_length",
            "n_positions",
            "max_sequence_length",
            "model_max_length",
        ],
        default=""
    )


def extract_moe_details(cfg):
    """
    Return Dense for normal dense Transformer models.
    For MoE models, collect the MoE-related fields from config.
    """
    moe_keys = [
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_intermediate_size",
        "first_k_dense_replace",
        "moe_layer_freq",
        "ep_size",
        "routed_scaling_factor",
        "topk_method",
        "n_group",
        "topk_group",
        "scoring_func",
        "norm_topk_prob",
    ]

    moe_details = {}

    for key in moe_keys:
        if key in cfg:
            moe_details[key] = cfg[key]

    if len(moe_details) == 0:
        return "Dense"

    return json.dumps(moe_details, ensure_ascii=False)


def config_to_csv_row(model_id, cfg):
    """
    Convert HuggingFace config.json into exactly the required CSV columns.
    """
    hidden_size = first_existing(
        cfg,
        [
            "hidden_size",
            "n_embd",
            "d_model",
        ],
    )

    num_layers = first_existing(
        cfg,
        [
            "num_hidden_layers",
            "n_layer",
            "num_layers",
        ],
    )

    num_attention_heads = first_existing(
        cfg,
        [
            "num_attention_heads",
            "n_head",
            "num_heads",
        ],
    )

    num_kv_heads = first_existing(
        cfg,
        [
            "num_key_value_heads",
            "n_kv_heads",
            "num_kv_heads",
        ],
        default=num_attention_heads,
    )

    mlp_size = first_existing(
        cfg,
        [
            "intermediate_size",
            "ffn_dim",
            "mlp_hidden_size",
            "moe_intermediate_size",
        ],
    )

    activation = first_existing(
        cfg,
        [
            "hidden_act",
            "activation_function",
            "activation",
        ],
    )

    norm_type = detect_norm_type(cfg)

    position_encoding = detect_position_encoding(cfg)

    context_length = extract_context_length(cfg)

    vocab_size = first_existing(
        cfg,
        [
            "vocab_size",
            "padded_vocab_size",
        ],
    )

    moe_details = extract_moe_details(cfg)

    return {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_size": mlp_size,
        "activation": activation,
        "norm_type": norm_type,
        "position_encoding": position_encoding,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "moe_details": moe_details,
    }


def write_architecture_to_md(md_file, model_id, cfg):
    """
    Append the architecture dump of a single model to arch.md.
    """
    md_file.write(f"# {model_id}\n\n")

    md_file.write("## Parameter shapes\n\n")

    try:
        model = get_model(model_id)

        md_file.write("```text\n")

        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                md_file.write(f"{name} {tuple(module.weight.shape)}\n")

        md_file.write("```\n\n")

    except Exception as e:
        md_file.write("```text\n")
        md_file.write("ERROR while building empty model / extracting architecture:\n")
        md_file.write(str(e))
        md_file.write("\n```\n\n")

    md_file.write("## config.json\n\n")
    md_file.write("```json\n")
    md_file.write(json.dumps(cfg, indent=2, ensure_ascii=False))
    md_file.write("\n```\n\n")
    md_file.write("---\n\n")


rows = []

with open("arch.md", "w", encoding="utf-8") as md_file:
    for model_id in models:
        print(f"Processing: {model_id}")

        try:
            cfg = get_model_cfg(model_id)

            row = config_to_csv_row(model_id, cfg)
            rows.append(row)

            write_architecture_to_md(md_file, model_id, cfg)

        except Exception as e:
            print(f"ERROR for {model_id}: {e}")

            error_cfg = {
                "error": str(e)
            }

            row = {
                "model_id": model_id,
                "hidden_size": "",
                "num_layers": "",
                "num_attention_heads": "",
                "num_kv_heads": "",
                "mlp_size": "",
                "activation": "",
                "norm_type": "",
                "position_encoding": "",
                "context_length": "",
                "vocab_size": "",
                "moe_details": json.dumps(error_cfg, ensure_ascii=False),
            }

            rows.append(row)

            md_file.write(f"# {model_id}\n\n")
            md_file.write("## ERROR\n\n")
            md_file.write("```text\n")
            md_file.write(str(e))
            md_file.write("\n```\n\n")
            md_file.write("---\n\n")


with open("models.csv", "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)


print("Done.")
print("Wrote arch.md")
print("Wrote models.csv")