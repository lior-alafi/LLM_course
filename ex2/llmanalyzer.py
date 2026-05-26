from huggingface_hub import hf_hub_download,login
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

import json
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

missing =['meta-llama/Llama-3.1-8B-Instruct',


          ]

def get_model_cfg(model_id):
    # try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="config.json"
        )
    # except:
    #
    #     return {}
        print(path)

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config# json.dumps(config, indent=2, ensure_ascii=False)

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


from contextlib import redirect_stdout, redirect_stderr

# with open("output2.txt", "w", encoding="utf-8") as f:
#     # with redirect_stdout(f), redirect_stderr(f):
#
#         for model_id in missing: #models:
#             print(f'model ID: {model_id}')
#             cfg = get_model_cfg(model_id)
#             arch = 'architectures'
#
#             if arch in cfg:
#                 print(arch)
#                 for arc in cfg[arch]:
#                     model = get_model(model_id)
#                     for name, module in model.named_modules():
#                         if hasattr(module, "weight"):
#                             print(name, tuple(module.weight.shape))
#
#                     print('----')
#             print(json.dumps(cfg, indent=2, ensure_ascii=False))
#             print('*'*10)

heb_set = set("אבגדהוזחטיכלמנסעפצקרשת".split())
for model_id in models:
    print(f'modelID: {model_id}')
    try:
        tok = AutoTokenizer.from_pretrained(model_id)

        data = json.loads(tok.backend_tokenizer.to_str())

        print(data['model'])
        for merge in data['model']['merges']:
            if set(merge).intersection(heb_set):
                print("support hebrew")
                break

    except:
        pass
    finally:
        print('=='*5)