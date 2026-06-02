import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# This must match OUTPUT_DIR from training.
ADAPTER_DIR = "qwen_lora_manual_adapter"


# -------------------------------------------------------
# 1. Load tokenizer
# -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------------
# 2. Load base model
# -------------------------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)


# -------------------------------------------------------
# 3. Load LoRA adapter
# -------------------------------------------------------
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

model.eval()


print("ADAPTER_DIR:", ADAPTER_DIR)
print("Current working dir:", os.getcwd())
print("Adapter files:", os.listdir(ADAPTER_DIR))
print("Active adapters:", model.active_adapters)
print("PEFT config:", model.peft_config)


# -------------------------------------------------------
# 4. Generate function
# -------------------------------------------------------
def generate_answer(model, tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer


# -------------------------------------------------------
# 5. Questions from tiny_debug.jsonl
# -------------------------------------------------------
questions = [
    "Explain why leaves fall from many trees in autumn.",
    "Explain why deserts are dry.",
    "Explain why fire needs oxygen.",
    "Explain why clouds look white.",
    "Write a clear and polite email asking a teacher to explain a difficult topic.",

    "Explain why the sky looks blue during the day.",
    "Give two advantages and two disadvantages of public transportation.",
    "Write a short email asking a professor for an extension on an assignment.",
    "Describe how to make a simple omelette.",
    "What is the difference between supervised and unsupervised learning?",
    "Summarize the story of Cinderella in three sentences.",
    "Suggest three ways to reduce smartphone distraction while studying.",
    "Explain what happens when water boils.",
    "Give a polite refusal to an invitation to a party.",
    "Turn the idea “practice makes progress” into advice for a student.",
]


for q in questions:
    print("=" * 80)
    print("question:")
    print(q)

    print("\nWITH LORA:")
    print(generate_answer(model, tokenizer, q))

    print("\nWITHOUT LORA:")
    with model.disable_adapter():
        print(generate_answer(model, tokenizer, q))