import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "qwen2_5_1_5b_lora"  # make sure this is the directory you saved


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()


questions = [
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
    messages = [
        {
            "role": "user",
            "content": q,
        },
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
            do_sample=False,  # deterministic, better for testing
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    print("=" * 80)
    print(f"question:\n{q}\n")
    print(f"answer:\n{answer}")