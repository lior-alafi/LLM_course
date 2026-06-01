import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_ids = [ 'Qwen/Qwen2.5-7B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3','']
#
# model_id = model_ids[1]
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
#
# # 'Answer only in fluent Hebrew. Do not use any other language.'
# prompt = "Explain quantum computing in one sentence."
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#
# # 4. Generate predictions
# outputs = model.generate(**inputs, max_new_tokens=250)
#
# # 5. Decode the generated tokens back to text
# result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# print(result)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
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
        "role": "system",
          "content": (
            "You must answer only in fluent, natural Hebrew. "
            "Do not use English. Do not use Chinese, Arabic, Korean, or any other language. "
            "The user will ask in English, but your answer must be in Hebrew only."
        ),
    },
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
    ).to(base_model.device)

    with torch.inference_mode():
        output_ids = base_model.generate(
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