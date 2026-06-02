import os
os.environ["PYTHONUTF8"] = "1"

import shutil
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# DATA_PATH = "tiny_debug.jsonl"
DATA_PATH = "train_new_5000.jsonl"
# This is the LoRA adapter directory.
# It does NOT contain the full Qwen model.
# It contains only the small LoRA weights.
OUTPUT_DIR = "qwen_lora_manual_adapter"


# -------------------------------------------------------
# 0. Clean old adapter directory
# -------------------------------------------------------
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)


# -------------------------------------------------------
# 1. Load tokenizer
# -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------
# dataset = load_dataset(
#     "json",
#     data_files=DATA_PATH,
#     split="train",
# )
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.shuffle(seed=42)

train_dataset = dataset.select(range(2000))
eval_dataset = dataset.select(range(2000, 2500))
print("Raw example:")
print(dataset[0])


# -------------------------------------------------------
# 3. Manual preprocessing
# -------------------------------------------------------
def preprocess(example):
    messages = example["messages"]

    user_message = None
    assistant_answer = None

    for msg in messages:
        if msg["role"] == "user":
            user_message = msg
        elif msg["role"] == "assistant":
            assistant_answer = msg["content"]

    if user_message is None:
        raise ValueError("Missing user message")

    if assistant_answer is None:
        raise ValueError("Missing assistant message")

    # Prompt is only the user turn, with generation prompt.
    # Qwen template will create something like:
    # <|im_start|>user
    # ...question...
    # <|im_end|>
    # <|im_start|>assistant
    prompt_text = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Full text = prompt + Hebrew answer + EOS
    full_text = prompt_text + assistant_answer + tokenizer.eos_token

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    # Ignore loss on the prompt.
    # The model is trained only to predict the assistant answer.
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
)

print("\nTokenized example:")
print(tokenized_dataset[0])

# Debug: decode input and labels
ex0 = tokenized_dataset[0]
print("\nDecoded full input:")
print(tokenizer.decode(ex0["input_ids"]))

label_ids = [
    token_id for token_id in ex0["labels"]
    if token_id != -100
]

print("\nDecoded part that receives loss:")
print(tokenizer.decode(label_ids))


# -------------------------------------------------------
# 4. Custom data collator
# -------------------------------------------------------
class DataCollatorForCausalLMWithLabels:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"]
            labels = f["labels"]

            pad_len = max_len - len(input_ids)

            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


data_collator = DataCollatorForCausalLMWithLabels(tokenizer)


# -------------------------------------------------------
# 5. Load base model
# -------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False


# -------------------------------------------------------
# 6. LoRA config
# -------------------------------------------------------
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.0,   # zero dropout because this is a tiny overfit sanity test
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#     ],
# )

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, peft_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()


# -------------------------------------------------------
# 7. Training config
# -------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # We intentionally overfit 5 examples.
    # If the pipeline is correct, the model should memorize them.
    # num_train_epochs=30,

    # per_device_train_batch_size=1,
    # gradient_accumulation_steps=1,

    # learning_rate=2e-4,
    # lr_scheduler_type="constant",
    warmup_steps=0,

    logging_steps=1,

    save_steps=1000,
    save_total_limit=1,

    bf16=torch.cuda.is_available(),
    fp16=False,

    optim="adamw_torch",
    report_to="none",

    remove_unused_columns=False,

num_train_epochs=3,
per_device_train_batch_size=2,
gradient_accumulation_steps=4,
learning_rate=1e-4,
lr_scheduler_type="cosine",
warmup_ratio=0.05,
weight_decay=0.01,
)


# -------------------------------------------------------
# 8. Trainer
# -------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()


# -------------------------------------------------------
# 9. Save LoRA adapter
# -------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nSaved LoRA adapter to: {OUTPUT_DIR}")
print("This directory is the adapter.")
print("It should contain adapter_config.json and adapter_model.safetensors.")