import os
os.environ["PYTHONUTF8"] = "1"
from datasets import load_dataset
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "train_2000.jsonl"
OUTPUT_DIR = "qwen2_5_1_5b_lora"


# -------------------------
# 1. Load tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# Qwen usually has pad_token unset or equal to eos; this is fine for SFT.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# 2. Load dataset
# -------------------------

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
eval_data = load_dataset("json", data_files='eval_500.jsonl', split="train")
# -------------------------
# 3. Load model
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False


# -------------------------
# 4. LoRA config
# -------------------------
# peft_config = LoraConfig(         # AGGRESSIVE!!!
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#
#     # Good target modules for Qwen/Llama-style transformer blocks
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
# )

# peft_config = LoraConfig(
#     r=4,
#     lora_alpha=8,
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj",
#         "v_proj",
#     ],
# )

#APPLES TO APPLES
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
# -------------------------
# 5. Training config
# -------------------------
# training_args = SFTConfig(
#     output_dir=OUTPUT_DIR,
#
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#
#     learning_rate=2e-4,
#     warmup_ratio=0.03,
#     lr_scheduler_type="cosine",
#
#     logging_steps=10,
#     save_steps=200,
#     save_total_limit=2,
#
#     max_length=1024,
#     packing=False,
#
#     bf16=torch.cuda.is_available(),
#     fp16=False,
#
#     optim="adamw_torch",
#     report_to="none",
# )

# training_args = SFTConfig(
#     output_dir=OUTPUT_DIR,
#
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#
#     learning_rate=5e-5,
#     warmup_ratio=0.1,
#     lr_scheduler_type="cosine",
#
#     logging_steps=1,
#     save_steps=50,
#     save_total_limit=2,
#
#     max_length=512,
#     packing=False,
#
#     bf16=torch.cuda.is_available(),
#     fp16=False,
#
#     optim="adamw_torch",
#     report_to="none",
#
#     assistant_only_loss=True,
# )

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    learning_rate=5e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    logging_steps=20,
    save_steps=200,
    save_total_limit=2,

    max_length=512,
    packing=False,

    bf16=torch.cuda.is_available(),
    fp16=False,

    optim="adamw_torch",
    report_to="none",

    eval_strategy="steps",
    eval_steps=100,
    assistant_only_loss=True,
)

# -------------------------
# 6. Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    eval_dataset = eval_data
)

print(dataset[0])
trainer.train()


# -------------------------
# 7. Save LoRA adapter
# -------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved LoRA adapter to {OUTPUT_DIR}")