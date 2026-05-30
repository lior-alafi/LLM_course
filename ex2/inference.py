from transformers import AutoModelForCausalLM, AutoTokenizer

model_ids = [ 'Qwen/Qwen2.5-7B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3']

model_id = model_ids[1]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "Explain quantum computing in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 4. Generate predictions
outputs = model.generate(**inputs, max_new_tokens=250)

# 5. Decode the generated tokens back to text
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(result)