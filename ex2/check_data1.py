import json
import random

path = "train_3500.jsonl"

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

for i, ex in random.sample(list(enumerate(rows)), 30):
    print("=" * 100)
    print("index:", i)
    for m in ex["messages"]:
        print(m["role"] + ":")
        print(m["content"])