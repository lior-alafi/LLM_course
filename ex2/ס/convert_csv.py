import json
import pandas as pd

# טעינת קובץ ה-CSV
csv_filename = "../constrained_questions.csv"
output_filename = "../decoding_outputs.jsonl"

df = pd.read_csv(csv_filename)

# יצירת רשימה של רשומות בפורמט המבוקש
jsonl_records = []
for index, row in df.iterrows():
    record = {
        "prompt": row["question"],
        "model": row["model_id"],
        "unconstrained_output": row["unconstrained"],
        "constrained_output": row["constrained"],
    }
    jsonl_records.append(record)

# כתיבה לקובץ JSONL שבו כל שורה היא אובייקט JSON עצמאי
# השימוש ב-ensure_ascii=False שומר על הטקסט בעברית קריא (ולא כקודים של Unicode)
with open(output_filename, "w", encoding="utf-8") as f:
    for record in jsonl_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"הקובץ הומר בהצלחה ונשמר תחת השם: {output_filename}")