import csv
import gc
import json

import torch

from xutils import HFChatGenerator

msg = "explain briefly what is supervised learning"
messages = [
    # {"role": "system", "content": "ענה רק בעברית."},
    {"role": "user", "content": msg},
]

questions = [
"Explain why the moon changes shape during the month.",
"Explain why rainbows appear after rain.",
"Explain why we need sleep.",
"Explain why plants need sunlight.",
"Explain why people recycle paper and plastic.",
"Explain why magnets stick to some metals.",
"Explain why brushing your teeth is important.",
"Explain why traffic lights are useful.",
"Explain why washing hands helps prevent illness.",

]

# with open('english_hebrew_questions_200.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
#         questions.append(data['question'])
#         if  len(questions) == 10:
#             break

csv_path = 'constrained_questions.csv'

with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['model_id', 'question', 'constrained', 'unconstrained'],
        quoting=csv.QUOTE_ALL
    )
    writer.writeheader()

model_inf = [{'id': 'mistralai/Mistral-7B-Instruct-v0.3','allowed_token_path': 'hebrew_allowed_tokens_mistral.json','prefix':"ענה בעברית"+"\n\n"},
             {'id': 'Qwen/Qwen2.5-7B-Instruct','allowed_token_path': 'hebrew_allowed_tokens_qwen.json','prefix':'answer only in Hebrew and check your output\n\n'} ]
prefix='answer only in Hebrew and check your output\n\n'
for m in model_inf:

    generator = HFChatGenerator( model_id=m['id'],
                                 allowed_token_json_path=m['allowed_token_path'] )

    generator.print_device_info()
    for question in questions:
        messages = [
            {"role": "user", "content": prefix+question},
        ]

        constrained_answer = generator.generate_constrained(
            messages,
            max_new_tokens=150,
            debug_template=True,
        )


        unconstrained_answer = generator.generate_unconstrained(
            messages,
            max_new_tokens=150,
            debug_template=True,
        )

        with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['model_id', 'question', 'constrained', 'unconstrained'],
                quoting=csv.QUOTE_ALL
            )

            writer.writerow({
                'model_id': m['id'],
                'question': question,
                'constrained': constrained_answer,
                'unconstrained': unconstrained_answer,
            })

    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()