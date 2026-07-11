from xutils import HFChatGenerator
import gc

msg = "explain briefly what is supervised learning"


model_inf = [{'id': 'mistralai/Mistral-7B-Instruct-v0.3','allowed_token_path': 'hebrew_allowed_tokens_mistral.json','prefix':"ענה בעברית"+"\n\n"},
             {'id': 'Qwen/Qwen2.5-7B-Instruct','allowed_token_path': 'hebrew_allowed_tokens_qwen.json','prefix':'answer only in Hebrew and check your output\n\n'} ]
for m in model_inf:
    generator = HFChatGenerator( model_id=m['id'],
                                 allowed_token_json_path=m['allowed_token_path'] )

    generator.print_device_info()

    # Apply this model's Hebrew-only instruction to BOTH generations, so the
    # constrained-vs-unconstrained comparison is actually about the token
    # masking (not about one of them never being asked to answer in Hebrew).
    messages = [
        {"role": "user", "content": m['prefix'] + msg},
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



    print(f'constrained: {constrained_answer}\nunconstrained: {unconstrained_answer}')
    del generator
    gc.collect()