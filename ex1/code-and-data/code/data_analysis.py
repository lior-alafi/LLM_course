from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer import TransformerLM

import data
eng_data = "../data/en/"
heb_data = "../data/he/"
fp_eng = r"..\final\eng\eng_model\best_model_seq128_bs32_L8_H4_E256_M1024_lr0p0004650772489.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(data_path, checkpoint_path):
    tokenizer, tokenized_data = data.load_data(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    params = ckpt["params"]
    model = TransformerLM(
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        embed_size=params["embed_size"],
        max_context_len=params["seq_len"],
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=params["mlp_hidden_size"],
        with_residuals=True,
        dropout=[params['dropout_rate'],params['dropout_rate'],params['dropout_rate']],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model,params,ckpt["best"],tokenizer

eng_model,eng_params,eng_best,tokenizer = load_model(eng_data,fp_eng)
print(eng_params)
print(eng_best)

for temperature in np.arange(0.1,1.0,0.2):
    print(f'temperature{temperature:.2f}')
    with torch.no_grad():
        for text in ['Hello','abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',"where art thou"]:
            print(f'prefix: {text}')
            text = tokenizer.tokenize(text)
            simple=tokenizer.detokenize(eng_model.sample_continuation(text, 500))
            complex=tokenizer.detokenize(eng_model.better_sample_continuation(text, 500,temperature,5))
            print(simple)
            print('#'*10)
            print(complex)
            print('*' * 10)

    print('-'*50)