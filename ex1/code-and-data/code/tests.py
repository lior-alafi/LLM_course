import torch
import attention
import torch.nn as nn
from transformer import TransformerLM
from data import load_data
from utils import load_best_model
from visualize import extract_and_plot2


def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([[1.,2.,3.],
                  [4.,5.,6.]]).unsqueeze(0) # a three-dim tensor
    
    b = torch.tensor([[7.,8.,9.],
                  [1.,2.,3.]]).unsqueeze(0) # a three-dim tensor
    
    expected_output = torch.tensor(  [[28.8675, 70.4365],
         [8.0829, 18.4752]]).unsqueeze(0) # a three-dim tensor
    # print(expected_output)
    A = attention.attention_scores(a, b)
    # print(A)
    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)

def test_init_weights_via_parameters():
     tmp = TransformerLM(4,4,4,4,4,4,True)
     for pn, p in tmp.named_parameters():
            print(f'{pn} {p}')
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.ones_(p.weight)
                torch.nn.init.zeros_(p.bias)
                assert torch.ones_like(p.weight) - p.weight == torch.zeros_like(p.weight)
            elif isinstance(p, nn.Linear):
                # You can look at initializers in torch.nn.init
                
                torch.nn.init.xavier_normal_(p.weight) #https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/weight-initialization-pytorch
                torch.nn.init.zeros_(p.bias)
                print("2")
            elif isinstance(p, nn.Embedding):
                # You can look at initializers in torch.nn.init
                torch.nn.init.normal_(p.weight, mean=0, std=0.02) #An Exploration of Word Embedding Initialization in Deep-Learning Tasks
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
                print("3")

def test_init_weights_via_modules():
        tmp = TransformerLM(4,4,4,4,4,4,True)
        
        
        for module in tmp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                assert isinstance(module, nn.Linear)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                assert isinstance(module, nn.Embedding)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                assert torch.sum(module.weight) == module.weight.shape[0]
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

import os
def load_best_model_from_dir(models_dir, data_path="../data/en/", device=None):
    tokenizer, tokenized_data = load_data(data_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    best_ckpt = None
    best_path = None
    best_val_loss = float("inf")

    for fname in os.listdir(models_dir):
        if not fname.endswith(".pth"):
            continue

        full_path = os.path.join(models_dir, fname)

        try:
            ckpt = torch.load(full_path, map_location="cpu")
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        if not isinstance(ckpt, dict):
            continue

        if "best" not in ckpt or "params" not in ckpt or "model_state_dict" not in ckpt:
            continue

        if ckpt.get("metric_type", "val_loss") != "val_loss":
            continue

        curr_val_loss = ckpt["best"]

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            best_ckpt = ckpt
            best_path = full_path

    if best_ckpt is None:
        raise ValueError(f"No valid checkpoint found in: {models_dir}")

    params = best_ckpt["params"]

    model = TransformerLM(
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        embed_size=params["embed_size"],
        max_context_len=params["seq_len"],
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=params["mlp_hidden_size"],
        with_residuals=True,
        dropout=params.get("dropout", [None, None, None]),
    )

    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded best model:")
    print(f"path: {best_path}")
    print(f"val_loss: {best_val_loss}")
    print(f"params: {params}")

    return model, tokenizer, best_ckpt, best_path


def test_best_model_attn():
    models_dir = r"C:\Users\liora\Documents\לימודים\תואר שני\שנה 2\סמסטר ב\טקסטים ורצפים\ex\ex1\code-and-data\models\eng\v1"

    model, tokenizer, ckpt, path = load_best_model_from_dir(
        models_dir=models_dir,
        data_path="../data/en/",
    )

    extract_and_plot2(
        model,
        tokenizer,
        prefix_text="For never was a story of more woe than this of Juliet and her Romeo",
        save_path="../attn_maps/attention_map3.png",
        max_len=32,
    )

test_best_model_attn()
# test_init_weights_via_modules()           
# test_attention_scores()