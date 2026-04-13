import torch
import attention
import torch.nn as nn
from transformer import TransformerLM

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
                # TODO initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                
                torch.nn.init.xavier_normal_(p.weight) #https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/weight-initialization-pytorch
                torch.nn.init.zeros_(p.bias)
                print("2")
            elif isinstance(p, nn.Embedding):
                # TODO initialize p.weight and p.bias (if it is not None).
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

# test_init_weights_via_modules()           
# test_attention_scores()