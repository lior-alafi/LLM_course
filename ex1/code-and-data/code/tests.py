import torch
import attention

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

test_attention_scores()