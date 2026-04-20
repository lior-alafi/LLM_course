from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False,dropout:float=None,attn_dropout:float=None):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        if dropout is not None:
            self.dropout_layer=nn.Dropout(p=dropout)
        else:
            self.dropout_layer=None
        if attn_dropout is not None:
            self.attn_dropout_layer=nn.Dropout(p=attn_dropout)
        else:
            self.attn_dropout_layer=None

    def forward(self, inputs, return_attn_maps=False):
        if self.with_residuals:
            x=inputs
            if return_attn_maps:
                sa,attn_maps=self.causal_attention(self.layer_norm_1(x), return_attn_maps,self.attn_dropout_layer)
            else:
                sa=self.causal_attention(self.layer_norm_1(x))
            if self.dropout_layer is not None:
                x=x+self.dropout_layer(sa)
            else:
                x=x+sa
            x=x+self.mlp(self.layer_norm_2(x))
        else:
            x = inputs
            x = self.layer_norm_1(x)
            if return_attn_maps:
                sa,attn_maps=self.causal_attention(x, return_attn_maps,self.attn_dropout_layer)
            else:
                sa=self.causal_attention(x)
            if self.dropout_layer is not None:
                x=self.dropout_layer(sa)
            else:
                x=sa
            x =self.layer_norm_2(x)
            x =self.mlp(x)
        if return_attn_maps:
            return x, attn_maps
        else:
            return x

class Embed(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size) # TODO set the right values
        self.position_embeddings = nn.Embedding(max_context_len, embed_size) # TODO set the right values
        self.max_context_len = max_context_len

    def forward(self, x):
        # raise Exception("Not implemented") # TODO implement.
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(torch.arange(x.size(1),device=x.device))
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            dropout:list[float]=[None,None,None],
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        dropout_1,dropout_2,dropout_3=dropout
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals,dropout_2,dropout_3) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        self.init_weights()
        if dropout_1 is not None:
            self.dropout=nn.Dropout(p=dropout_1)
        else:
            self.dropout=None

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs,return_attn_maps=False):
        x = self.embed(inputs)
        if self.dropout is not None:
            x=self.dropout(x)
        all_layers_maps=[]
        for layer in self.layers:
            if return_attn_maps:
                x, attn_maps = layer(x, return_attn_maps=True)
                all_layers_maps.append(attn_maps)
            else:
                x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        if return_attn_maps:
            stacked_attn_maps = torch.stack(all_layers_maps, dim=1)
            return logits, stacked_attn_maps
        return logits

    def init_weights(self):
        # initialize weights
        # TODO implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).

        # using modules instead 
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=self.word_prediction.weight.device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token,dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1).item()
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.
        feed_to_lm = prefix[:]
        generated = []

        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=self.word_prediction.weight.device))
                logits_for_last_token = logits[0][-1]

                scaled_logits = logits_for_last_token / temperature

                top_k_vals, top_k_indices = torch.topk(scaled_logits, topK)

                top_k_distribution = F.softmax(top_k_vals, dim=-1)

                sampled_idx_in_topk = torch.multinomial(top_k_distribution, num_samples=1).item()
                sampled_token = top_k_indices[sampled_idx_in_topk].item()

                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)

        return generated