import torch
import matplotlib.pyplot as plt



def plot_attention_maps(
    attn_maps,          # (B, n_layers, n_heads, N, N)
    tokens: list[str],
    sample_idx: int = 0,
    max_len: int = 32,
    save_path: str = None,
):
    n_layers = attn_maps.shape[1]
    n_heads = attn_maps.shape[2]

    seq_len = min(len(tokens), max_len)
    labels = tokens[:seq_len]

    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(3 * n_heads, 3 * n_layers),
        squeeze=False,
        constrained_layout=True,   # ✅ במקום tight_layout
    )

    fig.suptitle("Attention Maps (layer × head)", fontsize=14)

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attn = attn_maps[
                sample_idx,
                layer_idx,
                head_idx,
                :seq_len,
                :seq_len
            ].detach().cpu()

            ax = axes[layer_idx][head_idx]
            im = ax.imshow(attn, cmap="viridis", aspect="auto", vmin=0, vmax=1)

            ax.set_title(f"L{layer_idx+1} H{head_idx+1}", fontsize=8)

            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))

            ax.set_xticklabels(labels, fontsize=5, rotation=90)
            ax.set_yticklabels(labels, fontsize=5)

    # colorbar אחד לכל הגריד
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Attention weight")

    # ❌ לא להשתמש יותר בזה
    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()

    plt.close()

def extract_and_plot(model, tokenizer, prefix_text: str, save_path: str = None, max_len: int = 32):
    """
    Convenience wrapper: tokenize prefix_text, run a forward pass,
    and plot the resulting attention maps.
    """
    model.eval()
    with torch.no_grad():
        token_ids = tokenizer.tokenize(prefix_text)
        # Trim to model's context window
        token_ids = token_ids[-model.max_context_len:]
        input_tensor = torch.tensor([token_ids], dtype=torch.long,
                                    device=next(model.parameters()).device)
 
        _logits, attn_maps = model(input_tensor, return_attn_maps=True)
 
    # Decode each token id back to a readable string for the axis labels
    token_strings = [repr(tokenizer.vocab[t])[1:-1] for t in token_ids]
 
    plot_attention_maps(attn_maps, token_strings, sample_idx=0,
                        max_len=max_len, save_path=save_path)
    model.train()


def extract_and_plot2(model, tokenizer, prefix_text: str, save_path: str = None, max_len: int = 32):
    model.eval()

    with torch.no_grad():
        token_ids = tokenizer.tokenize(prefix_text)
        token_ids = token_ids[-model.max_context_len:]

        input_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=next(model.parameters()).device
        )

        logits, attn_maps = model(input_tensor, return_attn_maps=True)

    token_strings = [repr(tokenizer.vocab[t])[1:-1] for t in token_ids]

    plot_attention_maps(
        attn_maps,
        token_strings,
        sample_idx=0,
        max_len=max_len,
        save_path=save_path
    )

    return logits, attn_maps, token_ids, token_strings