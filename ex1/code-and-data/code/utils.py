import math
from datetime import datetime

import torch
from matplotlib import pyplot as plt
import os


def _sanitize_value_for_filename(v):
    if isinstance(v, float):
        s = f"{v:.10g}"
        s = s.replace(".", "p")
        s = s.replace("-", "m")
        return s
    return str(v)


def experiment_name(params):
    return (
        f"seq{_sanitize_value_for_filename(params['seq_len'])}"
        f"_bs{_sanitize_value_for_filename(params['batch_size'])}"
        f"_L{_sanitize_value_for_filename(params['n_layers'])}"
        f"_H{_sanitize_value_for_filename(params['n_heads'])}"
        f"_E{_sanitize_value_for_filename(params['embed_size'])}"
        f"_M{_sanitize_value_for_filename(params['mlp_hidden_size'])}"
        f"_lr{_sanitize_value_for_filename(params['learning_rate'])}"
    )


def save_best_model(model, params, best, metric_type='val_loss', epoch=0, out_dir="."):
    exp_name = experiment_name(params)
    model_path = f"{out_dir}/best_model_{exp_name}.pth"

    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best': best,
        'metric_type': metric_type,
        'params': params,
    }

    torch.save(payload, model_path)
    print(f"Saved best model to {model_path}")



def loss_plotter(train_losses,val_losses,params, out_dir="."):
    exp_name = experiment_name(params)

    # plotting after training ends
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = f"{out_dir}/loss_plot_{exp_name}_{timestamp}.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(train_losses)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Logging step (every 10 batches)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(val_losses)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Validation step (every 100 batches)")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    fig.suptitle(
        f"Training Curves | layers={params['n_layers']}, heads={params['n_heads']}, embed={params['embed_size']}, lr={params['learning_rate']}",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    plt.close(fig)


def split_data(tokenized_data, seq_len, train_ratio=0.9):
    #single file source
    if len(tokenized_data) == 1:
        full_seq = tokenized_data[0]

        if len(full_seq) < seq_len + 2:
            raise ValueError("Sequence too short for splitting")

        split_idx = int(train_ratio * len(full_seq))

        # avoid empty parts
        split_idx = max(seq_len + 1, split_idx)
        split_idx = min(len(full_seq) - (seq_len + 1), split_idx)

        train_data = [full_seq[:split_idx]]
        val_data = [full_seq[split_idx:]]

    else:

        split_idx = int(train_ratio * len(tokenized_data))

        if split_idx == 0:
            split_idx = 1

        train_data = tokenized_data[:split_idx]
        val_data = tokenized_data[split_idx:]

    train_data = [seq for seq in train_data if len(seq) > seq_len + 1]
    val_data = [seq for seq in val_data if len(seq) > seq_len + 1]

    if len(train_data) == 0:
        raise ValueError("train_data is empty after filtering")

    if len(val_data) == 0:
        print("Warning: val_data is empty")

    return train_data, val_data


import random

import random
import math


def _sample_param(x, param_name=None):
    # ערך בודד
    if not isinstance(x, (list, tuple)):
        return x

    # רשימה -> בחירה אקראית
    if isinstance(x, list):
        if len(x) == 0:
            return None
        return random.choice(x)

    # tuple באורך 2 -> טווח
    if isinstance(x, tuple) and len(x) == 2:
        a, b = x

        # learning_rate -> log-uniform
        if param_name == "learning_rate":
            # תיקון ערכים לא חיוביים
            a = max(a, 1e-8)
            b = max(b, 1e-8)

            low = min(a, b)
            high = max(a, b)

            log_low = math.log10(low)
            log_high = math.log10(high)
            return 10 ** random.uniform(log_low, log_high)

        # int range
        if isinstance(a, int) and isinstance(b, int):
            low = min(a, b)
            high = max(a, b)
            return random.randint(low, high)

        # float range
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            low = min(a, b)
            high = max(a, b)
            return random.uniform(low, high)

    # fallback
    return x


def parameters(seq_len, batch_size, n_layers, n_heads, embed_size, mlp_hidden_size, learning_rate):
    while True:
        chosen_seq_len = _sample_param(seq_len, "seq_len")
        chosen_batch_size = _sample_param(batch_size, "batch_size")
        chosen_n_layers = _sample_param(n_layers, "n_layers")
        chosen_n_heads = _sample_param(n_heads, "n_heads")
        chosen_embed_size = _sample_param(embed_size, "embed_size")
        chosen_learning_rate = _sample_param(learning_rate, "learning_rate")

        # fallbackים אם משהו יצא None
        if chosen_seq_len is None:
            chosen_seq_len = 128
        if chosen_batch_size is None:
            chosen_batch_size = 32
        if chosen_n_layers is None:
            chosen_n_layers = 6
        if chosen_n_heads is None:
            chosen_n_heads = 4
        if chosen_embed_size is None:
            chosen_embed_size = 128
        if chosen_learning_rate is None:
            chosen_learning_rate = 1e-4

        # תקן קומבינציה לא חוקית של heads/embed
        if chosen_n_heads <= 0:
            chosen_n_heads = 1

        if chosen_embed_size < chosen_n_heads:
            chosen_embed_size = chosen_n_heads

        # תקן embed_size כך שיהיה מתחלק ב-n_heads
        if chosen_embed_size % chosen_n_heads != 0:
            chosen_embed_size = ((chosen_embed_size + chosen_n_heads - 1) // chosen_n_heads) * chosen_n_heads

        # mlp_hidden_size
        if callable(mlp_hidden_size):
            chosen_mlp_hidden_size = mlp_hidden_size(chosen_embed_size)
        else:
            chosen_mlp_hidden_size = _sample_param(mlp_hidden_size, "mlp_hidden_size")
            if chosen_mlp_hidden_size is None:
                chosen_mlp_hidden_size = chosen_embed_size * 4

        # הגנות בסיסיות
        chosen_seq_len = max(2, int(chosen_seq_len))
        chosen_batch_size = max(1, int(chosen_batch_size))
        chosen_n_layers = max(1, int(chosen_n_layers))
        chosen_n_heads = max(1, int(chosen_n_heads))
        chosen_embed_size = max(chosen_n_heads, int(chosen_embed_size))
        chosen_mlp_hidden_size = max(1, int(chosen_mlp_hidden_size))
        chosen_learning_rate = max(1e-8, float(chosen_learning_rate))

        # התאמת batch size למודלים גדולים
        model_size_score = chosen_n_layers * chosen_embed_size

        if model_size_score >= 3000:
            chosen_batch_size = min(chosen_batch_size, 32)

        if model_size_score >= 5000:
            chosen_batch_size = min(chosen_batch_size, 16)

        return {
            'seq_len': chosen_seq_len,
            'batch_size': chosen_batch_size,
            'n_layers': chosen_n_layers,
            'n_heads': chosen_n_heads,
            'embed_size': chosen_embed_size,
            'mlp_hidden_size': chosen_mlp_hidden_size,
            'learning_rate': chosen_learning_rate
        }



def load_best_model(model_class, params=None, model_path=None, out_dir=".", device=None, strict=True):
    """
    Loads a model checkpoint saved by save_best_model()

    Args:
        model_class: model class (e.g., MyModel)
        params: if None, will be loaded from checkpoint
        model_path: full path to checkpoint file
        out_dir: directory where model is stored
        device: torch.device (if None -> auto-detect)
        strict: whether to strictly enforce state_dict matching

    Returns:
        model, checkpoint


    example:
    model, ckpt = load_best_model(
    TransformerLM,
    model_path="model.pth",
    device=device
)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        if params is None:
            raise ValueError("If model_path is None, params must be provided.")
        exp_name = experiment_name(params)
        model_path = os.path.join(out_dir, f"best_model_{exp_name}.pth")

    checkpoint = torch.load(model_path, map_location=device)

    if params is None:
        params = checkpoint['params']

    model = model_class(**params)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    model.to(device)
    model.eval()

    return model, checkpoint
