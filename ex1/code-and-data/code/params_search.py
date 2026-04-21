from __future__ import annotations

import random
import torch
from torch import optim

import data
import lm
from transformer import TransformerLM
from utils import save_best_model, parameters, loss_plotter, split_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = "../data/he/"
    gradient_clipping = 1.0
    num_batches_to_train = 600000
    num_trials = 1 #30

    validate_every = 100
    sample_every = 100
    log_every = 10
    val_steps = 20

    tokenizer, tokenized_data = data.load_data(data_path)

    for i in range(num_trials):
        print(f"\n===== Trial {i + 1}/{num_trials} =====")

        best_val_loss = float("inf")
        val_losses = []
        train_losses = []
        num_batches = 0

        params = parameters(
            seq_len=128, #[128,256],
            batch_size=128, #[32, 64, 128],
            n_layers=6,#[4, 6],
            n_heads=6,#[4, 6, 8],
            embed_size=132,#[64, 128, 192, 256, 384],
            mlp_hidden_size=lambda d: d * 4,
            learning_rate=2.014037949e-5,#(5e-7, 1e-4),
            dropout=[0.05,None,0.05]
           # [ random.choice([None,0.05,0.1]),random.choice([None,0.05,0.1]),random.choice([None,0.05,0.1])]
        )


        seq_len = params["seq_len"]
        batch_size = params["batch_size"]
        learning_rate = params["learning_rate"]

        train_data, val_data = split_data(tokenized_data, seq_len)

        train_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
        val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))
        val_batch_iter = data.batch_items(val_iter, batch_size)

        print(f"Training model with params: {params}")

        model: torch.nn.Module = TransformerLM(
            n_layers=params["n_layers"],
            n_heads=params["n_heads"],
            embed_size=params["embed_size"],
            max_context_len=params["seq_len"],
            vocab_size=tokenizer.vocab_size(),
            mlp_hidden_size=params["mlp_hidden_size"],
            with_residuals=True,
            dropout=params["dropout"],
        )

        model = model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=1e-7,
        )

        model.train()

        for batch in data.batch_items(train_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1

            if num_batches % log_every == 0:
                train_loss_value = loss.item()
                train_losses.append(train_loss_value)
                print(f"Seen {num_batches} batches. Last train loss: {train_loss_value:.6f}")

            if num_batches % sample_every == 0:
                model.eval()
                with torch.no_grad():
                    sampled = tokenizer.detokenize(
                        model.sample_continuation(tokenizer.tokenize("שלום"), 500)
                    )
                print(f"Model sample: '''{sampled}'''")
                print("")
                model.train()

            if num_batches % validate_every == 0:
                model.eval()

                with torch.no_grad():
                    val_loss_sum = 0.0

                    for _ in range(val_steps):
                        val_batch = next(val_batch_iter)
                        val_x, val_y = lm.batch_to_labeled_samples(val_batch)
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)

                        val_logits = model(val_x)
                        val_loss_sum += lm.compute_loss(val_logits, val_y).item()

                    curr_val_loss = val_loss_sum / val_steps

                val_losses.append(curr_val_loss)
                scheduler.step(curr_val_loss)

                if curr_val_loss < best_val_loss:
                    best_val_loss = curr_val_loss
                    save_best_model(
                        model,
                        params,
                        best_val_loss,
                        epoch=num_batches,
                        out_dir="../models/",
                    )

                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Val loss: {curr_val_loss:.6f} | "
                    f"best val loss: {best_val_loss:.6f} | "
                    f"lr: {current_lr:.8g}"
                )

                model.train()

        loss_plotter(train_losses, val_losses, params, "../figs/")