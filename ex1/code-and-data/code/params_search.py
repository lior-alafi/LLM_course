from __future__ import annotations
import matplotlib.pyplot as plt
from datetime import datetime

from utils import save_best_model,parameters,loss_plotter,split_data
if __name__ == "__main__":
    import lm
    import torch
    from torch import nn, optim
    from transformer import TransformerLM

    import data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seq_len = 128
    batch_size = 64
    data_path = "../data/en/"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0

    num_batches_to_train = 6000 #50000

    tokenizer, tokenized_data = data.load_data(data_path)

    #train, test split
    # train_data,val_data = split_data(tokenized_data, seq_len)

    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    # data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))









for i in range(30):

        best_val_loss = 1000
        val_losses = []
        train_losses = []

        num_batches = 0
        params = parameters(
            seq_len=[128],
            batch_size=[32, 64,128],
            n_layers=[4, 6, 8],
            n_heads=[4, 6, 8],
            embed_size=[64,128, 192, 256, 384],
            mlp_hidden_size=lambda d: d * 4,
            learning_rate=learning_rate
        )
        seq_len = params["seq_len"]
        batch_size = params["batch_size"]
        learning_rate = params["learning_rate"]


        train_data, val_data = split_data(tokenized_data, seq_len)

        train_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
        val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))
        print(f'training model with params: {params}')
        model: torch.nn.Module = TransformerLM(
            params['n_layers'],
            params['n_heads'],
            params['embed_size'],
            params['seq_len'],
            tokenizer.vocab_size(),
            params['mlp_hidden_size'],
            with_residuals=True,
        )

        model = model.to(device)


        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-7)
        model.train()
        for batch in data.batch_items(train_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            # model.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                train_losses.append(loss.item())
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.sample_continuation(tokenizer.tokenize("Hello"), 500)
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")
            if num_batches % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_batch = None
                    try:
                        val_batch = next(data.batch_items(val_iter, batch_size))
                    except StopIteration:
                        val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))
                        val_batch = next(data.batch_items(val_iter, batch_size))
                    val_x, val_y = lm.batch_to_labeled_samples(val_batch)
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                    val_logits = model(val_x)
                    val_loss = lm.compute_loss(val_logits, val_y)
                    curr_val_loss = val_loss.item()
                    val_losses.append(curr_val_loss)
                    scheduler.step(curr_val_loss)
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        save_best_model(model,params,best_val_loss,epoch=num_batches,out_dir='../models/')
                    print(f"Val loss: {curr_val_loss} best val loss: {best_val_loss}")

                model.train()

        loss_plotter(train_losses,val_losses,params,'../figs/')
