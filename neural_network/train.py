import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.mamba import Mamba
from utils import init_hidden, AudioSegmentDataset, Loss
from config import HyperParams, ModelParams
from itertools import cycle

if __name__== "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"using: {device}")

    # We are only training using the (original) 48 kHz samples. To train using 44.1 kHz, do: AudioSegmentDataset(directory="dataset/train", sr="44100", p_zero=0.05)
    # preprocess.py creates both 48 kHz and 44.1 kHz datasets for testing during evaluation.
    # (We could train on both 48 kHz and 44.1 kHz but the model does not degrade when changing the sampling scale as we learn a continuous-time state-space
    # and discretize it with bilinear transformation.)
    train_dataloader = DataLoader(AudioSegmentDataset(directory="dataset/train", sr="48000", p_zero=0.05), batch_size=HyperParams.batch_size, shuffle=True)
    val_dataloader = DataLoader(AudioSegmentDataset(directory="dataset/val", sr="48000", p_zero=0.0), batch_size=HyperParams.batch_size, shuffle=False)
    model = Mamba(ModelParams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HyperParams.lr)
    loss_fn = Loss(HyperParams.alpha)

    print(sum(p.numel() for p in model.parameters()), "model parameters")
    print("Model architecture:")
    print(model)
    print(f"training steps: {HyperParams.num_training_steps}, warm-up samples: {HyperParams.warm_up}, TBPTT samples: {HyperParams.tbptt_seq}, num updates per TBPTT: {HyperParams.seq_len // HyperParams.tbptt_seq}")

    writer = SummaryWriter()
    train_iterator = cycle(train_dataloader)
    pbar = tqdm(total=HyperParams.num_training_steps, desc="Training", position=0)
    train_loss = 0.0
    val_loss = 0.0
    global_step = 0
    chunks = HyperParams.seq_len // HyperParams.tbptt_seq
    best_loss = float('inf')
    while global_step < HyperParams.num_training_steps:
        batch = next(train_iterator)
        input_batch = batch['input'].unsqueeze(-1).to(device)
        target_batch = batch['target'].unsqueeze(-1).to(device)
        cond = batch['c'].to(device)

        model.train()
        h1, h2 = init_hidden(ModelParams.n_layers, input_batch.shape[0], ModelParams.ssm_size, device)
        with torch.no_grad():
            _, (h1, h2) = model(input_batch[:, :HyperParams.warm_up], (h1, h2), cond[:, :HyperParams.warm_up])

        current_step_loss = 0.0
        for i in range(chunks):
            optimizer.zero_grad()

            start = HyperParams.warm_up + i * HyperParams.tbptt_seq
            end = start + HyperParams.tbptt_seq
            input = input_batch[:, start:end]
            target = target_batch[:, start:end]
            c = cond[:, start:end]

            y, (h1, h2) = model(input, (h1, h2) , c)
            h1 = h1.detach()
            h2 = h2.detach()
            loss = loss_fn.compute(y, target)

            loss.backward()
            optimizer.step()
            current_step_loss += loss.item()
        train_loss += (current_step_loss / chunks)

        if (global_step + 1) % HyperParams.logging_interval == 0:
            with torch.no_grad():
                val_loss_total = 0.0
                model.eval()
                for batch in tqdm(val_dataloader, desc="Validating", leave=False) :
                    input_batch = batch['input'].unsqueeze(-1).to(device)
                    target_batch = batch['target'].unsqueeze(-1).to(device)
                    cond = batch['c'].to(device)

                    h1_val, h2_val = init_hidden(ModelParams.n_layers, input_batch.shape[0], ModelParams.ssm_size, device)
                    _, (h1_val, h2_val) = model(input_batch[:, :HyperParams.warm_up], (h1_val, h2_val), cond[:, :HyperParams.warm_up])

                    batch_loss = 0.0
                    for j in range(chunks):
                        start = HyperParams.warm_up + j * HyperParams.tbptt_seq
                        end = start + HyperParams.tbptt_seq
                        input = input_batch[:, start:end]
                        target = target_batch[:, start:end]
                        c = cond[:, start:end]

                        y, (h1_val, h2_val) = model(input, (h1_val, h2_val), c)
                        loss = loss_fn.compute(y, target)
                        batch_loss += loss.item()

                    val_loss_total += (batch_loss / chunks)
            val_loss = val_loss_total / len(val_dataloader)

            if val_loss < best_loss:
                torch.save(model.state_dict(), f"results/S5_mamba_model_{HyperParams.name}.pth")
                torch.save(optimizer.state_dict(), f"results/S5_mamba_optimizer_{HyperParams.name}.pth")
                best_loss = val_loss
            writer.add_scalar('Loss/val', val_loss, global_step)
            writer.add_scalar('Loss/train', train_loss / HyperParams.logging_interval, global_step)
            train_loss = 0.0
            val_loss = 0.0

        global_step += 1
        pbar.update(1)
    writer.close()
    pbar.close()