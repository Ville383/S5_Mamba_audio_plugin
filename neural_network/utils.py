import torch
from torch.utils.data import Dataset
import auraloss
import os
import pickle
from config import ModelParams, HyperParams
from tqdm import tqdm
import math
import random


def init_hidden(n_layers: int, batch_size: int, hidden_size: int, device: str):
    return (torch.zeros(n_layers, batch_size, hidden_size, dtype=torch.float32, device=device), torch.zeros(n_layers, batch_size, hidden_size, dtype=torch.float32, device=device))


class AudioSegmentDataset(Dataset):
    def __init__(self, directory, sr, p_zero):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(f'{sr}.pkl')]
        self.segment_len = HyperParams.warm_up + HyperParams.seq_len
        self.p_zero = p_zero
        print(f"Dataset initialized with {len(self.files)} files. Zero-sample probability: {self.p_zero:.2f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if random.random() < self.p_zero:
            input_tensor = torch.zeros(self.segment_len, dtype=torch.float32)
            target_tensor = torch.zeros(self.segment_len, dtype=torch.float32)
            c_tensor = torch.rand(self.segment_len, ModelParams.c_dim, dtype=torch.float32) * 2 - 1
        else:
            with open(self.files[idx], 'rb') as f:
                data = pickle.load(f)
            input_tensor = torch.from_numpy(data['input'])
            target_tensor = torch.from_numpy(data['target'])
            c_tensor = torch.from_numpy(data['c'])
        return {
            'input': input_tensor,
            'target': target_tensor,
            'c': c_tensor,
        }


class Loss:
    def __init__(self, alpha: float):
        self.loss_fn1 = torch.nn.MSELoss()
        self.loss_fn2 = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes = [1024, 512, 256], hop_sizes = [120, 50, 25], win_lengths = [440, 240, 100], mag_distance = "L1")
        self.alpha = alpha

    def compute(self, output: torch.Tensor, target: torch.Tensor):
        """
        Input:
            target (float32):   (B, L, 1)
            output (float32):   (B, L, 1)
        Return:
            loss (float32): (1)
        """
        return self.loss_fn1(output, target) + self.alpha * self.loss_fn2(output.permute(0,2,1), target.permute(0,2,1))


def ESR_loss(output, target, eps=1e-8):
    """
    ESR loss per sample in the batch.
    Args:
        output: (L)
        target: (L)
    Returns:
        Tensor: (1) ESR value
    """
    error_power = torch.sum((output - target) ** 2, dim=-1)
    signal_power = torch.sum(target ** 2, dim=-1)
    return error_power / (signal_power + eps)


def eval(model, loader, m_args: ModelParams, h_args: HyperParams, device):
    mr_stft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes = [1024, 512, 256], hop_sizes = [120, 50, 25], win_lengths = [440, 240, 100], mag_distance = "L1")
    model.eval()
    targets = []
    outputs = []
    if m_args.conj_sym:
        ssm_size = m_args.d_state // 2
    else:
        ssm_size = m_args.d_state
    chunks = h_args.seq_len // h_args.tbptt_seq
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_batch = batch['input'].unsqueeze(-1).to(device)
            target_batch = batch['target'].unsqueeze(-1).to(device)
            cond = batch['c'].to(device)
            h1, h2 = init_hidden(m_args.n_layers, input_batch.shape[0], ssm_size, device)
            _, (h1, h2) = model(input_batch[:, :h_args.warm_up], (h1, h2), cond[:, :h_args.warm_up])

            for i in range(chunks):
                start = h_args.warm_up + i * h_args.tbptt_seq
                end = start + h_args.tbptt_seq
                input = input_batch[:, start:end]
                target = target_batch[:, start:end]
                c = cond[:, start:end]

                y, (h1, h2) = model(input, (h1, h2), c)
                outputs.extend(list(torch.unbind(y.squeeze(-1), dim=0)))
                targets.extend(list(torch.unbind(target.squeeze(-1), dim=0)))

        outputs = torch.stack(outputs, dim=0)
        targets = torch.stack(targets, dim=0)
        loss = ESR_loss(outputs.flatten(), targets.flatten()).item()
        f_loss = mr_stft(outputs.unsqueeze(0), targets.unsqueeze(0)).item()

    print(f"ESR value: {loss}")
    print(f"ESR_dB value: {10*math.log10(loss)}")
    print(f"MR STFT value: {f_loss}")