import os
import numpy as np
import pickle
import torchaudio
from typing import List
from tqdm import tqdm
from config import HyperParams
from random import shuffle

def get_xy_pairs(file_path: str):
    def parse_conditioning(filename):
        parts = filename.split("_")
        d = int(parts[1][1])
        t = int(parts[2][1])
        d_scaled = 2 * (d / 4) - 1
        t_scaled = 2 * (t / 4) - 1
        return [d_scaled, t_scaled]
    
    x_dir = os.path.join(file_path, 'x')
    y_dir = os.path.join(file_path, 'y')
    y_dict = {f[1:]: f for f in os.listdir(y_dir)}

    pairs = []
    for x_file in os.listdir(x_dir):
        suffix = x_file[1:]
        y_file = y_dict.get(suffix)
        if y_file:
            c = parse_conditioning(y_file)
            x_path = os.path.join(x_dir, x_file)
            y_path = os.path.join(y_dir, y_file)
            pairs.append((x_path, y_path, c))

    return pairs


def process_and_save_pairs(pairs: List, save_dir: str, seq_len: int, warm_up: int, orig_sample_rate=48_000, sample_rate=44_100):
    save_dir_train = os.path.join(save_dir, "train")
    save_dir_val = os.path.join(save_dir, "val")
    save_dir_test = os.path.join(save_dir, "test")
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_val, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)

    idx = 0
    for x_path, y_path, cond in tqdm(pairs):
        x_wave, sr_x = torchaudio.load(x_path)
        y_wave, sr_y = torchaudio.load(y_path)
        assert sr_x == sr_y, f"Sample rates don't match for {x_path} and {y_path}"

        min_len = min(x_wave.shape[1], y_wave.shape[1])
        x_wave = x_wave[0, :min_len]
        y_wave = y_wave[0, :min_len]
        x_resampled = transform(x_wave.unsqueeze(0))[0]
        y_resampled = transform(y_wave.unsqueeze(0))[0]
        x_wave = x_wave.numpy().astype(np.float32)
        y_wave = y_wave.numpy().astype(np.float32)
        x_resampled = x_resampled.numpy().astype(np.float32)
        y_resampled = y_resampled.numpy().astype(np.float32)

        total_len = warm_up + seq_len
        num_segments_raw = (min_len - warm_up) // seq_len
        num_segments_res = (x_resampled.shape[0] - warm_up) // seq_len
        num_segments = min(num_segments_raw, num_segments_res)

        segments = []
        for i in range(num_segments):
            start = i * seq_len
            end = start + total_len
            segments.append((
                x_wave[start:end],
                y_wave[start:end],
                x_resampled[start:end],
                y_resampled[start:end]
            ))
        shuffle(segments)

        num_samples = len(segments)
        test_count = int(0.03 * num_samples)
        val_count = test_count + int(0.03 * num_samples)

        def save_split(split_name, subset, dir_path, c_array, base_idx):
            os.makedirs(dir_path, exist_ok=True)
            for i, (x0, y0, xr, yr) in enumerate(subset):
                # Save 48kHz version
                fname_48 = os.path.join(dir_path, f"sample_{base_idx}_{split_name}_{i}_48000.pkl")
                with open(fname_48, 'wb') as f:
                    pickle.dump({
                        'input': x0,
                        'target': y0,
                        'c': c_array,
                    }, f)

                # Save 44.1kHz version
                fname_44 = os.path.join(dir_path, f"sample_{base_idx}_{split_name}_{i}_44100.pkl")
                with open(fname_44, 'wb') as f:
                    pickle.dump({
                        'input': xr,
                        'target': yr,
                        'c': c_array,
                    }, f)

        c_array = np.array(cond, dtype=np.float32)
        c_array = np.tile(c_array, (total_len, 1))  # shape: (total_len, 2)
        save_split('test', segments[:test_count], save_dir_test, c_array, idx)
        save_split('val', segments[test_count:val_count], save_dir_val, c_array, idx)
        save_split('train', segments[val_count:], save_dir_train, c_array, idx)

        idx += 1


if __name__ == "__main__":
    DATA_PATH = "boss_od3_overdrive/overdrive/boss_od3"
    SAVE_PATH = "dataset"

    print(f"Using data from {DATA_PATH}, saving data to {SAVE_PATH}, each data point is {HyperParams.warm_up + HyperParams.seq_len} samples.")
    data = get_xy_pairs(DATA_PATH)
    process_and_save_pairs(data, SAVE_PATH, HyperParams.seq_len, HyperParams.warm_up)