from dataclasses import dataclass

@dataclass
class HyperParams:
    name = "2_layers_16_d_model_2_exp_f_64_d_state_4_blocks_conj_sym_p_zero_0.05_boss_od-3"

    # TBPTT (if you change these, run preprocess again)
    warm_up: int = 4096
    tbptt_seq: int = 4096
    seq_len: int = tbptt_seq * 15
    
    num_training_steps: int = 10_000
    logging_interval: int = 100

    batch_size: int = 32
    lr: float = 0.001

    # Loss
    alpha: float = 0.001 # weight on the MR-STFT

@dataclass
class ModelParams:
    # AUDIO CHANNELS, 1 for mono
    input_size: int = 1
    output_size: int = 1

    # S5/MAMBA BLOCK
    n_layers: int = 2
    d_model: int = 16 # Number of features for each layer
    d_state: int = 64 # Latent size, P in the S5 paper
    blocks: int = 4 # Number of blocks used for the initialization of A
    conj_sym: bool = True # Effectively cut state-space matrix sizes by half
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    expand_factor: int = 2 # Expansion factor in the Mamba architecture, 2 in the Mamba paper
    bias: bool = False # Whether to use bias in the Mamba projection layers

    d_inner: int = int(expand_factor * d_model)
    if conj_sym:
        ssm_size: int = d_state // 2
    else:
        ssm_size: int = d_state

    # FiLM conditioning
    c_dim: int = 2