import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelParams
from model.init_ssm import make_DPLR_HiPPO
from model.torch_parallel_scan import apply_ssm
from model.film import FiLM


class S5_SSM(nn.Module):
    def __init__(self, d_inner, ssm_size, Lambda_re_init, Lambda_im_init, V, Vinv, dt_min, dt_max, dt_init_floor, conj_sym):
        super(S5_SSM, self).__init__()
        self.conj_sym = conj_sym
        if conj_sym:
            local_P = 2*ssm_size
        else:
            local_P = ssm_size
        
        self.A_real = nn.Parameter(Lambda_re_init.type(torch.float32)) # (ssm_size)
        self.A_imag = nn.Parameter(Lambda_im_init.type(torch.float32))

        B_init = torch.randn(local_P, d_inner, dtype=torch.float32) * math.sqrt(1.0 / local_P)
        B = (Vinv @ B_init.to(Vinv.dtype)).type(torch.complex64)
        self.B_real = nn.Parameter(B.real) # (ssm_size, d_inner)
        self.B_imag = nn.Parameter(B.imag)

        C_init = torch.randn(d_inner, local_P, dtype=torch.complex64) * math.sqrt(1.0 / d_inner)
        C = (C_init.to(V.dtype) @ V).type(torch.complex64)
        self.C_real = nn.Parameter(C.real) # (d_inner, ssm_size)
        self.C_imag = nn.Parameter(C.imag)

        self.D = nn.Parameter(torch.randn(d_inner, dtype=torch.float32)) # (d_inner)

        log_dt = torch.rand(ssm_size, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt = torch.exp(log_dt).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.inv_dt = nn.Parameter(dt + torch.log(-torch.expm1(-dt))) # (ssm_size)
    
        self.step_rescale = 1.0
    
    def discretize_bilinear(self, A, B, dt):
        den = 1.0 - (dt / 2.0) * A
        num = 1.0 + (dt / 2.0) * A
        dA = num / den
        # Unsqueeze for broadcasting to multiply with B.
        dB = (dt / den).unsqueeze(1) * B
        return dA, dB

    def forward(self, u, h_real, h_imag):
        """
        Input:
            u:      (B, L, d_inner)
            h_real: (B, ssm_size)
            h_imag: (B, ssm_size)
        Returns:
            output: (B, L, d_inner)
        """
        # discretize using the bilinear transform
        A = self.A_real + 1j * self.A_imag
        B = self.B_real + 1j * self.B_imag
        C = self.C_real + 1j * self.C_imag
        h = h_real + 1j * h_imag
        dt = self.step_rescale * F.softplus(self.inv_dt)
        dA, dB = self.discretize_bilinear(A, B, dt)

        # Apply: h[n] = Ah[n-1]     + Bu[n]
        #        y[n] = real(Ch[n]) + Du[n]
        y, h_real, h_imag = apply_ssm(dA, dB, C, self.D, u, h, self.conj_sym)
        return y, h_real, h_imag

    def change_scale(self, step_rescale):
        self.step_rescale = step_rescale


class MambaBlock(nn.Module):
    def __init__(self, args: ModelParams):
        """A Mamba block, as described in the Mamba paper but with a S5 state-space."""
        super(MambaBlock, self).__init__()
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # Initialize state-space parameters A, B, C, D, dt
        assert args.d_state % args.blocks == 0, "d_state must be divisible by number of blocks"
        block_size = args.d_state // args.blocks
        ssm_size = args.d_state

        Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)

        if args.conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        Lambda = (Lambda * torch.ones((args.blocks, block_size))).ravel()
        V = torch.block_diag(*([V] * args.blocks))
        Vinv = torch.block_diag(*([Vc] * args.blocks))

        self.ssm = S5_SSM(args.d_inner, ssm_size, Lambda.real, Lambda.imag, V, Vinv, args.dt_min, args.dt_max, args.dt_init_floor, args.conj_sym)

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x, h_real, h_imag):
        """
        Input:
            x:      (B, L, d_model)
            h_real: (B, ssm_size)
            h_imag: (B, ssm_size)
        Returns:
            output: (B, L, d_model)
        """
        x_and_res = self.in_proj(x) # (B, L, 2 * d_inner)
        u, res = x_and_res.chunk(chunks=2, dim=-1)

        u = F.silu(u)
        y, h_real, h_imag = self.ssm(u, h_real, h_imag)
        y = y * F.silu(res)

        return self.out_proj(y), h_real, h_imag
    
    def change_scale(self, step_rescale):
        self.ssm.change_scale(step_rescale)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelParams):
        """Simple block wrapping a Mamba block with FiLM conditioning, RMSNorm, and a residual connection."""
        super(ResidualBlock, self).__init__()
        self.norm = nn.RMSNorm(args.d_model, eps = 1e-6)
        self.mamba = MambaBlock(args)
    
    def forward(self, x, h_real, h_imag, gamma, beta):
        """
        Input:
            x:           (B, L, d_model)
            h_real:      (B, ssm_size)
            h_imag:      (B, ssm_size)
            gamma, beta: (B, L, d_model)
        Returns:
            output:      (B, L, d_model)
        """
        tmp = gamma * x + beta # FiLM conditioning
        out, h_real, h_imag = self.mamba(self.norm(tmp), h_real, h_imag)
        return out + x, h_real, h_imag

    def change_scale(self, step_rescale):
        self.mamba.change_scale(step_rescale)


class Mamba(nn.Module):
    def __init__(self, args: ModelParams):
        super(Mamba, self).__init__()
        self.film_gen = FiLM(args.c_dim, args.d_model)

        self.in_proj = nn.Linear(args.input_size, args.d_model, bias=args.bias)

        self.mamba_blocks = nn.ModuleList([])
        for _ in range(args.n_layers):
            self.mamba_blocks.append(ResidualBlock(args))

        self.out_proj = nn.Linear(args.d_model, args.output_size, bias=args.bias)

        self.n_layers = args.n_layers

    def forward(self, x, h, c):
        """
        Input:
            x:           (B, L, input_size)
            h_real:      (B, ssm_size)
            h_imag:      (B, ssm_size)
            c:           (B, L, c_dim)
        Returns:
            output:      (B, L, output_size)
        """
        gamma, beta = self.film_gen(c)
        x = self.in_proj(x)

        h_real, h_imag = h
        for i in range(self.n_layers):
            x, h_real[i], h_imag[i] = self.mamba_blocks[i](x, h_real[i], h_imag[i], gamma, beta)
        
        return self.out_proj(x), (h_real, h_imag)
    
    def change_scale(self, step_rescale):
        for mamba_block in self.mamba_blocks:
            mamba_block.change_scale(step_rescale)