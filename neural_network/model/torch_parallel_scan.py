import torch
from s5 import binary_operator, associative_scan # https://github.com/i404788/s5-pytorch


def apply_ssm(dA: torch.Tensor, dB: torch.Tensor, C: torch.Tensor, D: torch.Tensor, u: torch.Tensor, h: torch.Tensor, conj_sym: bool):
    """ Compute the BxLxH output of discretized SSM given an BxLxH input.
        Args:
            dA       (complex64):       discretized diagonal state matrix (P)
            dB       (complex64):       discretized input matrix          (P, H)
            C        (complex64):       weight matrix                     (H, P)
            D        (float32):         weight vector                     (H)
            h        (complex64):       hidden state                      (B, P)
            u        (complex64):       input sequence of features        (B, L, H)
            conj_sym (bool):            enforce conjugate symmetry
        Returns:
            hs (float32):             the SSM outputs (S5 layer preactivations) (B, L, P)
            hidden_state (complex64): last hidden state from parallel scan      (B, P)
    """
    B = u.shape[0]
    L = u.shape[1]

    u_c = u.to(dB.dtype)
    dA_batch = dA.unsqueeze(0).unsqueeze(0).expand(B, L, -1) # (B, L, P)
    dB_batch = torch.einsum('ph,blh->blp', dB, u_c) # (B, L, P)
    # Initialize the first B_t with the hidden state h
    dB_batch[:, 0] = h * dA_batch[:, 0] + dB_batch[:, 0]
    
    _, y_scan = associative_scan(binary_operator, (dA_batch, dB_batch), axis=1)
    final_state = y_scan[:, -1]

    if conj_sym:
        return 2 * torch.einsum('hp,blp->blh', C, y_scan).real + D.unsqueeze(0).unsqueeze(0) * u, final_state.real, final_state.imag
    else:
        return torch.einsum('hp,blp->blh', C, y_scan).real + D.unsqueeze(0).unsqueeze(0) * u, final_state.real, final_state.imag