import torch
import torch.nn as nn

class FiLM(nn.Module):
    """FiLM Generator that produces gamma and beta from conditioning input.
       The same gamma and beta is shared between every layer."""
    def __init__(self, c_dim, d_model):
        super(FiLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 2 * d_model) # Output gamma and beta
        )

    def forward(self, c):
        """
        Args:
            c: Conditioning input tensor (batch_size, sequence_length, num_conditions)
        Returns:
            gamma: Scaling factor (batch_size, sequence_length, d_model)
            beta: Shifting factor (batch_size, sequence_length, d_model)
        """
        gamma_beta = self.fc(c)
        return torch.chunk(gamma_beta, 2, dim=-1)