import torch
import torch.nn as nn

class TemporalLSTM(nn.Module):
    '''
    Single-layer LSTM to extract temporal features of the given variable
    '''
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,          # one scalar per time-step
            hidden_size=hidden_dim,
            num_layers=1,          # single layer as per paper
            batch_first=True,
            bidirectional=False,   # unidirectional LSTM
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_nodes, time_steps)
               containing each node’s scalar time series
        Returns:
            feats: Tensor of shape (batch_size, num_nodes, hidden_dim)
                representing the final hidden state for each node
        """
        B, V, T = x.shape

        # Flatten nodes into the batch dimension: (B*V, T, 1) 1D convolution
        x_flat = x.view(B * V, T, 1)

        # outputs: (B*V, T, hidden_dim)
        # h_n: (num_layers, B*V, hidden_dim)
        # c_n: (num_layers, B*V, hidden_dim)
        output, (h_n, c_n) = self.lstm(x_flat)

        # (num_layers, B*V, hidden_dim) -> (B*V, hidden_dim)
        h_n = h_n.squeeze(0)

        # Convert shape (B, V, hidden_dim)
        feats = h_n.view(B, V, -1)

        return feats