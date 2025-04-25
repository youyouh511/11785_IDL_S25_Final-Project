import torch
import torch.nn as nn
import torch.nn.functional as F
from ..temporal_module import TemporalLSTM

class CausalGNN(nn.Module):
    """
    Graph Neural Network combining LSTM-derived node embeddings
    and a weighted directed adjacency for message passing.
    """
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        hidden_dim: int,
        adj_matrix: torch.Tensor
    ):
        super().__init__()
        self.num_nodes = num_nodes
        # Temporal encoder
        self.temporal = TemporalLSTM(hidden_dim)
        # Register adjacency as buffer
        self.register_buffer('adj', adj_matrix.float())   # (V, V)
        # Two graph-convolution layers (MLP on aggregated messages)
        self.gc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        # Final classifier\        
        self.classifier = nn.Linear(hidden_dim, 2)
        # Initialize weights
        for m in [self.gc1, self.gc2, self.classifier]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, V, T)
        Returns:
            logits: (batch_size, 2)
        """
        B, V, T = x.shape
        # 1) Node embeddings via LSTM -> (B, V, H)
        h = self.temporal(x)

        # 2) First message-passing: aggregate neighbor features
        adj = self.adj.unsqueeze(0).expand(B, -1, -1)  # (B, V, V)
        m1 = torch.bmm(adj, h)                        # (B, V, H)
        h1 = F.leaky_relu(self.gc1(m1))               # (B, V, H)

        # 3) Second message-passing
        m2 = torch.bmm(adj, h1)                       # (B, V, H)
        h2 = F.leaky_relu(self.gc2(m2))               # (B, V, H)

        # 4) Global readout: mean pool across nodes
        g = h2.mean(dim=1)                            # (B, H)

        # 5) Classification
        logits = self.classifier(g)                   # (B, 2)
        return logits