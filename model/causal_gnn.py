import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm import TemporalLSTM

class CausalGNN(nn.Module):
    """
    Graph Neural Network + LSTM-derived node embeddings + Weighted directed adjacency
    """
    def __init__(
        self,
        adj_matrix: torch.Tensor,    # normalized, masked adjacency matrix
        num_nodes: int = 7,
        hidden_dim: int = 256,
        
    ):
        """
        Args:
            adj_matrix: (V, V) noramlized, weighted adjacency matrix
            num_nodes: number of nodes in the graph, default to 7 as in the paper
            hidden_dim: dimension of the hidden state in LSTM, default to 256 as in the paper
        """
        super().__init__()

        self.H = hidden_dim                                 # hidden dimension

        self.num_nodes = num_nodes                          # channel, number of variables
        self.lstm = TemporalLSTM(hidden_dim)                # Temporal feature extractor

        self.register_buffer('adj', adj_matrix.float())     # register buffer for device compatibility
        
        # 2 layers of Conv to update temporal node features
        self.gc1 = nn.Conv1d(
            in_channel=hidden_dim, 
            out_channel=hidden_dim*2, 
            kernel_size=1)
        self.ln1 = nn.LayerNorm(hidden_dim*2)

        self.gc2 = nn.Conv1d(
            in_channel=hidden_dim*2, 
            out_channel=hidden_dim, 
            kernel_size=1)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)              # global average pooling of node features
        self.final_linear = nn.Linear(hidden_dim, 2)        # final linear binary classifier
        
        # Weight initialization
        for layer in [self.gc1, self.gc2, self.final_linear]:
            nn.init.xavier_uniform_(layer.weight)           # Apply Xavier normalization as in the paper
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)                  # Zero initialization for biases


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_nodes, seq_len)
                containing each node’s scalar time series
        Returns:
            logits: Tensor of shape (batch_size, 2)
                representing the predicted class probabilities
        """

        B, V, T = x.shape

        # Node embedding (B, V, hidden_dim)
        e = self.lstm(x)

        # Graph aggregation
        A = self.adj.unsqueeze(0).expand(B, -1, -1)             # (V, V) -> (B, V, V)
        agg = torch.bmm(A, e)                                   # (B, V, hidden_dim)

        # Graph convolution 1
        h1 = agg.view(B*V, self.H).unsqueeze(-1)                # (B*V, hidden_dim, 1)
        h1 = self.gc1(h1).squeeze(-1)                           # (B*V, hidden_dim*2)
        h1 = self.ln1(h1)                                       # (B*V, hidden_dim*2)
        h1 = F.leaky_relu(h1)                                   # (B*V, hidden_dim*2)

        # Graph convolution 2
        h2 = self.gc2(h1.unsqueeze(-1)).squeeze(-1)             # (B*V, hidden_dim)
        h2 = self.ln2(h2)                                       # (B*V, hidden_dim)
        h2 = F.leaky_relu(h2)                                   # (B*V, hidden_dim)

        # Reshape back and apply global average pooling
        feats = h2.view(B, V, -1)                               # (B, V, hidden_dim)
        pool = self.avgpool(feats.permute(0, 2, 1)).squeeze(-1) # (B, hidden_dim)
        logits = self.final_linear(pool)                        # (B, 2)

        # calculate confidence via softmax
        probs = F.softmax(logits, dim=1)                        # (B, 2)
        probs = probs[:, 1]                                     # confidence of positive targets

        return logits, probs