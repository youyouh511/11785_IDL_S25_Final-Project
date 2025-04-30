import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, GraphNorm
from model.lstm import TemporalLSTM

class CausalGNN(nn.Module):
    """
    Graph Neural Network + LSTM-derived node embeddings + Weighted directed adjacency
    """
    def __init__(
        self,
        adj_matrix: torch.Tensor,    # normalized, masked adjacency matrix
        num_nodes: int = 6,
        hidden_dim: int = 256,
        negative_slope: float = 0.2,
    ):
        """
        Args:
            adj_matrix: (V, V) noramlized, weighted adjacency matrix
            num_nodes: number of nodes in the graph, default to 7 as in the paper
            hidden_dim: dimension of the hidden state in LSTM, default to 256 as in the paper
        """
        super().__init__()

        self.H = hidden_dim                                 # hidden dimension
        self.neg_slp = negative_slope                       # dropout rate

        self.num_nodes = num_nodes                          # channel, number of variables
        self.lstm = TemporalLSTM(hidden_size=hidden_dim)                # Temporal feature extractor

        # This will move to device when model.to(device) is called
        self.register_buffer('adj', adj_matrix)     # register buffer for device compatibility
        
        # 2 layers of Conv to update temporal node features
        self.gc1 = DenseGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim*2, 
            )
        self.gn1 = GraphNorm(hidden_dim*2)

        self.gc2 = DenseGCNConv(
            in_channels=hidden_dim*2,
            out_channels=hidden_dim,
            )
        self.gn2 = GraphNorm(hidden_dim)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)              # global average pooling of node features
        self.final_linear = nn.Linear(hidden_dim, 2)        # final linear binary classifier
        
        # Weight initialization
        for layer in [self.final_linear]:
            nn.init.xavier_normal_(layer.weight)           # Apply Xavier normalization as in the paper
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)                  # Zero initialization for biases


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_nodes, seq_len)
                containing each node's scalar time series
        Returns:
            logits: Tensor of shape (batch_size, 2)
                representing the predicted class probabilities
        """

        B, V, T = x.shape

        # Node embedding
        e = self.lstm(x)                                        # (B, V, H)
    
        # Graph convolution #1
        h1 = self.gc1(e, self.adj)                              # (B, V, H) + (V, V) -> (B, V, 2H)
        h1 = F.leaky_relu(h1, negative_slope=self.neg_slp)

        # Graph convolution 2
        h2 = self.gc2(h1, self. adj)                            # -> (B, V, H)
        h2 = F.leaky_relu(h2, negative_slope=self.neg_slp)

        # Global average pooling & Binary classification
        feats = h2.permute(0, 2, 1)                             # -> (B, H, V)
        pool = self.avgpool(feats).squeeze(-1)                  # -> (B, H)
        logits = self.final_linear(pool)                        # (B, 2)

        # calculate confidence via softmax
        probs = F.softmax(logits, dim=-1)                       # (B, )
        probs = probs[:, 1]                                     # confidence of positive targets

        return logits, probs