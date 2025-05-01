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
        num_lstm: int = 1,
        num_gcn: int = 2,
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
        self.H       = hidden_dim
        self.neg_slp = negative_slope
        self.num_nodes = num_nodes

        # 1) temporal feature extractor
        self.lstm = TemporalLSTM(num_layers=num_lstm, hidden_dim=hidden_dim)

        # register adjacency
        self.register_buffer('adj', adj_matrix.float())

        # 2) graph conv + graph norm
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_gcn):
            in_ch = self.H * num_lstm if i == 0 else self.H
            self.gcn_layers.append(DenseGCNConv(in_channels=in_ch, out_channels=self.H))
            self.norm_layers.append(GraphNorm(self.H))


        # 3) readout
        self.avgpool     = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(self.H, 2)

        # initialize final classifier
        nn.init.xavier_normal_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, V, T) node time series
        Returns:
            logits: (B, 2) pre‐softmax
            probs : (B,)  positive‐class probability
        """
        B, V, T = x.shape
        # 1) node embeddings
        e = self.lstm(x)                

        # 2) GCN layers
        h = None
        for i, (gc, gn) in enumerate(zip(self.gcn_layers, self.norm_layers)):
            h_in = e if i == 0 else h
            h = gc(h_in, self.adj)
            h = gn(h)
            h = F.leaky_relu(h, negative_slope=self.neg_slp)

        # 3) global pool + classifier
        # permute to (B, H, V) for 1d‐pool over nodes
        feats = h.permute(0, 2, 1)      # → (B, H, V)
        pooled = self.avgpool(feats)    # → (B, H, 1)
        pooled = pooled.squeeze(-1)     # → (B, H)
        logits = self.final_linear(pooled)  # → (B, 2)

        # 4) positive‐class confidence
        probs = F.softmax(logits, dim=-1)[:, 1]  # → (B,)

        return logits, probs