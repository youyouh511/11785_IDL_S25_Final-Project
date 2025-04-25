import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGNN(nn.Module):
    """
    Two-layer temporal convolution + causal-graph pooling for binary classification.
    """
    def __init__(self, num_nodes: int, in_channels: int, hidden_dim: int, adj_matrix: torch.Tensor):
        super().__init__()

        self.num_nodes = num_nodes
        
        # Register adjacency as a buffer so it moves with the model to GPU/CPU automatically
        self.register_buffer('adj', adj_matrix.float())  # shape: (V, V)

        self.temporalConv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Binary linear classifier
        self.linear = nn.Linear(hidden_dim, 2)

        # Initialize all weights
        self._init_weights()


    def _init_weights(self):
        # Xavier uniform init for convolutional and linear layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, num_nodes, time_steps)
        B, V, T = x.shape
        # Reshape to apply 1D conv per node: (B*V, 1, T)
        h = x.view(B * V, 1, T)
        # First conv + remove last dim -> (B*V, hidden_dim)
        h = self.conv1(h).squeeze(-1)
        # Activation + normalize
        h = F.leaky_relu(self.ln1(h))
        # Prepare for second conv: add trailing dimension -> (B*V, hidden_dim, 1)
        h = h.unsqueeze(-1)
        # Second conv + squeeze -> (B*V, hidden_dim)
        h = self.conv2(h).squeeze(-1)
        h = F.leaky_relu(self.ln2(h))
        # Reshape back to (B, V, hidden_dim)
        h = h.view(B, V, -1)

        # Causal pooling: aggregate features from parent nodes
        # Expand adjacency to batch dimension -> (B, V, V)
        adj = self.adj.unsqueeze(0).expand(B, -1, -1)
        # Multiply adjacency (transpose) with node features -> (B, V, hidden_dim)
        parent_feats = torch.bmm(adj.transpose(1, 2), h)
        # Mean pooling across all nodes to get global feature -> (B, hidden_dim)
        global_feat = parent_feats.mean(dim=1)

        # Classifier to produce logits for two classes
        logits = self.classifier(global_feat)
        return logits