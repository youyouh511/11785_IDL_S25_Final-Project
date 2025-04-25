###########################################################################
# train.py                                                                 #
###########################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CausalGNN
from evaluate import evaluate


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int = 50,
          lr: float = 1e-5,
          weight_decay: float = 5e-6,
          device: torch.device = torch.device('cuda')) -> None:
    # Move model to the correct device
    model.to(device)
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_auprc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Training loop
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)       # send batch to device
            optimizer.zero_grad()
            logits = model(x)                       # forward pass
            loss = criterion(logits, y)             # compute loss
            loss.backward()                         # backpropagate
            optimizer.step()                        # update weights
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation step
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Val AUPRC={val_metrics['auprc']:.4f}, AUROC={val_metrics['auroc']:.4f}")

        # Save best model by AUPRC
        if val_metrics['auprc'] > best_auprc:
            best_auprc = val_metrics['auprc']
            torch.save(model.state_dict(), 'best_causal_gnn.pt')