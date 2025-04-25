###########################################################################
# evaluate.py                                                              #
###########################################################################
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model: torch.nn.Module,
             loader,
             device: torch.device = torch.device('cuda')) -> dict:
    # Ensure model is on device and in eval mode
    model.to(device)
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            # Compute probability of positive class
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ys.append(y.numpy())
            preds.append(probs)
    # Concatenate all batches
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    # Return both AUROC and AUPRC
    return {
        'auroc': roc_auc_score(ys, preds),
        'auprc': average_precision_score(ys, preds)
    }