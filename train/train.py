import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        device: torch.device | str = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        # 自动选择设备
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # 损失函数、优化器、调度器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    def train_epoch(self) -> tuple[float, float]:
        """单轮训练，返回 (平均 loss, accuracy)"""
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for x, y in tqdm(self.train_loader, desc="Train Batches", leave=False):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            # 支持 forward 返回 logits 或 (logits, probs)
            if isinstance(out, tuple):
                logits, _ = out
            else:
                logits = out

            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(y.cpu())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(
            torch.cat(all_targets).numpy(),
            torch.cat(all_preds).numpy()
        )
        return avg_loss, accuracy

    def evaluate(self, loader: DataLoader, mode: str = "Val") -> dict:
        """
        在给定 loader 上评估，返回 dict:
          - loss
          - accuracy
          - auroc
          - auprc
        """
        self.model.eval()
        ys, probs_list = [], []
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Eval {mode}", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, probs = out
                else:
                    logits = out
                    probs  = torch.softmax(logits, dim=1)

                loss = self.criterion(logits, y)
                total_loss += loss.item()

                # Accuracy 收集
                all_preds.append(logits.argmax(dim=1).cpu())
                all_targets.append(y.cpu())

                # AUROC/AUPRC 收集：动态取正类概率
                p = probs.cpu()
                if p.ndim == 1:
                    pos = p.numpy()
                else:
                    pos = p[:, 1].numpy()
                ys.append(y.cpu().numpy())
                probs_list.append(pos)

        # 拼接所有批次结果
        avg_loss = total_loss / len(loader)
        y_true  = np.concatenate(ys)
        y_score = np.concatenate(probs_list)
        y_pred  = torch.cat(all_preds).numpy()
        y_label = torch.cat(all_targets).numpy()

        metrics = {
            'loss':     avg_loss,
            'accuracy': accuracy_score(y_label, y_pred),
            'auroc':    float('nan'),
            'auprc':    float('nan'),
        }
        # 计算 AUROC/AUPRC（若单类则跳过）
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_score)
        except ValueError:
            pass
        try:
            metrics['auprc'] = average_precision_score(y_true, y_score)
        except ValueError:
            pass

        return metrics

    def fit(self, num_epochs: int):
        """完整训练流程，按 Val AUROC 保存最优模型"""
        best_auroc = -float('inf')
        best_state = None

        epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", leave=True)
        for epoch in epoch_bar:
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader, mode="Val")

            # lr 调度
            self.scheduler.step(val_metrics['loss'])

            # 更新进度条
            epoch_bar.set_description(
                f"Ep{epoch} | tr loss {train_loss:.3f} acc {train_acc:.3f} | "
                f"val loss {val_metrics['loss']:.3f} "
                f"auroc {val_metrics['auroc']:.3f} auprc {val_metrics['auprc']:.3f}"
            )

            # 打印本轮指标
            print(f"\nEpoch {epoch}:")
            print(f"  Train → loss {train_loss:.4f}, acc {train_acc:.4f}")
            print(
                f"  Val   → loss {val_metrics['loss']:.4f}, "
                f"acc {val_metrics['accuracy']:.4f}, "
                f"AUROC {val_metrics['auroc']:.4f}, AUPRC {val_metrics['auprc']:.4f}"
            )

            # 保存最佳模型
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                best_state = self.model.state_dict()

        print(f"\n🟢 Best Val AUROC: {best_auroc:.4f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def test(self):
        """在 test_loader 上评估并打印结果"""
        if self.test_loader is None:
            print("⚠️ No test loader provided.")
            return
        test_metrics = self.evaluate(self.test_loader, mode="Test")
        print(
            f"Test → loss {test_metrics['loss']:.4f}, "
            f"acc {test_metrics['accuracy']:.4f}, "
            f"AUROC {test_metrics['auroc']:.4f}, "
            f"AUPRC {test_metrics['auprc']:.4f}"
        )