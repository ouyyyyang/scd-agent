"""训练与推理工具。

- `Trainer`：封装 epoch 级训练循环、验证、进度条输出；支持类权重损失。
- `save_checkpoint / load_checkpoint`：模型权重 + 配置元信息的持久化。
- `predict_proba`：批量推理工具，返回 softmax 概率。
- `classification_report_from_loader`：在验证集上打印每类 precision/recall/F1。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 设备解析
# ---------------------------------------------------------------------

def resolve_device(pref: str = "auto") -> torch.device:
    """把配置里的设备字符串解析为 `torch.device`。"""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请改用 cpu 或 auto。")
        return torch.device("cuda")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"unknown device preference: {pref}")


# ---------------------------------------------------------------------
# 类权重
# ---------------------------------------------------------------------

def compute_class_weights(
    y: np.ndarray,
    num_classes: int,
    scheme: str = "inverse_freq",
) -> np.ndarray:
    """按类频率反比计算类权重，缓解 MIT-BIH 极端不平衡问题。

    Args:
        y: 整数标签数组。
        num_classes: 总类别数。
        scheme: `"inverse_freq"` (默认) 或 `"none"`。
    """
    if scheme == "none":
        return np.ones(num_classes, dtype=np.float32)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    # 避免除零：未出现的类给一个小计数
    counts = np.where(counts > 0, counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


# ---------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------

@dataclass
class EpochStats:
    loss: float
    acc: float


class Trainer:
    """最小可用的训练器。"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[np.ndarray] = None,
    ):
        self.model = model.to(device)
        self.device = device
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ----------------------------- train -----------------------------

    def _run_epoch(
        self, loader: DataLoader, train: bool, desc: str
    ) -> EpochStats:
        self.model.train(mode=train)
        total_loss = 0.0
        total_correct = 0
        total = 0
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for xb, yb in iterator:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train):
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            batch = yb.size(0)
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += batch
            iterator.set_postfix(
                loss=f"{total_loss/total:.4f}",
                acc=f"{total_correct/total:.4f}",
            )
        return EpochStats(total_loss / max(total, 1), total_correct / max(total, 1))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
    ) -> Dict[str, list]:
        """完整训练循环，返回各 epoch 的 loss/acc 曲线。"""
        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }
        for epoch in range(1, epochs + 1):
            tr = self._run_epoch(train_loader, train=True, desc=f"epoch {epoch}/{epochs} train")
            history["train_loss"].append(tr.loss)
            history["train_acc"].append(tr.acc)
            msg = f"epoch {epoch}/{epochs} | train loss={tr.loss:.4f} acc={tr.acc:.4f}"
            if val_loader is not None:
                vl = self._run_epoch(val_loader, train=False, desc=f"epoch {epoch}/{epochs} val  ")
                history["val_loss"].append(vl.loss)
                history["val_acc"].append(vl.acc)
                msg += f" | val loss={vl.loss:.4f} acc={vl.acc:.4f}"
            logger.info(msg)
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochStats:
        return self._run_epoch(loader, train=False, desc="evaluate")


# ---------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    meta: Dict[str, Any] | None = None,
) -> Path:
    """保存权重与元信息（类别列表、模型配置等）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)
    return path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """加载权重并返回 meta。

    我们信任自己保存的 checkpoint，显式 `weights_only=False` 以便
    兼容 PyTorch 2.6+ 的新默认值（否则 meta 里的复杂对象会被拒绝）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n请先运行训练，或检查 config.yaml 中 training.checkpoint_path。"
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})


# ---------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """批量推理并返回 softmax 概率矩阵。

    Args:
        X: `(N, win)` 或 `(N, C, win)` 数组。
    """
    model.eval()
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"X must be (N, win) or (N, C, win), got {X.shape}")
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    probs_list = []
    for s in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[s : s + batch_size]).float().to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


# ---------------------------------------------------------------------
# 评估报告
# ---------------------------------------------------------------------

@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(yb.numpy())
    if not ys:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(ys), np.concatenate(preds)


def classification_report_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> str:
    """在 loader 上跑一遍推理，打印每类 precision/recall/F1 + 混淆矩阵。"""
    y_true, y_pred = _collect_predictions(model, loader, device)
    if y_true.size == 0:
        return "(empty loader, no report)"
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_lines = ["confusion matrix (rows=true, cols=pred):"]
    header = "        " + " ".join(f"{c:>7}" for c in class_names)
    cm_lines.append(header)
    for name, row in zip(class_names, cm):
        cm_lines.append(f"{name:>7} " + " ".join(f"{v:>7d}" for v in row))
    return report + "\n" + "\n".join(cm_lines)
