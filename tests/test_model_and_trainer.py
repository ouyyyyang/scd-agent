import numpy as np
import torch
from torch.utils.data import DataLoader

from scd_agent.data.loader import BeatDataset
from scd_agent.models.cnn_lstm import CNN_LSTM
from scd_agent.training.trainer import (
    Trainer,
    classification_report_from_loader,
    compute_class_weights,
    load_checkpoint,
    predict_proba,
    resolve_device,
    save_checkpoint,
)


def test_model_forward_shape():
    model = CNN_LSTM(in_channels=1, num_classes=5, conv_channels=(8, 16), kernel_size=5, lstm_hidden=8, lstm_layers=1)
    x = torch.randn(3, 1, 360)
    out = model(x)
    assert out.shape == (3, 5)


def test_compute_class_weights_inverse_freq():
    y = np.array([0, 0, 0, 0, 1, 2])  # N:4, S:1, V:1
    w = compute_class_weights(y, num_classes=3, scheme="inverse_freq")
    assert w.shape == (3,)
    # 频率越低，权重越大
    assert w[1] > w[0] and w[2] > w[0]
    # 未出现的类走 "1" 兜底，不应该报错
    w2 = compute_class_weights(y, num_classes=5, scheme="inverse_freq")
    assert w2.shape == (5,)

    w_none = compute_class_weights(y, num_classes=3, scheme="none")
    assert np.all(w_none == 1.0)


def test_trainer_one_epoch_and_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 360)).astype(np.float32)
    y = rng.integers(0, 2, size=16).astype(np.int64)
    ds = BeatDataset(X, y, channels=1)
    loader = DataLoader(ds, batch_size=4)
    model = CNN_LSTM(in_channels=1, num_classes=2, conv_channels=(8, 16), kernel_size=5, lstm_hidden=8, lstm_layers=1)

    device = resolve_device("cpu")
    trainer = Trainer(model, device=device, lr=1e-3, class_weights=np.array([1.0, 1.5], dtype=np.float32))
    history = trainer.fit(loader, loader, epochs=1)
    assert len(history["train_loss"]) == 1 and len(history["val_loss"]) == 1

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(model, ckpt, meta={"classes": ["A", "B"]})

    model2 = CNN_LSTM(in_channels=1, num_classes=2, conv_channels=(8, 16), kernel_size=5, lstm_hidden=8, lstm_layers=1)
    meta = load_checkpoint(model2, ckpt)
    assert meta["classes"] == ["A", "B"]

    probs = predict_proba(model2, X, device=device)
    assert probs.shape == (16, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    report = classification_report_from_loader(model2, loader, device=device, class_names=["A", "B"])
    assert "precision" in report and "confusion matrix" in report


def test_beat_dataset_shape_mismatch_errors():
    X = np.zeros((4, 360), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    # 隐式变成 (4,1,360)，channel=2 应报错（不做静默广播）
    import pytest

    with pytest.raises(ValueError):
        BeatDataset(X, y, channels=2)
    with pytest.raises(ValueError):
        BeatDataset(X, np.zeros(3, dtype=np.int64), channels=1)
