"""
Training loop for HAM10000 binary classifier.
Returns per-epoch metrics and saves the best checkpoint by val F1 (melanoma).
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    epochs: int = 15,
    lr: float = 1e-4,
    class_weights: torch.Tensor = None,
) -> list[dict]:
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / total
        train_loss /= total

        # --- val ---
        val_metrics = _evaluate_loop(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": round(scheduler.get_last_lr()[0], 6),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1_mel={val_metrics['f1_mel']:.3f} "
            f"val_auc={val_metrics['auc']:.3f} | {elapsed:.0f}s"
        )

        if val_metrics["f1_mel"] >= best_f1:
            best_f1 = val_metrics["f1_mel"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def _evaluate_loop(model, loader, criterion, device) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score, recall_score

    model.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss_sum += loss.item() * len(labels)
            total += len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return {
        "loss":    round(loss_sum / total, 4),
        "f1_mel":  round(f1_score(all_labels, all_preds, pos_label=1, zero_division=0), 4),
        "recall_mel": round(recall_score(all_labels, all_preds, pos_label=1, zero_division=0), 4),
        "auc":     round(roc_auc_score(all_labels, all_probs), 4),
    }
