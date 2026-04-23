"""
Final evaluation on the test set. Loads the best checkpoint and produces
a full classification report, confusion matrix, and ROC curve.
Saves results to the experiment run directory.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader


CLASS_NAMES = ["nv", "mel"]


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    checkpoint: str = "best_model.pt",
) -> dict:
    ckpt_path = run_dir / checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
    )
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        "auc":        round(auc, 4),
        "f1_mel":     round(report["mel"]["f1-score"], 4),
        "recall_mel": round(report["mel"]["recall"], 4),
        "precision_mel": round(report["mel"]["precision"], 4),
        "f1_nv":      round(report["nv"]["f1-score"], 4),
        "accuracy":   round(report["accuracy"], 4),
    }

    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2)
    )

    _plot_confusion_matrix(all_labels, all_preds, run_dir)
    _plot_roc_curve(all_labels, all_probs, auc, run_dir)

    return metrics


def _plot_confusion_matrix(labels, preds, run_dir: Path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc_curve(labels, probs, auc: float, run_dir: Path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Melanoma vs Nevus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
