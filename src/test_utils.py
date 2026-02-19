import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
)


def run_test(model, dataset, device, batch_size: int = 16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels, dtype=int)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    return all_labels, all_preds, all_probs


def plot_test(labels, preds, probs, output_dir: str, title_prefix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # --- Confusion matrix ---
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"], cmap = plt.cm.Blues)
    disp.plot(ax=axes[0], colorbar=True)
    axes[0].set_title(f"{title_prefix}Confusion Matrix")

    # --- ROC / AUC ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="navy", linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{title_prefix}ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "test_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Test plot saved to {plot_path}")

    # --- Text report ---
    report = classification_report(labels, preds, target_names=["Benign", "Malignant"])
    print(report)
    report_path = os.path.join(output_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
