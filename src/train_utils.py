import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from test_utils import run_test, plot_test


def train_epoch(model, loader, criterion, optimizer, device, epoch: int, num_epochs: int, output_dir: str) -> float:
    model.train()
    total_loss = 0.0
    history = []

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        history.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    pd.DataFrame(history, columns=["loss"]).to_csv(
        os.path.join(output_dir, f"epoch{epoch + 1}.csv"), index=False
    )
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device, epoch: int, num_epochs: int) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", unit="batch", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            total_loss += criterion(outputs, labels).item()
            all_preds.append((torch.sigmoid(outputs) >= 0.5).float())
            all_labels.append(labels)

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    tp = ((preds_cat == 1) & (labels_cat == 1)).sum().item()
    tn = ((preds_cat == 0) & (labels_cat == 0)).sum().item()
    fp = ((preds_cat == 1) & (labels_cat == 0)).sum().item()
    fn = ((preds_cat == 0) & (labels_cat == 1)).sum().item()
    n = len(preds_cat)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "val_loss": total_loss / len(loader),
        "acc": (tp + tn) / n if n else 0.0,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def run_training(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset,
    device,
    criterion,
    optimizer,
    num_epochs: int,
    start_epoch: int,
    output_dir: str,
    model_dir: str,
    val_metrics: list,
    best_val_f1: float,
    title_prefix: str,
    phase_switch_at: int = None,
    on_phase_switch=None,
) -> None:
    best_model_path = None
    current_optimizer = optimizer

    for epoch in range(start_epoch, num_epochs):
        if phase_switch_at is not None and on_phase_switch is not None and epoch == phase_switch_at:
            current_optimizer = on_phase_switch(model)
            print(f"Phase switch at epoch {epoch + 1}: backbone unfrozen with differential LRs.")

        avg_train_loss = train_epoch(model, train_loader, criterion, current_optimizer, device, epoch, num_epochs, output_dir)
        metrics = validate_epoch(model, val_loader, criterion, device, epoch, num_epochs)
        val_metrics.append(metrics)

        epoch_model_path = os.path.join(model_dir, f"epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved to {epoch_model_path}")

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val F1: {best_val_f1:.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | Acc: {metrics['acc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}"
        )

    pd.DataFrame(val_metrics).to_csv(os.path.join(output_dir, "val_loss.csv"), index=False)
    print("Training complete. Logs saved to", output_dir)
    if best_model_path:
        print(f"Best model (this run) saved at {best_model_path} with val F1: {best_val_f1:.4f}")
    else:
        print(f"No improvement over previous best F1 ({best_val_f1:.4f}) in this run.")

    weights_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, f"epoch{num_epochs}.pt")
    print(f"\nRunning post-training evaluation on validation set using {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    labels, preds, probs = run_test(model, val_dataset, device)
    plot_test(labels, preds, probs, output_dir, title_prefix=title_prefix)
