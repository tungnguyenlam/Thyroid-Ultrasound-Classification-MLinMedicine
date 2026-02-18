import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import pandas as pd
from simple_cnn import Simple2DConvNN
from datasets_utils import get_datasets
from utils import get_device, get_previous_epoch_count, delete_previous_run
from train_utils import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--delete-pre", action="store_true")
    args = parser.parse_args()

    num_epochs = args.epochs
    train_batch_size = 8
    output_dir = "output/model_cnn"
    model_dir = "model/model_cnn"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if args.delete_pre:
        delete_previous_run(output_dir, model_dir)

    prev_epochs = get_previous_epoch_count(output_dir, model_dir)
    if prev_epochs > 0 and not args.delete_pre:
        if num_epochs <= prev_epochs:
            print(f"Already trained {prev_epochs} epoch(s). Requested {num_epochs}. Nothing to do.")
            return
        print(f"Resuming from epoch {prev_epochs + 1} to {num_epochs}.")
    else:
        print(f"Starting fresh training for {num_epochs} epoch(s).")

    device = get_device()
    print(f"Device: {device}")

    print("Loading datasets...")
    train_dataset, val_dataset, _ = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size * 2, shuffle=False, num_workers=2, prefetch_factor=2)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    model = Simple2DConvNN().to(device)
    if prev_epochs > 0 and not args.delete_pre:
        resume_path = os.path.join(model_dir, f"epoch{prev_epochs}.pt")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Loaded weights from {resume_path}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    val_loss_file = os.path.join(output_dir, "val_loss.csv")
    if prev_epochs > 0 and not args.delete_pre and os.path.exists(val_loss_file):
        val_metrics = pd.read_csv(val_loss_file).to_dict("records")
        best_val_recall = max(r["recall"] for r in val_metrics)
    else:
        val_metrics = []
        best_val_recall = -1.0

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        start_epoch=prev_epochs,
        output_dir=output_dir,
        model_dir=model_dir,
        val_metrics=val_metrics,
        best_val_recall=best_val_recall,
        title_prefix="SimpleCNN val | ",
    )


if __name__ == "__main__":
    main()
