import sys
import os

# Add parent directory to sys.path to import modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simple_cnn import Simple2DConvNN
import torch
from torch.utils.data import DataLoader
from datasets_utils import get_datasets
from utils import get_device
import pandas as pd
from tqdm import tqdm


def main():
    # Configuration
    num_epochs = 20
    train_batch_size = 8
    output_dir = "output/model_cnn"
    model_dir = "model/model_cnn"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Device
    device = get_device()
    print(f"Device: {device}")

    # Data
    print("Loading datasets...")
    train_dataset, val_dataset, _ = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size * 2, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Model, Loss, Optimizer
    model = Simple2DConvNN().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    val_losses = []
    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        train_loss_history = []
        total_train_loss = 0

        # Tqdm progress bar for training loop
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch"
        )

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_train_loss += loss_val
            train_loss_history.append(loss_val)

            # Update progress bar
            train_pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Calculate average train loss
        avg_train_loss = total_train_loss / len(train_loader)

        # Save epoch loss history
        epoch_loss_df = pd.DataFrame(train_loss_history, columns=["loss"])
        epoch_loss_df.to_csv(
            os.path.join(output_dir, f"epoch{epoch + 1}.csv"), index=False
        )

        # Validation
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
            unit="batch",
            leave=False,
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save model after each epoch
        epoch_model_path = os.path.join(model_dir, f"epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved to {epoch_model_path}")

        # Check if this is the best model based on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

        # Log summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    # Save validation loss history
    val_loss_df = pd.DataFrame(val_losses, columns=["val_loss"])
    val_loss_df.to_csv(os.path.join(output_dir, "val_loss.csv"), index=False)
    print("Training complete. Logs saved to", output_dir)
    print(f"Best model saved at {best_model_path} with val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
