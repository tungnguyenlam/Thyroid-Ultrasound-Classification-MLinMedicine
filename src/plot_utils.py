import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def plot_losses(output_dir: str) -> None:
    epoch_files = glob.glob(os.path.join(output_dir, "epoch*.csv"))
    epoch_files.sort(key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.csv')[0]))

    if not epoch_files:
        print(f"No epoch files found in {output_dir}")
        return

    plt.figure(figsize=(12, 10))
    all_train_losses = []
    for file in epoch_files:
        df = pd.read_csv(file)
        all_train_losses.extend(df['loss'].tolist())

    plt.subplot(2, 2, 1)
    plt.plot(all_train_losses, label='Training Loss')
    plt.title('Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    val_loss_file = os.path.join(output_dir, "val_loss.csv")
    if os.path.exists(val_loss_file):
        val_df = pd.read_csv(val_loss_file)
        n_val = len(val_df)
        epochs = range(1, n_val + 1)

        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_df['val_loss'], label='Val Loss', color='orange')
        plt.title('Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        if 'acc' in val_df.columns:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, val_df['acc'], label='Accuracy', color='green')
            plt.title('Validation Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(epochs, val_df['recall'], label='Recall (malignant)', color='red')
            plt.plot(epochs, val_df['precision'], label='Precision (malignant)', color='blue')
            plt.title('Recall & Precision per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
    else:
        print(f"No validation loss file found: {val_loss_file}")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
