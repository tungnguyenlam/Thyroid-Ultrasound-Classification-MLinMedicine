import os
import glob


def get_dataset_path():
    import kagglehub

    path = kagglehub.dataset_download("sowmyaabirami/thyroid-ultrasound-dataset")
    return path


def get_device():
    import torch

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def get_previous_epoch_count(output_dir: str, model_dir: str) -> int:
    epoch_csvs = glob.glob(os.path.join(output_dir, "epoch*.csv"))
    epoch_pts = glob.glob(os.path.join(model_dir, "epoch*.pt"))
    counts = []
    for f in epoch_csvs + epoch_pts:
        base = os.path.basename(f)
        try:
            n = int(base.replace("epoch", "").split(".")[0])
            counts.append(n)
        except ValueError:
            pass
    return max(counts) if counts else 0


def delete_previous_run(output_dir: str, model_dir: str) -> None:
    for f in glob.glob(os.path.join(output_dir, "epoch*.csv")):
        os.remove(f)
    val_loss_file = os.path.join(output_dir, "val_loss.csv")
    if os.path.exists(val_loss_file):
        os.remove(val_loss_file)
    for f in glob.glob(os.path.join(model_dir, "epoch*.pt")):
        os.remove(f)
    best_model = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(best_model):
        os.remove(best_model)
    print("Deleted previous run files.")
