def get_dataset_path():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sowmyaabirami/thyroid-ultrasound-dataset")
    return path

def get_device():
    import torch
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return device

