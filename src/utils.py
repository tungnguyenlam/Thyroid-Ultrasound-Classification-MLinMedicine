def get_dataset_path():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sowmyaabirami/thyroid-ultrasound-dataset")
    return path
