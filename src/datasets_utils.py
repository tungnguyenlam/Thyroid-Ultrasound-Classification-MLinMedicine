from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((500, 700)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1), shear=15),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.08))
])

val_transform = transforms.Compose([
    transforms.Resize((500, 700)),
])

class ThyroidUltrasoundDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transformer = transform if transform is not None else val_transform
        self.true_count = sum(labels)
        self.false_count = len(labels) - sum(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = decode_image(self.image_paths[idx], mode=ImageReadMode.GRAY)
        image = self.transformer(image).float() / 255.0
        label = self.labels[idx]

        return image, label

def get_datasets():
    from utils import get_dataset_path
    import os
    from sklearn.model_selection import train_test_split

    path = get_dataset_path()
    malignant_path = os.path.join(path, 'malignant')
    benign_path = os.path.join(path, 'benign')

    malignant_img_path_list = [os.path.join(malignant_path, name) for name in os.listdir(malignant_path)]
    benign_img_path_list = [os.path.join(benign_path, name) for name in os.listdir(benign_path)]

    all_path_list = []
    all_label_list = []

    for path in malignant_img_path_list:
        all_path_list.append(path)
        all_label_list.append(True)

    for path in benign_img_path_list:
        all_path_list.append(path)
        all_label_list.append(False)

    train_paths, test_paths, train_labels, test_labels = train_test_split(all_path_list, all_label_list, test_size = 0.2, random_state = 42, stratify=all_label_list)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size = 0.125, random_state = 42, stratify=train_labels)

    train_dataset = ThyroidUltrasoundDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ThyroidUltrasoundDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = ThyroidUltrasoundDataset(test_paths, test_labels, transform=val_transform)

    return train_dataset, val_dataset, test_dataset

    





    