import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path

LABEL_NAMES = ['label1', 'label2', 'label3']  # Replace with your actual label names

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.data = []
        self.dataset_path = dataset_path
        self.to_tensor = ToTensor()

        for fname in Path(dataset_path).glob('*.png'):  # Assuming images are in .png format
            label = fname.stem.split('_')[-1]  # Assuming label is part of the filename
            if label in LABEL_NAMES:
                image = Image.open(fname)
                label_id = LABEL_NAMES.index(label)
                self.data.append((self.to_tensor(image), label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path: str, num_workers: int = 0, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return torch.tensor(accuracy)