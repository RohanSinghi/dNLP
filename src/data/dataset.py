import torch
from torch.utils.data import Dataset

class DecaDataset(Dataset):
    def __init__(self, split):
        # Replace with actual data loading logic
        self.data = [{'input_ids': torch.randn(10), 'labels': torch.randint(0, 5, (1,)).item()} for _ in range(100)] # Dummy data
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
