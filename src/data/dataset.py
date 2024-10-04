import torch
from torch.utils.data import Dataset
import os

class TextDataset(Dataset):
    def __init__(self, data_path, text_preprocessing_func=None):
        self.data_path = data_path
        self.texts, self.labels = self.load_data()
        self.text_preprocessing_func = text_preprocessing_func
        if self.text_preprocessing_func:
            self.texts = [self.text_preprocessing_func(text) for text in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

    def load_data(self):
        texts = []
        labels = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming each line is "text\tlabel"
                try:
                    text, label = line.strip().split('\t')
                    texts.append(text)
                    labels.append(int(label))
                except ValueError:
                    print(f"Skipping line: {line.strip()} due to format error.")
                    continue

        return texts, labels


if __name__ == '__main__':
    # Create a dummy dataset file for testing purposes.
    dummy_data = ["This is a positive example.\t1", "This is a negative example.\t0", "Another positive example.\t1"]
    dummy_file_path = 'dummy_dataset.txt'
    with open(dummy_file_path, 'w') as f:
        f.write('\n'.join(dummy_data))

    # Example usage
    dataset = TextDataset(dummy_file_path)
    print(f"Dataset size: {len(dataset)}")
    text, label = dataset[0]
    print(f"First example: Text: {text}, Label: {label}")

    # Cleanup the dummy file
    os.remove(dummy_file_path)