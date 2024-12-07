import torch
from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoded_text['input_ids'].flatten(), 'attention_mask': encoded_text['attention_mask'].flatten(), 'label': torch.tensor(label)}