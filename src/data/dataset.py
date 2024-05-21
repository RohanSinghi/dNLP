```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from src.data.fields import TextField, LabelField
from src.utils.utils import normalize_text  # Import normalize_text from utils

class DecaDataset(Dataset):
    def __init__(self, data, fields, preprocess=True):
        self.data = data
        self.fields = fields
        self.preprocess = preprocess
        self.examples = self._create_examples(data)

    def _create_examples(self, data):
        examples = []
        for item in data:
            # Use the mapping from string field names to Field objects
            example = {field_name: fields[field_name].process(item[field_name]) for field_name in fields}
            examples.append(example)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def preprocess_text(self, text):
        if self.preprocess:
            return normalize_text(text)  # Use normalize_text function
        return text

    @staticmethod
    def collate_fn(batch):
        # Implements the collation function, and handles padding.
        collated_batch = {}
        for field_name in batch[0]:
          field_values = [item[field_name] for item in batch]
          if isinstance(field_values[0], torch.Tensor):
            collated_batch[field_name] = torch.stack(field_values)
          else:
            collated_batch[field_name] = field_values
        return collated_batch
```