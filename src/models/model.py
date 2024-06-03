import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def save(self, path):
        """Saves the model's state dictionary to the given path.

        Args:
            path (str): The path where the model should be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    @classmethod
    def load(cls, path, *args, **kwargs):
        """Loads a model from the given path.

        Args:
            path (str): The path from where the model should be loaded.
            *args: Arguments passed to the model's constructor.
            **kwargs: Keyword arguments passed to the model's constructor.

        Returns:
            BaseModel: A new instance of the model with the loaded state dictionary.
        """
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')

        return model