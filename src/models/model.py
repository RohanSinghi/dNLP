'''Module for defining the base model class.'''

import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models."

    def __init__(self):
        """Initializes the BaseModel.
        """
        super().__init__()

    def forward(self, x):
        """Forward pass of the model.

        :param x: The input tensor.
        :type x: torch.Tensor
        :raises NotImplementedError: If the forward method is not implemented in the subclass.
        """
        raise NotImplementedError