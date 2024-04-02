'''Module for training models.
'''
import torch
import torch.optim as optim

class Trainer:
    """A class for training models.

    Attributes:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (torch.nn.Module): The loss function.
    """
    def __init__(self, model, learning_rate):
        """Initializes a new Trainer instance.

        :param model: The model to train.
        :type model: torch.nn.Module
        :param learning_rate: The learning rate.
        :type learning_rate: float
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, data_loader):
        """Trains the model for one epoch.

        :param data_loader: The data loader.
        :type data_loader: torch.utils.data.DataLoader
        """
        self.model.train()
        for inputs, labels in data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()