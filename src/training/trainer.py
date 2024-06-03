import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader, epoch_num: int):
        self.model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch_num}')):
            # Move data to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(**batch)

            # Calculate loss
            loss = self.criterion(outputs, batch['labels'])

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int, save_path: str = None):
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_dataloader, epoch)
            val_loss = self.evaluate(val_dataloader)

            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('Saving model...')
                if save_path:
                    self.model.save(save_path)


    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
                # Move data to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                # Calculate loss
                loss = self.criterion(outputs, batch['labels'])

                total_loss += loss.item()

        return total_loss / len(dataloader)