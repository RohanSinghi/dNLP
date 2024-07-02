import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, criterion, device, validation_data=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.validation_data = validation_data

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            # Move data to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(batch)

            # Calculate loss
            loss = self.criterion(outputs, batch['labels'])

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)

                # Calculate loss
                loss = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train(self, train_loader, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

            if self.validation_data:
                val_loss = self.validate(self.validation_data)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print('Saving model...')
                    torch.save(self.model.state_dict(), 'model.pth') # Basic model saving
            else:
                print('No validation data provided.')