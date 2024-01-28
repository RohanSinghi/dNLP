import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = config['batch_size']
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            inputs = batch['text'].to(self.device)
            targets = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['text'].to(self.device)
                targets = batch['label'].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('Saved best model')

