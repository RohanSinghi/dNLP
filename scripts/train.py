import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import DecaDataset # or whatever your dataset class is called
from src.models.model import SimpleModel # or whatever your model class is called
from src.training.trainer import Trainer

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_dataset = DecaDataset('train') # Replace with your actual data loading logic
val_dataset = DecaDataset('val') # Replace with your actual data loading logic
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = SimpleModel(input_size=10, hidden_size=20, output_size=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Initialize trainer
trainer = Trainer(model, optimizer, criterion, device, validation_data=val_loader)

# Train model
trainer.train(train_loader, num_epochs)