import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import SimpleDataset  # Replace with your actual Dataset class
from src.models.model import BaseModel  # Replace with your actual Model class
from src.training.trainer import Trainer
from src.utils.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data.')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to the validation data.')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/model.pth', help='Path to save the trained model.')
    parser.add_argument('--model_load_path', type=str, default=None, help='Path to load a pretrained model.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')

    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    train_dataset = SimpleDataset(args.train_data_path)  # Replace with your actual data loading logic
    val_dataset = SimpleDataset(args.val_data_path)  # Replace with your actual data loading logic
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model, optimizer, and criterion
    model = BaseModel()  # Replace with your actual model initialization
    if args.model_load_path:
      model = BaseModel.load(args.model_load_path)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    trainer = Trainer(model, optimizer, criterion, args.device)
    trainer.train(train_dataloader, val_dataloader, args.num_epochs, args.model_save_path)

if __name__ == '__main__':
    main()