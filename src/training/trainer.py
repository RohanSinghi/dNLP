import torch
import torch.nn as nn
from tqdm import tqdm
from src.evaluation.evaluate import evaluate_and_log  # Import the evaluation function

class Trainer:
    def __init__(self, model, optimizer, train_dataloader, val_dataloader, device, logger, epochs):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.logger = logger
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        step = 0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for batch in tqdm(self.train_dataloader, desc="Training"):
                self.optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'target'}
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.logger.log({"train/loss": loss.item()}, step=step)
                # Log the loss

                step += 1

            # Evaluate after each epoch
            evaluate_and_log(self.model, self.val_dataloader, self.device, self.logger, step=step)
            self.model.train()
