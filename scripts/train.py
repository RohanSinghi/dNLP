import torch
import yaml
from src.data.dataset import TextDataset
from src.data.vocabulary import Vocabulary
from src.models.model import get_model
from src.training.trainer import Trainer
import pickle

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load vocabulary
    with open(config['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)

    # Create datasets
    train_dataset = TextDataset(config['dataset_path'], vocab)
    val_dataset = TextDataset(config['validation_dataset_path'], vocab)

    # Get model
    model = get_model(config['model_name'], len(vocab), config['embedding_dim'], config['hidden_dim'])

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)

    # Train the model
    trainer.train(config['num_epochs'])

if __name__ == '__main__':
    main()
