import torch
from src.data.dataset import NLPDataset
from src.models.model import SimpleModel
from src.training.trainer import train, evaluate
from src.utils.utils import get_logger, load_config

logger = get_logger(__name__)

def main():
    config = load_config('config.yaml')
    logger.info(f'Loaded config: {config}')

    # Example data (replace with your actual data loading)
    data = [{'text': 'This is a positive review', 'label': 1}, {'text': 'This is a negative review', 'label': 0}]

    # Example tokenizer (replace with your actual tokenizer)
    tokenizer = lambda x: x.split()

    dataset = NLPDataset(data, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])

    model = SimpleModel(vocab_size=1000, embedding_dim=100, hidden_dim=200, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        train_loss, train_acc = train(model, dataloader, optimizer, 'cpu')
        logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

if __name__ == '__main__':
    main()