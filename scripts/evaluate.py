import torch
import argparse
from torch.utils.data import DataLoader
from src.data.dataset import DecaNLPDataloader  # Assuming you have or will implement this.
from src.utils.utils import load_model
from src.evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation (cuda or cpu).')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)
    model.to(args.device)

    # Load the test dataset
    test_dataset = DecaNLPDataloader(args.test_data_path, split='test')  # TODO: Adapt to your dataset implementation
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # no shuffling in evaluation!
    
    # Evaluate the model
    accuracy = evaluate(model, test_dataloader, args.device)

    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
