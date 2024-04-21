import torch
from torch.utils.data import DataLoader
from src.utils.utils import load_model
from tqdm import tqdm

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
            targets = batch['target'].to(device)
            
            outputs = model(inputs)

            predicted = torch.argmax(outputs, dim=1)

            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def evaluate_and_log(model, dataloader, device, logger, step):
    accuracy = evaluate(model, dataloader, device)
    logger.log({"eval/accuracy": accuracy}, step=step)
    print(f"Evaluation Accuracy: {accuracy:.4f}")
