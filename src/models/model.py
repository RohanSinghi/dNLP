import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output


class BiggerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.dropout(output)
        output = self.fc(output)
        return output



def get_model(model_name, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
    if model_name == 'BaseModel':
        return BaseModel(vocab_size, embedding_dim, hidden_dim)
    elif model_name == 'BiggerModel':
        return BiggerModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
