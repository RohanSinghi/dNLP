```python
# src/models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use lstm_out[:, -1, :] to get the last time step's output
        output = self.fc(lstm_out[:, -1, :])
        return output

    def predict(self, x):
        # A separate predict method for inference
        self.eval()
        with torch.no_grad():
           embedded = self.embedding(x)
           lstm_out, _ = self.lstm(embedded)
           output = self.fc(lstm_out[:, -1, :])
           probabilities = F.softmax(output, dim=1)
        return probabilities
```