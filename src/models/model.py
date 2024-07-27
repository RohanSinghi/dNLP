import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: [batch size, seq len] ; Assuming text is already tokenized and indexed
        embedded = self.embedding(text)
        # embedded: [batch size, seq len, embedding dim]
        embedded = embedded.mean(dim=1) #Average across sequence length
        # embedded: [batch size, embedding dim]

        hidden = self.fc1(embedded)
        # hidden: [batch size, hidden dim]
        hidden = self.relu(hidden)

        output = self.fc2(hidden)
        # output: [batch size, output dim]

        return output

if __name__ == '__main__':
    # Example Usage
    vocab_size = 10000  # Example vocabulary size
    embedding_dim = 100
    hidden_dim = 200
    output_dim = 2  # Binary classification

    model = FeedForwardNN(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Create some dummy input data
    batch_size = 32
    seq_len = 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Pass the input through the model
    output = model(dummy_input)

    # Print the output shape
    print("Output shape:", output.shape)
