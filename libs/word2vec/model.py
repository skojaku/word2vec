from torch import nn
from .dataset import Word2VecDataset
import torch


class Word2Vec(nn.Module):
    def __init__(self, window=5, vector_size=32, epochs=10):
        super().__init__()
        self.window = window
        self.vector_size = vector_size
        self.epochs = epochs

        self.in_vec = None
        self.in_out = None

    def fit(self, input_text):
        dataset = Word2VecDataset(input_text, window=self.window)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Get the list of unique words
        self.vocab = dataset.get_vocab()

        # word2int
        self.word2int = {word: i for i, word in enumerate(self.vocab)}

        # Initialize the input and output vectors
        self.in_vec = nn.Embedding(len(self.vocab), self.vector_size)
        self.out_vec = nn.Embedding(len(self.vocab), self.vector_size)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_function = nn.BCELoss()

        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                optimizer.zero_grad()

                i_vec = self.forward(inputs)
                o_vec = self.forward_out(targets)

                predictions = torch.sum(i_vec * o_vec, dim=1).squeeze()

                loss = loss_function(predictions, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader)}")
        return self

    def transform(self):
        emb = self.forward(self.vocab)
        return emb

    def forward(self, word):
        # Convert to one-hot encoidng
        i = self.word2int[word]
        return self.in_vec(i)

    def forward_out(self, word):
        # Convert to one-hot encoidng
        # Convert to one-hot encoidng
        i = self.word2int[word]
        return self.out_vec(i)

print("model.py has been loaded")